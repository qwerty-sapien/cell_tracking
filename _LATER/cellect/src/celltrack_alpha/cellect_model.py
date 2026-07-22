"""Clean-room CELLECT network components.

The implementation follows the architecture published in Nature Methods and
its supplementary figure.  The released numeric checkpoint uses an
eight-channel CEN input (three confidence logits, their softmax, and two image
frames), whereas Supplementary Table 3 describes six channels.  The released
checkpoint is the compatibility contract here because cross-modality transfer
depends on loading those weights exactly; :data:`PUBLICATION_DISCREPANCY`
records the distinction explicitly.

This module contains no loader for, or import from, the GPL reference source.
Only ordinary PyTorch layers and numeric ``state_dict`` checkpoints are used.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


EMBEDDING_CHANNELS = 64
CHECKPOINT_OUTPUT_CHANNELS = 6
CHECKPOINT_CEN_INPUT_CHANNELS = 8
PUBLISHED_CHECKPOINT_PARAMETERS = 2_541_972
PUBLISHED_CHECKPOINT_SHA256 = (
    "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"
)
PUBLISHED_CHECKPOINT_FILENAME = "U-ext+-x3rd-149.0-4.6540.pth"
PUBLICATION_DISCREPANCY = {
    "supplementary_table_3": {
        "main_confidence_channels": 4,
        "cen_input_channels": 6,
    },
    "released_checkpoint": {
        "main_confidence_channels": 3,
        "division_channels": 2,
        "radius_channels": 1,
        "cen_input_channels": 8,
    },
    "primary_contract": "released-checkpoint",
}


class CellectModelError(RuntimeError):
    """Raised when a model or numeric checkpoint violates the frozen contract."""


class CellectOutputs(NamedTuple):
    """Five maps returned for the first frame of a two-frame input."""

    confidence: torch.Tensor
    enhanced_confidence: torch.Tensor
    embedding: torch.Tensor
    division: torch.Tensor
    radius: torch.Tensor


class InterframeMatcherOutputs(NamedTuple):
    """Association scores and the two-class division prediction."""

    association_logits: torch.Tensor
    division_logits: torch.Tensor


class _NormActConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm3d(channels)
        self.activation = nn.LeakyReLU()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(self.activation(self.norm(value)))


class _ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(value)))


class _UpsampleConv(nn.Module):
    """Anisotropic decoder step: preserve Z and double both lateral axes."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.input_norm = nn.InstanceNorm3d(in_channels)
        self.activation = nn.LeakyReLU()
        self.upsample = nn.Upsample(scale_factor=(2, 2, 1), mode="nearest")
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.output_norm = nn.InstanceNorm3d(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.activation(self.input_norm(value))
        value = self.upsample(value)
        value = self.conv(value)
        return self.activation(self.output_norm(value))


class _Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.conv_mid = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.conv_out = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout3d(p=0.6)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.conv_in(value)
        value = self.conv_mid(self.activation(residual))
        value = self.conv_out(self.activation(self.dropout(value)))
        value = value + residual
        context = self.activation(value)
        return self.activation(self.norm(value)), context


class _MainDownBlock(nn.Module):
    """Stride-convolution residual block used by the primary backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_output = normalize_output
        stride = (2, 2, 1)
        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.refine = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.skip = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.transform = _NormActConv(out_channels)
        self.dropout = nn.Dropout3d(p=0.6)
        self.output_norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.refine(self.down(value)) + self.skip(value)
        value = self.transform(residual)
        value = self.transform(self.dropout(value))
        value = value + residual
        return self.activation(self.output_norm(value)) if self.normalize_output else value


class _CenterDownBlock(nn.Module):
    """Smaller stride-convolution residual block used by the CEN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_output = normalize_output
        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(2, 2, 1),
            padding=1,
            bias=False,
        )
        self.transform = _NormActConv(out_channels)
        self.dropout = nn.Dropout3d(p=0.6)
        self.output_norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.down(value)
        value = self.transform(residual)
        value = self.transform(self.dropout(value))
        value = value + residual
        return self.activation(self.output_norm(value)) if self.normalize_output else value


class _DecoderStage(nn.Module):
    def __init__(
        self,
        fused_channels: int,
        projected_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.fuse = _ConvNormAct(fused_channels, fused_channels)
        self.project = nn.Conv3d(
            fused_channels,
            projected_channels,
            kernel_size=1,
            bias=False,
        )
        self.up = _UpsampleConv(projected_channels, output_channels)

    def forward(self, value: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fuse(torch.cat((value, context), dim=1))
        value = self.up(self.project(fused))
        return value, fused


class _EncoderDecoderCore(nn.Module):
    """Shared U-Net mechanics for the primary model and the CEN."""

    def _decode(
        self,
        value: torch.Tensor,
        contexts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.bottleneck_up(value)
        value = self.activation(self.bottleneck_norm(self.bottleneck_project(value)))
        value, _ = self.decoders[0](value, contexts[3])
        value, coarse = self.decoders[1](value, contexts[2])
        value, fine = self.decoders[2](value, contexts[1])
        return torch.cat((value, contexts[0]), dim=1), coarse, fine

    def _deep_supervision(
        self,
        coarse: torch.Tensor,
        fine: torch.Tensor,
    ) -> torch.Tensor:
        coarse_logits = self.deep_coarse(coarse)
        coarse_logits = F.interpolate(coarse_logits, scale_factor=(2, 2, 1), mode="nearest")
        fine_logits = self.deep_fine(fine)
        combined = coarse_logits + fine_logits
        return F.interpolate(combined, scale_factor=(2, 2, 1), mode="nearest")


class CenterEnhancementNet(_EncoderDecoderCore):
    """Lightweight 3D U-Net that sharpens the center-confidence map."""

    def __init__(self, in_channels: int = CHECKPOINT_CEN_INPUT_CHANNELS) -> None:
        super().__init__()
        channels = (2, 4, 8, 16, 32)
        self.stem = _Stem(in_channels, channels[0])
        self.encoders = nn.ModuleList(
            _CenterDownBlock(
                channels[index],
                channels[index + 1],
                normalize_output=index < 3,
            )
            for index in range(4)
        )
        self.bottleneck_up = _UpsampleConv(channels[4], channels[3])
        self.bottleneck_project = nn.Conv3d(channels[3], channels[3], kernel_size=1, bias=False)
        self.bottleneck_norm = nn.InstanceNorm3d(channels[3])
        self.activation = nn.LeakyReLU()
        self.decoders = nn.ModuleList(
            (
                _DecoderStage(32, 16, 8),
                _DecoderStage(16, 8, 4),
                _DecoderStage(8, 4, 2),
            )
        )
        self.final_fuse = _ConvNormAct(4, 4)
        self.output = nn.Conv3d(4, 5, kernel_size=1, bias=False)
        self.deep_coarse = nn.Conv3d(16, 5, kernel_size=1, bias=False)
        self.deep_fine = nn.Conv3d(8, 5, kernel_size=1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value, context_1 = self.stem(value)
        contexts = [context_1]
        for block in self.encoders[:-1]:
            value = block(value)
            contexts.append(value)
        value = self.encoders[-1](value)
        decoded, coarse, fine = self._decode(value, tuple(contexts))
        direct = self.output(self.final_fuse(decoded))
        return direct + self._deep_supervision(coarse, fine)


class CellectBackbone(_EncoderDecoderCore):
    """Checkpoint-compatible clean-room CELLECT 3D backbone."""

    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        if in_channels != 2:
            raise CellectModelError("CELLECT requires exactly two consecutive input frames")
        channels = (8, 16, 32, 64, 128)
        self.center_enhancer = CenterEnhancementNet()
        self.stem = _Stem(in_channels, channels[0])
        self.encoders = nn.ModuleList(
            _MainDownBlock(
                channels[index],
                channels[index + 1],
                normalize_output=index < 3,
            )
            for index in range(4)
        )
        self.bottleneck_up = _UpsampleConv(channels[4], channels[3])
        self.bottleneck_project = nn.Conv3d(channels[3], channels[3], kernel_size=1, bias=False)
        self.bottleneck_norm = nn.InstanceNorm3d(channels[3])
        self.activation = nn.LeakyReLU()
        self.decoders = nn.ModuleList(
            (
                _DecoderStage(128, 64, 32),
                _DecoderStage(64, 32, 16),
                _DecoderStage(32, 16, 8),
            )
        )
        self.segmentation_fuse = _ConvNormAct(16, 16)
        self.segmentation_out = nn.Conv3d(
            16,
            CHECKPOINT_OUTPUT_CHANNELS,
            kernel_size=1,
            bias=False,
        )
        self.feature_fuse = _ConvNormAct(16, 32)
        self.feature_project = nn.Conv3d(32, 32, kernel_size=1, bias=False)
        self.embedding_fuse = _ConvNormAct(38, 32)
        self.embedding_out = nn.Conv3d(32, EMBEDDING_CHANNELS, kernel_size=1, bias=False)
        self.deep_coarse = nn.Conv3d(64, CHECKPOINT_OUTPUT_CHANNELS, kernel_size=1, bias=False)
        self.deep_fine = nn.Conv3d(32, CHECKPOINT_OUTPUT_CHANNELS, kernel_size=1, bias=False)

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != PUBLISHED_CHECKPOINT_PARAMETERS:
            raise CellectModelError(
                f"clean-room parameter count changed: {parameter_count} != "
                f"{PUBLISHED_CHECKPOINT_PARAMETERS}"
            )

    def forward(self, paired_frames: torch.Tensor) -> CellectOutputs:
        if paired_frames.ndim != 5 or paired_frames.shape[1] != 2:
            raise CellectModelError("input must have shape (B,2,Y,X,Z)")
        if paired_frames.shape[2] % 16 or paired_frames.shape[3] % 16:
            raise CellectModelError("lateral input dimensions must be divisible by 16")
        if not bool(torch.isfinite(paired_frames).all().item()):
            raise CellectModelError("input contains non-finite values")

        value, context_1 = self.stem(paired_frames)
        contexts = [context_1]
        for block in self.encoders[:-1]:
            value = block(value)
            contexts.append(value)
        value = self.encoders[-1](value)
        decoded, coarse, fine = self._decode(value, tuple(contexts))
        deep_logits = self._deep_supervision(coarse, fine)
        segmentation = self.segmentation_out(self.segmentation_fuse(decoded)) + deep_logits

        feature = self.feature_project(self.feature_fuse(decoded))
        embedding = self.embedding_out(
            self.embedding_fuse(torch.cat((feature, deep_logits), dim=1))
        )
        confidence = segmentation[:, :3]
        center_input = torch.cat(
            (confidence, torch.softmax(confidence, dim=1), paired_frames),
            dim=1,
        )
        enhanced = self.center_enhancer(center_input)
        radius = torch.relu(segmentation[:, 5:])
        outputs = CellectOutputs(
            confidence=confidence,
            enhanced_confidence=enhanced,
            embedding=embedding,
            division=segmentation[:, 3:5],
            radius=radius,
        )
        if any(not bool(torch.isfinite(item).all().item()) for item in outputs):
            raise CellectModelError("CELLECT produced non-finite output values")
        return outputs


class CellectMatcher(nn.Module):
    """Five-candidate MLP used for intra- and interframe classification."""

    def __init__(self, output_channels: int) -> None:
        super().__init__()
        if output_channels not in (6, 8):
            raise CellectModelError("matcher output must be 6 (intra) or 8 (inter)")
        width = EMBEDDING_CHANNELS
        self.output_channels = output_channels
        self.context_projection = nn.Linear(width * 5, width)
        self.difference_projection = nn.Linear(width * 5, width)
        self.similarity_projection = nn.Linear(width * 5, width)
        self.spatial_projection = nn.Linear(3 * 5, width)
        self.size_projection = nn.Linear(6 * 5, width)
        self.division_projection = nn.Linear(6 * 5, width)
        self.compatibility_bn_x = nn.BatchNorm1d(width)
        self.compatibility_bn_y = nn.BatchNorm1d(width)
        self.context_norm = nn.BatchNorm1d(width)
        self.difference_norm = nn.BatchNorm1d(width)
        self.similarity_norm = nn.BatchNorm1d(width)
        self.spatial_norm = nn.BatchNorm1d(width)
        self.size_norm = nn.BatchNorm1d(width)
        self.division_norm = nn.BatchNorm1d(width)
        self.fused_norm = nn.BatchNorm1d(width * 2)
        self.pair_conv_norm = nn.BatchNorm2d(8)
        self.fused_projection = nn.Linear(width * 6, width * 2)
        self.hidden_projection = nn.Linear(width * 2, width)
        self.output_layer = nn.Linear(width, output_channels)
        self.pair_conv_wide = nn.Conv2d(
            1,
            8,
            kernel_size=(2, 5),
            padding=(0, 2),
            bias=False,
        )
        self.pair_conv_narrow = nn.Conv2d(
            1,
            8,
            kernel_size=(2, 3),
            padding=(0, 1),
            bias=False,
        )
        self.pair_projection = nn.Linear(width * 8 * 2, width)
        # Retained only because both released MLP checkpoints contain this
        # unused numeric branch.
        self.compatibility_output = nn.Linear(width + output_channels, 2)
        self.activation = nn.LeakyReLU()

    @staticmethod
    def _validated_inputs(
        anchor: torch.Tensor,
        candidates: torch.Tensor,
        spatial_offsets: torch.Tensor,
        size_features: torch.Tensor,
        division_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if anchor.ndim == 2:
            anchor = anchor.unsqueeze(1)
        batch = anchor.shape[0]
        expected = {
            "anchor": (batch, 1, EMBEDDING_CHANNELS),
            "candidates": (batch, 5, EMBEDDING_CHANNELS),
            "spatial_offsets": (batch, 5, 3),
            "size_features": (batch, 5, 6),
            "division_features": (batch, 5, 6),
        }
        observed = {
            "anchor": tuple(anchor.shape),
            "candidates": tuple(candidates.shape),
            "spatial_offsets": tuple(spatial_offsets.shape),
            "size_features": tuple(size_features.shape),
            "division_features": tuple(division_features.shape),
        }
        for label, shape in expected.items():
            if observed[label] != shape:
                raise CellectModelError(f"{label} shape {observed[label]} != {shape}")
        values = (anchor, candidates, spatial_offsets, size_features, division_features)
        if any(not bool(value.is_floating_point()) for value in values):
            raise CellectModelError("matcher inputs must be floating-point tensors")
        if any(not bool(torch.isfinite(value).all().item()) for value in values):
            raise CellectModelError("matcher input contains non-finite values")
        return values

    def forward(
        self,
        anchor: torch.Tensor,
        candidates: torch.Tensor,
        spatial_offsets: torch.Tensor,
        size_features: torch.Tensor,
        division_features: torch.Tensor,
    ) -> torch.Tensor | InterframeMatcherOutputs:
        anchor, candidates, spatial_offsets, size_features, division_features = (
            self._validated_inputs(
                anchor,
                candidates,
                spatial_offsets,
                size_features,
                division_features,
            )
        )
        pair_features = []
        for candidate_index in range(5):
            pair = torch.cat((anchor, candidates[:, candidate_index].unsqueeze(1)), dim=1)
            pair = pair.unsqueeze(1)
            wide = self.activation(self.pair_conv_norm(self.pair_conv_wide(pair)))
            narrow = self.activation(self.pair_conv_norm(self.pair_conv_narrow(pair)))
            pair_features.append(
                self.pair_projection(torch.cat((wide, narrow), dim=1).reshape(len(anchor), -1))
            )
        context = torch.cat(pair_features, dim=1)
        difference = (candidates - anchor).reshape(len(anchor), -1)
        anchor_norm = torch.sqrt((anchor * anchor).sum(dim=2))
        candidate_norm = torch.sqrt((candidates * candidates).sum(dim=2))
        normalized_components = (
            anchor * candidates / (anchor_norm * candidate_norm).unsqueeze(-1)
        ).reshape(len(anchor), -1)

        branches = (
            self.activation(self.context_norm(self.context_projection(context))),
            self.activation(self.difference_norm(self.difference_projection(difference))),
            self.activation(
                self.similarity_norm(self.similarity_projection(normalized_components))
            ),
            self.activation(
                self.spatial_norm(self.spatial_projection(spatial_offsets.reshape(len(anchor), -1)))
            ),
            self.activation(
                self.size_norm(self.size_projection(size_features.reshape(len(anchor), -1)))
            ),
            self.activation(
                self.division_norm(
                    self.division_projection(division_features.reshape(len(anchor), -1))
                )
            ),
        )
        fused = torch.cat(branches, dim=1)
        fused = self.fused_norm(self.activation(self.fused_projection(fused)))
        logits = self.output_layer(self.activation(self.hidden_projection(fused)))
        if self.output_channels == 6:
            return logits
        return InterframeMatcherOutputs(logits[:, :6], logits[:, 6:])


def _map(prefix: str, clean_prefix: str, suffix: str = "weight") -> tuple[str, str]:
    return f"{prefix}.{suffix}", f"{clean_prefix}.{suffix}"


def _backbone_weight_map() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []

    def add_unet(legacy_root: str, clean_root: str, *, main: bool) -> None:
        legacy = f"{legacy_root}." if legacy_root else ""
        clean = f"{clean_root}." if clean_root else ""
        pairs.extend(
            (
                _map(f"{legacy}conv3d_c1_1", f"{clean}stem.conv_in"),
                _map(f"{legacy}conv3d_c1_2", f"{clean}stem.conv_mid"),
                _map(f"{legacy}lrelu_conv_c1.1", f"{clean}stem.conv_out"),
            )
        )
        for index, level in enumerate(range(2, 6)):
            pairs.append(_map(f"{legacy}conv3d_c{level}", f"{clean}encoders.{index}.down"))
            if main:
                pairs.extend(
                    (
                        _map(
                            f"{legacy}conv3d_c{level}s1",
                            f"{clean}encoders.{index}.refine",
                        ),
                        _map(
                            f"{legacy}conv3d_c{level}s2",
                            f"{clean}encoders.{index}.skip",
                        ),
                    )
                )
            pairs.append(
                _map(
                    f"{legacy}norm_lrelu_conv_c{level}.2",
                    f"{clean}encoders.{index}.transform.conv",
                )
            )
        pairs.extend(
            (
                _map(
                    f"{legacy}norm_lrelu_upscale_conv_norm_lrelu_l0.3",
                    f"{clean}bottleneck_up.conv",
                ),
                _map(f"{legacy}conv3d_l0", f"{clean}bottleneck_project"),
            )
        )
        for index, level in enumerate(range(1, 4)):
            pairs.extend(
                (
                    _map(
                        f"{legacy}conv_norm_lrelu_l{level}.0",
                        f"{clean}decoders.{index}.fuse.conv",
                    ),
                    _map(f"{legacy}conv3d_l{level}", f"{clean}decoders.{index}.project"),
                    _map(
                        f"{legacy}norm_lrelu_upscale_conv_norm_lrelu_l{level}.3",
                        f"{clean}decoders.{index}.up.conv",
                    ),
                )
            )
        if main:
            pairs.extend(
                (
                    _map(f"{legacy}conv_norm_lrelu_l4.0", f"{clean}segmentation_fuse.conv"),
                    _map(f"{legacy}conv3d_l4", f"{clean}segmentation_out"),
                    _map(f"{legacy}conv_norm_lrelu_f.0", f"{clean}feature_fuse.conv"),
                    _map(f"{legacy}conv3d_f", f"{clean}feature_project"),
                    _map(f"{legacy}conv_norm_lrelu_f1.0", f"{clean}embedding_fuse.conv"),
                    _map(f"{legacy}conv3d_f1", f"{clean}embedding_out"),
                )
            )
        else:
            pairs.extend(
                (
                    _map(f"{legacy}conv_norm_lrelu_l4.0", f"{clean}final_fuse.conv"),
                    _map(f"{legacy}conv3d_l4", f"{clean}output"),
                )
            )
        pairs.extend(
            (
                _map(f"{legacy}ds2_1x1_conv3d", f"{clean}deep_coarse"),
                _map(f"{legacy}ds3_1x1_conv3d", f"{clean}deep_fine"),
            )
        )

    add_unet("lnet", "center_enhancer", main=False)
    add_unet("", "", main=True)
    return tuple(pairs)


BACKBONE_CHECKPOINT_KEY_MAP = _backbone_weight_map()


def _load_numeric_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise CellectModelError(f"numeric checkpoint does not exist: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older packaged torch
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and isinstance(value.get("state_dict"), dict):
        value = value["state_dict"]
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        raise CellectModelError("checkpoint is not a tensor state dictionary")
    return value


def load_backbone_checkpoint(model: CellectBackbone, checkpoint: Path) -> None:
    """Load the released numeric checkpoint through a frozen key translation."""

    source = _load_numeric_state(checkpoint)
    expected_legacy = {legacy for legacy, _clean in BACKBONE_CHECKPOINT_KEY_MAP}
    if set(source) != expected_legacy:
        missing = sorted(expected_legacy - set(source))
        extra = sorted(set(source) - expected_legacy)
        raise CellectModelError(f"checkpoint key mismatch; missing={missing}, extra={extra}")
    translated = OrderedDict(
        (clean, source[legacy]) for legacy, clean in BACKBONE_CHECKPOINT_KEY_MAP
    )
    model.load_state_dict(translated, strict=True)


_MATCHER_PREFIX_MAP = {
    "l1": "context_projection",
    "l2": "difference_projection",
    "l3": "similarity_projection",
    "l4": "spatial_projection",
    "l5": "size_projection",
    "l6": "division_projection",
    "bnx": "compatibility_bn_x",
    "bny": "compatibility_bn_y",
    "bn1": "context_norm",
    "bn2": "difference_norm",
    "bn3": "similarity_norm",
    "bn4": "spatial_norm",
    "bn5": "size_norm",
    "bn6": "division_norm",
    "bn7": "fused_norm",
    "bnc": "pair_conv_norm",
    "fc": "fused_projection",
    "fc2": "hidden_projection",
    "out": "output_layer",
    "c1": "pair_conv_wide",
    "c2": "pair_conv_narrow",
    "cx": "pair_projection",
    "outc": "compatibility_output",
}


def load_matcher_checkpoint(model: CellectMatcher, checkpoint: Path) -> None:
    """Load either released MLP checkpoint without importing reference code."""

    source = _load_numeric_state(checkpoint)
    translated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in source.items():
        prefix, separator, suffix = key.partition(".")
        clean_prefix = _MATCHER_PREFIX_MAP.get(prefix)
        if clean_prefix is None or not separator:
            raise CellectModelError(f"unexpected matcher checkpoint key: {key}")
        translated[f"{clean_prefix}.{suffix}"] = tensor
    model.load_state_dict(translated, strict=True)


def load_published_backbone(
    checkpoint: Path,
    *,
    device: torch.device | str = "cpu",
) -> CellectBackbone:
    """Instantiate the clean-room model and load a numeric pretrained checkpoint."""

    model = CellectBackbone()
    load_backbone_checkpoint(model, Path(checkpoint))
    return model.to(device)


def matcher_input_features(
    anchor_position: torch.Tensor,
    candidate_positions: torch.Tensor,
    anchor_size: torch.Tensor,
    candidate_sizes: torch.Tensor,
    anchor_division: torch.Tensor,
    candidate_divisions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the published spatial, size, and division candidate descriptors."""

    if anchor_position.ndim != 2 or anchor_position.shape[1] != 3:
        raise CellectModelError("anchor positions must have shape (N,3)")
    batch = len(anchor_position)
    if candidate_positions.shape != (batch, 5, 3):
        raise CellectModelError("candidate positions must have shape (N,5,3)")
    if anchor_size.shape not in ((batch,), (batch, 1)):
        raise CellectModelError("anchor sizes must have shape (N,) or (N,1)")
    if candidate_sizes.shape != (batch, 5):
        raise CellectModelError("candidate sizes must have shape (N,5)")
    if anchor_division.shape != (batch, 2) or candidate_divisions.shape != (batch, 5, 2):
        raise CellectModelError("division probabilities must have shapes (N,2) and (N,5,2)")
    spatial = torch.abs(candidate_positions - anchor_position[:, None, :])
    distance = torch.linalg.vector_norm(spatial, dim=2)
    anchor_size = anchor_size.reshape(batch, 1)
    size = torch.stack(
        (
            candidate_sizes - anchor_size,
            candidate_sizes / (anchor_size + 1.0),
            distance,
            candidate_sizes - distance,
            (candidate_sizes > distance).to(candidate_sizes.dtype),
            candidate_sizes / (distance + 1e-4),
        ),
        dim=2,
    )
    anchor_probability = anchor_division[:, None, :].expand(-1, 5, -1)
    interaction = torch.stack(
        (
            (anchor_probability[:, :, 0] > 0.5)
            & (candidate_divisions[:, :, 1] > 0.5),
            torch.zeros_like(candidate_divisions[:, :, 0], dtype=torch.bool),
        ),
        dim=2,
    ).to(candidate_divisions.dtype)
    interaction[:, :, 1] = (
        anchor_probability[:, :, 0] * candidate_divisions[:, :, 1]
    )
    division = torch.cat((anchor_probability, candidate_divisions, interaction), dim=2)
    return spatial, size, division


__all__ = [
    "BACKBONE_CHECKPOINT_KEY_MAP",
    "CHECKPOINT_CEN_INPUT_CHANNELS",
    "CellectBackbone",
    "CellectMatcher",
    "CellectModelError",
    "CellectOutputs",
    "CenterEnhancementNet",
    "EMBEDDING_CHANNELS",
    "InterframeMatcherOutputs",
    "PUBLICATION_DISCREPANCY",
    "PUBLISHED_CHECKPOINT_FILENAME",
    "PUBLISHED_CHECKPOINT_PARAMETERS",
    "PUBLISHED_CHECKPOINT_SHA256",
    "load_backbone_checkpoint",
    "load_matcher_checkpoint",
    "load_published_backbone",
    "matcher_input_features",
]
