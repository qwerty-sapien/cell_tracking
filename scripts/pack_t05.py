#!/usr/bin/env python3
"""Build the private T-05 released-detector distillation training package.

T-05 keeps the released 3D U-Net frozen.  Only the small released detection
head and existing sparse auxiliary head are trainable.  Annotated centres add
positive-only supervision while a frozen copy of the released detector anchors
the student logits at every voxel; therefore unlabeled voxels are never treated
as negatives.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kaggle_kernel" / "t04.py"
PYTHON = ROOT / "kaggle_kernel" / "t05.py"
NOTEBOOK = ROOT / "kaggle_kernel" / "t05.ipynb"
META = ROOT / "kaggle_kernel" / "t05.meta.json"
PACK = ROOT / "kaggle_kernel" / "t05.pack.json"
TRIAL = ROOT / "reports" / "g02" / "t05-trial.json"
TEST = ROOT / "tests" / "test_t05.py"
MAX_CODE_BYTES = 1_000_000
BASE_SHA256 = "ab1dba1c8500830c1762fd358b232e6c5c60492a2d93a83b65a3cdc0c8c7390e"


def canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} is not an object")
    return value


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def trial_bytes() -> tuple[dict[str, Any], bytes]:
    trial = read_json(TRIAL)
    expected = {
        "batch_size": 1,
        "center_loss": "annotated-positive-plus-released-logit-distillation",
        "center_supervision": "annotated-only",
        "center_weight": 0.1,
        "detector_policy": "released_teacher_distill",
        "division_enabled": False,
        "epochs": 1,
        "fit_fraction": 0.1,
        "fold": 0,
        "learning_rate": 1e-5,
        "max_steps": 1000,
        "released_detector_frozen": False,
        "scenario": "universal",
        "seed": 17,
        "teacher_weight": 1.0,
        "unfreeze": 0,
    }
    if trial.get("artifact") != "t05-private-trial" or trial.get("status") != "package_not_launched":
        raise ValueError("T-05 trial state is invalid")
    if trial.get("candidate") != expected:
        raise ValueError("T-05 candidate identity drift")
    if trial.get("authorization") != {
        "approved_by_user": True, "concurrency": 1, "maximum_charge_usd": 0.3,
        "maximum_hours": 0.75, "private_only": True, "public_effects": False,
        "scored_submission": False,
    }:
        raise ValueError("T-05 trial authorization drift")
    if trial.get("package_contract") != {
        "code_package_under_bytes": MAX_CODE_BYTES, "released_checkpoint_input_required": True,
        "runtime_self_contained": False, "support_input_required": True,
    }:
        raise ValueError("T-05 package boundary drift")
    return trial, canonical(trial)


OVERRIDE = r'''
# T-05 override: frozen-backbone released-detector distillation.
DETECTOR_POLICY = "released_teacher_distill"
CENTER_WEIGHT = 0.1
TEACHER_WEIGHT = 1.0
DISTILL_CENTER_LOSS = "annotated-positive-plus-released-logit-distillation"

class TemporalAdapterModel(nn.Module):
    """Frozen released features with a teacher-anchored detector adaptation."""
    def __init__(self, backbone: nn.Module, detect_head: nn.Conv3d) -> None:
        super().__init__()
        self.backbone = backbone
        self.detect_head = detect_head
        self.aux_head = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.GELU(), nn.Conv3d(32, 5, 1))
        self.specialists = nn.ModuleDict({"sparse_clean": _Specialist(), "dense_crowded": _Specialist(), "low_snr_depth": _Specialist()})
        self.set_unfreeze_depth(0)
    @staticmethod
    def scenarios() -> tuple[str, ...]: return SCENARIOS
    def set_unfreeze_depth(self, depth: int) -> dict[str, int]:
        if depth != 0: raise ValueError("T-05 requires a frozen released backbone")
        for parameter in self.backbone.parameters(): parameter.requires_grad = False
        for parameter in self.detect_head.parameters(): parameter.requires_grad = True
        self.backbone.eval(); self.detect_head.train(True)
        return self.parameter_accounting()
    def parameter_accounting(self) -> dict[str, int]:
        parameters = tuple(self.parameters()); total = sum(p.numel() for p in parameters)
        trainable = sum(p.numel() for p in parameters if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total-trainable,
                "backbone_trainable": sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
                "detect_head_trainable": sum(p.numel() for p in self.detect_head.parameters() if p.requires_grad),
                "aux_head_trainable": sum(p.numel() for p in self.aux_head.parameters() if p.requires_grad)}
    def train(self, mode: bool=True) -> "TemporalAdapterModel":
        super().train(mode); self.backbone.eval(); self.detect_head.train(mode); return self
    def forward(self, images: torch.Tensor, scenario: str="universal") -> torch.Tensor:
        if scenario not in SCENARIOS: raise ValueError(f"unknown scenario: {scenario}")
        if not isinstance(images, torch.Tensor) or images.ndim != 6 or images.shape[2] != 1 or min(images.shape) < 1: raise ValueError("images must have nonempty shape (B,T,1,Z,Y,X)")
        if not _finite(images): raise ValueError("images must be finite")
        batch, steps, _, z, y, x = images.shape; features = self.backbone(images)
        if tuple(features.shape) != (batch, steps, 32, z, y, x): raise ValueError("released backbone returned an incompatible feature shape")
        flat = features.reshape(batch*steps, 32, z, y, x); universal = torch.cat((self.detect_head(flat), self.aux_head(flat)), dim=1)
        output = universal if scenario == "universal" else self.specialists[scenario](flat, universal, self.aux_head)
        if tuple(output.shape) != (batch*steps, 6, z, y, x): raise RuntimeError("adapter output schema violation")
        return output.reshape(batch, steps, 6, z, y, x)

def multitask_loss(output: torch.Tensor, targets: SparseTargets, division_enabled: bool=False, *,
                   center_weight: float=CENTER_WEIGHT, teacher_logits: torch.Tensor | None=None,
                   teacher_weight: float=TEACHER_WEIGHT) -> dict[str, torch.Tensor]:
    if output.device != targets.device: raise TargetError("prediction and sparse targets must share a device")
    if tuple(output.shape) != tuple(targets.center.shape[:2]) + (6,) + tuple(targets.center.shape[3:]): raise TargetError("output and target shapes differ")
    if targets.center_supervision_mask.dtype is not torch.bool or targets.center_supervision_mask.device != output.device or tuple(targets.center_supervision_mask.shape) != tuple(targets.center.shape): raise TargetError("center supervision mask is invalid")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0 for value in (center_weight, teacher_weight)): raise TargetError("loss weights must be nonnegative finite scalars")
    if teacher_weight > 0 and (teacher_logits is None or tuple(teacher_logits.shape) != tuple(output[:, :, :1].shape) or teacher_logits.device != output.device): raise TargetError("teacher logits must match the center channel")
    center_terms = F.binary_cross_entropy_with_logits(output[:, :, :1], targets.center, reduction="none")
    center = _masked_mean(center_terms, targets.center_supervision_mask)
    error = (output[:, :, 1:4]-targets.displacement).abs().sum(dim=2, keepdim=True); displacement = _masked_mean(error, targets.displacement_mask)
    logged = output[:, :, 4:5].clamp(-10.,10.); variance = _masked_mean(.5*(torch.exp(-logged)*error+logged), targets.displacement_mask)
    division = _masked_mean(F.binary_cross_entropy_with_logits(output[:, :, 5:6], targets.division, reduction="none"), targets.division_mask) if division_enabled else output.new_zeros(())
    teacher = F.mse_loss(output[:, :, :1], teacher_logits.detach()) if teacher_logits is not None else output.new_zeros(())
    return {"total": displacement+variance+division+float(center_weight)*center+float(teacher_weight)*teacher,
            "center": center, "teacher": teacher, "displacement": displacement, "log_variance": variance, "division": division}

def _identity(args: argparse.Namespace) -> dict[str, Any]:
    return {"fold":args.fold,"scenario":args.scenario,"unfreeze":args.unfreeze,"lr":args.lr,"fraction":args.fraction,"seed":args.seed,
            "detector_policy":DETECTOR_POLICY,"center_weight":CENTER_WEIGHT,"teacher_weight":TEACHER_WEIGHT,
            "center_supervision":CENTER_SUPERVISION,"center_loss":DISTILL_CENTER_LOSS,"split_sha256":SPLIT_SHA256,
            "provenance_sha256":PROVENANCE_SHA256,"registry_sha256":REGISTRY_SHA256,"registry_projection_sha256":REGISTRY_PROJECTION_SHA256,
            "weights_sha256":WEIGHTS_SHA256,"config_sha256":CONFIG_SHA256}

def _build_released_model(args: argparse.Namespace, device: torch.device) -> TemporalAdapterModel:
    backbone = TemporalUNet3D(in_channels=1, out_channels=EXPECTED_CONFIG["unet_out_channels"], layers=EXPECTED_CONFIG["unet_layers"])
    detector, _ = load_released_core(backbone, args.weights_path, args.config_path)
    model = TemporalAdapterModel(backbone, detector); model.set_unfreeze_depth(args.unfreeze); return model.to(device)

def _released_teacher(model: TemporalAdapterModel, device: torch.device) -> nn.Conv3d:
    if any(parameter.requires_grad for parameter in model.backbone.parameters()): raise LaunchError("T-05 teacher requires a frozen backbone")
    teacher = nn.Conv3d(32, 1, kernel_size=1).to(device); teacher.load_state_dict(model.detect_head.state_dict(), strict=True)
    for parameter in teacher.parameters(): parameter.requires_grad = False
    teacher.eval(); return teacher

def _batch_loss(model: TemporalAdapterModel, batch: Mapping[str, Any], device: torch.device, teacher: nn.Conv3d) -> Mapping[str, torch.Tensor]:
    frames = batch["frames"].to(device=device, dtype=torch.float32, non_blocking=device.type=="cuda"); output = model(frames, scenario="universal")
    with torch.no_grad():
        b,t,_,z,y,x = frames.shape; features = model.backbone(frames).reshape(b*t,32,z,y,x); teacher_logits = teacher(features).reshape(b,t,1,z,y,x)
    targets = build_sparse_targets(output,batch["nodes"],batch["edges"],batch["voxel_size_um"])
    return multitask_loss(output,targets,division_enabled=False,center_weight=CENTER_WEIGHT,teacher_logits=teacher_logits,teacher_weight=TEACHER_WEIGHT)

def train_steps(model: TemporalAdapterModel, optimizer: torch.optim.Optimizer, batches: Iterable[Mapping[str, Any]], *, device: torch.device, deadline: float, max_steps: int, teacher: nn.Conv3d, scaler: Any|None=None, checkpoint_callback: Callable[[Mapping[str, Any], Any|None], None]|None=None) -> tuple[dict[str, Any], Any|None]:
    if max_steps < 0: raise LaunchError("remaining steps cannot be negative")
    enabled = USE_MIXED_PRECISION and device.type=="cuda"; scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda",enabled=enabled)
    sums={"total":0.,"center":0.,"teacher":0.,"displacement":0.,"log_variance":0.,"division":0.}; by_epoch={}; completed=0; window_ids=[]; per_step=[]; status="completed"; model.train()
    for batch in batches:
        if completed >= max_steps: break
        if time.monotonic() >= deadline: status="time_cap"; break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",enabled=enabled): losses=_batch_loss(model,batch,device,teacher)
        bad=[name for name,value in losses.items() if not _finite_tensor(value)]
        if bad: raise LaunchError("nonfinite loss components: "+",".join(bad))
        if enabled: scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
        else: losses["total"].backward()
        bad=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is not None and not _finite_tensor(p.grad)]
        if bad: raise LaunchError("nonfinite gradients: "+",".join(bad[:8]))
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
        if time.monotonic() >= deadline: status="time_cap"; break
        if enabled: scaler.step(optimizer); scaler.update()
        else: optimizer.step()
        if not all(_finite_tensor(p.detach()) for p in model.parameters()): raise LaunchError("nonfinite parameter after optimizer step")
        completed+=1; window_ids.append(str(batch.get("window_id",completed))); epoch=int(batch.get("epoch",1))
        if epoch < 1: raise LaunchError("batch epoch must be positive")
        aggregate=by_epoch.setdefault(epoch,{"steps":0,"sums":{key:0. for key in sums}}); aggregate["steps"]+=1
        for key in sums: sums[key]+=float(losses[key].detach().cpu()); aggregate["sums"][key]+=float(losses[key].detach().cpu())
        per_step.append({"step":completed,"epoch":epoch,"window_id":window_ids[-1],"loss_components":{key:float(losses[key].detach().cpu()) for key in sums}})
        if checkpoint_callback is not None and completed%CHECKPOINT_EVERY_STEPS==0: checkpoint_callback(_step_result("running",completed,window_ids,per_step,sums,by_epoch),scaler)
    return _step_result(status,completed,window_ids,per_step,sums,by_epoch),scaler

def _run_bounded_training(args: argparse.Namespace, selected: Sequence[SourceRecord], device_name: str) -> dict[str, Any]:
    _validate_arg_contract(args)
    if not args.armed: raise LaunchError("non-dry launch requires explicit --armed before data access")
    if args.data_root is None: raise LaunchError("--data-root is required when dry-run is disabled")
    device=torch.device(device_name); identity={**_identity(args),"selected_sources":[record.json() for record in selected]}; _seed_all(args.seed)
    started=float(getattr(args,"started_monotonic",time.monotonic())); deadline=started+max(0.,args.maximum_hours*3600.)
    model=_build_released_model(args,device)
    if any(parameter.requires_grad for parameter in model.backbone.parameters()) or not any(parameter.requires_grad for parameter in model.detect_head.parameters()): raise LaunchError("T-05 model training policy drift")
    teacher=_released_teacher(model,device); trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable: raise LaunchError("no trainable parameters")
    optimizer=torch.optim.AdamW(trainable,lr=args.lr); scaler=torch.amp.GradScaler("cuda",enabled=USE_MIXED_PRECISION and device.type=="cuda")
    prior_steps=prior_elapsed=0
    if args.resume is not None: prior_steps,prior_elapsed,scaler=_load_resume(args.resume,identity,model,optimizer,scaler)
    deadline-=prior_elapsed
    if prior_steps >= args.max_steps:
        empty={key:0. for key in ("total","center","teacher","displacement","log_variance","division")}; result={"status":"completed","steps":0,"windows":[],"per_step":[],"epoch_metrics":[],"loss_components":empty,"source_window_stats":{}}
    elif time.monotonic() >= deadline:
        empty={key:0. for key in ("total","center","teacher","displacement","log_variance","division")}; result={"status":"time_cap","steps":0,"windows":[],"per_step":[],"epoch_metrics":[{"epoch":1,"steps":0,"loss_components":empty}],"loss_components":empty,"source_window_stats":{}}
    else:
        source=SparseWindowSource(args.data_root,EXPECTED_CONFIG["downsample"]); batches=source.batches(selected,completed_windows=prior_steps,deadline=deadline,epochs=args.epochs)
        def checkpoint_callback(progress: Mapping[str,Any],progress_scaler: Any|None)->None:
            metrics=_write_outputs(args.output_dir,identity,model,optimizer,progress_scaler,progress,started,selected,prior_steps,prior_elapsed,args.epochs,args.max_steps); print(json.dumps({"status":"running","steps":metrics["completed_steps"],"output":str(args.output_dir)},sort_keys=True),flush=True)
        result,scaler=train_steps(model,optimizer,batches,device=device,deadline=deadline,max_steps=args.max_steps-prior_steps,teacher=teacher,scaler=scaler,checkpoint_callback=checkpoint_callback); result["source_window_stats"]=dict(source.window_stats)
    return _write_outputs(args.output_dir,identity,model,optimizer,scaler,result,started,selected,prior_steps,prior_elapsed,args.epochs,args.max_steps)

ARG_FIELDS = ("dry_run","fold","output_dir","resume","armed","scenario","unfreeze","lr","fraction","seed","epochs","max_steps","maximum_hours","maximum_charge","detector_policy","center_weight","teacher_weight")
def _validate_arg_contract(args: argparse.Namespace) -> None:
    missing=[name for name in ARG_FIELDS if not hasattr(args,name)]
    if missing: raise LaunchError("argument namespace is incomplete: "+",".join(missing))
    if not isinstance(args.dry_run,bool) or not isinstance(args.armed,bool) or args.armed != (not args.dry_run): raise LaunchError("armed argument contract mismatch")
    if not isinstance(args.output_dir,Path) or (args.resume is not None and not isinstance(args.resume,Path)): raise LaunchError("path argument contract mismatch")
    if (args.fold not in (0,1,2) or args.scenario != "universal" or args.unfreeze != 0 or args.detector_policy != DETECTOR_POLICY or args.center_weight != CENTER_WEIGHT or args.teacher_weight != TEACHER_WEIGHT): raise LaunchError("candidate argument contract mismatch")
    numeric=(args.lr,args.fraction,args.maximum_hours,args.maximum_charge)
    if not all(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(float(value)) for value in numeric): raise LaunchError("numeric argument contract mismatch")
    if not all(isinstance(value,int) and not isinstance(value,bool) for value in (args.seed,args.epochs,args.max_steps)): raise LaunchError("integer argument contract mismatch")

def _exact_candidate(args: argparse.Namespace) -> None:
    c=trial_record()["candidate"]
    got={"fold":args.fold,"scenario":args.scenario,"unfreeze":args.unfreeze,"learning_rate":args.lr,"fit_fraction":args.fraction,"seed":args.seed,"epochs":args.epochs,"max_steps":args.max_steps,"batch_size":1,"division_enabled":False,"center_supervision":CENTER_SUPERVISION,"center_loss":DISTILL_CENTER_LOSS,"released_detector_frozen":False,"detector_policy":DETECTOR_POLICY,"center_weight":CENTER_WEIGHT,"teacher_weight":TEACHER_WEIGHT}
    if got != c or args.maximum_hours != .75 or args.maximum_charge != .30: raise LaunchError("armed candidate identity or cap drift")
    immutable=trial_record()["immutable_inputs"]
    if immutable["registry_sha256"] != REGISTRY_SHA256 or immutable["registry_projection_sha256"] != REGISTRY_PROJECTION_SHA256 or immutable["checkpoint_sha256"] != WEIGHTS_SHA256 or immutable["checkpoint_config_sha256"] != CONFIG_SHA256: raise LaunchError("trial immutable input drift")

def _args(argv: Sequence[str]|None=None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description="private released-detector distillation trainer"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--fold",type=int,choices=(0,1,2),default=0); p.add_argument("--output-dir",type=Path,default=KOUT); p.add_argument("--resume",type=Path)
    a=p.parse_args(argv); a.armed=not a.dry_run; a.scenario="universal"; a.unfreeze=0; a.lr=1e-5; a.fraction=.1; a.seed=17; a.epochs=1; a.max_steps=1000; a.maximum_hours=.75; a.maximum_charge=.30; a.detector_policy=DETECTOR_POLICY; a.center_weight=CENTER_WEIGHT; a.teacher_weight=TEACHER_WEIGHT; _validate_arg_contract(a); return a

def main(argv: Sequence[str]|None=None) -> int:
    args=_args(argv); args.started_monotonic=time.monotonic(); _validate_budget(args.maximum_hours,args.maximum_charge); runtime=_runtime_preflight(args)
    split,provenance=embedded_manifest("split"),embedded_manifest("provenance"); registry=runtime_registry_projection(); selected_groups=select_fit_groups(args.fold,split,provenance,args.fraction,args.seed); _,forbidden=_fold_data(args.fold,split,provenance); selected=resolve_fit_records(selected_groups,registry,forbidden)
    if not args.dry_run:
        _exact_candidate(args)
        if args.output_dir.resolve() != KOUT: raise LaunchError("armed output root must be private Kaggle working/t05")
        args.data_root=KROOT if KROOT.is_dir() else KALT
        if not args.data_root.is_dir(): raise LaunchError("competition train root is absent")
        args.weights_path=_find_input("edge_predictor_best.pth",WEIGHTS_BYTES,WEIGHTS_SHA256); args.config_path=_sibling_config(args.weights_path); verify_released_files(args.weights_path,args.config_path)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pre={"artifact":"prelaunch-v1","status":"not_launched" if args.dry_run else "armed","identity":{**_identity(args),"selected_sources":[r.json() for r in selected]},"source_isolation":{"fit_only":True,"selected_group_ids":[r.group_id for r in selected],"selected_train_video_ids":[r.video_id for r in selected],"held_out_used":False,"audit_used":False,"test_used":False},"execution":{"device":"cuda" if torch.cuda.is_available() else "cpu","mixed_precision":USE_MIXED_PRECISION,"gradient_clip_norm":1.,"epochs":1,"max_steps":1000,"batch_size":1,"dry_run":args.dry_run,"cap_scope":"notebook-start-through-checkpoint"},"training_policy":{"backbone_frozen":True,"detector_policy":DETECTOR_POLICY,"center_supervision":CENTER_SUPERVISION,"center_loss":DISTILL_CENTER_LOSS,"center_weight":CENTER_WEIGHT,"teacher_weight":TEACHER_WEIGHT,"division_enabled":False},"runtime_preflight":runtime,"spend_gate":{"maximum_hours":.75,"maximum_charge":.30,"cap_hours":MAX_HOURS_CAP,"cap_charge":MAX_CHARGE_CAP,"approved_spend":not args.dry_run}}
    digest=_atomic_json(args.output_dir/"prelaunch.json",pre)
    if args.dry_run: print(json.dumps({"status":"dry","prelaunch_sha256":digest,"sources":len(selected)},sort_keys=True)); return 0
    print(json.dumps({"status":"armed","output":str(KOUT),"sources":len(selected)},sort_keys=True),flush=True); result=_run_bounded_training(args,selected,"cuda"); print(json.dumps({"status":result["status"],"steps":result["completed_steps"],"output":str(KOUT)},sort_keys=True),flush=True); return 0
'''


def source_bytes(trial: bytes) -> bytes:
    raw = BASE.read_bytes()
    if digest(raw) != BASE_SHA256:
        raise ValueError("T-04 base source drift; T-05 must be regenerated deliberately")
    source = raw.decode("utf-8")
    encoded = base64.b85encode(trial)
    source, count = re.subn(r"^TRIAL_BYTES = base64\.b85decode\(b'[^']*'\)$", "TRIAL_BYTES = base64.b85decode(" + repr(encoded) + ")", source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("T-05 trial payload replacement is ambiguous")
    source, count = re.subn(r"^TRIAL_SHA256 = '[0-9a-f]{64}'$", f"TRIAL_SHA256 = '{digest(trial)}'", source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("T-05 trial digest replacement is ambiguous")
    source = source.replace("/kaggle/working/t04", "/kaggle/working/t05")
    if "/kaggle/working/t04" in source:
        raise ValueError("T-04 output path remained in T-05 source")
    marker = "\nif __name__ == '__main__': raise SystemExit(main([]))\n"
    if source.count(marker) != 1:
        raise ValueError("T-04 entrypoint marker is ambiguous")
    source = source.replace(marker, OVERRIDE + marker)
    ast.parse(source)
    result = source.encode("utf-8")
    if len(result) >= MAX_CODE_BYTES:
        raise ValueError("T-05 source exceeds Kaggle's 1 MB cap")
    return result


def notebook_bytes(source: bytes) -> bytes:
    value = {"cells": [{"cell_type": "code", "execution_count": None, "id": "t05-entry", "metadata": {}, "outputs": [], "source": source.decode("utf-8").splitlines(keepends=True)}],
             "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "private_only": True, "runtime_self_contained": False, "support_input_required": True, "t05_script_sha256": digest(source)}, "nbformat": 4, "nbformat_minor": 5}
    result = canonical(value)
    if b"".join(line.encode("utf-8") for line in value["cells"][0]["source"]) != source or len(result) >= MAX_CODE_BYTES:
        raise ValueError("T-05 notebook is not an exact compact source container")
    return result


def metadata(trial: Mapping[str, Any], trial_data: bytes, source: bytes) -> bytes:
    return canonical({"artifact": "t05-meta-v1", "candidate": trial["candidate"], "entrypoint": "t05.py", "notebook": "t05.ipynb", "output_root": "/kaggle/working/t05", "private_only": True, "public_effect": False, "network": False, "submission": False, "runtime_self_contained": False, "support_input_required": True, "source": {"bytes": len(source), "sha256": digest(source)}, "trial": {"path": "reports/g02/t05-trial.json", "sha256": digest(trial_data)}, "base": {"path": "kaggle_kernel/t04.py", "sha256": BASE_SHA256}, "training_policy": {"backbone_frozen": True, "detector_policy": "released_teacher_distill", "positive_only_center_labels": True, "teacher_logit_distillation": True}})


def package_bytes(source: bytes, notebook: bytes, meta: bytes, trial: bytes) -> bytes:
    files = {PYTHON: source, NOTEBOOK: notebook, META: meta, TRIAL: trial, TEST: TEST.read_bytes(), Path(__file__): Path(__file__).read_bytes()}
    value = {"artifact": "t05-pack-v1", "code_package_bytes": len(source) + len(notebook), "code_package_under_bytes": MAX_CODE_BYTES, "notebook": {"runtime_self_contained": False, "source_cell_sha256": digest(source), "support_input_required": True}, "package_files": {path.relative_to(ROOT).as_posix(): {"bytes": len(data), "sha256": digest(data)} for path, data in sorted(files.items(), key=lambda item: item[0].as_posix())}, "runtime": {"fp32_only": True, "network": False, "private_only": True, "submission": False, "released_checkpoint_input_required": True, "support_input_required": True}}
    if value["code_package_bytes"] >= MAX_CODE_BYTES:
        raise ValueError("T-05 code package exceeds Kaggle's 1 MB cap")
    return canonical(value)


def build() -> dict[Path, bytes]:
    trial, trial_data = trial_bytes()
    source = source_bytes(trial_data)
    notebook = notebook_bytes(source)
    meta = metadata(trial, trial_data, source)
    return {TRIAL: trial_data, PYTHON: source, NOTEBOOK: notebook, META: meta, PACK: package_bytes(source, notebook, meta, trial_data)}


def check() -> None:
    expected = build()
    for path, data in expected.items():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"T-05 package drift: {path.relative_to(ROOT)}")
    package = read_json(PACK)
    for name, identity in package["package_files"].items():
        data = (ROOT / name).read_bytes()
        if identity != {"bytes": len(data), "sha256": digest(data)}:
            raise ValueError(f"T-05 listed package identity drift: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    if parser.parse_args().check:
        check()
    else:
        for path, data in build().items():
            atomic_write(path, data)
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
