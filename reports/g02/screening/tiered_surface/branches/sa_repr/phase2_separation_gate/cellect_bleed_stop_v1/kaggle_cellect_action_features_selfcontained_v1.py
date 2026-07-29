#!/usr/bin/env python3
"""Generate only frozen 44b6 CELLECT action features on one bounded Kaggle GPU run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

# BEGIN INLINE CONTENT-ADDRESSED STEP 0
import os
from collections import Counter, defaultdict

EMBEDDED_ASSET_CONTRACT = json.loads('{"checkpoint":{"bytes":10197221,"container_bytes":10199040,"container_sha256":"a8110e0d5c796ad54bc51bf6fd8f11c332848c9e9ef378cd23bd9575aa0af559","filename":"cellect_uext_x3rd_149_0_4p6540.pth","sha256":"f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"},"dataset":"onebitbrain/g02-cellect-r7-published-assets-v1","dataset_version":6,"population":{"bytes":57161,"sha256":"4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"},"request_count":1391458,"request_file_count":77,"requests":[{"action_class":"delete","action_count":29232,"bytes":304379,"identity":"train:44b6_0c582fdc","ordinal":0,"sha256":"b6f1f256f2853d785b80459e4b5e3ac66edc5db4bc0479f4c1c0ba3d9700d220"},{"action_class":"delete","action_count":18486,"bytes":189938,"identity":"train:44b6_0db75fae","ordinal":1,"sha256":"f66a905949732ac70aed169edcc72b41a77305db075549bdbed1aed4c818552c"},{"action_class":"delete","action_count":50294,"bytes":494454,"identity":"train:44b6_12dfb391","ordinal":2,"sha256":"b3df081f8e3fea2f3ee9b37e6f053824858c240294cb52965679875c2420f57a"},{"action_class":"delete","action_count":32334,"bytes":307419,"identity":"train:44b6_144b256d","ordinal":3,"sha256":"e04d65978ae559f496b77b10d475159b9b18f41a18a338d139381db4dc559216"},{"action_class":"delete","action_count":20662,"bytes":204665,"identity":"train:44b6_1574802b","ordinal":4,"sha256":"3523dc924ab10b247f48efe210e8e06c24bccc3d8a99e5293defd373d540076e"},{"action_class":"delete","action_count":80345,"bytes":835438,"identity":"train:44b6_18ced818","ordinal":5,"sha256":"0b32e4b870ce8f69dd775d71cd02a1b84f18b435d349a5d51562bdae55566217"},{"action_class":"delete","action_count":39739,"bytes":405317,"identity":"train:44b6_1d530831","ordinal":6,"sha256":"9a86197618b83d75a9b1190c84bc639738d2eeddf3c3b8cf0711d33709a1d091"},{"action_class":"delete","action_count":43139,"bytes":444634,"identity":"train:44b6_2a2eff9f","ordinal":7,"sha256":"d7d42bb6beb7f144e2cc32977a296fa4dbd3fef39a3dfa6cca67166563c999cf"},{"action_class":"delete","action_count":38086,"bytes":384740,"identity":"train:44b6_3a861e03","ordinal":8,"sha256":"2cbf8066eba943ec9a2ef597a23a7f4e38ab48b61ee17e7da63d8d6367d08efb"},{"action_class":"delete","action_count":36141,"bytes":366925,"identity":"train:44b6_40c45f5a","ordinal":9,"sha256":"96742564f19c6095d0d35df6c440803290c1f930631494ac2e53a4c5d77559c3"},{"action_class":"delete","action_count":36373,"bytes":361371,"identity":"train:44b6_415c0a3a","ordinal":10,"sha256":"5e33b498419890708fd186d1357f1021eeaf865b70bc4d0f4b702c3a15cc2fe6"},{"action_class":"delete","action_count":36819,"bytes":372530,"identity":"train:44b6_551a5dba","ordinal":11,"sha256":"f0e05f2b0ae6ca6b21af3429e599035340221543651606491245903a0f6905fd"},{"action_class":"delete","action_count":12918,"bytes":128665,"identity":"train:44b6_5740d24b","ordinal":12,"sha256":"725501f9104bda2f6c91b6f6f478fc77e784b240db011ee58e4979d44c8219fe"},{"action_class":"delete","action_count":18879,"bytes":196409,"identity":"train:44b6_587a1e22","ordinal":13,"sha256":"6c7a41cbcb6a4c55f4cd37c27eb964895c7019f3f5d13e813619bd6e17087d12"},{"action_class":"delete","action_count":25851,"bytes":266716,"identity":"train:44b6_5f15d135","ordinal":14,"sha256":"aeabb1195be51bcedef2bf16c0b91f640bf609e2716aeabe48331bdd1c9571b6"},{"action_class":"delete","action_count":31715,"bytes":295511,"identity":"train:44b6_706092f0","ordinal":15,"sha256":"a8262668cea93c801d426589405a49df22ac49815a9513c6f891aabb1c47a389"},{"action_class":"delete","action_count":48316,"bytes":470150,"identity":"train:44b6_7e557709","ordinal":16,"sha256":"77afd8db1a90b19b8344eba69d5afe5067883be7fc4f49e00640e83a53f389d5"},{"action_class":"delete","action_count":16443,"bytes":166739,"identity":"train:44b6_808952d6","ordinal":17,"sha256":"036ce2042a03bcd0e37bc70ba6af74dd49722aa8fe95853e6d10205eeb982d53"},{"action_class":"delete","action_count":11348,"bytes":113159,"identity":"train:44b6_81c256f0","ordinal":18,"sha256":"86ad64ec733dda90bcb8f32657c08d73430582597396e087ab7740b9818e6a51"},{"action_class":"delete","action_count":29858,"bytes":300813,"identity":"train:44b6_8f5ab931","ordinal":19,"sha256":"87406ec75502769d7819cf101cdc16edd482165e571685d108d2111eddc377c1"},{"action_class":"delete","action_count":25160,"bytes":259016,"identity":"train:44b6_90724892","ordinal":20,"sha256":"233e66e98a307b338a451679ddd5c6d3a8cddb22be5be24d9ae6597b08525488"},{"action_class":"delete","action_count":42496,"bytes":436981,"identity":"train:44b6_949adeb1","ordinal":21,"sha256":"8543bcb469b67c33f8e762e3c0d627a656cb63bd86e1f22b0facf03d573ac398"},{"action_class":"delete","action_count":4740,"bytes":46357,"identity":"train:44b6_95029e92","ordinal":22,"sha256":"6ea7162c64e1fd353738053cda9f4e0169c6a493d65be24ad58253ac70b21503"},{"action_class":"delete","action_count":37722,"bytes":389185,"identity":"train:44b6_97725144","ordinal":23,"sha256":"0f9f47170c2ea79f4a4bbaf66644e52df0a394b0c7edcb0188e8d1a77604db26"},{"action_class":"delete","action_count":42597,"bytes":406235,"identity":"train:44b6_9bfa6a0a","ordinal":24,"sha256":"55089938a381fc340b87d76a72056ed9f81353c4b81f447441bc7d4d5e5d55de"},{"action_class":"delete","action_count":56227,"bytes":578953,"identity":"train:44b6_a2bb48bb","ordinal":25,"sha256":"20cc9c153d8d81009a280cf302ba63c1e71b9fbffa406964a88e0f10eeb47766"},{"action_class":"delete","action_count":28864,"bytes":279484,"identity":"train:44b6_abf82518","ordinal":26,"sha256":"971a3b5b5ed290a3c2ba7d3f34340a6d20a65c1ab86b31e7a201a4ac105166a4"},{"action_class":"delete","action_count":42279,"bytes":405752,"identity":"train:44b6_b2c44266","ordinal":27,"sha256":"81727e6c7c9a3724e854332b0577eca71f1b160aed7783fc875210e812efe1eb"},{"action_class":"delete","action_count":20562,"bytes":210501,"identity":"train:44b6_c15fded2","ordinal":28,"sha256":"39567f5b5155ec82e624bff0e86aec608531be42b015aa5a6f12f578e4316cf1"},{"action_class":"delete","action_count":46091,"bytes":475866,"identity":"train:44b6_c50204e0","ordinal":29,"sha256":"a18bf6d067d2267ba63921586a799e9b0361f8cc43e9e0ea6c0878960f7fb5fc"},{"action_class":"delete","action_count":35857,"bytes":361436,"identity":"train:44b6_c771cb04","ordinal":30,"sha256":"094d3b6ba61827bb2c4fb962ce2292e1190e141087a6e154d006f6be9ca4d975"},{"action_class":"delete","action_count":44735,"bytes":467001,"identity":"train:44b6_cf8fed6b","ordinal":31,"sha256":"3038c2c63045d1b4918058905ac6b3a9c7e91be37e060d7330f8770c155b7d89"},{"action_class":"delete","action_count":36504,"bytes":366307,"identity":"train:44b6_d29c9ab2","ordinal":32,"sha256":"b0622dc8fe5e09c89aea00297f8dbbcf9f9eae47b19cf3bc4e50780368921c7d"},{"action_class":"delete","action_count":5675,"bytes":60571,"identity":"train:44b6_d754aa59","ordinal":33,"sha256":"e7ab115aff3605c47381411351f5e1342539596c82ed353fe34c9a03acf886b7"},{"action_class":"delete","action_count":23454,"bytes":240895,"identity":"train:44b6_d78e09d9","ordinal":34,"sha256":"1d10c7982e900047eb77c5796a1bf721f2b4d90b9171a9e097239cee8a9ddec3"},{"action_class":"delete","action_count":17003,"bytes":174468,"identity":"train:44b6_db3c847b","ordinal":35,"sha256":"2322fe80662e4c3dbc8a511c5c3b8c2393e83f3164c1c45a514103b590f34997"},{"action_class":"delete","action_count":29145,"bytes":272150,"identity":"train:44b6_e28840c6","ordinal":36,"sha256":"54040b3f902160103949b920890f589bbede32ecc06f9b2bc70fcd568f7e98b0"},{"action_class":"delete","action_count":14334,"bytes":151423,"identity":"train:44b6_e29f0176","ordinal":37,"sha256":"1ccdbc99e3e8d2a2c7293ffa58ef7afbd55aef2c9c2c5d9c38cfd57c95f31f4a"},{"action_class":"delete","action_count":72007,"bytes":751066,"identity":"train:44b6_e31261b4","ordinal":38,"sha256":"f5c7e584ff25eeaf08222f194f27e2c95e6b04a8e16177643ff0977ae8901d6a"},{"action_class":"delete","action_count":23352,"bytes":242181,"identity":"train:44b6_e35b117d","ordinal":39,"sha256":"48820035adf8946a679ef97ee850342a846e7704116e43ae487589b68bd38691"},{"action_class":"delete","action_count":39715,"bytes":408061,"identity":"train:44b6_eb2880fc","ordinal":40,"sha256":"c1b664b9b17c9421d5c5d15c70ea60fd4d337a4257435a900a5d9299832ca157"},{"action_class":"occupied_swap","action_count":1023,"bytes":12996,"identity":"train:44b6_0db75fae","ordinal":41,"sha256":"2f090ddccc83b459c4e9194012d857d06987a21e0937a2fc11e652b061a1bc4d"},{"action_class":"occupied_swap","action_count":1637,"bytes":19772,"identity":"train:44b6_12dfb391","ordinal":42,"sha256":"a85174f80ea9f9e7251c974ef901f6a8425ebd26a6140f5d23fdd3d52ecc6825"},{"action_class":"occupied_swap","action_count":618,"bytes":8829,"identity":"train:44b6_144b256d","ordinal":43,"sha256":"3c3847b31d96fe83ae7b0c05cf08c3996aefd03d07d7fe8a69897d84e8e28b84"},{"action_class":"occupied_swap","action_count":664,"bytes":9152,"identity":"train:44b6_1574802b","ordinal":44,"sha256":"a1067ff6f162e01f4084837962d603edcea28aca4a975cfef1473e2fff7af6d4"},{"action_class":"occupied_swap","action_count":4633,"bytes":52342,"identity":"train:44b6_18ced818","ordinal":45,"sha256":"a61d5c3bc35c841286083d8b592be23fb0efa19755bba26f445f82309ba0361e"},{"action_class":"occupied_swap","action_count":1707,"bytes":20524,"identity":"train:44b6_1d530831","ordinal":46,"sha256":"753c5ba4064c549eccd04f2ffb698dd78600bbae0d8ee68baac5f3e8f95e6fdc"},{"action_class":"occupied_swap","action_count":844,"bytes":11407,"identity":"train:44b6_3a861e03","ordinal":47,"sha256":"b44379b4d4671ab2210121c36dc621a26898e9654f1d1bf12374209ec1c7b4fd"},{"action_class":"occupied_swap","action_count":2119,"bytes":24647,"identity":"train:44b6_40c45f5a","ordinal":48,"sha256":"b3938f85e853266a8c39bb3fe8a56469b4163bae068d3f6e61179a186d95375b"},{"action_class":"occupied_swap","action_count":1046,"bytes":13511,"identity":"train:44b6_415c0a3a","ordinal":49,"sha256":"a1f944ac8ec08154eba124d9e899d4530343b4cf246cc1c4a157c717b5aa999c"},{"action_class":"occupied_swap","action_count":1815,"bytes":21535,"identity":"train:44b6_551a5dba","ordinal":50,"sha256":"6561e09961317bdbb08d88a68609bf05186f18750b03c20901ab49bbc6d39d55"},{"action_class":"occupied_swap","action_count":511,"bytes":7404,"identity":"train:44b6_5740d24b","ordinal":51,"sha256":"1f4e4fa543c84a1f11201d7600a9b2e6a771706ce073e53a8b9395613b37923d"},{"action_class":"occupied_swap","action_count":727,"bytes":9913,"identity":"train:44b6_587a1e22","ordinal":52,"sha256":"7a651c5710b97547119accbcf330d9530b7f13bdbdcee4a33a12183bc1dd78c4"},{"action_class":"occupied_swap","action_count":1027,"bytes":13083,"identity":"train:44b6_5f15d135","ordinal":53,"sha256":"b93fd584bfd1e577112f86cd4de1e3f63a58acbbae43fc9ffa31e03837380e7d"},{"action_class":"occupied_swap","action_count":1039,"bytes":13490,"identity":"train:44b6_7e557709","ordinal":54,"sha256":"a7be36313f2db781e99c04a69f106b2b3608cf46da20eb9a9142ab3b0ace5f28"},{"action_class":"occupied_swap","action_count":532,"bytes":7672,"identity":"train:44b6_808952d6","ordinal":55,"sha256":"9ec7d0810a2e1a5d43e5246cdb20f3e54801500f2ffa2e2e57e8519bd192bf3d"},{"action_class":"occupied_swap","action_count":141,"bytes":3314,"identity":"train:44b6_81c256f0","ordinal":56,"sha256":"b6758590748485dfca0ccd4ccfc0e4e8620acd53117d13e7cace60d201994721"},{"action_class":"occupied_swap","action_count":1649,"bytes":19698,"identity":"train:44b6_8f5ab931","ordinal":57,"sha256":"f63d8c3ef50620517263a51b6dcb01c50ac755e3f1560b774ae946fd6e590f61"},{"action_class":"occupied_swap","action_count":756,"bytes":10256,"identity":"train:44b6_90724892","ordinal":58,"sha256":"84b762acc3aa2d51df6fa7d727ca18b9dd04e7509ca38ffeababbffc639148e9"},{"action_class":"occupied_swap","action_count":2211,"bytes":25919,"identity":"train:44b6_949adeb1","ordinal":59,"sha256":"4a71ece881c0a261c6f3a1861ee802e750a42b4aa2f7954841e9c2c0a0f90f40"},{"action_class":"occupied_swap","action_count":627,"bytes":8918,"identity":"train:44b6_97725144","ordinal":60,"sha256":"92f1fee15a6b62782694b84bd7c4349686cbd0e01824ba195118046e706f0bcd"},{"action_class":"occupied_swap","action_count":1297,"bytes":16069,"identity":"train:44b6_9bfa6a0a","ordinal":61,"sha256":"98391daf61e557916acdf28b12b3a56205654dc73cb611b9645b7c0d2a0e4656"},{"action_class":"occupied_swap","action_count":3725,"bytes":41951,"identity":"train:44b6_a2bb48bb","ordinal":62,"sha256":"1b445e6ccdcad1c63f4f41856a845fbb3a675fa0c0fd394e58a2452e32f3c565"},{"action_class":"occupied_swap","action_count":895,"bytes":11610,"identity":"train:44b6_abf82518","ordinal":63,"sha256":"995ed9aa8625836402bdf66ebc669275860494d1c6bd8329d2888e06f78b54ce"},{"action_class":"occupied_swap","action_count":622,"bytes":8767,"identity":"train:44b6_c15fded2","ordinal":64,"sha256":"ff4114e1888750919e64c5960cda765420178adabf71185fd485136d06a5caa1"},{"action_class":"occupied_swap","action_count":3140,"bytes":35714,"identity":"train:44b6_c50204e0","ordinal":65,"sha256":"84149e7781d32c2c840c652acaf570fafe138bc3af5126ed08d05e668c9efd61"},{"action_class":"occupied_swap","action_count":1458,"bytes":17766,"identity":"train:44b6_c771cb04","ordinal":66,"sha256":"8470705a299ca11934e054cee9c90464d47b38abbd4bfc89ae63d28c0f97d7f0"},{"action_class":"occupied_swap","action_count":1019,"bytes":13318,"identity":"train:44b6_cf8fed6b","ordinal":67,"sha256":"8dd4329c6c28bf711660a683fe872c3bd302c30c8a9a23052b8597e777ca62d1"},{"action_class":"occupied_swap","action_count":565,"bytes":8317,"identity":"train:44b6_d29c9ab2","ordinal":68,"sha256":"24b8c6ebd8fbb9214a829a87b822cb3dee01dc4164406c1d47c9e8b880fded11"},{"action_class":"occupied_swap","action_count":72,"bytes":2378,"identity":"train:44b6_d754aa59","ordinal":69,"sha256":"6b6b1b5c906ea2dc47cc966ad4d3f4c0f74d0d942d096d61c8dfe62b7f23a6c8"},{"action_class":"occupied_swap","action_count":1062,"bytes":13530,"identity":"train:44b6_d78e09d9","ordinal":70,"sha256":"c4b868fe1604fad988be35937b191273b6280037a5639f29ba4a6d3c0363e4c5"},{"action_class":"occupied_swap","action_count":573,"bytes":8164,"identity":"train:44b6_db3c847b","ordinal":71,"sha256":"6c250db6874f406cd62221d2c67842b0d8e64cea70146f4e566369382aabffe7"},{"action_class":"occupied_swap","action_count":547,"bytes":8005,"identity":"train:44b6_e28840c6","ordinal":72,"sha256":"6283b81cbe45b77ef98c888fb50cc590d8e29ca7d0a5f5df70007ec87bd2d31e"},{"action_class":"occupied_swap","action_count":776,"bytes":10293,"identity":"train:44b6_e29f0176","ordinal":73,"sha256":"02edc887365cbd46f26196b81452bdaaaea47f026ef3567f4216454da24eef25"},{"action_class":"occupied_swap","action_count":2851,"bytes":33346,"identity":"train:44b6_e31261b4","ordinal":74,"sha256":"9efdccabd987a29d281da932c70bcb93d8e748434d007412b643ccfb6e646639"},{"action_class":"occupied_swap","action_count":947,"bytes":12351,"identity":"train:44b6_e35b117d","ordinal":75,"sha256":"2132e2c09a7ccc546ffaab239daf7924333cebf60d6a43f6809d2396eb54c535"},{"action_class":"occupied_swap","action_count":1086,"bytes":14042,"identity":"train:44b6_eb2880fc","ordinal":76,"sha256":"ce07348d0fc86265e2c4ca13fed326a0632ececc72a75adf690f50f1223707fc"}],"safe_loader":{"bytes":31010,"filename":"cellect_model.py","sha256":"2394797f4cb680cd45136c83792df67a3ee8915b6925994a6135d4425998b0c0"},"schema":"celltrack-alpha.cellect-embedded-runtime-inputs.v1","wheels":[{"bytes":21592,"filename":"donfig-0.8.1.post1-py3-none-any.whl","sha256":"2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d"},{"bytes":8879163,"filename":"numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl","sha256":"c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3"},{"bytes":319617,"filename":"zarr-3.2.1-py3-none-any.whl","sha256":"f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7"}]}')
EMBEDDED_ASSET_CONTRACT_SHA = "e84a62dc12a92fdb323ac4a5edf2927a8a33163d16f1ca06f213a2a9320c3bba"


class Step0InputError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _inventory(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise Step0InputError(f"input root is not a directory: {root}")
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file():
            by_hash[sha256(candidate)].append(candidate)
    return dict(by_hash)


def _unique_content(
    by_hash: dict[str, list[Path]],
    record: dict[str, object],
    *,
    label: str,
) -> Path:
    expected_sha256 = record.get("sha256")
    expected_bytes = record.get("bytes")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise Step0InputError(f"{label} has invalid embedded pins")
    matches = by_hash.get(expected_sha256, [])
    if len(matches) != 1:
        raise Step0InputError(
            f"{label} content resolution expected one match, found {len(matches)}"
        )
    path = matches[0]
    if path.stat().st_size != expected_bytes:
        raise Step0InputError(f"{label} byte count drift")
    return path


def _materialize(
    source: Path,
    work_root: Path,
    relative: Path,
    record: dict[str, object],
) -> Path:
    target = work_root / "cellect_step0_materialized" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
        raise Step0InputError(f"materialized input drift: {relative}")
    return target


def _checkpoint(
    by_hash: dict[str, list[Path]],
    contract: dict[str, object],
    work_root: Path,
) -> Path:
    record = contract["checkpoint"]
    direct = by_hash.get(str(record["sha256"]), [])
    if len(direct) > 1:
        raise Step0InputError("checkpoint content resolution found duplicates")
    if len(direct) == 1:
        if direct[0].stat().st_size != record["bytes"]:
            raise Step0InputError("checkpoint byte count drift")
        return _materialize(
            direct[0], work_root, Path(str(record["filename"])), record
        )
    containers = by_hash.get(str(record["container_sha256"]), [])
    if len(containers) != 1:
        raise Step0InputError(
            f"checkpoint container resolution expected one match, found {len(containers)}"
        )
    if containers[0].stat().st_size != record["container_bytes"]:
        raise Step0InputError("checkpoint container byte count drift")
    matches: list[bytes] = []
    try:
        archive = tarfile.open(containers[0], mode="r:")
    except tarfile.TarError as error:
        raise Step0InputError("checkpoint container is not an uncompressed tar") from error
    with archive:
        for member in archive:
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise Step0InputError("checkpoint tar member is unreadable")
            payload = handle.read()
            if hashlib.sha256(payload).hexdigest() == record["sha256"]:
                if len(payload) != record["bytes"]:
                    raise Step0InputError("checkpoint tar member byte count drift")
                matches.append(payload)
    if len(matches) != 1:
        raise Step0InputError(
            f"checkpoint tar-member resolution expected one match, found {len(matches)}"
        )
    target = work_root / "cellect_step0_materialized" / str(record["filename"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(matches[0])
    if sha256(target) != record["sha256"]:
        raise Step0InputError("checkpoint changed during materialization")
    return target


def run_cellect_step0(input_root: Path, work_root: Path) -> dict[str, object]:
    """Resolve every required input from code-embedded bytes/SHA pins only."""
    started = time.monotonic()
    contract = EMBEDDED_ASSET_CONTRACT
    if _canonical_sha256(contract) != EMBEDDED_ASSET_CONTRACT_SHA:
        raise Step0InputError("embedded asset contract hash drift")
    requests = contract.get("requests")
    if (
        not isinstance(requests, list)
        or len(requests) != contract.get("request_file_count")
        or contract.get("request_file_count") != 77
        or contract.get("request_count") != 1391458
    ):
        raise Step0InputError("embedded frozen request census drift")
    by_hash = _inventory(input_root)

    population = _unique_content(
        by_hash, contract["population"], label="population"
    )
    safe_loader_source = _unique_content(
        by_hash, contract["safe_loader"], label="safe loader"
    )
    safe_loader = _materialize(
        safe_loader_source,
        work_root,
        Path(str(contract["safe_loader"]["filename"])),
        contract["safe_loader"],
    )
    checkpoint = _checkpoint(by_hash, contract, work_root)

    wheels: dict[str, str] = {}
    for record in contract["wheels"]:
        source = _unique_content(
            by_hash, record, label=f"offline wheel {record['filename']}"
        )
        wheel = _materialize(
            source, work_root, Path("wheels") / str(record["filename"]), record
        )
        wheels[str(record["filename"])] = str(wheel)

    resolved_requests: list[dict[str, object]] = []
    observed_actions = 0
    seen_semantics: set[tuple[str, str]] = set()
    class_files: Counter[str] = Counter()
    class_actions: Counter[str] = Counter()
    for ordinal, record in enumerate(requests):
        action_class = record.get("action_class")
        identity = record.get("identity")
        action_count = record.get("action_count")
        if action_class not in {"delete", "occupied_swap"}:
            raise Step0InputError("embedded request class drift")
        if not isinstance(identity, str) or not identity.startswith("train:44b6_"):
            raise Step0InputError("embedded request identity drift")
        if not isinstance(action_count, int) or action_count <= 0:
            raise Step0InputError("embedded request count drift")
        semantic = (action_class, identity)
        if semantic in seen_semantics:
            raise Step0InputError("embedded request semantic duplicate")
        seen_semantics.add(semantic)
        path = _unique_content(
            by_hash, record, label=f"request payload {ordinal}"
        )
        resolved_requests.append(
            {
                "ordinal": ordinal,
                "action_class": action_class,
                "identity": identity,
                "action_count": action_count,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "resolved_path": str(path),
            }
        )
        observed_actions += action_count
        class_files[action_class] += 1
        class_actions[action_class] += action_count
    if observed_actions != contract["request_count"]:
        raise Step0InputError("resolved request action total drift")
    if dict(class_files) != {"delete": 41, "occupied_swap": 36}:
        raise Step0InputError("resolved request class-file census drift")
    if dict(class_actions) != {"delete": 1345497, "occupied_swap": 45961}:
        raise Step0InputError("resolved request class-action census drift")

    result = {
        "schema": "celltrack-alpha.cellect-embedded-step0.v1",
        "status": "PASS_SELF_CONTAINED_EXPANDED_MOUNT",
        "elapsed_seconds": time.monotonic() - started,
        "asset_contract_sha256": EMBEDDED_ASSET_CONTRACT_SHA,
        "inventory_file_count": sum(len(paths) for paths in by_hash.values()),
        "population_manifest_path": str(population),
        "population_manifest_sha256": contract["population"]["sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": contract["checkpoint"]["sha256"],
        "safe_loader_path": str(safe_loader),
        "safe_loader_sha256": contract["safe_loader"]["sha256"],
        "wheels": wheels,
        "request_file_count": len(resolved_requests),
        "request_count": observed_actions,
        "class_file_counts": dict(class_files),
        "class_action_counts": dict(class_actions),
        "request_records": resolved_requests,
        "external_manifest_files_required": 0,
        "heavy_work_started": False,
    }
    print(json.dumps({"step": 0, **result}, sort_keys=True), flush=True)
    return result

# END INLINE CONTENT-ADDRESSED STEP 0

BOOT_STARTED = time.monotonic()

WEIGHT_SHA = "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"
SAFE_WEIGHT_NAME = "cellect_uext_x3rd_149_0_4p6540.pth"
POPULATION_MANIFEST_SHA = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
CELLECT_INPUT_ROOT = Path(os.environ.get(
    "CELLECT_INPUT_ROOT_OVERRIDE",
    "/kaggle/input/datasets/onebitbrain/g02-cellect-r7-published-assets-v1",
))
WHEEL_SHA256 = {
    "donfig-0.8.1.post1-py3-none-any.whl": (
        "2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d"
    ),
    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": (
        "c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3"
    ),
    "zarr-3.2.1-py3-none-any.whl": (
        "f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7"
    ),
}


class FeatureError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_tar_member(name: str) -> bool:
    member = Path(name)
    return (
        bool(name)
        and name.isascii()
        and not member.is_absolute()
        and ".." not in member.parts
        and member.name == SAFE_WEIGHT_NAME
    )


def resolve_weight_by_content(
    input_root: Path,
    work_root: Path,
    expected_sha256: str = WEIGHT_SHA,
) -> Path:
    """Resolve the one exact checkpoint from direct files or uncompressed tar members."""
    started = time.monotonic()
    observations: list[dict[str, object]] = []
    matches: list[tuple[str, Path]] = []

    for suffix in ("*.pth", "*.bin"):
        for candidate in sorted(input_root.rglob(suffix)):
            if not candidate.is_file():
                continue
            digest = sha256(candidate)
            observations.append(
                {
                    "kind": "file",
                    "path": str(candidate),
                    "bytes": candidate.stat().st_size,
                    "sha256": digest,
                }
            )
            if digest == expected_sha256:
                matches.append((f"file:{candidate}", candidate))

    extraction_root = work_root / "cellect_checkpoint_step0"
    extraction_root.mkdir(parents=True, exist_ok=True)
    for archive_ordinal, archive in enumerate(sorted(input_root.rglob("*.tar"))):
        if not archive.is_file():
            continue
        try:
            handle = tarfile.open(archive, mode="r:")
        except tarfile.TarError as error:
            raise FeatureError(f"invalid uncompressed tar {archive}: {error}") from error
        with handle:
            for member_ordinal, member in enumerate(handle):
                if not member.isfile():
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise FeatureError(f"cannot read tar member {archive}:{member.name}")
                temporary = extraction_root / (
                    f"candidate_{archive_ordinal:04d}_{member_ordinal:04d}.partial"
                )
                digest = hashlib.sha256()
                size = 0
                with temporary.open("wb") as destination:
                    for block in iter(lambda: extracted.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                        destination.write(block)
                observed_sha256 = digest.hexdigest()
                observations.append(
                    {
                        "kind": "tar_member",
                        "path": str(archive),
                        "member": member.name,
                        "bytes": size,
                        "sha256": observed_sha256,
                    }
                )
                if observed_sha256 != expected_sha256:
                    temporary.unlink()
                    continue
                if not _safe_tar_member(member.name):
                    temporary.unlink()
                    raise FeatureError(
                        f"matching checkpoint uses unsafe or unexpected tar member name: {member.name!r}"
                    )
                resolved = extraction_root / SAFE_WEIGHT_NAME
                if resolved.exists():
                    resolved = extraction_root / (
                        f"{archive_ordinal:04d}_{member_ordinal:04d}_{SAFE_WEIGHT_NAME}"
                    )
                temporary.replace(resolved)
                matches.append((f"tar:{archive}:{member.name}", resolved))

    if len(matches) != 1:
        summary = json.dumps(
            {
                "expected_sha256": expected_sha256,
                "match_count": len(matches),
                "matches": [source for source, _path in matches],
                "observations": observations,
            },
            sort_keys=True,
        )
        for _source, path in matches:
            if extraction_root in path.parents and path.exists():
                path.unlink()
        raise FeatureError(f"checkpoint content resolution failed: {summary}")

    source, resolved = matches[0]
    if source.startswith("file:"):
        destination = extraction_root / SAFE_WEIGHT_NAME
        shutil.copyfile(resolved, destination)
        resolved = destination
    if sha256(resolved) != expected_sha256:
        raise FeatureError("checkpoint changed during step-0 resolution")
    print(
        json.dumps(
            {
                "step": 0,
                "checkpoint_source": source,
                "resolved_path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": expected_sha256,
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return resolved


# Step 0 authenticates every required input from embedded pins before
# NumPy, torch, zarr, dependency installation, model loading, or inference.
STEP0 = (
    run_cellect_step0(
        CELLECT_INPUT_ROOT,
        Path(os.environ.get("CELLECT_WORK_ROOT_OVERRIDE", "/kaggle/working")),
    )
    if __name__ == "__main__"
    else None
)
RESOLVED_WEIGHT = Path(str(STEP0["checkpoint_path"])) if STEP0 is not None else None
if __name__ == "__main__" and os.environ.get("CELLECT_STEP0_ONLY") == "1":
    print(json.dumps({"step0_only": True, "status": STEP0["status"]}, sort_keys=True))
    raise SystemExit(0)

import numpy as np
import torch

if importlib.util.find_spec("zarr") is None:
    if STEP0 is None:
        raise RuntimeError("step-0 input preflight did not run")
    wheels = STEP0["wheels"]
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheels["donfig-0.8.1.post1-py3-none-any.whl"]),
            str(
                wheels[
                    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
                ]
            ),
            str(wheels["zarr-3.2.1-py3-none-any.whl"]),
        ]
    )

import zarr

AMENDMENT_SHA = "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e"
SUPPLIER_SHA = "8d34445eafa19dcd135b59747c6ccb4314b0e2ca19f28e80fcfa752bab9d3664"
MODEL_SOURCE_SHA = "2394797f4cb680cd45136c83792df67a3ee8915b6925994a6135d4425998b0c0"
ASSET_DATASET_VERSION = 6
TILE_ZYX = (16, 64, 64)
BATCH = 8
HARD_KILL_SECONDS = 5 * 3600
SOFT_STOP_SECONDS = HARD_KILL_SECONDS - 300
OUTPUT = Path("/kaggle/working/cellect_action_features")


def locate_one(pattern: str) -> Path:
    matches = sorted(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FeatureError(f"expected one {pattern}, found {len(matches)}")
    return matches[0]


def action_features(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 64:
        raise FeatureError("embedding pair shape drift")
    difference = source - target
    src_norm = np.linalg.norm(source, axis=1)
    tgt_norm = np.linalg.norm(target, axis=1)
    denominator = np.maximum(src_norm * tgt_norm, 1e-8)
    centered_source = source - source.mean(axis=1, keepdims=True)
    centered_target = target - target.mean(axis=1, keepdims=True)
    corr_denominator = np.maximum(
        np.linalg.norm(centered_source, axis=1)
        * np.linalg.norm(centered_target, axis=1),
        1e-8,
    )
    return np.stack(
        (
            np.sum(source * target, axis=1) / denominator,
            np.linalg.norm(difference, axis=1) / 8.0,
            np.mean(np.abs(difference), axis=1),
            np.mean(source * target, axis=1),
            src_norm / 8.0,
            tgt_norm / 8.0,
            (src_norm - tgt_norm) / 8.0,
            np.sum(centered_source * centered_target, axis=1) / corr_denominator,
        ),
        axis=1,
    ).astype(np.float32)


def permutation(values: np.ndarray, node_ids: np.ndarray, identity: str, frame: int) -> np.ndarray:
    unique_ids, first = np.unique(node_ids, return_index=True)
    vectors = values[first]
    seed = f"R-SEPARATION-FALSIFIER-44B6{SUPPLIER_SHA}{identity}{frame}"
    order = sorted(
        range(len(unique_ids)),
        key=lambda index: (
            hashlib.sha256(f"{seed}{int(unique_ids[index])}".encode("ascii")).digest(),
            int(unique_ids[index]),
        ),
    )
    replacement = {}
    if order:
        for position, index in enumerate(order):
            replacement[int(unique_ids[index])] = vectors[order[(position + 1) % len(order)]]
    return np.stack([replacement[int(node)] for node in node_ids]).astype(np.float32)


def preprocess(volume: torch.Tensor) -> torch.Tensor:
    positive = volume > 0
    if not bool(positive.any().item()):
        raise FeatureError("tile contains no positive intensity")
    minimum = volume[positive].amin()
    return torch.log1p(torch.where(volume < minimum, minimum, volume) + 1900.0)


def count_tile_jobs(request: dict[str, np.ndarray]) -> int:
    source_keys = np.column_stack(
        (
            request["source_t"],
            request["source_zyx"][:, 0] // TILE_ZYX[0],
            request["source_zyx"][:, 1] // TILE_ZYX[1],
            request["source_zyx"][:, 2] // TILE_ZYX[2],
        )
    )
    target_keys = np.column_stack(
        (
            request["target_t"] - 1,
            request["target_zyx"][:, 0] // TILE_ZYX[0],
            request["target_zyx"][:, 1] // TILE_ZYX[1],
            request["target_zyx"][:, 2] // TILE_ZYX[2],
        )
    )
    return int(len(np.unique(np.concatenate((source_keys, target_keys)), axis=0)))


def group_endpoint_indices(
    frames: np.ndarray, coordinates: np.ndarray, *, target_role: bool
) -> dict[tuple[int, int, int, int], np.ndarray]:
    keys = np.column_stack(
        (
            frames - int(target_role),
            coordinates[:, 0] // TILE_ZYX[0],
            coordinates[:, 1] // TILE_ZYX[1],
            coordinates[:, 2] // TILE_ZYX[2],
        )
    )
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return {
        tuple(int(value) for value in key): order[offsets[index] : offsets[index + 1]]
        for index, key in enumerate(unique)
    }


def infer_video(
    model: torch.nn.Module,
    array: zarr.Array,
    request: dict[str, np.ndarray],
    started: float,
    completed_before: int,
    total_planned_jobs: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    count = len(request["source_id"])
    source_embedding = np.empty((count, 64), dtype=np.float32)
    target_embedding = np.empty((count, 64), dtype=np.float32)
    source_written = np.zeros(count, dtype=bool)
    target_written = np.zeros(count, dtype=bool)
    source_groups = group_endpoint_indices(
        request["source_t"], request["source_zyx"], target_role=False
    )
    target_groups = group_endpoint_indices(
        request["target_t"], request["target_zyx"], target_role=True
    )
    empty = np.empty(0, dtype=np.int64)
    tile_jobs = [
        (
            key[0],
            key[1] * TILE_ZYX[0],
            key[2] * TILE_ZYX[1],
            key[3] * TILE_ZYX[2],
            source_groups.get(key, empty),
            target_groups.get(key, empty),
        )
        for key in sorted(source_groups.keys() | target_groups.keys())
    ]
    if len(tile_jobs) != count_tile_jobs(request):
        raise FeatureError("tile grouping drift")
    completed = 0
    for offset in range(0, len(tile_jobs), BATCH):
        if time.monotonic() - started >= SOFT_STOP_SECONDS:
            raise FeatureError("five-hour hard-kill guard approached")
        jobs = tile_jobs[offset : offset + BATCH]
        volumes = []
        for frame, z0, y0, x0, _src, _tgt in jobs:
            forward = np.asarray(
                array[frame : frame + 2, z0 : z0 + 16, y0 : y0 + 64, x0 : x0 + 64],
                dtype=np.float32,
            )
            volumes.extend((forward, forward[::-1].copy()))
        tensor = torch.from_numpy(np.stack(volumes)).permute(0, 1, 3, 4, 2).to("cuda")
        with torch.inference_mode():
            embedding = model(preprocess(tensor)).embedding.float().cpu().numpy()
        for job_index, (_frame, z0, y0, x0, src, tgt) in enumerate(jobs):
            forward = embedding[2 * job_index]
            reverse = embedding[2 * job_index + 1]
            if len(src):
                point = request["source_zyx"][src] - np.asarray((z0, y0, x0))
                source_embedding[src] = forward[:, point[:, 1], point[:, 2], point[:, 0]].T
                source_written[src] = True
            if len(tgt):
                point = request["target_zyx"][tgt] - np.asarray((z0, y0, x0))
                target_embedding[tgt] = reverse[:, point[:, 1], point[:, 2], point[:, 0]].T
                target_written[tgt] = True
        completed += len(jobs)
        global_completed = completed_before + completed
        if global_completed >= min(64, total_planned_jobs):
            elapsed = time.monotonic() - started
            projection = elapsed / global_completed * total_planned_jobs
            if projection >= HARD_KILL_SECONDS:
                raise FeatureError(f"authenticated projection exceeds five hours: {projection / 3600:.3f}")
    if not source_written.all() or not target_written.all():
        raise FeatureError("endpoint feature coverage is incomplete")
    return source_embedding, target_embedding, len(tile_jobs)


def main() -> int:
    started = BOOT_STARTED
    if RESOLVED_WEIGHT is None or STEP0 is None:
        raise FeatureError("step-0 all-input preflight did not run")
    if not torch.cuda.is_available():
        raise FeatureError("CUDA is required for the bounded feature sample")
    model_source = Path(str(STEP0["safe_loader_path"]))
    weight = RESOLVED_WEIGHT
    if sha256(weight) != WEIGHT_SHA:
        raise FeatureError("CELLECT weight hash mismatch")
    if sha256(model_source) != MODEL_SOURCE_SHA:
        raise FeatureError("CELLECT safe-loader hash mismatch")
    specification = importlib.util.spec_from_file_location(
        "cellect_authenticated_model", model_source
    )
    if specification is None or specification.loader is None:
        raise FeatureError("cannot create CELLECT safe-loader module specification")
    model_module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = model_module
    specification.loader.exec_module(model_module)
    load_published_backbone = model_module.load_published_backbone

    model = load_published_backbone(weight, device="cuda").eval()
    request_records = STEP0["request_records"]
    if not request_records:
        raise FeatureError("step-0 returned no frozen feature requests")
    planned_jobs = 0
    for record in request_records:
        request_path = Path(str(record["resolved_path"]))
        with np.load(request_path, allow_pickle=False) as loaded:
            request = {key: loaded[key] for key in loaded.files}
        planned_jobs += count_tile_jobs(request)
    if planned_jobs < 64:
        raise FeatureError("insufficient tile jobs for the frozen runtime projection")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    files = []
    total_jobs = 0
    for ordinal, record in enumerate(request_records, start=1):
        request_path = Path(str(record["resolved_path"]))
        action_class = str(record["action_class"])
        identity = str(record["identity"])
        dataset = identity.split(":", 1)[1]
        raw = locate_one(f"{dataset}.zarr")
        array = zarr.open_array(str(raw / "0"), mode="r")
        with np.load(request_path, allow_pickle=False) as loaded:
            request = {key: loaded[key] for key in loaded.files}
        source, target, jobs = infer_video(
            model,
            array,
            request,
            started,
            total_jobs,
            planned_jobs,
        )
        a1 = action_features(source, target)
        a0_source = np.empty_like(source)
        a0_target = np.empty_like(target)
        for frame in np.unique(request["source_t"]):
            index = np.flatnonzero(request["source_t"] == frame)
            a0_source[index] = permutation(source[index], request["source_id"][index], identity, int(frame))
        for frame in np.unique(request["target_t"]):
            index = np.flatnonzero(request["target_t"] == frame)
            a0_target[index] = permutation(target[index], request["target_id"][index], identity, int(frame))
        a0 = action_features(a0_source, a0_target)
        destination = OUTPUT / action_class / (
            identity.replace(":", "__", 1) + ".npz"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, a1=a1, a0=a0)
        files.append(
            {
                "identity": identity,
                "action_class": action_class,
                "count": len(a1),
                "request_sha256": sha256(request_path),
                "output": str(destination.relative_to(OUTPUT)),
                "output_sha256": sha256(destination),
                "tile_jobs": jobs,
            }
        )
        total_jobs += jobs
        print(f"{ordinal}/{len(request_records)} {action_class} {identity} actions={len(a1)} jobs={jobs}", flush=True)
    if total_jobs != planned_jobs:
        raise FeatureError(f"tile-job plan drift: {total_jobs} != {planned_jobs}")
    manifest = {
        "schema": "celltrack-alpha.cellect-action-features.v1",
        "status": "COMPLETE",
        "amendment_sha256": AMENDMENT_SHA,
        "supplier_sha256": SUPPLIER_SHA,
        "weight_sha256": WEIGHT_SHA,
        "safe_loader_sha256": MODEL_SOURCE_SHA,
        "asset_dataset_version": ASSET_DATASET_VERSION,
        "device": torch.cuda.get_device_name(0),
        "tile_zyx": list(TILE_ZYX),
        "batch": BATCH,
        "action_feature_names": [
            "cosine",
            "euclidean_over_sqrt64",
            "mean_absolute_difference",
            "mean_product",
            "source_norm_over_sqrt64",
            "target_norm_over_sqrt64",
            "norm_difference_over_sqrt64",
            "centered_correlation",
        ],
        "files": files,
        "total_tile_jobs": total_jobs,
        "planned_tile_jobs": planned_jobs,
        "elapsed_seconds": time.monotonic() - started,
        "full_pool_cache": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(files), "jobs": total_jobs, "seconds": manifest["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
