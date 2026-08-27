#!/usr/bin/env python3
"""Static and attested prelaunch guard for G-02 Kaggle kernels."""

from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any
import zipfile
import zlib


HEAVY_IMPORTS = {
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "tracksdata",
    "zarr",
}
FORBIDDEN_DISCOVERY_SNIPPETS = (
    '.glob("**/',
    ".glob('**/",
    ".rglob(",
)
PRISTINE_PUBLIC_V32_NOTEBOOK_SHA256 = (
    "3f995a97a51c9517bdf8d8da2f606ffd184a51c841de87c962d911a336b7847d"
)
PROJECT_ROOT = Path(__file__).resolve().parents[5]
VISUAL_RECEIPT_SCHEMA = "celltrack-alpha.local-visual-preflight.v1"
VISUAL_SIDECAR_SCHEMA = "celltrack-alpha.local-visual-sidecar.v1"
VISUAL_INPUT_MANIFEST_SCHEMA = "celltrack-alpha.local-visual-input-manifest.v1"
REMOTE_VISUAL_ONLY_PRELAUNCH_SCHEMA = "celltrack-alpha.remote-visual-only-prelaunch.v1"
REMOTE_VISUAL_ONLY_CONTRACT_SCHEMA = "celltrack-alpha.gpu-visual-only-contract.v1"
REMOTE_VISUAL_ONLY_EXCEPTION_SCHEMA = "celltrack-alpha.remote-visual-only-exception.v1"
REMOTE_VISUAL_REVIEW_RECEIPT_SCHEMA = "celltrack-alpha.remote-visual-review.v1"
REMOTE_VISUAL_ONLY_EXECUTION_PROFILE = "remote_visual_only_resource_stop.v1"
CERTIFIED_SWEEP_WRAPPER_EXECUTION_PROFILE = "certified_sweep_wrapper.v1"
M7R6_NATIVE_T4_CERT_EXECUTION_PROFILE = "m7r6_native_t4_whole_pipeline_cert.v1"
M7R6_NATIVE_T4_CERT_ROUTE = {
    "kernel_id": "onebitbrain/m7r6-native-t4-whole-pipeline-cert-v1",
    "code_sha256": "c3e5b29e351e97214631313d327ef9c9dd2976dc36deee664241e6fa2552c775",
    "metadata_sha256": "39f0a049b584c4637f9ce60dd01d154812deda003d22554f2e7f4077967277c3",
    "package_tree_sha256": "17df098684f712d47e7b32382a8b992bac7c16a83daec4632d4d1840b8d72fb1",
    "package_inventory_sha256": "d1f28c579c6a5e6e6fccd00a6717ed64f5ff6db093687de8976ae2dae84abc8d",
    "m7r5_source_sha256": "9a5a1429d95ba04900e27b5fa043085cb0de4d54fa92dbe7050a071b8664d5ac",
    "m7r5_terminal_sha256": "849ec6df0b68b7ba3ba84edb45b971341a916aab8f304d7027f7d1dc3486296a",
    "m7r5_rejected_receipt_sha256": "64839f2bd56a49b9d1a84eb852939af04c74f3b768f5f68a91383547aeb763dd",
    "authenticated_child_sha256": "c217eed8ae26139aba627fd26ceffd989bf5a2f1fea76685eff41b265abd20ea",
    "ship_profile_sha256": "56eb4c74808f0e59dbab8e6e490e77d9929a8f6e230ac0f6d115eb8dc9b85270",
    "experiment_plan_sha256": "58955bf1247a779918835ea4bf1a1e18c7f03373d96116ebe5c1993279d8cc88",
    "downloaded_evidence_sealer_sha256": "f4d841f40b4eabb3f129e3a42cdaf546b79dfc847f6731b5d8357a5eef878b72",
}
M7R7_NATIVE_RUNTIME_DIAGNOSTIC_EXECUTION_PROFILE = (
    "m7r7_native_runtime_tuple_diagnostic.v1"
)
M7R7_NATIVE_RUNTIME_DIAGNOSTIC_ROUTE = {
    "kernel_id": "onebitbrain/m7r7-native-runtime-tuple-diagnostic-v1",
    "code_sha256": "51fb000a576dfae685aade489112ba46403a262391c6309ac0e00c87f4f16b86",
    "metadata_sha256": "1d975f481729dc2a625903367901007a6bb2839bf5b8dfd53532645b5d948624",
    "package_tree_sha256": "992329cfd050750af84459af47e44209bf37177e5e91462255ff56673ca0f0c4",
    "package_inventory_sha256": "ef6f39853455bbef705e26f5040280197c221af3f23b9a91b2193ee8d0ac84e3",
    "candidate_inventory_sha256": "7a8cca144014f811a3e2be06027aeb3804d081e5d9008b34df767846ea131cf1",
    "environment_sha256": "48c5f4ba443930f15c2d15fd1370f6910281d9f06d2d46536d729ca75ec985ea",
    "experiment_plan_sha256": "d756c835d23dbbd0fec22384c4bdf14df56258c83870c5e6a455d789f457854d",
    "experiment_validation_sha256": "ea8e368b25c29e0c53627f46049fd334e57cf9434d2017eea71e072cba96db23",
    "local_env_probe_sha256": "87b9538913e3eb1f8233b994801cdc6d80e7d28c3df9dbdcb114bd995ace1d1c",
    "ship_profile_sha256": "57580cfce5c482272786024bd0fc3f60a38bcc7b7d2e671faf857d1c0e6ad7c1",
    "ship_certificate_path": (
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r7_native_runtime_tuple_diagnostic/M7R7_SHIP_CERTIFICATION.json"
    ),
    "ship_certificate_sha256": "180582f81faa0fa7ac72ab5b748cfd2877dadcb134a322effb3a0f97c50713f4",
    "ship_certificate_content_sha256": "595c84d11c98c79e9ae0b8d8b29e58c08d845f89ae52dac95892cdfa464ed746",
    "downloaded_runtime_sealer_sha256": "4dba04621387a3c14c8652aaa684a671f948b451087d16aae45bcdaa78fece9b",
    "m7r6_terminal_path": (
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r6_native_t4_whole_pipeline_cert/M7R6_NATIVE_T4_V1_TERMINAL.json"
    ),
    "m7r6_terminal_sha256": "fd96d448aa0370dace640d9ddc4e4de7ab9d8dd8e2e9665a7ad61d7f727041a4",
    "m7r6_status_sha256": "e30dcb307bd7f36b5c6e8369659d1038c6cb043efc758311e84e60e50253486f",
    "sentinel_roundtrip_path": (
        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
        "d2k5_zarr_cv2_dependency_carrier/DATASET_V1_ROUNDTRIP_RECEIPT.json"
    ),
    "sentinel_roundtrip_sha256": "4d8313c647a7090e68f35a7f85a0ad9f14fe9506ac470fcf23c9e874df3bc646",
    "sentinel_mount_correction_path": (
        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
        "d2k5_zarr_cv2_dependency_carrier/DATASET_V1_MOUNT_LAYOUT_CORRECTION.json"
    ),
    "sentinel_mount_correction_sha256": "a9cbc4ba18021a6f45618e1a99a48dce442b7dc0e0dae84193b68bb3a9d5f8f7",
}
M12R4_NO_TARGET_SYNTHETIC_CERT_EXECUTION_PROFILE = (
    "m12r4_no_target_synthetic_model_cert.v1"
)
M12R4_NO_TARGET_SYNTHETIC_CERT_ROUTE = {
    "kernel_id": "onebitbrain/m12r3-native-t4-forward-backward-v1",
    "code_sha256": "2da5b694c881de6a1fa9577a12c27072620ce6ed9e43d6a41817966708d8def6",
    "metadata_sha256": "9a2cd1e6dbb6f08b579c829ca95b2bc57efc57814816c42da51ff96d45f52c09",
    "package_tree_sha256": "d975f7a0a38aeaf1edd19ca78dbb615fcc3aa9a073177d1f77d6b15167538a38",
    "package_inventory_sha256": "ab5bcf093641bf5e3ae9f3625d54a8bfa2574e9e75e92827f06dc4f20cb02c24",
    "candidate_inventory_sha256": "5de6bdb5eb72d95d8f5a4db7056373c303adb071642267156fce30a0d290b4fa",
    "contract_sha256": "981f9f5a15642185cf47ddd1609c832f7d83996cd1033f0818a33990046a4742",
    "minimal_registry_sha256": "b015c11015e9b88cbb21f28f386a84d4a6d9a73c2075c279f555b88a36d5858d",
    "package_equivalence_sha256": "11d3f66df1ee98f6e1360f35775386de15aef39b490f730befc8e97beb3dd20f",
    "experiment_plan_sha256": "32ec71f9bb3d58beb32bdbf2f865d4bcf5338b414ced81c31c6475a269130a78",
    "experiment_validation_sha256": "eac8f32d738f2e6ac8a7765aebe373e061086728afaac7c8baa7e67d370da000",
    "one_shot_sha256": "e9e079144d94734cc16623b863f18b48a77e3a819922d459a45522930c242efb",
    "monitor_sha256": "23e90a06a83e3988365088a49241532eb70621600eedde2c371c7bab1c2b5c81",
    "downloaded_evidence_sealer_sha256": "64253c524659101dbee13f5118c8fe6067b1e38c4a99009d4958ba0fc6b99e5c",
    "m12r1_terminal_sha256": "4d87cd493f71c51be30e515f644870dd43cb8b1000b34a1eadab6454cb34acdc",
    "m12r1_source_sha256": "be51ff3c4596d5082f5273f2e205a8aba7bb8f127c8780e0506a280ea1b26224",
    "m12r1_config_sha256": "0d7327e78d2d0b028b059740a3400bfae98117b52053204135b36c90eba83326",
    "m12r1_synthetic_sha256": "10d66d6949819f81bcf09c153b59916323a5a0dd34c05dd7c6ef8a910193aa8b",
    "m12r2_terminal_sha256": "864fc8a941f292ef3242ff5e477cb7402fca2a5b91e9c7d4f5da431f7c196fb8",
    "m12r2_rejection_sha256": "f46e33afe8efedc1f0ca288ce463d3d11d982ce2071631897b8c54030625f501",
    "m12r3_terminal_sha256": "68565acb8790fa89fb8b29f6bcd5ce40447a25e14007e4ff44e171caefe98302",
    "m12r3_profile_sha256": "d6c2ecf6d3ac7ebe99aef793fce997b29c2607d7825df0e90a0a589acd10e32b",
    "m12r3_sentinel_audit_sha256": "32fb65b1a0edeb9ed004b6ef2bdb91179d0896899b4f997aa0f5fd7d63fa05ac",
    "m12r3_environment_sha256": "6f81fd2cfe06c6af0700076f4bb9e29b1921777cafda5030eb05099b39ea06d4",
    "m12r3_ship_rejection_sha256": "7ff4ad1be93868d46d56c6a2a59f9ac2cabf6849df3d130dd1ed39d50c5b074e",
    "m7r7_terminal_sha256": "4558e67de20cf455b05ae8304f4b7862278d99e4340938d64673f1980a21f5de",
    "m7r7_runtime_seal_sha256": "71dcb5333200643f9525de8c468a18fba17addcfbc92abc704bb03fdd3552a8b",
    "m7r7_runtime_receipt_sha256": "a494acebd281b7d06dfb272064e39447c2d50355723d9588222c010b6898f271",
    "m7r7_runtime_tuple_sha256": "6a982da092de01994011d060bd86a0cc607ef0d97150eb760d550f2ca5c71168",
}
M12R5_NO_TARGET_SYNTHETIC_CERT_EXECUTION_PROFILE = (
    "m12r5_self_orchestrating_no_target_synthetic_cert.v1"
)
M12R5_NO_TARGET_SYNTHETIC_CERT_ROUTE = {
    "kernel_id": "onebitbrain/m12r5-self-orchestrating-no-target-cert-v1",
    "code_sha256": "1e609ae36959cb62a8539db4112714d660836b0142687067d31821270b672fa3",
    "metadata_sha256": "9fa50adc6784f65942c75c30c1bc7308f409b138f604ada63d63746c22ebec45",
    "package_freeze_sha256": "8b0042721a454cc3e87eb540129e8f3fe85c458822e8acf69a3d8eea8378ae5c",
    "package_tree_sha256": "3152fdfcbe38077cc9284136a0938c8caba8636826a7038456c2d0599ee75a20",
    "package_inventory_sha256": "79ab4727ed60b723b555d8e2cb57219098f5a125c30e27447b1dedd0b187c2f1",
    "candidate_inventory_sha256": "ad92b19c2bd3701fef262769a5ae3767b31029953bfd837e68dd2424209b0706",
    "provider_profile_sha256": "dda6da82e929b6e8ce07cd7a23a40e27ec36a0c662094adca1d75b1a000ff2d9",
    "transport_equivalence_sha256": "8cb054f18391259b9c6cfc3a9e5bc97a5e29211a395fc1bfe1e77fee7f7fa8b7",
    "static_certification_sha256": "631f2a95f20ff7dc7bdef7572a9cdba600fa430bd82d780919d75948b6406bad",
    "no_target_contract_sha256": "918e1c9b95489a72933443631496ebbee88cde5dc6b5b2b29b85ffcc78e8b2e2",
    "experiment_plan_sha256": "d96d08679b85926adfda14bcfa2134ec7622e7e2db834a3271090a1c8c1349e7",
    "experiment_validation_sha256": "9a436485add729075c713d967cc99dc42c58e0ea0907e65af493d49b6e62df44",
    "downloaded_evidence_sealer_sha256": "c14b2ef4af3651061ad0a7c0c1668ef1090b2a56541df2d7e4ccf0de9e97a7aa",
    "contract_sha256": "3a92594c94eaa109efe158d876bb8d7f93b432597e4d89d9ba321edefd9e3313",
    "minimal_registry_sha256": "9747be0423ba8f0b914cecc56b9447e9b5c4be26cdac1b7376f86adc423bbd22",
    "m12r3_child_sha256": "2da5b694c881de6a1fa9577a12c27072620ce6ed9e43d6a41817966708d8def6",
    "m12r1_terminal_sha256": "4d87cd493f71c51be30e515f644870dd43cb8b1000b34a1eadab6454cb34acdc",
    "m12r1_source_sha256": "be51ff3c4596d5082f5273f2e205a8aba7bb8f127c8780e0506a280ea1b26224",
    "m12r1_config_sha256": "0d7327e78d2d0b028b059740a3400bfae98117b52053204135b36c90eba83326",
    "m12r1_synthetic_sha256": "10d66d6949819f81bcf09c153b59916323a5a0dd34c05dd7c6ef8a910193aa8b",
    "m12r2_terminal_sha256": "864fc8a941f292ef3242ff5e477cb7402fca2a5b91e9c7d4f5da431f7c196fb8",
    "m12r3_terminal_sha256": "68565acb8790fa89fb8b29f6bcd5ce40447a25e14007e4ff44e171caefe98302",
    "m12r3_sentinel_sha256": "32fb65b1a0edeb9ed004b6ef2bdb91179d0896899b4f997aa0f5fd7d63fa05ac",
    "m12r4_terminal_sha256": "d4484aad61992f8c7d2d417ef26444d566b7d3c199d467376a3f71de2ed9de52",
    "m7r7_terminal_sha256": "4558e67de20cf455b05ae8304f4b7862278d99e4340938d64673f1980a21f5de",
    "m7r7_seal_sha256": "71dcb5333200643f9525de8c468a18fba17addcfbc92abc704bb03fdd3552a8b",
    "m7r7_receipt_sha256": "a494acebd281b7d06dfb272064e39447c2d50355723d9588222c010b6898f271",
    "m7r7_runtime_tuple_sha256": "6a982da092de01994011d060bd86a0cc607ef0d97150eb760d550f2ca5c71168",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M12R6_REMOTE_CERTIFICATION_EXECUTION_PROFILE = (
    "m12r6_exact_no_target_remote_certification.v1"
)
M12R6_REMOTE_CERTIFICATION_EXCEPTION_ALLOWLIST = {
    "m12r6-exact-no-target-native-t4-cert-v1": {
        "kernel_id": "onebitbrain/m12r5-self-orchestrating-no-target-cert-v1",
        "execution_profile": M12R6_REMOTE_CERTIFICATION_EXECUTION_PROFILE,
        "code_sha256": "1e609ae36959cb62a8539db4112714d660836b0142687067d31821270b672fa3",
        "record_path": (
            "dag_prompts/2026-08-23-model-clade-renewal-v1/"
            "m12r6_remote_certification_exception/"
            "M12R6_REMOTE_CERTIFICATION_EXCEPTION.json"
        ),
        "record_sha256": "85e938f8b3fd487d8e33ba61abf5e529459afc2abd9125dd5b4c189b65e150b1",
    }
}
M12R6_REMOTE_CERTIFICATION_ROUTE = {
    "kernel_id": "onebitbrain/m12r5-self-orchestrating-no-target-cert-v1",
    "code_sha256": "1e609ae36959cb62a8539db4112714d660836b0142687067d31821270b672fa3",
    "metadata_sha256": "9fa50adc6784f65942c75c30c1bc7308f409b138f604ada63d63746c22ebec45",
    "package_freeze_sha256": "8b0042721a454cc3e87eb540129e8f3fe85c458822e8acf69a3d8eea8378ae5c",
    "package_tree_sha256": "3152fdfcbe38077cc9284136a0938c8caba8636826a7038456c2d0599ee75a20",
    "package_inventory_sha256": "79ab4727ed60b723b555d8e2cb57219098f5a125c30e27447b1dedd0b187c2f1",
    "exception_sha256": "85e938f8b3fd487d8e33ba61abf5e529459afc2abd9125dd5b4c189b65e150b1",
    "registry_sha256": "7e4cbb4a0a03fc8afa8c7af37f292981db3953a5efa6d89e39654fa22a77e704",
    "candidate_inventory_sha256": "2a2c03a7c3cc4c58c0906851918a840e696f6dc09519142cdaccb90b2620c127",
    "experiment_plan_sha256": "5345b1a6f841c1239627fdcebf6383d12a6d201e34abadeadc5cb95a0f5f68b2",
    "experiment_validation_sha256": "99baae7dafa89c4d0307ad61d88ebdbe2bbbfd6543bbf20c46627e5aaa4bea0d",
    "contract_sha256": "5ef0a240834509ec78c9d672f8d7ba8e36bc09e037409231698edd798fbd5319",
    "one_shot_sha256": "92cb8d9b8ecd12a71465cbeb0f8075a9a07aa2f06c0aaec1e4752ff1147e092f",
    "monitor_sha256": "7579af5531f9bf8b83af978347976cd88a2a401d8e32959f236ddd9856a7d6b2",
    "reconcile_sha256": "f2413f93b3e37ca0e445a27b4508b6647b91e01a04c253475f63122b3d65f147",
    "downloaded_evidence_sealer_sha256": "46eba8b807a5fceb4b87e116035d89738e48c1333a7b54b5e9d0c3b246d0f89f",
    "m12r5_terminal_sha256": "31adff72da1041c6bc29ea8e7cae7918349b273e79b612f8040fb768a80fa3c8",
    "m12r5_static_freeze_sha256": "f36dc982538c94792a2e8bc85b3440e0fbe22d219351ff82875dbed71938ea34",
    "m12r5_inline_sha256": "c8c307e4de6a325c603fc627502ffaa9a9d35ccf4ba6f686dbcb748bf67ebd04",
    "m12r5_profile_sha256": "dda6da82e929b6e8ce07cd7a23a40e27ec36a0c662094adca1d75b1a000ff2d9",
    "m12r5_no_target_sha256": "918e1c9b95489a72933443631496ebbee88cde5dc6b5b2b29b85ffcc78e8b2e2",
    "m12r5_equivalence_sha256": "8cb054f18391259b9c6cfc3a9e5bc97a5e29211a395fc1bfe1e77fee7f7fa8b7",
    "m12r4_terminal_sha256": "d4484aad61992f8c7d2d417ef26444d566b7d3c199d467376a3f71de2ed9de52",
    "m12r3_terminal_sha256": "68565acb8790fa89fb8b29f6bcd5ce40447a25e14007e4ff44e171caefe98302",
    "m12r3_sentinel_sha256": "32fb65b1a0edeb9ed004b6ef2bdb91179d0896899b4f997aa0f5fd7d63fa05ac",
    "m7r7_terminal_sha256": "4558e67de20cf455b05ae8304f4b7862278d99e4340938d64673f1980a21f5de",
    "m7r7_runtime_seal_sha256": "71dcb5333200643f9525de8c468a18fba17addcfbc92abc704bb03fdd3552a8b",
    "m7r7_runtime_receipt_sha256": "a494acebd281b7d06dfb272064e39447c2d50355723d9588222c010b6898f271",
    "m7r7_runtime_tuple_sha256": "6a982da092de01994011d060bd86a0cc607ef0d97150eb760d550f2ca5c71168",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M12R7_CANONICAL_REGISTRY_EXECUTION_PROFILE = (
    "m12r7_canonical_registry_remote_certification.v1"
)
M12R7_CANONICAL_REGISTRY_ROUTE = {
    "kernel_id": "onebitbrain/m12r5-self-orchestrating-no-target-cert-v1",
    "code_sha256": "1e609ae36959cb62a8539db4112714d660836b0142687067d31821270b672fa3",
    "metadata_sha256": "9fa50adc6784f65942c75c30c1bc7308f409b138f604ada63d63746c22ebec45",
    "package_freeze_sha256": "8b0042721a454cc3e87eb540129e8f3fe85c458822e8acf69a3d8eea8378ae5c",
    "package_tree_sha256": "3152fdfcbe38077cc9284136a0938c8caba8636826a7038456c2d0599ee75a20",
    "package_inventory_sha256": "79ab4727ed60b723b555d8e2cb57219098f5a125c30e27447b1dedd0b187c2f1",
    "contract_sha256": "0741e335cc2a7d88ad877faf7b07c073a6c2d7ef89825cb74e0972fad4268896",
    "registry_sha256": "dfe1e2c7eff147da69e3df4e8f73301cdd3b334369ec8edc9a977582a774fc2b",
    "candidate_inventory_sha256": "32a0a5f5c95b1602227e370431a05f8b50a2684885ea8f5bdc87abd5c3e5dca0",
    "transport_equivalence_sha256": "3789d23ab98d3808cd9cb3e4c0c4b2297ef66f8f33bfd1b420c35bc83f8b25e0",
    "experiment_plan_sha256": "5ead057371140a0fc945900ff2a68a708414bb6e15ad2bcaa505fa5903cb369e",
    "experiment_validation_sha256": "086ea9be4774710cae354b993b8d1ed64c7620b24c40ea97a7610958c7869a5c",
    "m12r6_blocker_sha256": "f156f86829f98b47265a8a1415a853d00d381ba92ad822d5ebcbb9d72bba900a",
    "m12r6_terminal_sha256": "447cd77c78c8fe35ee3b12e96aaa3fba629dd087b3fdb3f88da98105b1b8af0a",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M12R8_SYNTHETIC_VISUAL_EXECUTION_PROFILE = (
    "m12r8_synthetic_control_visual_preflight.v1"
)
M12R8_SYNTHETIC_VISUAL_ROUTE = {
    "kernel_id": "onebitbrain/m12r5-self-orchestrating-no-target-cert-v1",
    "code_sha256": "1e609ae36959cb62a8539db4112714d660836b0142687067d31821270b672fa3",
    "metadata_sha256": "9fa50adc6784f65942c75c30c1bc7308f409b138f604ada63d63746c22ebec45",
    "package_freeze_sha256": "8b0042721a454cc3e87eb540129e8f3fe85c458822e8acf69a3d8eea8378ae5c",
    "package_tree_sha256": "3152fdfcbe38077cc9284136a0938c8caba8636826a7038456c2d0599ee75a20",
    "package_inventory_sha256": "79ab4727ed60b723b555d8e2cb57219098f5a125c30e27447b1dedd0b187c2f1",
    "contract_sha256": "2de75904a7eb97cdf17a8dadec66bcb7b6c80c73f097a45c54e727552b90c84f",
    "registry_sha256": "f8a242ecec8c179497b2b0009ba3ec94bb7802d1a4fc171967d0fab2d48fbcf9",
    "candidate_inventory_sha256": "caf73d156d729a218b84f50b2950a29a2dd36ac2738099cf000ab7315fa25bc2",
    "synthetic_control_visual_binding_sha256": "ebe718e0974b6cb731fb2e9072296bc1e6e654abea5ad93ddb11bacc8ee7eabf",
    "experiment_plan_sha256": "53c250ab15fad065bfcfd4ec7124556d24acb3e7954b9a57729344bbdd3ffa12",
    "experiment_validation_sha256": "1ba22970f8053a08eb183fa843c49a0ebec9577b8a433ef7d24d8aea270497b8",
    "m12r7_blocker_sha256": "4d9166faa110e582d3572225abda79538f1f7557745c80dcfca243499c8a4985",
    "m12r7_terminal_sha256": "2bbaba0877b4abb0e785f94b2318bc097ababdcc7ec6cb124020b4bca1400559",
    "m12r1_source_sha256": "be51ff3c4596d5082f5273f2e205a8aba7bb8f127c8780e0506a280ea1b26224",
    "m12r1_config_sha256": "0d7327e78d2d0b028b059740a3400bfae98117b52053204135b36c90eba83326",
    "m12r1_runner_sha256": "4f6f49536e6759c16bdd29cb4147e4d6f9e473631dee969bb887d1b3af966d64",
    "m12r1_synthetic_receipt_sha256": "10d66d6949819f81bcf09c153b59916323a5a0dd34c05dd7c6ef8a910193aa8b",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M7R8_NATIVE_T4_CERT_EXECUTION_PROFILE = "m7r8_native_t4_whole_pipeline_cert.v1"
M7R8_NATIVE_T4_CERT_ROUTE = {
    "kernel_id": "onebitbrain/m7r8-native-t4-whole-pipeline-cert-v1",
    "code_sha256": "a7819ce80b85f9433a7c8918411bafdf6bb011eb7077be5651b4889bdf02598b",
    "metadata_sha256": "4de80d32683510374022a85edeb6ff84a064abe3936b3439cdc1558fcff2fddd",
    "freeze_sha256": "0c36b3e50662b3481f00c3172ca2a93410431dd1af86e8d4382b6e1f33e0fc26",
    "package_tree_sha256": "803253db4eeda967128725b7894bffa57319429facb02ef21cedcefe4eb95ed3",
    "package_inventory_sha256": "39ee5f560f2e1684594bbd4ff085176ef9199f45a250729764951089be883cd5",
    "parent_m7r6_code_sha256": "c3e5b29e351e97214631313d327ef9c9dd2976dc36deee664241e6fa2552c775",
    "m7r6_terminal_sha256": "fd96d448aa0370dace640d9ddc4e4de7ab9d8dd8e2e9665a7ad61d7f727041a4",
    "m7r7_terminal_sha256": "4558e67de20cf455b05ae8304f4b7862278d99e4340938d64673f1980a21f5de",
    "m7r7_runtime_seal_sha256": "71dcb5333200643f9525de8c468a18fba17addcfbc92abc704bb03fdd3552a8b",
    "m7r7_runtime_receipt_sha256": "a494acebd281b7d06dfb272064e39447c2d50355723d9588222c010b6898f271",
    "m7r7_runtime_tuple_sha256": "6a982da092de01994011d060bd86a0cc607ef0d97150eb760d550f2ca5c71168",
    "runtime_tuple_proof_sha256": "4e1552c4822ae427a7ee5bf61eaf0edad600bb9389ba0f541c7b860229d8f1b3",
    "candidate_inventory_sha256": "60d01584901fe81a511000dbd6125f9040dc6e2c17b67abdbeb5d228367059e4",
    "environment_sha256": "a5af15b9798d88eec6407c1494589b30cf8b85782390f86e0a82dc243216f2d9",
    "dependency_audit_sha256": "05c4fc9d4167b48aa20599c2327eb93b326627ca633d2c65dc417adf9466bf7f",
    "experiment_plan_sha256": "e6d95c8cc4f3ed67257cc5df4a6974441022f5b5ed4a4b22bacc5d6446c68fc2",
    "experiment_validation_sha256": "10b405c6c3a9f89602a263b6746e9c9435ae5a2de6a8682a523af763d60ad76a",
    "ship_profile_sha256": "c18e8f06821722b81ae897a8ed8aac793f4208e927142e8f48e551f31f8d6616",
    "downloaded_evidence_sealer_sha256": "adf2a565d24c5cae8f5a461e2c545c0e7474be3f3b0a5f22da820a8dcd38831f",
}
M8R13_NATIVE_T4_EXPORTER_EXECUTION_PROFILE = (
    "m8r13_native_t4_exporter_provider_cert.v1"
)
M8R13_NATIVE_T4_EXPORTER_ROUTE = {
    "kernel_id": "onebitbrain/m8r13-native-t4-exporter-cert-v1",
    "code_sha256": "db41bdc9d9c6713ab75834dda15284dee7ffa547ffefab2a5ee98bb7fee9f280",
    "metadata_sha256": "aa62a8cdac61ab1dd189ffb0531a9ddce09143bb3d592a4da7ca297decd1c382",
    "package_freeze_sha256": "19f815be7bfde03211959ee4748d1497a1f4887718caaeab8d393259017dc1e8",
    "package_tree_sha256": "d145d2bab35ee03f3867c83e3d7ac7e4ac339da368d61fb448a1e465ecd33174",
    "package_inventory_sha256": "e928342b44b5432e18cdd96a31f72d3545f6b9e8fdea24dd4b440de019fa082b",
    "parent_kernel_sha256": "aa71825d8044ea16b8a509418cdfc24ea31190b27cfaa4dcb06a51ccb66c7704",
    "parent_package_freeze_sha256": "3de1a6222af118392fd889baa2c3ae4862e02c89288685dffb06c2c25504f03a",
    "evidence_contract_sha256": "a0d34fee90d74b2a7281707329d8e415e974e2c0127f5f13273c7cf618cc91dd",
    "carrier_roundtrip_sha256": "a01e723933452521c9e00214f87b1f5b30daf89141d29ff68e9433340cf286ac",
    "carrier_action_execution_sha256": "669bc2ae4844b81df38d9ab5252d2044cd929121d8acd58b1fbf706314bddb2e",
    "carrier_authorization_sha256": "735affb06f09d23a96a0050d46545baae41663b692d368faa33aa602398f90da",
    "transport_equivalence_sha256": "659d2bd78554f9497271ba3539f2eb02bb986df9859020fc4d49e71ac76cdeb9",
    "noconflict_report_sha256": "caa469a3db2ef307158817a00f536fdddb686dd27ba6393751c82e4e59e5396d",
    "provider_profile_sha256": "f76297433f16b79d419604742cff7bef07ab35cf4ee2b7c49b9b222f981a04ed",
    "candidate_inventory_sha256": "b02094af2fad4bb1ce3476aa4790e30b18e2653d1082daf8d6261b37e71ef287",
    "static_readiness_sha256": "379ae1a04855d41702b85e640c3b7a59570b49d018268847afb87398145628f8",
    "experiment_plan_sha256": "878856d21fec8b0d86a85629311d775f6a71a6034147a47d42fed61d5eceda10",
    "experiment_validation_sha256": "7cafd207d4572ac1b6486f3887ae58cacc10d2790d0361d36466161c28bef887",
    "downloaded_evidence_sealer_sha256": "9f8154af2110e1b4a78e3b654683f94bef97d4a8c911c1d5dc60b30b3b1242aa",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M8R14_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE = (
    "m8r14_postilp_target_feature_provider_cert.v1"
)
M8R14_POSTILP_TARGET_FEATURE_ROUTE = {
    "kernel_id": "onebitbrain/m8r14-postilp-target-feature-cert-v1",
    "code_sha256": "12433e47620351912571ea33ac26d65686a00f5f5e79f668d764b92b2dc14578",
    "metadata_sha256": "6045d2dc149ffabd3d54dcddf8af95596fa960fbd9088401e278d95202a78ea1",
    "package_freeze_sha256": "c1f90ee9d474827458eb45d23af2b6df6289fe152b0d72aa6943b87687039061",
    "package_tree_sha256": "6cf57d77f7acc7fd7f60080ebf78e6865dc42c6df907c93610c3a34ea5455309",
    "package_inventory_sha256": "cbddfd017c699279827e25c5d0d4a06135784913947fdea1b2c0df2a939676c7",
    "candidate_inventory_sha256": "e0bd44aded5d4304b4b77330a8d72a1b25c4b23475bb97ec800f778b65468800",
    "source_equivalence_sha256": "7ed6c97c7d1455e4026462108a269a4aca6f225637947d9be3439022798b8bba",
    "local_falsifier_sha256": "8b2742427ad910cca005395d9dffebb0599a9830bb8fd788549f959f83914b62",
    "noconflict_report_sha256": "3eb2519435b62df001f49b7110eee3832fc86fa0c71f6a5dcf839df1a9a7611e",
    "provider_profile_sha256": "5e96cf20906380360f3fca643344e989ff81054aadbec59300bef1e507195484",
    "static_readiness_sha256": "572d8dccef18e1afe204fae7232f68dd6c19dad639fcb41e5436f5a593d39e49",
    "evidence_contract_sha256": "7eb2163748ea8aaabd5b8708723a29dd763d5e239df5173edd116fb728aac3b5",
    "experiment_plan_sha256": "ccc200c684bdaf315e1c29e6e13bb6f98f77704f0caa9dacfda3fb063c1fd2e6",
    "experiment_validation_sha256": "1663e662bce8b55ef7da3e79e47faa8086c1159ceb4abf9bfa21c06d0370297c",
    "downloaded_evidence_sealer_sha256": "4da099283183126edd5be13e14ae44f1b868dabcdc69324584a6156acd7a55e5",
    "postilp_contract_sha256": "f31bce1a41a69acc5725d52851ef87abdd3c3061ddf6b693bc620ec78ba2b176",
    "r13_terminal_sha256": "0de6ba1e955589895127b1db0100be72a0229ac012a1813875a6c5016dd5c05f",
    "carrier_manifest_sha256": "da78e03642756e0648bd7896c0b1829d2909d0f5b4ae06fc7950280c87250a10",
    "checkpoint_sha256": "38a303d4eb24ec008ed94893d4355853f5ade42ced40eb0364ee1af55b19b6a6",
    "combined_source_sha256": "7a10013b6aefb4e64c16a3c9bd6957d8ed6d397a1e1cd9770605836264c7eec2",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M8R15_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE = (
    "m8r15_postilp_target_feature_provider_cert.v1"
)
M8R15_POSTILP_TARGET_FEATURE_ROUTE = {
    "schema": "celltrack-alpha.m8r15-central-prelaunch.v1",
    "execution_profile": M8R15_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE,
    "kernel_id": "onebitbrain/m8r15-postilp-target-feature-cert-v1",
    "lane_owner": "M8R15_POSTILP_TARGET_FEATURE_CERTIFICATION_ONLY",
    "candidate_id": "M8R15-M8-POSTILP-TARGET-FEATURE-EXPORT-v1",
    "code_sha256": "d05621cb6814fe12433edc8b673137b9140e3dab4e16d52a48c9c2745e84dcf1",
    "metadata_sha256": "a4ae921a6caad8953fb2b16126b16ae77a8a940366792e780e7a57b41ce3161d",
    "package_freeze_sha256": "fd66f0cae8e13c82255ee8a9a0699887984f1b4b6609e6e726780f17c5f83c6b",
    "package_tree_sha256": "99bd881c88399818944181826ca3a2f63ae2d49b3b3708e35ed51c774301aa18",
    "package_inventory_sha256": "b33eb405bfd4437e5b23c5037622250c2fa51ffaf837e5980942ec5c845275cc",
    "source_equivalence_sha256": "a2e21c052fa85b8fa45c1d08cb03725f79cf9237c6b2deb30b30ebaec6ec420e",
    "ship_certification_sha256": "090759d804f3683212a7dddbab8a6a1127eec810b1c0d3781997a332677ea0d4",
    "ship_receipt_content_sha256": "4f8102ffba3122c1d75f6465a284f863c28f06385e4864cd5ee915f27c3211a6",
    "noconflict_report_sha256": "f6b1a1117c70d6c6fa04bfe77a0853a47f2b27379b22df8f3322a8a361eeb630",
    "provider_profile_sha256": "08905395af0c60633b4c01145eab866c41d2ab04f1756ec0c49c62faaf847c2e",
    "static_readiness_sha256": "67204e1012adff70e98bbf30a6408e8a15992b7b0bcc8f28e61fd24922ccb695",
    "evidence_contract_sha256": "5f49f764a82087858dcd31b53832551b6b76855b37ed3b39bc291f0ca84433a8",
    "experiment_plan_sha256": "9fe1f2c285c22da33c9ee41fe97ee28880577b29ce183c3bfb23c56420e256e9",
    "experiment_validation_sha256": "ecbe24c7a97455dc3c88227a7a12297e3e3ca24f41b3241d847714208a5944fd",
    "downloaded_evidence_sealer_sha256": "64330ea727e0c6bdb2ca884b3f1ba49ebe51e6e0344c73790cd6621e849f9552",
    "postilp_contract_sha256": "f31bce1a41a69acc5725d52851ef87abdd3c3061ddf6b693bc620ec78ba2b176",
    "checkpoint_sha256": "38a303d4eb24ec008ed94893d4355853f5ade42ced40eb0364ee1af55b19b6a6",
    "combined_source_sha256": "cc0764561ac82554305ebdba0e8475be5c6245464f24d959e871d0132952d0b2",
    "parent_terminal_sha256": "44540ea3abe077d31230a2edd1694294d8a9368459091a9f128b2a526dc6af86",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    "frames_per_identity": 100,
    "completed_units": 200,
    "configuration_count": 1,
    "seed": 104729,
    "hpo": False,
    "one_shot": True,
    "certification_only": True,
    "model_execution": True,
    "target_input": True,
    "training_permitted": False,
    "scoring_permitted": False,
    "submission_output_permitted": False,
    "production_output_permitted": False,
    "whole_run_hard_bound_seconds": 27900,
    "downloaded_evidence_required_for_any_successor": True,
}
M8R16_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE = (
    "m8r16_postilp_target_feature_provider_cert.v1"
)
M8R16_POSTILP_TARGET_FEATURE_ROUTE = {
    "schema": "celltrack-alpha.m8r16-central-prelaunch.v1",
    "execution_profile": M8R16_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE,
    "kernel_id": "onebitbrain/m8r16-postilp-target-feature-cert-v1",
    "lane_owner": "M8R16_POSTILP_TARGET_FEATURE_CERTIFICATION_ONLY",
    "candidate_id": "M8R16-M8-POSTILP-TARGET-FEATURE-EXPORT-v1",
    "code_sha256": "2791cd68f1a52043d5ba235f7895711970cfee9f5c59605605abecaa199992ba",
    "metadata_sha256": "76f729dc1541fc7020e5043407166651a8e55d0209ff4c6db7a7b2b309ee6390",
    "package_freeze_sha256": "19a925d9c061f7e88150b0b8fcedf2e36faec62d8c900e790aec37b3df43a6e9",
    "package_tree_sha256": "ccfb53f373d788e496a971e99aed01b856a56064501b7bc89b9d3515bc8640eb",
    "package_inventory_sha256": "776252916a7f03d88168c2c87ebec1c7e223e6e2c5962816aabcae94715e7cf6",
    "source_equivalence_sha256": "52c638432dd11aeab845096b36ee0ba3377f960f31c0f2034b42094ac8fc5b86",
    "ship_certification_sha256": "93479a683b575b5d557de4ab5d7f8f4d5ad85e8efbf17a8d723a346707f25593",
    "ship_receipt_content_sha256": "167222c7e27a0f63f2e17b8a10fb892d48e7310b9ab7c936533338753d084ad8",
    "noconflict_report_sha256": "9d48f8a59cc82d0b33f9e8f2d3ca16daaec18e69cb772f64701df1f045db4cba",
    "provider_profile_sha256": "db476cedc3b977a16b72e51320b84835cb5226ab50cd55a89c86c0ba6b63a1b4",
    "static_readiness_sha256": "da7bf9069ac43f09a96bbefd271397d8193e603a95bf7d7737c41f7424de9076",
    "evidence_contract_sha256": "1da4db8c97f3fd06ffbbd160006a3a2d8d3801915a7fb14787cb84190e69e98e",
    "experiment_plan_sha256": "940a9d87eb70eeeee705ba999ed1cb69847ea4d83374e521492dc70b116fa578",
    "experiment_validation_sha256": "bef7bd0c8bfcc684b414df0669ab38fdef16e332ed2e3dd74e7c56c54b3e8b5c",
    "downloaded_evidence_sealer_sha256": "7a8a8c19f569ecb66209e32e3542abae4f09edc94cf6c483e391ca42dd76521d",
    "postilp_contract_sha256": "f31bce1a41a69acc5725d52851ef87abdd3c3061ddf6b693bc620ec78ba2b176",
    "checkpoint_sha256": "38a303d4eb24ec008ed94893d4355853f5ade42ced40eb0364ee1af55b19b6a6",
    "combined_source_sha256": "4b1972be91519e37d444de5217cfdf9f1de35a82d20f2c9c03ba1de85f8a2c9f",
    "parent_terminal_sha256": "44540ea3abe077d31230a2edd1694294d8a9368459091a9f128b2a526dc6af86",
    "causal_terminal_sha256": "b52ad39875a7df1c32413a98e2be01b403f2a678189a0d3c59064e4536509c99",
    "causal_failure_log_sha256": "9053017d6b2eac6ab3f9bebb800142ea23408e19a4d918a8beb126841dd75747",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    "frames_per_identity": 100,
    "completed_units": 200,
    "configuration_count": 1,
    "seed": 104729,
    "hpo": False,
    "one_shot": True,
    "certification_only": True,
    "model_execution": True,
    "target_input": True,
    "training_permitted": False,
    "scoring_permitted": False,
    "submission_output_permitted": False,
    "production_output_permitted": False,
    "whole_run_hard_bound_seconds": 27900,
    "downloaded_evidence_required_for_any_successor": True,
}
M8R17_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE = (
    "m8r17_postilp_target_feature_provider_cert.v1"
)
M8R17_POSTILP_TARGET_FEATURE_ROUTE = {
    "schema": "celltrack-alpha.m8r17-central-prelaunch.v1",
    "execution_profile": M8R17_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE,
    "kernel_id": "onebitbrain/m8r17-postilp-target-feature-cert-v1",
    "lane_owner": "M8R17_POSTILP_TARGET_FEATURE_CERTIFICATION_ONLY",
    "candidate_id": "M8R17-M8-POSTILP-TARGET-FEATURE-EXPORT-v1",
    "code_sha256": "b58e0293f8d685794a00bb33f773f30ddcabbd9828f19fa482e9dc1bcfbb5af9",
    "metadata_sha256": "b37802a0da33e555e19bd81fea06c4f7419caa80c0bff673f294e5c938dd6b79",
    "package_freeze_sha256": "79bcdde3ae9b2defdd9473bd2fc3b0d8a8579772a7668eee416af57732c31215",
    "package_tree_sha256": "21834cc3ba5392bd56800fbee78781ec1209e45be03086208d3afbd0ceeeaca2",
    "package_inventory_sha256": "3ece60a063f713fe3ed07bc30adfb59db64aea0ce222146ead7e8153eb7c447f",
    "source_equivalence_sha256": "efd5785f195c9b8ec05fcff9164b27782792dccf56c38b198cd46f5152f48df9",
    "late_tail_preflight_sha256": "57b101af61d89f5e60110defeea3f15caaecf7dcbc916c25f1f009d85e76c191",
    "ship_certification_sha256": "edb59a3155bb365a6b82b6a9a4d86af7aba2eb09e599409f0290312d20c53cd2",
    "ship_receipt_content_sha256": "95dbda50222923b8079452f8717a794ccddce2bc807789a9eb949bd2b8a2e24c",
    "noconflict_report_sha256": "79dac17db6aedd37186d4bada457926ce8161ca54a258119089d029da70bc01f",
    "provider_profile_sha256": "b546304c776841537f7322249094b0c895bc1c8afc277482f96e05a38737158a",
    "static_readiness_sha256": "7353f4466557c0643533ddf19c00802e59ce7e930eb4207733763d14ff6c7f44",
    "evidence_contract_sha256": "6944de1e6f3fdfc901febdb56801194178571996b350aa3e66addb92b5fa1ca1",
    "experiment_plan_sha256": "7591d5fcd3ed30dc957678e4719784e90215a25baa4b9c83b0613ed855028021",
    "experiment_validation_sha256": "c81a85892a88fab9905c8239b58fbc3d72f975e262d87d5002a3046188e20f9e",
    "downloaded_evidence_sealer_sha256": "8dfd0c2015bc8b545d0bd10eb55081ebec863137820d941447f524c5ea24f306",
    "postilp_contract_sha256": "f31bce1a41a69acc5725d52851ef87abdd3c3061ddf6b693bc620ec78ba2b176",
    "checkpoint_sha256": "38a303d4eb24ec008ed94893d4355853f5ade42ced40eb0364ee1af55b19b6a6",
    "combined_source_sha256": "ebd19717d3823fad855ce98d993f765380719254b7efb90412534ddc69e1298e",
    "parent_terminal_sha256": "7a41f428b02236374bf27446c72f55d5eac743ac1a9fe4ab82e704d04f9141a0",
    "causal_terminal_sha256": "7a41f428b02236374bf27446c72f55d5eac743ac1a9fe4ab82e704d04f9141a0",
    "causal_failure_sha256": "a29820a76b43aefeae3849c93a48490786aa2257321b57b8f605f7c83aa4a448",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
    "frames_per_identity": 100,
    "completed_units": 200,
    "configuration_count": 1,
    "seed": 104729,
    "hpo": False,
    "one_shot": True,
    "certification_only": True,
    "model_execution": True,
    "target_input": True,
    "training_permitted": False,
    "scoring_permitted": False,
    "submission_output_permitted": False,
    "production_output_permitted": False,
    "whole_run_hard_bound_seconds": 27900,
    "downloaded_evidence_required_for_any_successor": True,
}
M8R20_FULL21_TARGET_FEATURE_EXECUTION_PROFILE = (
    "m8r20_full21_target_feature_provider_cert.v1"
)
M8R20_FULL21_TARGET_FEATURE_ROUTE = {
    "candidate_id": "M8R20-EXACT-DUAL-COMPETITION-ROOT-CANONICALIZATION-v1",
    "checkpoint_sha256": "38a303d4eb24ec008ed94893d4355853f5ade42ced40eb0364ee1af55b19b6a6",
    "code_sha256": "b92e07e26d250368f9274a8a14edda5066bd4f6c315e216534759f8404949113",
    "completed_units": 2100,
    "configuration_count": 1,
    "downloaded_evidence_required_for_any_successor": True,
    "downloaded_evidence_sealer_sha256": "4f7c803bbead40e7e96b1a703ccb2ddd40eb5dfcac620a82e9987e50133ad683",
    "evidence_sealer_repair_sha256": "3c9463db654d677ba474a4d81458956934c3589b7d7e8b720b3e78256bc8e30c",
    "execution_profile": M8R20_FULL21_TARGET_FEATURE_EXECUTION_PROFILE,
    "expected_version": 1,
    "experiment_plan_sha256": "1b46f40e605edf338088b07012488d88a3b01c180b652b896f42f1397014dfa2",
    "experiment_validation_sha256": "e4050657058406310d3ed5a6ad166461dc518cea4c3d47a456f7dff515384061",
    "external_action_performed": False,
    "full21_raw_metadata_sha256": "f6c515fd1f92252434eab0f3d8b479dd835a354d8ba6bac852d5255bade8eae1",
    "full_export_permitted": True,
    "graph_efficacy_permitted": False,
    "hpo": False,
    "kernel_id": "onebitbrain/m8r20-full21-target-feature-export-v1",
    "lane_owner": "M8R20_FULL21_TARGET_FEATURE_PROVIDER_CERTIFICATION_ONLY",
    "launch_permitted": True,
    "lmt_sha256": "35376648ab432e2d7c901b89e5efeb3542c154af5c1dff8fdc3dd681949fb478",
    "local_material_resource_receipt_sha256": "9c34ec93c66b290c11360721187053f0de1a7ca418882ee228a8a0706d8873d2",
    "metadata_sha256": "0dbfe6251a5cb46fb9a59eaf2563aafb4faf7ee24f00fd7afca77f122d56101d",
    "model_execution": True,
    "one_shot": True,
    "outcome_access_permitted": False,
    "package_freeze_sha256": "2b4a3ea00bf1863b34c42ff5f2cfe5632a24cd57e4741a050e463174f23e19bd",
    "package_inventory_sha256": "19704f7cb83637632fe8534acb8e626e15ada0ff3344c99c7146afe3adeab253",
    "package_tree_sha256": "21d34af407f56571603c773b48495864d1d19e12ef4a0e23acfd7c6fa4914e7c",
    "parent_terminal_sha256": "d9cd31bdf242994e7102d650a5bdf9d951f7ce3445a85551fdd5db2c6885ce08",
    "production_output_permitted": False,
    "projected_remote_upper_seconds": 16605.500544906823,
    "remote_preflight_kernel_id": "onebitbrain/m8r20-full21-target-feature-export-v1",
    "schema": "celltrack-alpha.m8r20-central-prelaunch.v1",
    "scoring_permitted": False,
    "seed": 104729,
    "ship_certification_sha256": "876719dab75d41b6ac8f589543d0b581e71de2531f1faefa5ca9394ef53786ee",
    "ship_profile_sha256": "1c701a92c6a665e6c3cf0f9e2ee05451d053ff049c31aeebec4d720bfce3984f",
    "ship_receipt_content_sha256": "6694f69ec68f2ba3399b70a0f80788a7618ef86a338854b88b2f380662fe4e3d",
    "static_readiness_sha256": "d8c94bf37ee3531fd65b4b91cfa0fdf268d51226f3e363399869ad0efd09b912",
    "submission_output_permitted": False,
    "target_input": True,
    "training_permitted": False,
    "transport_equivalence_sha256": "b26e8231af013ec9282f1cbebf98abce513f63747051acad255ea7e6fc9ceffa",
    "whole_run_hard_bound_seconds": 27900,
}
M7R13_PROVIDER_CERT_EXECUTION_PROFILE = "m7r13_full21_provider_cert.v1"
M7R13_PROVIDER_CERT_ROUTE = {
    "kernel_id": "onebitbrain/m7r13-full21-v32-m6-provider-cert-v1",
    "code_sha256": "72e6a1ba91a896607553ac4ff95f19d7137e7c15e0a4a83dbc67f168b85122d7",
    "metadata_sha256": "b6bb1ae3900da0762f92599855435fc2b307c7825011730940d2f526c7b7e02b",
    "package_freeze_sha256": "2ff154857f999deb0284acf300e6a2988837ff4e706c602688e8cf3619176943",
    "package_tree_sha256": "5a120682b89595d607d9741b8d53757935a860c3c5e8bf7ae5200976cee333f4",
    "package_inventory_sha256": "cc13a195c59bb239d942ca0b377adfc11e58b74c41f016a48df9a8f35f662d83",
    "m7r12_code_sha256": "e1b36fe81062f011f5ab04b7144ca8bf9e8522b0c16e32e505679264fb56e38f",
    "m7r12_freeze_sha256": "cb680f3f439cfc72e239697de4b25e65560e98bf129b89e439e9b7b75b77c8b8",
    "m7r12_canary_sha256": "2c34727c4b871fb10111e356e405b48f737b12833aee1027c11c2bb85993613d",
    "m7r12_full_profile_sha256": "fb3f7f5001d80f0a5f06bacf8e6c1ee45273f0c46ea0f1f648d63f3e82fa63ec",
    "fit_manifest_sha256": "c5009ec69dff95bdfad3ba81a29d8281d1ea47ccaae7e5715c39124c61f21c4b",
    "m7r7_runtime_seal_sha256": "71dcb5333200643f9525de8c468a18fba17addcfbc92abc704bb03fdd3552a8b",
    "m7r7_runtime_tuple_sha256": "6a982da092de01994011d060bd86a0cc607ef0d97150eb760d550f2ca5c71168",
    "candidate_inventory_sha256": "a1c555f480deefe54bdc96288cb78fca605751fcd2182f364ad3358ce8f46a30",
    "transport_equivalence_sha256": "9d4211efe4833147010a7de27f6fb2b74d16842c9cbfd7ef20563c80c69aa556",
    "experiment_plan_sha256": "111ad4c5263b5653ddab3437dcac344ea642492fd34b51a55e895011b0daa1fd",
    "experiment_validation_sha256": "a1fa44d3e15ab473eeb9beb3d90f298b800763dfcba7976ee566f1f9a79784ee",
    "ship_profile_sha256": "e5b8e66738c7eff156619b01581dc5265defdc2f575dfe7725cd7d6b1925a185",
    "downloaded_evidence_sealer_sha256": "1591d8ec67f416d3173cb25be8ef2f5d47f60e4e2f601fc7112378c701bb30e2",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "provider_prefix_sha256": "10fcf18069577d9c103d436c9c8b2f7d0ac76f38bb89610506e03807dfe3f32c",
}
M7R14_RAW_METADATA_EXECUTION_PROFILE = "m7r14_full21_raw_metadata_transport.v1"
M7R14_RAW_METADATA_LAUNCH_EXECUTION_PROFILE = (
    "m7r14_full21_raw_metadata_transport.launch.v1"
)
M7R14_RAW_METADATA_ROUTE = {
    "kernel_id": "onebitbrain/m7r14-full21-v32-m6-provider-cert-v1",
    "code_sha256": "5e5e3cbf36aece14b5805bc795815623cb1395aaaca43435052ec18bdc8a51ba",
    "metadata_sha256": "6f6fbe53171afa74ffe9527631b855a5e385a2a174206dc70836de87307858eb",
    "package_freeze_sha256": "0a1db2fac5b2f1fb4cb81aee3792959f5a7f9646b462ab34a00ea40142ac4033",
    "package_tree_sha256": "655935ac2bb9d82682d44f4a07283ae6ff7689dd7cee460c38a5c13cadbb018e",
    "package_inventory_sha256": "b79480d109ae745e1477d5ef225858ecd131687924297d28305a8f9c53a0a96e",
    "parent_m7r13_code_sha256": "72e6a1ba91a896607553ac4ff95f19d7137e7c15e0a4a83dbc67f168b85122d7",
    "parent_m7r13_terminal_sha256": "6109fc26d11dc7e89c59a46cc5d11a1e6c182d1f6799e5fbba4ee8600a7032c9",
    "parent_m7r13_input_auth_sha256": "ef0baef70d68f554eecc15bb1a09bd319fb3e1a481e34e2d18262393f18aa46a",
    "fit_manifest_sha256": "c5009ec69dff95bdfad3ba81a29d8281d1ea47ccaae7e5715c39124c61f21c4b",
    "dataset_registry_sha256": "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158",
    "raw_metadata_manifest_sha256": "f6c515fd1f92252434eab0f3d8b479dd835a354d8ba6bac852d5255bade8eae1",
    "causal_audit_sha256": "c344fd12fff550b19179dd46766b2227b54073b8f681ca82895254425dd66726",
    "transport_equivalence_sha256": "4ffa3ef58b22b8922365bf43505d5a8f864a59beced4810afc1784de556333c4",
    "candidate_inventory_sha256": "2541cd89a9c5c4e9ef4df5d1304e16cba4c2e4034f2eb68e38458990a4397444",
    "experiment_plan_sha256": "3ebde26182b5f61d81ef995a3e94200762a4f30df2dbf820b1cc5cd1632c07df",
    "experiment_validation_sha256": "d5895dd9cd3a6a3d320cb495a24165b7bdbb559d6a75db5cfc23c7de57851f4d",
    "ship_profile_sha256": "8d2466028c82a060844ad25fa2d7797e1b68c785af52ead6c02a9a2b71c5303c",
    "provider_prefix_sha256": "0e75dd8a5c2b30f313d5187318f04d4bf25e871e119c946e431bc0e3cba8db96",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M7R14_RAW_METADATA_LAUNCH_ROUTE = {
    **M7R14_RAW_METADATA_ROUTE,
    "m10r5_terminal_seal_sha256": "c5358ae4e32d14b516ee73340195254ba39ca6582be3c5809eac2ad6096a2af6",
    "m10r5_terminal_classification_sha256": "a04bb114d30d881a8ac88162fd0042de92174998bd97c003ba9390bafb4756fd",
}
M7R15_ACTION_SUPPLY_EXECUTION_PROFILE = "m7r15_full21_action_supply_census.v1"
M7R15_ACTION_SUPPLY_ROUTE = {
    "kernel_id": "onebitbrain/m7r15-full21-action-supply-census-v1",
    "code_sha256": "8162f7c9f44061d682b363d2d5e118cf946666e118c6cd523f8ec7512c7b2905",
    "metadata_sha256": "29fb4da5d7a279bf5039ccbb694be2d1fd7aebd15363eafa076ae4e98ab0522d",
    "package_freeze_sha256": "89f3387da2dd87afd30c3f79ee51f3664802b033a737e863f4c2ee4afe685307",
    "package_tree_sha256": "d92ef3c6b8f1b1168efacc5599dada3b80d0bba45e938553b319b25bb154e849",
    "package_inventory_sha256": "6b774150532dcd27dcc604dc60179e76b9246ef6a0c24a3ab7d9642a7e55a162",
    "child_sha256": "1844416c8995f60e251f9887ae3fca329444fb03b0c644adc3610a7163aae7e5",
    "source_equivalence_sha256": "bc5a24d72de49a582cb47e31cc56dab5bed869be1b604d53c59c8700b820adde",
    "environment_sha256": "24816cf64555170e07b99ff2cae91dac1a10c1809546646c52e81914f64350ad",
    "candidate_inventory_sha256": "c80032b21e769ada00453b03f167a9714f294e9087e00ada052c524858645d83",
    "experiment_plan_sha256": "e94fa3d46917669a616731c8acf28878c172d475f7a73452f5253dbf91ed540d",
    "experiment_validation_sha256": "b61034f664fe548e55a7530370e998e06acee3c69c1e2060c26c979907909dd3",
    "ship_profile_sha256": "f7aa59c50badfa23fb384a4a30561e8d09ba2e2ee616199253ba78fc6e555a4f",
    "downloaded_evidence_sealer_sha256": "7e0a38e588b6637892a7c3777b1a95f3939c1d84b060615f6300081e1599d080",
    "noconflict_report_sha256": "c5919857d2afb0b05d0c877d2dd684db4b83b414c44f762936e76cce8d0286e3",
    "local_analysis_probe_sha256": "7dfc5ec5c7888499ae830a9f109eb772d95ccdaa98eddc9d847977af6ee2204c",
    "m7r14_terminal_sha256": "fd6ce8cbd3167fc4a8e7a0f1de14fa6e675b5c689fc0b5868f949415eb0eac81",
    "m7r14_local_seal_sha256": "a9b0261eb440f94abcdeafdf03e8993248afe76bfdecaa87550503eb8b5ff458",
    "m7r14_alias_sha256": "50bb2f27fa27cb2b79b0346930bb35ad6ce3860383e595e6d769dff2f7d695ac",
    "m7r14_evidence_zip_sha256": "29c4677c3e824e8330c8521e94c7681a4b8d4108a0a85d259f7bdd7327824684",
    "m7r12_code_sha256": "e1b36fe81062f011f5ab04b7144ca8bf9e8522b0c16e32e505679264fb56e38f",
    "m7r12_schema_sha256": "0a327a968f585eea0b6a1949eb6bce3de17c2da8a70833bc9cd27fdd6e5f09e8",
    "m7r12_profile_sha256": "fb3f7f5001d80f0a5f06bacf8e6c1ee45273f0c46ea0f1f648d63f3e82fa63ec",
    "m7r11_boundary_sha256": "cefcd626786a8e1254117232346864743a56b3b7ab6ffed0100b70f506967a41",
    "fit_manifest_sha256": "c5009ec69dff95bdfad3ba81a29d8281d1ea47ccaae7e5715c39124c61f21c4b",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M10R3_PROVIDER_CERT_EXECUTION_PROFILE = "m10r3_self_orchestrating_provider_cert.v1"
M10R3_PROVIDER_CERT_ROUTE = {
    "kernel_id": "onebitbrain/m10r3-aniso3d-dualslot-provider-cert-v1",
    "code_sha256": "97e9d1f81252da99d3c2a4419fc13391b2ffaa3f14a168ad1e4468c9fe18116b",
    "metadata_sha256": "3846a4c1235229419a72b7d23d2b69fb4717cd3dbbac490e0162544e80d0a39e",
    "package_freeze_sha256": "7be9433e3fe63484de9d3ed9603601f2efd070b8b123526743066f8ad3e00d8d",
    "package_tree_sha256": "65f6c632a374b22d3300c32dd41bc136d69231dc36cfeb655a7e69dbd85ad623",
    "package_inventory_sha256": "9746c56d6c8bea85b093dead267e6d72fc56e0be0b4d16680502bff5290b180b",
    "m10r2_child_sha256": "8b852073bd49751c01b090dfbf6c41765f777e21d955e9ef0d1351b54d8901d6",
    "m10r2_static_freeze_sha256": "8cd1b1ab565a680906269c50f1edaa79d2ee170e5e53ff53a47f16ac4d99ca8c",
    "m7r7_runtime_seal_sha256": "71dcb5333200643f9525de8c468a18fba17addcfbc92abc704bb03fdd3552a8b",
    "m7r7_runtime_tuple_sha256": "6a982da092de01994011d060bd86a0cc607ef0d97150eb760d550f2ca5c71168",
    "candidate_inventory_sha256": "8e706559eebca2844e90864d643ec30bb5d967480c244bb0571de0e5fd530e78",
    "provider_profile_sha256": "b3df18726234bf10165a6c0a53814a56e2d1c95ffc58ea7d0e88f38c5f338475",
    "transport_equivalence_sha256": "49e5e65671a81d3ad12e5fcc97386a5dfd4be4f6dfd2be8f4a3670f2ed1a3e18",
    "static_certification_sha256": "2af04184ab494bee963f99fe80dcaea4988f6b52e03ca320e8ebff8800aacdd9",
    "experiment_plan_sha256": "d5aec95a190ab3d65f5354113aace6ba95d96b1c358107ed56625f3ea77afb5e",
    "experiment_validation_sha256": "3aed63cafc8bf7283aa65ee7606bceb4c98fddf5a994a7e67a8bb699afca6815",
    "downloaded_evidence_sealer_sha256": "dbfee54cbb1ea7b396218713b12fb430624f4f1b99707ce6b1b94e0d44937f01",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
M10R5_PROVIDER_CERT_EXECUTION_PROFILE = "m10r5_gather_native_t4_provider_cert.v1"
M10R5_PROVIDER_CERT_ROUTE = {
    "kernel_id": "onebitbrain/m10r5-aniso3d-dualslot-provider-cert-v1",
    "code_sha256": "55eb36949b637ea458cdf40c784822870ee5efe507e6aec47789c95de31ceb8c",
    "metadata_sha256": "01c0c0ea4d8c73794e52a8e13d15729ac9f7f5aa06371e0fbfcd3f147aca7362",
    "package_freeze_sha256": "088bd7c2f1877de09f37c0067de1d6c628b9cc6a18f37e76469968ea0bc0ca71",
    "package_tree_sha256": "fd4edb0edba4f551a4475d38e0aebf3b2ee4d8b1be9329b2d4b7c9528af9b310",
    "package_inventory_sha256": "5f58c01a112c36843f6b3cea7dc2c7361dfecc6ea21ce462e919916ad335cdd6",
    "child_sha256": "ea1bfc72da286e616caf480705256a9598e2cbab2bd448d8ae573d66a1589238",
    "equivalent_model_source_sha256": "5b20e8dddf1b89168e60c1831cb0bf2f8f9909c99594254571c29b62f22244b2",
    "equivalence_binding_sha256": "3062fb362dba50d60e4bc466eca8b20cdb47ed05c2558027e127829a8f051322",
    "provider_profile_sha256": "00d5fb7d9072b000c9b3c8e5ebd1c328d2acc976e026d8d2158bf14a72fdaccd",
    "static_certification_sha256": "c72d0c8ac11f9b1808476f1ea2c4a762fce3b298b12113e8e7bf26e9b45e01ce",
    "experiment_plan_sha256": "33e3bbd85c1a5e334a2e0417b261665f2db024419a26506f392b3c53518f679c",
    "experiment_validation_sha256": "aad3263001431707eabd716da28ec6e4dc09e501d05cf11ceca4dab0c964cb3a",
    "downloaded_evidence_sealer_sha256": "854c0a3adea2760bdba99974a7f1c50f87719ff4fd49442f3452393f542dfd79",
    "m10r4_static_freeze_sha256": "152999fc4e80a1bedfc31bc091e9bf2f021f3e5d2d64c87241331f49976845e0",
    "m10r3_terminal_failure_sha256": "d3193c1757d94aaf1ffdb5cb3e9940d8d273ac1a4339f9296892a16f54b4f007",
    "m10r3_post_reconciliation_sha256": "e21bb6d4e664730112f03c258abc4e08b7934ece8a6861f63bbb1b053b483d40",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
}
D10_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE = (
    "d10_current_policy_native_t4_certification.v1"
)
D10_CURRENT_POLICY_NATIVE_CERT_ROUTE = {
    "kernel_id": "onebitbrain/f2-d-v10-native-cert-current-v1",
    "code_sha256": "c5bb135f9535dc10ef3d5697ebb4accf49df72be55583049e7331c7002a5e790",
    "metadata_sha256": "be72b3baf2bf69d222c42c996c1ac392bfe4920314bc3f7696294f542de423b8",
    "package_tree_sha256": "39a1476c75557196eba4bd8c1752df673cbb7d28125070a81bde35742b534bbc",
    "package_inventory_sha256": "33e3e632f0e3e8cbc022d8509607924e3e2c7127abc88b1562b80f300922cd3e",
    "package_freeze_sha256": "a734580d2061a9169b81b509fe2d76017df4d0174638c93ccb9eba7c634be64b",
    "profile_sha256": "ecfccf5048791483a84e123807d73eb3dba5359bec5ff69d92478e5f86f17ae4",
    "static_readiness_sha256": "d47f0919b1ffb623047f0fcd68c66a30bdc7c7bc6dbcedc93a7100d72260224e",
    "downloaded_evidence_sealer_sha256": "6a7e5218e37356ddd643b29a864dcf80790720191a68e35ca4d965de7123741c",
    "source_kernel_sha256": "da1b2ddc5c4a4a4b79f72c6e57393857b9e91dfe82739b2a0e9790392487bf6f",
    "source_freeze_sha256": "62b7e1cfa179ac62ec8ae14724e7b2afab95a99cf1f94faa398c5acc33703718",
    "source_metadata_sha256": "9dd1a2c7db6ba9a77add78b5998d91a30a36eafb554e5633d30afad76f219070",
    "parent_profile_sha256": "cea1d52cd44fee3f6f8f0ec09f1fa13b0073135b3abea399548447ead1c0ffc8",
    "current_ship_skill_sha256": "09a07046fc229d88559e2c58e98f2f6e629eca2921d808272190b746c2e4eb2f",
    "current_certifier_sha256": "afffd901c68576bfe5755d96ed00664763a4befaedd81cd1ef5fd0c5a41ee077",
    "ship_skill_forward_test_receipt_sha256": "8429089d748146597b6827947538f093818fdd58142557dfed8976ab9591a9f2",
    "ship_skill_forward_test_receipt_content_sha256": "821fb8eb833a3f1deadfbb1ef8889b7c77afb640fb59a29ca1cb9ae8a12f895b",
    "target_entrypoint_sha256": "4c4aa7b94ea985625229746e2bde4f7c136f59cee17b01b2bf425ef7ec78974d",
    "target_metadata_sha256": "991d74aff0b2cd358df282448a57ee4cb20db6eab79bdf2c5683823ef8d54435",
    "target_package_inventory_sha256": "7304c79ba829db7b8dc8487398b916b5afb5a0afd8fa572ee620c97b988161de",
    "lmt_sha256": "35376648ab432e2d7c901b89e5efeb3542c154af5c1dff8fdc3dd681949fb478",
    "identities": [
        "train:44b6_cf2536e8",
        "train:44b6_3bb3690f",
        "train:44b6_aaf8b0ea",
        "train:44b6_c8e2a523",
        "train:44b6_87bba6c4",
        "train:44b6_996155de",
    ],
}
D11_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE = (
    "d11_current_policy_native_t4_certification.v1"
)
D11_CURRENT_POLICY_NATIVE_CERT_ROUTE = {
    "kernel_id": "onebitbrain/f2-d-v11-native-cert-current-v1",
    "code_sha256": "7f12e64f864c59bbe62990ae9b38137d0ccf1e10cf41b26744120998a016dd97",
    "metadata_sha256": "028125adbef49938e2a9e35054e5777d843a500edb7ecae7912848d1921221e3",
    "package_tree_sha256": "16c0a407fb11b6154dea99a9796d05ac24f53b78475545a9d1c28e48d23cf371",
    "package_inventory_sha256": "a40729051fb19b432961051a096354cac01d31a5fd94a7ade1896b9497b26fd0",
    "package_freeze_sha256": "af5d30e9a8b03a8fe939ee83fc77c6514e0720cbb742cc8d2e8bb1c35efb0b1b",
    "profile_sha256": "f8199e3f37eabddc2b3d0c46ea185dc3c331731e73c64ba9bab231e0e59681d5",
    "static_readiness_sha256": "9ca76f40696dbb03846f37f1aa315282421b431c78865347217dee12226535c7",
    "downloaded_evidence_sealer_sha256": "54957c3769c9c83d649fe2080284d4afeb1a0b1fa46b45c23eacb953e96bac95",
    "v3_static_check_sha256": "415afb6987a5f8c198904066a95786ce4c326a7fe7c8aa34ca150539199abe88",
    "v3_installation_receipt_sha256": "6667e1e498e22d0b544e0a5d660f8514a31d994a4e98034bd95eb2f05e42e66a",
    "current_ship_skill_sha256": "012054a3d949691c904675ebf9005c4174d0a097e118a9e6c35e26bb9a452f66",
    "current_certifier_sha256": "696bb7c4c97afeeed13c4d0c31913c6da2ffbf257bf85f7b0167dd5c3c97bca6",
    "target_entrypoint_sha256": "0dbfac708e8e41555b6f1ac312863844ac8970a72882cd244320474fc302919e",
    "d10_target_entrypoint_sha256": "4c4aa7b94ea985625229746e2bde4f7c136f59cee17b01b2bf425ef7ec78974d",
    "scientific_source_sha256": "da1b2ddc5c4a4a4b79f72c6e57393857b9e91dfe82739b2a0e9790392487bf6f",
    "target_metadata_sha256": "991d74aff0b2cd358df282448a57ee4cb20db6eab79bdf2c5683823ef8d54435",
    "target_package_inventory_sha256": "c74b8973c6b9a16daf7e50f444b2f959aad54b4bdc3bb6c05f77b60589294e82",
    "source_freeze_sha256": "62b7e1cfa179ac62ec8ae14724e7b2afab95a99cf1f94faa398c5acc33703718",
    "d10_terminal_sha256": "15cbcfd266406b4c2f391b6cd0ffafa14f6e77d2be1f13c5abdbeea28a950134",
    "d10_causal_sha256": "93569a2f171b50ab9d2eca339cf6a743e1e484be69fb305f2e900291c5e578b4",
    "lmt_sha256": "35376648ab432e2d7c901b89e5efeb3542c154af5c1dff8fdc3dd681949fb478",
    "identities": D10_CURRENT_POLICY_NATIVE_CERT_ROUTE["identities"],
}
D16_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE = (
    "d16_current_policy_native_t4_certification.v1"
)
D16_CURRENT_POLICY_NATIVE_CERT_ROUTE = {'kernel_id': 'onebitbrain/f2-d-v16-native-cert-current-v1',
 'code_sha256': '698ede8576064e1d4d6f8f5f11e69207cae14f0dd8ab047000376bde7b6f5dab',
 'metadata_sha256': '130510f8b3a43e74f5cb1604a5ae37e28e52b05d60d0324598ec987c82cbfb7f',
 'package_tree_sha256': '511f70ae7a0013ee6d147c7a52f255d6f7308f1bd828f4ca479b3c53e97dfed6',
 'package_inventory_sha256': 'a8f163eced4bb2a91f12926d5d23495ca7e080398e66dc6edd4aa3449b5b82fd',
 'package_freeze_sha256': '809fb757fc871ed1772358a7b4d601119529cc493073ef30e253a3e01fa65059',
 'profile_sha256': '3fa7597ba97d658938839d6e673fe0be92a2c391c311a17e913c123d3bfd74b2',
 'static_readiness_sha256': '796dfe88bc06d17ebd5dd7e7974eb3aead178cf891466c3a547092deb6eec3f6',
 'downloaded_evidence_sealer_sha256': '9f9b86c74da053d8a40f75bd899f8e12de0e8cdfe069140506b8fe5961f6649b',
 'v3_static_check_sha256': 'd871671792c5463cb0764d8bb8321fab2551635824d05b34d6ae20ba6705ec03',
 'v3_installation_receipt_sha256': '6667e1e498e22d0b544e0a5d660f8514a31d994a4e98034bd95eb2f05e42e66a',
 'current_ship_skill_sha256': '012054a3d949691c904675ebf9005c4174d0a097e118a9e6c35e26bb9a452f66',
 'current_certifier_sha256': '696bb7c4c97afeeed13c4d0c31913c6da2ffbf257bf85f7b0167dd5c3c97bca6',
 'target_entrypoint_sha256': 'da53fe3d710faa25e2d7e8cfd5299264bfa511e79c800a1635a01beabcd4e8c6',
 'scientific_source_sha256': 'da1b2ddc5c4a4a4b79f72c6e57393857b9e91dfe82739b2a0e9790392487bf6f',
 'target_metadata_sha256': '2d52b43e9f484a0407aa1f76b165b6399c4de4a040faa70cbed0260a76264cbc',
 'target_package_inventory_sha256': '9cfda684459dd5e4f93cc2fe6460b81fa2d25168c9835f4ca946be05807fe26e',
 'source_freeze_sha256': '62b7e1cfa179ac62ec8ae14724e7b2afab95a99cf1f94faa398c5acc33703718',
 'd11_terminal_sha256': '0bc916518fa1376c567b8ef0b8e55bd10941738a2932fa8d75f98dbba8258906',
 'd11_causal_sha256': '1010d7ce8532697c179c5edba58097c1de7b69cc47550d6cc127cbacf9751626',
 'lmt_sha256': '35376648ab432e2d7c901b89e5efeb3542c154af5c1dff8fdc3dd681949fb478',
 'identities': ['train:44b6_cf2536e8',
                'train:44b6_3bb3690f',
                'train:44b6_aaf8b0ea',
                'train:44b6_c8e2a523',
                'train:44b6_87bba6c4',
                'train:44b6_996155de']}
F2_CERTIFIED_SWEEP_CONTRACT_SCHEMA = (
    "celltrack-alpha.f2-certified-sweep-wrapper-contract.v1"
)
CELLECT_TWO_CARRIER_EXECUTION_PROFILE = (
    "cellect_remote_visual_two_carrier_release_probe.v1"
)
CELLECT_P100_EXECUTION_PROFILE = (
    "cellect_remote_visual_three_carrier_p100_release_probe.v1"
)
LINAJEA_LEGACY_GPU_EXECUTION_PROFILE = "linajea_legacy_tf1_gpu_visual.v1"
CELLECT_TWO_CARRIER_KERNEL_ID = "onebitbrain/g02-cellect-native-gpu-visual-v2"
CELLECT_TWO_CARRIER_MOUNT_REPAIR_KERNEL_ID = (
    "onebitbrain/g02-cellect-native-gpu-visual-v3"
)
CELLECT_P100_KERNEL_ID = "onebitbrain/g02-cellect-native-gpu-visual-v4-p100"
CELLECT_P100_REAP_KERNEL_ID = (
    "onebitbrain/g02-cellect-native-gpu-visual-v5-p100-reap"
)
CELLECT_P100_TYPED_REAP_KERNEL_ID = (
    "onebitbrain/g02-cellect-native-gpu-visual-v6-p100-typed-reap"
)
CELLECT_P100_KERNEL_IDS = {
    CELLECT_P100_KERNEL_ID,
    CELLECT_P100_REAP_KERNEL_ID,
    CELLECT_P100_TYPED_REAP_KERNEL_ID,
}
LINAJEA_LEGACY_GPU_KERNEL_ID = "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v3"
LINAJEA_LEGACY_GPU_MOUNT_REPAIR_KERNEL_ID = (
    "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v4"
)
LINAJEA_PROVIDER_ROOT_KERNEL_ID = (
    "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v7"
)
LINAJEA_HOST_DRIVER_KERNEL_ID = (
    "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v8"
)
LINAJEA_DEVICE_ROUTING_KERNEL_ID = (
    "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v9"
)
ARCHIVED_SHIP_CERTIFIER_SHA256 = (
    "49d8ea390e67c0aa8a084f119e143ce878197119de81c4e72202ffdbb83d955f"
)
LINAJEA_CANONICAL_PROVIDER_KERNEL_IDS = {
    LINAJEA_PROVIDER_ROOT_KERNEL_ID,
    LINAJEA_HOST_DRIVER_KERNEL_ID,
    LINAJEA_DEVICE_ROUTING_KERNEL_ID,
}
CELLECT_TWO_CARRIER_STEP0_MODE = "cellect_two_carrier_release_probe_step0.v1"
CELLECT_P100_STEP0_MODE = "cellect_three_carrier_p100_release_probe_step0.v1"
LINAJEA_LEGACY_GPU_STEP0_MODE = "linajea_opaque_runtime_release_probe_step0.v1"
PRIVATE_CARRIER_ROUNDTRIP_SCHEMA = "celltrack-alpha.private-carrier-roundtrip.v1"
PRIVATE_DATASET_EXTERNAL_ACTION_APPROVAL_SCHEMA = (
    "celltrack-alpha.owner-external-action-approval.v1"
)
CELLECT_TWO_CARRIER_ROLES = ("model_assets", "cp312_release_dependencies")
CELLECT_P100_CARRIER_ROLES = (
    "model_assets",
    "cp312_release_dependencies",
    "p100_torch_runtime",
)
LINAJEA_TWO_CARRIER_ROLES = ("tf1_runtime_opaque", "cp312_release_dependencies")
LINAJEA_REQUIRED_VIDEO_VIEWS = (
    "side_1_tracking",
    "side_2_tracking",
    "dual_consensus_tracking",
)
REMOTE_VISUAL_PENDING_STATUS = "REMOTE_VISUAL_PENDING"
D28_CLOSED_SCHEMA_NATIVE_CERT_EXECUTION_PROFILE = "d28_fresh_reconciliation_time_native_t4_certification.v1"
D28_CLOSED_SCHEMA_NATIVE_CERT_ROUTE = {'kernel_id': 'onebitbrain/f2-d28-native-cert-fresh-reconciliation-time-v1',
 'code_sha256': '1504f1656594e59788ad55357039cfd725247f84971a06afc38bcd263455223c',
 'metadata_sha256': '9949c051c255ccefa634c327980f4943d706435a06c77be623823aaac0cac0de',
 'package_tree_sha256': '3995593a56d67614fb0fa8fbbd883b454056d8928153691a3f328869ba366e6e',
 'package_inventory_sha256': '68f91649ef3f8cd894273681003dad76e2a39d698e4f55bc55675c732b543c0f',
 'package_freeze_sha256': '821537b6ae65af0452c696da3cbea3ec05b0e9b959a71dd4fadf4dd5cba61ca7',
 'd24_profile_sha256': 'd5db744102748a07109584649a35825c74cbc53545f4ea7b21125680e99eddcf',
 'd24_static_terminal_sha256': '7615f31dce76c625b0eccc0b158bbc83eb5eba10b1bca79902dac96d0cebd99d',
 'd24_breaker_sha256': '2bd27b250f76a75edec1c62184b031aaa8168e74b305a71191d8f33a720e22dc',
 'd24_central_nogo_sha256': '35ff73e0f0bc54ecd195b9c87f18dec27da3af6186a4a662eabbf94a369ba35f',
 'd25_independent_nogo_sha256': '8594556d7fa8dfdade5f142cbb7349ba0bad6c7408c5fcf9c1743511691ce691',
 'd26_independent_nogo_sha256': '35a0f0253f4101b07c62d3676396901eb534e9fab41a1f8e5fd738275f525d7c',
 'd27_independent_nogo_sha256': '3ef1196ebacd69a5fe01d6b384677b07bfb84d9b7e80a9c3f96f4a13d3719977',
 'reconciliation_time_policy_sha256': '7c79fa3d8bcba922916c99f1be64ee9c32a15067fc1a36ebed8746155019eee4',
 'd24_source_freeze_sha256': 'a4074a78772fa3a557c256c2cffa3c29435e3c0bd96b3c55ad6cf01e295516a6',
 'd24_runtime_contract_sha256': '01702a2fb5a0f165cde1380aa40e845e1d4d6c029bf8cad197f9fdb23eca6fba',
 'd24_certifier_copy_sha256': '1cc8408c691c4ab1c7a1f4e9710bbe5c6c8d8a7e7ebd6547f17f92d7cab89466',
 'd24_executable_sha256': '1d7235a0f6989a3304311925eb8fe46be1e6a3bf835df030fa147fe6ebac4e21',
 'target_entrypoint_sha256': '8c11157d42bb5e7862fb56ca616e5b60786c02bf715ef4e6796d9bd3e87ee037',
 'target_metadata_sha256': '1d972732cf6f1189b5fb879d9b5892cd602f4c299164bdb22f115a07bb23aafc',
 'target_package_inventory_sha256': '79f6cb82f0348047cffe899f25743118067eccb15fdc6637db30c34e6f19ee0c',
 'lmt_sha256': '35376648ab432e2d7c901b89e5efeb3542c154af5c1dff8fdc3dd681949fb478',
 'expected_metadata': {'code_file': 'kernel.py',
                       'competition_sources': ['biohub-cell-tracking-during-development'],
                       'dataset_sources': ['pilkwang/biohub-tracking-support-pack-50ep-v1',
                                           'pilkwang/biohub-temporal-unet3d-seed314159-v1'],
                       'enable_gpu': True,
                       'enable_internet': False,
                       'enable_tpu': False,
                       'id': 'onebitbrain/f2-d28-native-cert-fresh-reconciliation-time-v1',
                       'is_private': True,
                       'kernel_type': 'script',
                       'language': 'python',
                       'machine_shape': 'NvidiaTeslaT4',
                       'model_sources': [],
                       'title': 'f2-d28-native-cert-fresh-reconciliation-time-v1'}}
D19_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE = (
    "d19_current_policy_native_t4_certification.v1"
)
D19_CURRENT_POLICY_NATIVE_CERT_ROUTE = {'kernel_id': 'onebitbrain/f2-d-v19-native-cert-current-v1',
 'code_sha256': '150897c37b11a1e4d6fdd72df44867f79216e35c6ff4cffdb896f09afb004bd1',
 'metadata_sha256': 'd9f6450200e3f0d4e3654fd73c35725d0bc2e6512903d940d7d9c468652f328c',
 'package_tree_sha256': '590cb16def9b8cd584572edcc2b6217cd69428069436a757b235a29586285dbb',
 'package_inventory_sha256': 'aec21ed4df452da04285bab1db8f066f05daf61c20b2d52311faf8b0cae15255',
 'package_freeze_sha256': 'cf8720162a8ceda6d43c2788b532ca2d5fbaa3800eff0f1d53ec582c615a836f',
 'profile_sha256': '308867e118d228d422f94d98664a12f31d50f67e78ae17c1d2375eff5997d81f',
 'static_readiness_sha256': '79c243ec66d13d1c1b263a0894a18e26cf95bd90127c9066aec04015d2fcd13b',
 'static_terminal_sha256': '6023eb5043b41be24276b1cdcb2161b3d60c8214768ed267f6ceea67ebd3a657',
 'downloaded_evidence_sealer_sha256': 'c6a2d2e6971c3005a495ef6c631b3f95c2443c8e976f49554efde40b525ef440',
 'v3_static_check_sha256': 'fec8f3646ba84dcf00ab7695f8b89f613819870c989116bdc88435e12d5ef12a',
 'v3_installation_receipt_sha256': '6667e1e498e22d0b544e0a5d660f8514a31d994a4e98034bd95eb2f05e42e66a',
 'current_ship_skill_sha256': '012054a3d949691c904675ebf9005c4174d0a097e118a9e6c35e26bb9a452f66',
 'current_certifier_sha256': '696bb7c4c97afeeed13c4d0c31913c6da2ffbf257bf85f7b0167dd5c3c97bca6',
 'target_entrypoint_sha256': '8c11157d42bb5e7862fb56ca616e5b60786c02bf715ef4e6796d9bd3e87ee037',
 'scientific_source_sha256': 'da1b2ddc5c4a4a4b79f72c6e57393857b9e91dfe82739b2a0e9790392487bf6f',
 'target_metadata_sha256': '4b590cd13dea83ca4c43f59fcf677a396b04371b449c079ac34c5343573c48d3',
 'target_package_inventory_sha256': 'd53ad6fa64a9936455d7a734916b506b4ddf860e3ffc5109ccdb61fbbe9eec53',
 'source_freeze_sha256': 'cf8720162a8ceda6d43c2788b532ca2d5fbaa3800eff0f1d53ec582c615a836f',
 'd18_breaker_sha256': '859e89246267a560a04c35279ad03bc63be546ece4f363c71b06d2df67d4c33b',
 'd19_breaker_audit_sha256': '8ab0259e6b6db90780f7d08def88932e81019994e709e7c313fa0a0a7e7c23e5',
 'lmt_sha256': '35376648ab432e2d7c901b89e5efeb3542c154af5c1dff8fdc3dd681949fb478',
 'identities': ['train:44b6_cf2536e8',
                'train:44b6_3bb3690f',
                'train:44b6_aaf8b0ea',
                'train:44b6_c8e2a523',
                'train:44b6_87bba6c4',
                'train:44b6_996155de']}
ASSET_TRANSPORT_EXECUTION_PROFILE = "cellect_asset_transport_step0_only.v1"
REMOTE_VISUAL_ONLY_COMPETITION_ROUTE = {
    "mode": "d2k2_direct_competition_source.v1",
    "kernel_id": "onebitbrain/g02-cellect-visual-only-v2",
    "competition_source": "biohub-cell-tracking-during-development",
    "mount_root": "/kaggle/input/biohub-cell-tracking-during-development",
    "contract_kernel_properties": {
        "enable_gpu": True,
        "enable_internet": False,
    },
    "metadata": {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [],
        "enable_gpu": True,
        "enable_internet": False,
        "id": "onebitbrain/g02-cellect-visual-only-v2",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "title": "g02-cellect-visual-only-v2",
    },
}
# The sole prepared repair lane.  It is intentionally *not* in the central
# exception allowlist yet: its package and root-authored record do not exist at
# this amendment point.  Adding a route merely lets the guard inspect a later
# immutable package; it grants no launch authority on its own.
REMOTE_VISUAL_ONLY_D2K3_COMPETITION_ROUTE = {
    **REMOTE_VISUAL_ONLY_COMPETITION_ROUTE,
    "mode": "d2k3_direct_competition_source.v1",
    "kernel_id": "onebitbrain/g02-cellect-visual-only-v3",
    "metadata": {
        **REMOTE_VISUAL_ONLY_COMPETITION_ROUTE["metadata"],
        "id": "onebitbrain/g02-cellect-visual-only-v3",
        "title": "g02-cellect-visual-only-v3",
    },
}
REMOTE_VISUAL_ONLY_D2K5_COMPETITION_ROUTE = {
    "surface": "d2k5_declared_asset",
    "mode": "d2k5_declared_asset_and_competition_source.v1",
    "kernel_id": "onebitbrain/g02-cellect-native-gpu-visual-v1",
    "competition_source": "biohub-cell-tracking-during-development",
    "competition_mount_aliases": [
        "/kaggle/input/biohub-cell-tracking-during-development",
        "/kaggle/input/competitions/biohub-cell-tracking-during-development",
    ],
    "asset_source": "onebitbrain/g02-cellect-native-threehead-assets-v1",
    "asset_mount_root": (
        "/kaggle/input/datasets/onebitbrain/"
        "g02-cellect-native-threehead-assets-v1/cellect_assets"
    ),
    "contract_kernel_properties": {
        "enable_gpu": True,
        "enable_internet": False,
    },
    "metadata": {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": ["onebitbrain/g02-cellect-native-threehead-assets-v1"],
        "enable_gpu": True,
        "enable_internet": False,
        "id": "onebitbrain/g02-cellect-native-gpu-visual-v1",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "title": "g02-cellect-native-gpu-visual-v1",
    },
}
REMOTE_VISUAL_ONLY_COMPETITION_ROUTES = {
    route["kernel_id"]: route
    for route in (
        REMOTE_VISUAL_ONLY_COMPETITION_ROUTE,
        REMOTE_VISUAL_ONLY_D2K3_COMPETITION_ROUTE,
        REMOTE_VISUAL_ONLY_D2K5_COMPETITION_ROUTE,
    )
}
D2K2_PACKAGE_CONTRACT_SCHEMA = "celltrack-alpha.cellect-d2k2-contract.v1"
D2K2_PACKAGE_MANIFEST_SCHEMA = "celltrack-alpha.cellect-d2k2-package.v1"
D2K2_CANDIDATE_ID = "CELLECT-NATIVE-3HEAD-X3RD-v1"
D2K2_FIXED_IDENTITIES = (
    "train:44b6_d754aa59",
    "train:44b6_18ced818",
)
D2K2_REQUIRED_SOURCE_MARKERS = (
    "discover_competition_root",
    "D2K2_STEP0_ASSIGNMENT_COMPLETE",
    "D2K2_HEAVY_WORK_BEGIN",
    "D2K2_VISUAL_RELEASE_COMPLETE",
    "RESOURCE_CAP_ABORT.json",
    "RESOURCE_MONITOR_ABORT.json",
    "D2K2_CUDA_EXIT_RECEIPT",
)
D2K2_RESOURCE_POLICY = {
    "native_child_host_rss_cap_bytes": 29_000_000_000,
    "cuda_per_process_cap_rule": "min(29000000000, floor(0.90 * cuda:0_total_memory_bytes))",
}
D2K3_INLINE_STEP0_MODE = "d2k3_step0_before_native_release.v1"
D2K5_INLINE_STEP0_MODE = "d2k5_step0_before_declared_carrier_native_release.v1"
D2K5_ASSET_MANIFEST_SHA256 = (
    "f018575564e29de6f973ddc2b0b1a5562956648da7e57ea076f604f408c48575"
)
IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*:[A-Za-z0-9][A-Za-z0-9_-]*")
# Deliberately empty by default.  A non-model exception is a code-reviewed central
# grant, not a contract author choosing a plausible profile name.  Entries are
# ``allowlist_id: (kernel_id, execution_profile, exact_code_sha256)``.
NON_MODEL_EXECUTION_ALLOWLIST: dict[str, tuple[str, str, str]] = {
    # D2K4 reads only the declared private GPL-noticed carrier and hashes its
    # immutable records.  It has no model imports or competition mount, and the
    # exact source hash is independently certified by ship-kaggle-kernels.
    "cellect-d2k4-asset-transport-v1": (
        "onebitbrain/g02-cellect-asset-transport-canary-v1",
        "cellect_asset_transport_step0_only.v1",
        "a0a861957aedfe41e1f0f2fbda0944a284220e584da00d9f6614d42e2cb56829",
    ),
}
# A remote model visual canary may bypass the *local MP4* prerequisite only after
# an independent, root-authored amendment pins this exact record.  This is
# empty by default; any production entry below is a reviewed amendment.  Contract
# data cannot populate or widen it.
#
# ``exception_id: {record_path, record_sha256, kernel_id, execution_profile}``
REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST: dict[str, dict[str, str]] = {
    "cellect-d2k2-remote-visual-only-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k2_kaggle_visual/REMOTE_VISUAL_ONLY_EXCEPTION.json"
        ),
        "record_sha256": "c227291a53ede50742058adbdae26f70b3d568128a275b09a5468e49f7e8722b",
        "kernel_id": "onebitbrain/g02-cellect-visual-only-v2",
        "execution_profile": "remote_visual_only_resource_stop.v1",
    },
    "cellect-d2k3-remote-visual-only-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k3_kaggle_visual/REMOTE_VISUAL_ONLY_EXCEPTION.json"
        ),
        "record_sha256": "84f7e4ba2f3710c43e20162c98ae6195b143cdfee2261fc19cff0e280db9765d",
        "kernel_id": "onebitbrain/g02-cellect-visual-only-v3",
        "execution_profile": "remote_visual_only_resource_stop.v1",
    },
    "cellect-d2k5-remote-visual-only-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/REMOTE_VISUAL_ONLY_EXCEPTION.json"
        ),
        "record_sha256": "eb0ddf68b89a0bdabf37cdf22cd663b313e1b17cc2a9d8cb6b4a11d9db0eb4f2",
        "kernel_id": "onebitbrain/g02-cellect-native-gpu-visual-v1",
        "execution_profile": "remote_visual_only_resource_stop.v1",
    },
    "cellect-d2k5-v2-two-carrier-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v2_successor/REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "02053872a9ea66ed695031368d245cf3f2d778daee45a06a36ff35ba7b19addf",
        "kernel_id": CELLECT_TWO_CARRIER_KERNEL_ID,
        "execution_profile": CELLECT_TWO_CARRIER_EXECUTION_PROFILE,
    },
    "cellect-d2k5-v3-mount-repair-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v3_mount_root/REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "4efce8b0b65973f40b114a9c6d0dff85bad57b4957d7a5c420ee1f88fd45c240",
        "kernel_id": CELLECT_TWO_CARRIER_MOUNT_REPAIR_KERNEL_ID,
        "execution_profile": CELLECT_TWO_CARRIER_EXECUTION_PROFILE,
    },
    "cellect-d2k5-v4-p100-three-carrier-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v4_p100_sm60/"
            "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "40b0b26d37e0527842ed9b67081c8f6d86975c22e79243f3078d060c59b2927c",
        "kernel_id": CELLECT_P100_KERNEL_ID,
        "execution_profile": CELLECT_P100_EXECUTION_PROFILE,
    },
    "cellect-d2k5-v5-p100-monitor-reap-three-carrier-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v5_p100_monitor_reap/"
            "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "cc2814faf6372ff4903aa375f8fb01ead264e924331d1f9eb2e5008e62cd585e",
        "kernel_id": CELLECT_P100_REAP_KERNEL_ID,
        "execution_profile": CELLECT_P100_EXECUTION_PROFILE,
    },
    "cellect-d2k5-v6-p100-typed-terminal-reap-three-carrier-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v6_p100_typed_terminal_reap/"
            "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "41f842e80da54089cd8e6e46b7ee9be453fafaca972fadf3a9a7118f386d6763",
        "kernel_id": CELLECT_P100_TYPED_REAP_KERNEL_ID,
        "execution_profile": CELLECT_P100_EXECUTION_PROFILE,
    },
    "linajea-v4-r3-legacy-gpu-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/"
            "REMOTE_VISUAL_ONLY_EXCEPTION_V4_R3_ROOT_APPROVED.json"
        ),
        "record_sha256": "d74e99281370c1a78ef98786ba061e9b3730be0fd3df1adf4d027d465b032c1b",
        "kernel_id": LINAJEA_LEGACY_GPU_KERNEL_ID,
        "execution_profile": LINAJEA_LEGACY_GPU_EXECUTION_PROFILE,
    },
    "linajea-v4-mount-repair-legacy-gpu-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/mount_root_repair_v4/"
            "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "edb72263947eae56ca9b61ac7534a85b8bc9b78944cebc8b17812efce81b7b95",
        "kernel_id": LINAJEA_LEGACY_GPU_MOUNT_REPAIR_KERNEL_ID,
        "execution_profile": LINAJEA_LEGACY_GPU_EXECUTION_PROFILE,
    },
    "linajea-v7-runtime-output-hardened-legacy-gpu-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/runtime_output_hardening_v7/"
            "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "54b0e37c52a26782523ebe581c7834a88bd2aaf74610d64372d06dd6b1480bba",
        "kernel_id": LINAJEA_PROVIDER_ROOT_KERNEL_ID,
        "execution_profile": LINAJEA_LEGACY_GPU_EXECUTION_PROFILE,
    },
    "linajea-v8-host-driver-path-legacy-gpu-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/host_driver_path_v8/"
            "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "5debbdeedf0e0357fb1f9d70cf4aa1f44bd26673ea41a50e441b265693258b11",
        "kernel_id": LINAJEA_HOST_DRIVER_KERNEL_ID,
        "execution_profile": LINAJEA_LEGACY_GPU_EXECUTION_PROFILE,
    },
    "linajea-v9-device-routing-legacy-gpu-remote-visual-v1": {
        "record_path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/device_routing_v9/"
            "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
        ),
        "record_sha256": "cdc1ab56cf32edf1abb546b1a2e4eea60efed87702110da5f639ff12e5bd022c",
        "kernel_id": LINAJEA_DEVICE_ROUTING_KERNEL_ID,
        "execution_profile": LINAJEA_LEGACY_GPU_EXECUTION_PROFILE,
    },
}
# ``runner_id: {path, sha256}``.  Entries are independently reviewed central
# grants; contract data cannot add, replace, or modify them.
TRUSTED_LOCAL_RUNNERS: dict[str, dict[str, str]] = {
    "cellect-native-local-runner-v1": {
        "path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/runner/"
            "implementation/cellect_native_local_runner.py"
        ),
        "sha256": "fa483b8db9ff2ea587e9e474a0c7e2444ce69dcb371dac74e5799707f64985e5",
    },
    "ot2-v3-local-runner-v1": {
        "path": (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "ot2_v3_runtime/implementation/ot2_v3_local_runner.py"
        ),
        "sha256": "f961d266bdc94e2aa8a1c73c6b42a740d72947bfd8242e26f6de0db2327b3f36",
    },
    "f2-v32-child-local-runner-v1": {
        "path": (
            "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
            "f2_v32_child_local_visual_v1/runner/f2_v32_child_local_runner.py"
        ),
        "sha256": "ee222bf8a1db6ce933811bf8b8579d94dea7972889f8f8824904b540ce357373",
    },
}
MODEL_EXECUTION_IMPORTS = {
    "celltrack_alpha",
    "onnxruntime",
    "torch",
    "torchvision",
    "trackastra",
    "tracksdata",
}
MODEL_EXECUTION_MARKERS = (
    "load_state_dict",
    "torch.load",
    "model.predict",
    "model.forward",
    "run_inference",
    "predict_tracks",
    "tracking_model",
)
VISUAL_FAILURE_TAXONOMY = {
    "border_failures",
    "drift_or_jump_failures",
    "false_divisions",
    "fragmentation",
    "hallucinated_links",
    "identity_switches",
    "missed_divisions",
    "missed_links",
}

F2_SWEEP_IDENTITIES = (
    "train:44b6_cf2536e8",
    "train:44b6_3bb3690f",
    "train:44b6_aaf8b0ea",
    "train:44b6_c8e2a523",
    "train:44b6_87bba6c4",
    "train:44b6_996155de",
)
F2_CHILD_VISUAL_IDENTITIES = (
    "train:44b6_d754aa59",
    "train:44b6_18ced818",
)
F2_CHILD_VISUAL_EXECUTION_PROFILE = "f2_v32_child_cpu_visual.v1"
F2_CHILD_VISUAL_CONTRACT_SCHEMA = "celltrack-alpha.f2-child-local-visual-contract.v1"
F2_CHILD_VISUAL_KERNEL_ID = "onebitbrain/f2-v32-child-visual-v1"
F2_CHILD_VISUAL_MODEL_ID = "public-v32-dual-seed-control"
F2_CHILD_VISUAL_LOCAL_RUN_ID = "f2-v32-child-cpu-six-frames-v1"
F2_CHILD_VISUAL_RUNNER_ID = "f2-v32-child-local-runner-v1"
F2_CHILD_VISUAL_DOCKER_IMAGE_ID = (
    "sha256:143b518e7ebac16c724fbedfa53033768f162502f2f1bfd003bb4d5640c06614"
)
F2_CHILD_VISUAL_FRAMES_PER_IDENTITY = 6
F2_CHILD_VISUAL_DOCKER_MEMORY_MIB = 4096
F2_CHILD_VISUAL_MAXIMUM_COMBINED_RSS_MIB = 8192.0
F2_CHILD_VISUAL_PRIMARY_CHECKPOINT = (
    "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
    "weights/unet_transformer/split_0/edge_predictor_best.pth"
)
F2_CHILD_VISUAL_PRIMARY_CHECKPOINT_SHA256 = (
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
)
F2_CHILD_VISUAL_SECONDARY_CHECKPOINT = (
    "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_secondary/"
    "weights/unet_transformer/split_0/edge_predictor_best.pth"
)
F2_CHILD_VISUAL_SECONDARY_CHECKPOINT_SHA256 = (
    "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
)
F2_CHILD_VISUAL_PRIMARY_MANIFEST = (
    "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_primary/"
    "ARTIFACT_MANIFEST.json"
)
F2_CHILD_VISUAL_PRIMARY_MANIFEST_SHA256 = (
    "bc20f1f04cfb682af3b27a836ce9a57f44a2fe27508dc59039200b2a23188103"
)
F2_CHILD_VISUAL_SECONDARY_MANIFEST = (
    "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/provider_secondary/"
    "ARTIFACT_MANIFEST.json"
)
F2_CHILD_VISUAL_SECONDARY_MANIFEST_SHA256 = (
    "7f01c0b2f9491606a1b543339e072230e25317d9c70fb9a18aea59e002a995eb"
)
F2_CHILD_VISUAL_RAW_ROOT = (
    "dag_prompts/2026-08-16-five-lane-alpha-swarm-v1/ot2_runtime/inputs"
)
F2_CHILD_VISUAL_FFMPEG_SHA256 = (
    "1332dc2de372bade9a8a63da0d6cdfab9de97fcefbae707bcc0b0506e1203327"
)
F2_SWEEP_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "D": tuple(
        {
            "id": f"D_det{value}",
            "family": "D",
            "delta": {"DET_THRESHOLD": value},
            "environment": {"F2_DET_THRESHOLD": value},
        }
        for value in ("0.9375", "0.95", "0.96", "0.975", "0.98", "0.99")
    )
    + ({"id": "CONTROL_V32", "family": "CONTROL", "delta": {}, "environment": {}},),
    "E": tuple(
        [
            {
                "id": f"E_sew{value}",
                "family": "E",
                "delta": {"SECONDARY_EDGE_WEIGHT": value},
                "environment": {"F2_SECONDARY_EDGE_WEIGHT": value},
            }
            for value in ("0.10", "0.20", "0.25")
        ]
        + [
            {
                "id": f"E_sdw{value}",
                "family": "E",
                "delta": {"SECONDARY_DETECTION_WEIGHT": value},
                "environment": {"F2_SECONDARY_DETECTION_WEIGHT": value},
            }
            for value in ("0.4", "0.6")
        ]
        + [
            {
                "id": f"E_dset{value}",
                "family": "E",
                "delta": {"DUAL_SEED_EDGE_THRESHOLD": value},
                "environment": {"F2_DUAL_SEED_EDGE_THRESHOLD": value},
            }
            for value in ("0.44", "0.52")
        ]
        + [
            {
                "id": "E_lowmargin0.45",
                "family": "E",
                "delta": {"SECONDARY_LOW_MARGIN_MAX": "0.45"},
                "environment": {"F2_SECONDARY_LOW_MARGIN_MAX": "0.45"},
            },
            {"id": "CONTROL_V32", "family": "CONTROL", "delta": {}, "environment": {}},
        ]
    ),
}
F2_SWEEP_KERNELS = {
    "D": {
        "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v5",
        "code_file": "f2_direct_d_sweep_kernel.py",
        "lane_owner": "F2_DIRECT_D_SWEEP",
        "output_root": "f2_direct_d_sweep_v1",
    },
    "E": {
        "kernel_id": "onebitbrain/f2-v32-direct-e-sweep-v5",
        "code_file": "f2_direct_e_sweep_kernel.py",
        "lane_owner": "F2_DIRECT_E_SWEEP",
        "output_root": "f2_direct_e_sweep_v1",
    },
}
F2_SWEEP_V6_KERNELS = {
    "D": {
        "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v6",
        "code_file": "f2_direct_d_sweep_kernel.py",
        "lane_owner": "F2_DIRECT_D_SWEEP",
        "output_root": "f2_direct_d_sweep_v1",
    },
}
F2_SWEEP_V7_KERNELS = {
    "D": {
        "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v7",
        "code_file": "f2_direct_d_sweep_kernel.py",
        "lane_owner": "F2_DIRECT_D_SWEEP",
        "output_root": "f2_direct_d_sweep_v1",
    },
}
F2_SWEEP_V8_KERNELS = {
    "D": {
        "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v8",
        "code_file": "f2_direct_d_sweep_kernel.py",
        "lane_owner": "F2_DIRECT_D_SWEEP",
        "output_root": "f2_direct_d_sweep_v1",
    },
}
F2_SWEEP_V9_KERNELS = {
    "D": {
        "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v9",
        "code_file": "f2_direct_d_sweep_kernel.py",
        "lane_owner": "F2_DIRECT_D_SWEEP",
        "output_root": "f2_direct_d_sweep_v1",
    },
}
F2_SWEEP_V10_KERNELS = {
    "D": {
        "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v10",
        "code_file": "f2_direct_d_sweep_kernel.py",
        "lane_owner": "F2_DIRECT_D_SWEEP",
        "output_root": "f2_direct_d_sweep_v1",
    },
}
F2_D_V6_NATIVE_T4_VISUAL_EXECUTION_PROFILE = "f2_d_v6_native_t4_visual.v1"
F2_D_V6_NATIVE_PRODUCTION_CONTRACT_SCHEMA = (
    "celltrack-alpha.f2-d-v6-native-production-prelaunch.v1"
)
F2_D_V6_NATIVE_VISUAL_IDENTITIES = (
    "train:44b6_d754aa59",
    "train:44b6_18ced818",
)
F2_D_V6_NATIVE_VISUAL_FRAMES_PER_IDENTITY = 6
F2_D_V6_NATIVE_TRANSPORT_ENVELOPE = {
    "physical_gpu_count": 2,
    "physical_gpu_name": "Tesla T4",
    "physical_gpu_compute_capability": "7.5",
    "application_cuda_visible_devices": "0",
    "application_visible_gpu_count": 1,
    "science_or_ilp_modified": False,
}
F2_D_V6_NATIVE_BINDINGS = {
    "code_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "kernel_f2_direct_d_sweep_v6/f2_direct_d_sweep_kernel.py"
    ),
    "code_sha256": "69b320872d9d804febcb46510714c7900895cb192a2f2b41fccfe6a7692d4589",
    "embedded_child_sha256": "e3ad0020157b3014c1044f2f887ebc4088b6fdfdef1c0f489f372985eacef20f",
    "frames_per_identity": F2_D_V6_NATIVE_VISUAL_FRAMES_PER_IDENTITY,
    "identities": list(F2_D_V6_NATIVE_VISUAL_IDENTITIES),
    "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v6",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "metadata_sha256": "7605c3a2b3317b0430078ca034fb4885307c23b4dd88586b22baa6ee15dfb2f0",
    "native_audit_sha256": "e9c7ea51c4f4de7fafd8b7c171b9593860cb66911da8c2ad75bb43611e84fa77",
    "native_bundle_sha256": "95f359726279cdaed536c508441921165efc66593b0e832263f3f81602057a63",
    "package_inventory_sha256": "6f8c7b052ce81320e73ea2332890d6ba97ad6a3ae517aba08d892e476a9a74f6",
    "profile_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "F2_DIRECT_D_SWEEP_V6_PREFLIGHT_PROFILE.json"
    ),
    "profile_sha256": "d1ec7b3bdc86bb2ce306fb773105e6c6ead6756ea70b54e5e0767cd4f3d4d3ef",
    "source_freeze_sha256": "622769daab4f6965b8dee22ffc3bcf74536f91e283630457448df472e0f61060",
    "transport_execution_envelope": F2_D_V6_NATIVE_TRANSPORT_ENVELOPE,
}
F2_D_V6_NATIVE_REQUIREMENTS = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v6_native_visual_prelaunch/CENTRAL_GUARD_REQUIREMENTS.json"
    ),
    "sha256": "cfbbff9f4a3b719e4546e2ee1553fc0b02648cf974bde38eb5ab0cdfd61c58bd",
}
F2_D_V6_NATIVE_SEALER = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v6_native_visual_prelaunch/seal_downloaded_evidence.py"
    ),
    "sha256": "0ff08b80993ccf8f16ed3bf1281f980a4e1806d01ce3282fbc60976564ba41e5",
}
F2_D_V7_NATIVE_T4_VISUAL_EXECUTION_PROFILE = "f2_d_v7_native_t4_visual.v1"
F2_D_V7_NATIVE_PRODUCTION_CONTRACT_SCHEMA = (
    "celltrack-alpha.f2-d-v7-native-production-prelaunch.v1"
)
F2_D_V7_NATIVE_VISUAL_IDENTITIES = F2_D_V6_NATIVE_VISUAL_IDENTITIES
F2_D_V7_NATIVE_VISUAL_FRAMES_PER_IDENTITY = (
    F2_D_V6_NATIVE_VISUAL_FRAMES_PER_IDENTITY
)
F2_D_V7_NATIVE_TRANSPORT_ENVELOPE = dict(F2_D_V6_NATIVE_TRANSPORT_ENVELOPE)
F2_D_V7_NATIVE_BINDINGS = {
    "code_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "kernel_f2_direct_d_sweep_v7/f2_direct_d_sweep_kernel.py"
    ),
    "code_sha256": "e5abc0ce30eb7ac15eddb2be791954af74e5c192e1b1fd2d0270b1f8d9e5ffb9",
    "embedded_child_sha256": "50eb4816ae852832025e3b96ea036356a197c06d936f3a8950226106decceb40",
    "frames_per_identity": F2_D_V7_NATIVE_VISUAL_FRAMES_PER_IDENTITY,
    "identities": list(F2_D_V7_NATIVE_VISUAL_IDENTITIES),
    "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v7",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "metadata_sha256": "2fa997ad33c94afd821432c1f416b68267336308d0b93085be310a3c71d66ac4",
    "native_audit_sha256": "1f393b3a7692ad61544045cf6ffd206ac1c118e0decb6fe618a263d2fd3fc5b1",
    "native_bundle_sha256": "abdc40ed6e584e02f0e09d9ff3bf8449486e3149a0a1a2f0c0cb5dfa263e9e9d",
    "native_transport_audit_sha256": (
        "291281a66ad026395e01d14ac7cda0cef3b795ac677e74df24f870d8c4156a04"
    ),
    "native_transport_sha256": (
        "6c806636de35b2a601c8e694825d76f38b79a38251994f0e898119bb5edbc49e"
    ),
    "package_inventory_sha256": (
        "ed7c81a967426a82c68d8b52945ef288c1e179860913a22be045883c91758af6"
    ),
    "profile_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "F2_DIRECT_D_SWEEP_V7_PREFLIGHT_PROFILE.json"
    ),
    "profile_sha256": "985b9a0fd839c221ed2e8f3151fd8573b4dda76efbe1070bda90de90bddbee44",
    "source_freeze_sha256": "672b82a2e40fb656e5bc2a232db3bd8d5270cbbc972d67f59ad283e5232378ca",
    "transport_execution_envelope": F2_D_V7_NATIVE_TRANSPORT_ENVELOPE,
}
F2_D_V7_NATIVE_REQUIREMENTS = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v7_native_visual_prelaunch/CENTRAL_GUARD_REQUIREMENTS.json"
    ),
    "sha256": "f4eccc094fe633e0d9294c8ddbe2e427eba5cd9bc82faaaf52755dbe08c35628",
}
F2_D_V7_NATIVE_SEALER = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v7_native_visual_prelaunch/seal_downloaded_evidence.py"
    ),
    "sha256": "644a8db475262937502a1f1d96e94ee6ec693dbe34e29d25a7ddfaf4d345d79a",
}
F2_D_V8_NATIVE_T4_VISUAL_EXECUTION_PROFILE = "f2_d_v8_native_t4_visual.v1"
F2_D_V8_NATIVE_PRODUCTION_CONTRACT_SCHEMA = (
    "celltrack-alpha.f2-d-v8-native-production-prelaunch.v1"
)
F2_D_V8_NATIVE_VISUAL_IDENTITIES = F2_D_V6_NATIVE_VISUAL_IDENTITIES
F2_D_V8_NATIVE_VISUAL_FRAMES_PER_IDENTITY = (
    F2_D_V6_NATIVE_VISUAL_FRAMES_PER_IDENTITY
)
F2_D_V8_NATIVE_TRANSPORT_ENVELOPE = dict(F2_D_V6_NATIVE_TRANSPORT_ENVELOPE)
F2_D_V8_NATIVE_BINDINGS = {
    "code_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "kernel_f2_direct_d_sweep_v8/f2_direct_d_sweep_kernel.py"
    ),
    "code_sha256": "d742f51783e7638ff3a2dcbddeb017ad0382080450d1d08f30347ea3b5c91f7a",
    "embedded_child_sha256": "a7c29aefd7613ff4687e11d04607aecd067225778bf061d0171809226e395467",
    "frames_per_identity": F2_D_V8_NATIVE_VISUAL_FRAMES_PER_IDENTITY,
    "identities": list(F2_D_V8_NATIVE_VISUAL_IDENTITIES),
    "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v8",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "metadata_sha256": "e2886a809345c1665dc1bbb500ec3a7c71c5d47135c5c9d6d4f8176bbecdd569",
    "native_audit_sha256": "0a4c32b8ffeaf890362edbf4fe8b045247b0cc134bbf046f847956299df3ba4d",
    "native_bundle_sha256": "9340def365eb4a03851560c0c1436f7ad57d1e29870e1ed1d909287066f8d7b3",
    "native_transport_audit_sha256": (
        "08c4d4518c223a16462a3452f07e74ec6af0a0f621021f7b7cbb81d7b73a5671"
    ),
    "native_transport_sha256": (
        "55cf867b3e96751aeb3bbf069c7d4a8149c7ec3e42f84a38852f031184dbad3a"
    ),
    "package_inventory_sha256": (
        "fbe577532b9c4d93578ff5a72e41010c307b311cb7c6dd08e3a059c7260a3ae5"
    ),
    "parent_v7_rejection_sha256": (
        "44e4b84f9e609ac8195b5130aac8c0de89a73c1ae1d777acd55e9ecc9ab40f68"
    ),
    "profile_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "F2_DIRECT_D_SWEEP_V8_PREFLIGHT_PROFILE.json"
    ),
    "profile_sha256": "8398fade50ddd54723c36f67408dd3a207704ef12fa120d12d5ced2cf80aad28",
    "source_freeze_sha256": "0ee24c51701ba92783c027e5684a30a14e915a2b8a3401b9bf222f7e8192a251",
    "transport_execution_envelope": F2_D_V8_NATIVE_TRANSPORT_ENVELOPE,
}
F2_D_V8_NATIVE_REQUIREMENTS = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v8_native_visual_prelaunch/CENTRAL_GUARD_REQUIREMENTS.json"
    ),
    "sha256": "b994ebf86226abb55e5710a291717167e938ef3adbe779b0fdacc852c926cc59",
}
F2_D_V8_NATIVE_SEALER = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v8_native_visual_prelaunch/seal_downloaded_evidence.py"
    ),
    "sha256": "87ff84e0ddb8ad1273962e698a82f21e4ea595ab167869227f9c2ccb75809cdd",
}
F2_D_V9_NATIVE_T4_VISUAL_EXECUTION_PROFILE = "f2_d_v9_native_t4_visual.v1"
F2_D_V9_NATIVE_PRODUCTION_CONTRACT_SCHEMA = (
    "celltrack-alpha.f2-d-v9-native-production-prelaunch.v1"
)
F2_D_V9_NATIVE_VISUAL_IDENTITIES = F2_D_V6_NATIVE_VISUAL_IDENTITIES
F2_D_V9_NATIVE_VISUAL_FRAMES_PER_IDENTITY = (
    F2_D_V6_NATIVE_VISUAL_FRAMES_PER_IDENTITY
)
F2_D_V9_NATIVE_TRANSPORT_ENVELOPE = dict(F2_D_V6_NATIVE_TRANSPORT_ENVELOPE)
F2_D_V9_NATIVE_BINDINGS = {
    "code_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "kernel_f2_direct_d_sweep_v9/f2_direct_d_sweep_kernel.py"
    ),
    "code_sha256": "3c81ca3b5c088f061e0f3835953556c0bd3aafeb36ed8bfe3cf864333d8b63e1",
    "embedded_child_sha256": "a1bc799fbdc61e83ef044f18827dcd56a4c30401e4a24aaf507bf4a92ccf3fcc",
    "frames_per_identity": F2_D_V9_NATIVE_VISUAL_FRAMES_PER_IDENTITY,
    "full_support_install_plan_sha256": (
        "44d759547070cca311a89b3676e3a949045f6b242443f2263d39b8593c3553d1"
    ),
    "identities": list(F2_D_V9_NATIVE_VISUAL_IDENTITIES),
    "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v9",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "metadata_sha256": "ca6c62a1957f6bdedc8c2579e63c114f17d8bedae99d9446761c80c9c02a27f0",
    "native_audit_sha256": "fa644a5af7d6f7e4d8db13ef74d093fa0e8082cd667ad9ed10fbf9670e4db6ba",
    "native_bundle_sha256": "c48b52c5ba3af07e98f7827c0af5c865155d6cd3b9dd46804459737e325602e9",
    "native_transport_audit_sha256": (
        "ee8bd563a97d690e30ffe070d8c74cee66953377da1aef25ba469dd58e876f7a"
    ),
    "native_transport_sha256": (
        "b3431211cf6662260856ceba9918bd97b03300b2752ec70df34b085d75bfd70f"
    ),
    "package_inventory_sha256": (
        "e2929647561e51de2607f53aad77a25def9641db9a3849d8f57427f1e2b1c61f"
    ),
    "parent_v8_rejection_sha256": (
        "7b48ca347954487d2e6991e3bf5872d86f951e58773d61fb1b4990ee7fd1496a"
    ),
    "profile_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "F2_DIRECT_D_SWEEP_V9_PREFLIGHT_PROFILE.json"
    ),
    "profile_sha256": "5d8061a80bff26b639ae84b2c9ce876ed08b0b28902cb7e4303ca729f028806f",
    "source_freeze_sha256": "898d7cc6204753a2713f70696a9a41cc0969207dd9f844b811dd8f8e7b851789",
    "transport_execution_envelope": F2_D_V9_NATIVE_TRANSPORT_ENVELOPE,
}
F2_D_V9_NATIVE_REQUIREMENTS = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v9_native_visual_prelaunch/CENTRAL_GUARD_REQUIREMENTS.json"
    ),
    "sha256": "bf0aca9f656e5904a8e4d8358ff52a945267ccef8b21c7df6f52340f7d2a048a",
}
F2_D_V9_NATIVE_SEALER = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v9_native_visual_prelaunch/seal_downloaded_evidence.py"
    ),
    "sha256": "4c3a6808ece4296b63cd7961e6da326e9d67b4588f7265c382fad411feac6c7b",
}
F2_D_V10_NATIVE_T4_VISUAL_EXECUTION_PROFILE = "f2_d_v10_native_t4_visual.v1"
F2_D_V10_NATIVE_PRODUCTION_CONTRACT_SCHEMA = (
    "celltrack-alpha.f2-d-v10-native-production-prelaunch.v1"
)
F2_D_V10_NATIVE_VISUAL_IDENTITIES = F2_D_V6_NATIVE_VISUAL_IDENTITIES
F2_D_V10_NATIVE_VISUAL_FRAMES_PER_IDENTITY = (
    F2_D_V6_NATIVE_VISUAL_FRAMES_PER_IDENTITY
)
F2_D_V10_NATIVE_TRANSPORT_ENVELOPE = dict(F2_D_V6_NATIVE_TRANSPORT_ENVELOPE)
F2_D_V10_NATIVE_BINDINGS = {
    "certifier_base_sha256": (
        "49d8ea390e67c0aa8a084f119e143ce878197119de81c4e72202ffdbb83d955f"
    ),
    "certifier_sha256": (
        "5529b376752a68a80f827018b3796807a206ed722afaca5c400a56e118ccffc0"
    ),
    "code_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "kernel_f2_direct_d_sweep_v10/f2_direct_d_sweep_kernel.py"
    ),
    "code_sha256": "da1b2ddc5c4a4a4b79f72c6e57393857b9e91dfe82739b2a0e9790392487bf6f",
    "embedded_child_sha256": "e417536c8ae03fc176bce7dd7df5fcd3367182f6dce3435784fc4eca21b880ed",
    "frames_per_identity": F2_D_V10_NATIVE_VISUAL_FRAMES_PER_IDENTITY,
    "full_support_install_plan_sha256": (
        "44d759547070cca311a89b3676e3a949045f6b242443f2263d39b8593c3553d1"
    ),
    "identities": list(F2_D_V10_NATIVE_VISUAL_IDENTITIES),
    "kernel_id": "onebitbrain/f2-v32-direct-d-sweep-v10",
    "lmt_sha256": "1ff54f1da2242a6ce7b596cce5b6b71129ece266dfe5888b0a5ca3884cc1069b",
    "metadata_sha256": "9dd1a2c7db6ba9a77add78b5998d91a30a36eafb554e5633d30afad76f219070",
    "native_audit_sha256": "3885d56a1703c911d9e63c682c80500db7a6ad0de2fb545bacc6b5954e9aa422",
    "native_bundle_sha256": "40dfb8b206173e9361315440df32e6e302a2a2fc2d21bd4f797bf5628ebc7c62",
    "native_transport_audit_sha256": (
        "dbdee82d4a53e66b2a060fa8c0b67b390fdff7288c3f59c1d7ec1e08ba7da0ae"
    ),
    "native_transport_sha256": (
        "57b3ac3b7ed114b487bee5ac70fc05eb1d73abcd411d61cfa6de0b1c459f522d"
    ),
    "package_inventory_sha256": (
        "4acd042bc67a1faa8cf7558d97b25d7a1502502c7fbba44c0aef1cffcae23980"
    ),
    "parent_v9_rejection_sha256": (
        "5370c55e18d0ec25ec673258e44a9b8ef40e15ff75123c53ecd10362bcfc174c"
    ),
    "profile_relative": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "F2_DIRECT_D_SWEEP_V10_PREFLIGHT_PROFILE.json"
    ),
    "profile_sha256": "cea1d52cd44fee3f6f8f0ec09f1fa13b0073135b3abea399548447ead1c0ffc8",
    "runtime_projection_patch_sha256": (
        "3fef87f23dc4a31a5275b3a928d2818cf26e1997cc37c0d1226be7a1d63c2226"
    ),
    "source_freeze_sha256": "62b7e1cfa179ac62ec8ae14724e7b2afab95a99cf1f94faa398c5acc33703718",
    "transport_execution_envelope": F2_D_V10_NATIVE_TRANSPORT_ENVELOPE,
}
F2_D_V10_NATIVE_REQUIREMENTS = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v10_native_visual_prelaunch/CENTRAL_GUARD_REQUIREMENTS.json"
    ),
    "sha256": "d266e596c2b5c6e2470732c2e3cba7716e05edc24f8302e4807abf3c3369282a",
}
F2_D_V10_NATIVE_SEALER = {
    "path": (
        "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
        "f2_d_v10_native_visual_prelaunch/seal_downloaded_evidence.py"
    ),
    "sha256": "8461f380295e45a023171e078e71a4f8f74af1a10295841da7ccbb028ed9966f",
}
TRUSTED_REMOTE_VISUAL_RUNNERS: dict[str, dict[str, str]] = {
    "f2-d-v6-native-t4-visual-v1": {
        "path": (
            "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
            "f2_d_v6_native_visual_prelaunch/interactive_native_visual.py"
        ),
        "sha256": "96d74fa087ead2d01b771d0e588a3cc5cdaa8312d8838788228ede77c5dc179d",
    },
    "f2-d-v7-native-t4-visual-v1": {
        "path": (
            "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
            "f2_d_v7_native_visual_prelaunch/interactive_native_visual.py"
        ),
        "sha256": "5ebbd7c43230dc3686d96dace39e9fff572e5bad0cfd61566473bbb1ab21f093",
    },
    "f2-d-v8-native-t4-visual-v1": {
        "path": (
            "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
            "f2_d_v8_native_visual_prelaunch/interactive_native_visual.py"
        ),
        "sha256": "6c93241708307743738ed908c0cbe5df10fa479b9778b8d1f505f217fd62a69e",
    },
    "f2-d-v9-native-t4-visual-v1": {
        "path": (
            "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
            "f2_d_v9_native_visual_prelaunch/interactive_native_visual.py"
        ),
        "sha256": "38cf673fa739717ceef61c49d10db66167cad41b2fa360913f0c11ea3e45a143",
    },
    "f2-d-v10-native-t4-visual-v1": {
        "path": (
            "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
            "f2_d_v10_native_visual_prelaunch/interactive_native_visual.py"
        ),
        "sha256": "482e29123979cbcc18ff52cbf2ef3533c63e887516bd1a692becda34e03f6312",
    },
}

F2_D_NATIVE_VISUAL_ROUTES: dict[str, dict[str, Any]] = {
    "v6": {
        "version": "v6",
        "bindings": F2_D_V6_NATIVE_BINDINGS,
        "execution_profile": F2_D_V6_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
        "production_contract_schema": F2_D_V6_NATIVE_PRODUCTION_CONTRACT_SCHEMA,
        "runner_id": "f2-d-v6-native-t4-visual-v1",
        "requirements": F2_D_V6_NATIVE_REQUIREMENTS,
        "sealer": F2_D_V6_NATIVE_SEALER,
        "native_exact_files": {
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v6_native_interactive_cert/interactive_native_certify.py"
            ): F2_D_V6_NATIVE_BINDINGS["native_bundle_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v6_native_interactive_cert/INTERACTIVE_CERTIFICATION_AUDIT.json"
            ): F2_D_V6_NATIVE_BINDINGS["native_audit_sha256"],
        },
        "execution_schema": (
            "celltrack-alpha.f2-d-v6-native-t4-visual-execution.v1"
        ),
        "native_result_schema": (
            "celltrack-alpha.f2-d-v6-native-interactive-result.v1"
        ),
        "download_manifest_schema": (
            "celltrack-alpha.f2-d-v6-native-visual-download.v1"
        ),
        "download_decode_schema": (
            "celltrack-alpha.f2-d-v6-downloaded-mp4-full-decode.v1"
        ),
        "human_review_schema": (
            "celltrack-alpha.f2-d-v6-native-visual-human-review.v1"
        ),
        "requirements_bind_sealer": False,
        "production_enabled": True,
    },
    "v7": {
        "version": "v7",
        "bindings": F2_D_V7_NATIVE_BINDINGS,
        "execution_profile": F2_D_V7_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
        "production_contract_schema": F2_D_V7_NATIVE_PRODUCTION_CONTRACT_SCHEMA,
        "runner_id": "f2-d-v7-native-t4-visual-v1",
        "requirements": F2_D_V7_NATIVE_REQUIREMENTS,
        "sealer": F2_D_V7_NATIVE_SEALER,
        "native_exact_files": {
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v7_native_interactive_cert/interactive_native_certify.py"
            ): F2_D_V7_NATIVE_BINDINGS["native_bundle_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v7_native_interactive_cert/INTERACTIVE_CERTIFICATION_AUDIT.json"
            ): F2_D_V7_NATIVE_BINDINGS["native_audit_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v7_native_interactive_cert/notebook_cell_transport.py"
            ): F2_D_V7_NATIVE_BINDINGS["native_transport_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v7_native_interactive_cert/NOTEBOOK_CELL_TRANSPORT_AUDIT.json"
            ): F2_D_V7_NATIVE_BINDINGS["native_transport_audit_sha256"],
        },
        "execution_schema": (
            "celltrack-alpha.f2-d-v7-native-t4-visual-execution.v1"
        ),
        "native_result_schema": (
            "celltrack-alpha.f2-d-v7-native-interactive-result.v1"
        ),
        "download_manifest_schema": (
            "celltrack-alpha.f2-d-v7-native-visual-download.v1"
        ),
        "download_decode_schema": (
            "celltrack-alpha.f2-d-v7-downloaded-mp4-full-decode.v1"
        ),
        "human_review_schema": (
            "celltrack-alpha.f2-d-v7-native-visual-human-review.v1"
        ),
        "requirements_bind_sealer": True,
        "production_enabled": False,
        "terminal_rejected_certificate": {
            "path": (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v7_native_interactive_cert/downloaded_native_t4/extracted/"
                "F2_DIRECT_D_SWEEP_V7_NATIVE_CERTIFICATION.json"
            ),
            "sha256": (
                "44e4b84f9e609ac8195b5130aac8c0de89a73c1ae1d777acd55e9ecc9ab40f68"
            ),
        },
    },
    "v8": {
        "version": "v8",
        "bindings": F2_D_V8_NATIVE_BINDINGS,
        "execution_profile": F2_D_V8_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
        "production_contract_schema": F2_D_V8_NATIVE_PRODUCTION_CONTRACT_SCHEMA,
        "runner_id": "f2-d-v8-native-t4-visual-v1",
        "requirements": F2_D_V8_NATIVE_REQUIREMENTS,
        "sealer": F2_D_V8_NATIVE_SEALER,
        "native_exact_files": {
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v8_native_interactive_cert/interactive_native_certify.py"
            ): F2_D_V8_NATIVE_BINDINGS["native_bundle_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v8_native_interactive_cert/INTERACTIVE_CERTIFICATION_AUDIT.json"
            ): F2_D_V8_NATIVE_BINDINGS["native_audit_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v8_native_interactive_cert/notebook_cell_transport.py"
            ): F2_D_V8_NATIVE_BINDINGS["native_transport_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v8_native_interactive_cert/NOTEBOOK_CELL_TRANSPORT_AUDIT.json"
            ): F2_D_V8_NATIVE_BINDINGS["native_transport_audit_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v7_native_interactive_cert/downloaded_native_t4/extracted/"
                "F2_DIRECT_D_SWEEP_V7_NATIVE_CERTIFICATION.json"
            ): F2_D_V8_NATIVE_BINDINGS["parent_v7_rejection_sha256"],
        },
        "execution_schema": (
            "celltrack-alpha.f2-d-v8-native-t4-visual-execution.v1"
        ),
        "native_result_schema": (
            "celltrack-alpha.f2-d-v8-native-interactive-result.v1"
        ),
        "download_manifest_schema": (
            "celltrack-alpha.f2-d-v8-native-visual-download.v1"
        ),
        "download_decode_schema": (
            "celltrack-alpha.f2-d-v8-downloaded-mp4-full-decode.v1"
        ),
        "human_review_schema": (
            "celltrack-alpha.f2-d-v8-native-visual-human-review.v1"
        ),
        "requirements_bind_sealer": True,
        "production_enabled": False,
        "terminal_rejected_certificate": {
            "path": (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v8_native_interactive_cert/downloaded_native_t4/extracted/"
                "F2_DIRECT_D_SWEEP_V8_NATIVE_CERTIFICATION.json"
            ),
            "sha256": (
                "7b48ca347954487d2e6991e3bf5872d86f951e58773d61fb1b4990ee7fd1496a"
            ),
        },
    },
    "v9": {
        "version": "v9",
        "bindings": F2_D_V9_NATIVE_BINDINGS,
        "execution_profile": F2_D_V9_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
        "production_contract_schema": F2_D_V9_NATIVE_PRODUCTION_CONTRACT_SCHEMA,
        "runner_id": "f2-d-v9-native-t4-visual-v1",
        "requirements": F2_D_V9_NATIVE_REQUIREMENTS,
        "sealer": F2_D_V9_NATIVE_SEALER,
        "native_exact_files": {
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v9_native_interactive_cert/interactive_native_certify.py"
            ): F2_D_V9_NATIVE_BINDINGS["native_bundle_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v9_native_interactive_cert/INTERACTIVE_CERTIFICATION_AUDIT.json"
            ): F2_D_V9_NATIVE_BINDINGS["native_audit_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v9_native_interactive_cert/notebook_cell_transport.py"
            ): F2_D_V9_NATIVE_BINDINGS["native_transport_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v9_native_interactive_cert/NOTEBOOK_CELL_TRANSPORT_AUDIT.json"
            ): F2_D_V9_NATIVE_BINDINGS["native_transport_audit_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v8_native_interactive_cert/downloaded_native_t4/extracted/"
                "F2_DIRECT_D_SWEEP_V8_NATIVE_CERTIFICATION.json"
            ): F2_D_V9_NATIVE_BINDINGS["parent_v8_rejection_sha256"],
        },
        "execution_schema": (
            "celltrack-alpha.f2-d-v9-native-t4-visual-execution.v1"
        ),
        "native_result_schema": (
            "celltrack-alpha.f2-d-v9-native-interactive-result.v1"
        ),
        "download_manifest_schema": (
            "celltrack-alpha.f2-d-v9-native-visual-download.v1"
        ),
        "download_decode_schema": (
            "celltrack-alpha.f2-d-v9-downloaded-mp4-full-decode.v1"
        ),
        "human_review_schema": (
            "celltrack-alpha.f2-d-v9-native-visual-human-review.v1"
        ),
        "requirements_bind_sealer": True,
        "production_enabled": False,
        "terminal_rejected_certificate": {
            "path": (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v9_native_interactive_cert/downloaded_native_t4/extracted/"
                "F2_DIRECT_D_SWEEP_V9_NATIVE_CERTIFICATION.json"
            ),
            "sha256": (
                "5370c55e18d0ec25ec673258e44a9b8ef40e15ff75123c53ecd10362bcfc174c"
            ),
        },
    },
    "v10": {
        "version": "v10",
        "bindings": F2_D_V10_NATIVE_BINDINGS,
        "execution_profile": F2_D_V10_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
        "production_contract_schema": F2_D_V10_NATIVE_PRODUCTION_CONTRACT_SCHEMA,
        "runner_id": "f2-d-v10-native-t4-visual-v1",
        "requirements": F2_D_V10_NATIVE_REQUIREMENTS,
        "sealer": F2_D_V10_NATIVE_SEALER,
        "native_exact_files": {
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v10_native_interactive_cert/interactive_native_certify.py"
            ): F2_D_V10_NATIVE_BINDINGS["native_bundle_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v10_native_interactive_cert/INTERACTIVE_CERTIFICATION_AUDIT.json"
            ): F2_D_V10_NATIVE_BINDINGS["native_audit_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v10_native_interactive_cert/notebook_cell_transport.py"
            ): F2_D_V10_NATIVE_BINDINGS["native_transport_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v10_native_interactive_cert/NOTEBOOK_CELL_TRANSPORT_AUDIT.json"
            ): F2_D_V10_NATIVE_BINDINGS["native_transport_audit_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v10_native_interactive_cert/runtime_projection_patch.py"
            ): F2_D_V10_NATIVE_BINDINGS["runtime_projection_patch_sha256"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v9_native_interactive_cert/downloaded_native_t4/extracted/"
                "F2_DIRECT_D_SWEEP_V9_NATIVE_CERTIFICATION.json"
            ): F2_D_V10_NATIVE_BINDINGS["parent_v9_rejection_sha256"],
        },
        "execution_schema": (
            "celltrack-alpha.f2-d-v10-native-t4-visual-execution.v1"
        ),
        "native_result_schema": (
            "celltrack-alpha.f2-d-v10-native-interactive-result.v1"
        ),
        "download_manifest_schema": (
            "celltrack-alpha.f2-d-v10-native-visual-download.v1"
        ),
        "download_decode_schema": (
            "celltrack-alpha.f2-d-v10-downloaded-mp4-full-decode.v1"
        ),
        "human_review_schema": (
            "celltrack-alpha.f2-d-v10-native-visual-human-review.v1"
        ),
        "requirements_bind_sealer": True,
        "production_enabled": False,
        "block_reason": "pending_exact_native_eligible_certificate",
    },
}


class GuardError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GuardError(f"{path}: expected JSON object")
    return payload


def title_slug(title: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def _project_file(
    raw_path: Any,
    *,
    project_root: Path,
    label: str,
    suffix: str | None = None,
    allow_empty: bool = False,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise GuardError(f"{label} must be a nonempty repository-relative path")
    root = project_root.resolve()
    unresolved = root / raw_path
    if unresolved.is_symlink():
        raise GuardError(f"{label} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GuardError(f"{label} escapes the repository root") from exc
    if path.is_symlink() or not path.is_file():
        raise GuardError(f"{label} must be a regular file")
    if not allow_empty and path.stat().st_size <= 0:
        raise GuardError(f"{label} must be a nonempty regular file")
    if suffix is not None and path.suffix.lower() != suffix:
        raise GuardError(f"{label} must end in {suffix}")
    return path


def _relative(path: Path, root: Path, label: str) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise GuardError(f"{label} escapes the repository root") from exc


def _package_inventory(kernel_dir: Path, project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    package = kernel_dir.resolve()
    _relative(package, root, "kernel package")
    if kernel_dir.is_symlink() or not package.is_dir():
        raise GuardError("kernel package must be a non-symbolic-link directory")
    files: list[dict[str, Any]] = []
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise GuardError(f"kernel package contains symbolic link: {path}")
        if not path.is_file():
            continue
        files.append(
            {
                "path": _relative(path, root, "kernel package file"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not files:
        raise GuardError("kernel package contains no regular files")
    rendered = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": _relative(package, root, "kernel package"),
        "file_count": len(files),
        "files": files,
        "tree_sha256": hashlib.sha256(rendered).hexdigest(),
    }


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise GuardError(f"{label} must be a nonempty RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GuardError(f"{label} is not a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise GuardError(f"{label} must include a timezone")
    return parsed


def _validate_mp4_container(path: Path, label: str) -> dict[str, Any]:
    """Require a probed video stream and a complete decode, not box-like bytes."""

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise GuardError("ffprobe and ffmpeg are required for local visual validation")
    try:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=index,codec_type,codec_name,nb_frames,nb_read_frames,duration",
                "-show_entries",
                "format=format_name,duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardError(f"{label} could not be probed: {exc}") from exc
    if probe.returncode != 0:
        raise GuardError(f"{label} failed ffprobe: {probe.stderr.strip()}")
    try:
        payload = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        raise GuardError(f"{label} returned malformed ffprobe JSON") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams:
        raise GuardError(f"{label} has no video stream")
    frames: list[int] = []
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        raw = stream.get("nb_read_frames", stream.get("nb_frames"))
        try:
            frames.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not frames or max(frames) <= 0:
        raise GuardError(f"{label} has zero decodable video frames")
    try:
        decode = subprocess.run(
            [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardError(f"{label} could not be decoded: {exc}") from exc
    if decode.returncode != 0:
        raise GuardError(f"{label} failed full video decode: {decode.stderr.strip()}")
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    try:
        duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError) as exc:
        raise GuardError(f"{label} has malformed duration metadata") from exc
    if duration <= 0:
        raise GuardError(f"{label} has nonpositive duration")
    return {
        "video_stream_count": len(streams),
        "decoded_frame_count": max(frames),
        "duration_seconds": duration,
    }


def _sha_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise GuardError(f"{label} must be a lowercase SHA-256")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTITY.fullmatch(value) is None:
        raise GuardError(
            f"{label} must be split:dataset with nonempty alphanumeric/_/- tokens"
        )
    return value


def _identity_set(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise GuardError(f"{label} must be a nonempty identity list")
    identities = [_identity(item, f"{label} item {index}") for index, item in enumerate(value)]
    if len(identities) != len(set(identities)):
        raise GuardError(f"{label} contains duplicate identities")
    return identities


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bound_project_asset(
    payload: dict[str, Any],
    *,
    path_key: str,
    sha_key: str,
    project_root: Path,
    label: str,
) -> tuple[Path, str]:
    path = _project_file(payload.get(path_key), project_root=project_root, label=label)
    expected = _sha_field(payload.get(sha_key), f"{label} hash")
    if sha256(path) != expected:
        raise GuardError(f"{label} hash drift")
    return path, expected


def _bound_reference(
    reference: Any,
    *,
    project_root: Path,
    label: str,
    suffix: str | None = None,
) -> tuple[Path, str]:
    """Resolve one hash-bound, repository-local evidence reference."""

    if not isinstance(reference, dict):
        raise GuardError(f"{label} reference is missing or malformed")
    path = _project_file(
        reference.get("path"),
        project_root=project_root,
        label=label,
        suffix=suffix,
    )
    expected = _sha_field(reference.get("sha256"), f"{label} hash")
    if sha256(path) != expected:
        raise GuardError(f"{label} hash drift")
    return path, expected


def _lmt_policy(project_root: Path) -> tuple[Path, str, dict[str, Any]]:
    """Read the current repository LMT policy without importing a launch tool."""

    path = project_root.resolve() / "LMT.md"
    if path.is_symlink() or not path.is_file():
        raise GuardError("current LMT.md is missing or not a regular file")
    source = path.read_text(encoding="utf-8")
    begin = "<!-- LMT_POLICY_JSON_BEGIN -->"
    end = "<!-- LMT_POLICY_JSON_END -->"
    try:
        block = source.split(begin, 1)[1].split(end, 1)[0].strip()
        if block.startswith("```json"):
            block = block[len("```json") :].strip()
        if block.endswith("```"):
            block = block[:-3].strip()
        policy = json.loads(block)
    except (IndexError, json.JSONDecodeError) as exc:
        raise GuardError(f"current LMT.md policy is malformed: {exc}") from exc
    if not isinstance(policy, dict) or policy.get("schema") != "cell-tracking.autonomy-limits.v1":
        raise GuardError("current LMT.md policy schema is unsupported")
    local = policy.get("local_resource_safety")
    if not isinstance(local, dict):
        raise GuardError("current LMT.md local resource policy is missing")
    limit = local.get("maximum_combined_rss_mib")
    if not isinstance(limit, (int, float)) or isinstance(limit, bool) or limit <= 0:
        raise GuardError("current LMT.md combined RSS ceiling is malformed")
    return path, sha256(path), policy


def _ship_package_inventory_sha256(kernel_dir: Path) -> str:
    """Match ship-kaggle-kernels' exact package-inventory digest contract."""

    if kernel_dir.is_symlink() or not kernel_dir.is_dir():
        raise GuardError("kernel package must be a non-symbolic-link directory")
    records: list[dict[str, Any]] = []
    for path in sorted(kernel_dir.rglob("*")):
        if path.is_symlink():
            raise GuardError(f"kernel package contains symbolic link: {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(kernel_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    rendered = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _reference_dict(path: Path, digest: str, project_root: Path, label: str) -> dict[str, str]:
    return {"path": _relative(path, project_root, label), "sha256": digest}


def _remote_review_lock(
    value: Any,
    *,
    identities: list[str],
    label: str,
    require_full_decode: bool = False,
) -> dict[str, Any]:
    """Require remote artefact retrieval and local review before later science."""

    if not isinstance(value, dict):
        raise GuardError(f"{label} is missing or malformed")
    expected = {
        "remote_mp4_retrieval_required": True,
        "local_visual_review_required": True,
        "score_or_promotion_permitted_before_review": False,
        "remote_review_receipt_schema": REMOTE_VISUAL_REVIEW_RECEIPT_SCHEMA,
    }
    if require_full_decode:
        expected["full_decode_required"] = True
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise GuardError(f"{label} drift: {key}")
    observed = _identity_set(value.get("required_remote_mp4_identities"), f"{label} identities")
    if observed != identities:
        raise GuardError(f"{label} identities differ from the fixed identity list")
    return {
        **expected,
        "required_remote_mp4_identities": observed,
        "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_REVIEW",
    }


def _remote_exception_summary(
    *,
    exception_id: str | None = None,
) -> dict[str, Any]:
    return {
        "required": True,
        "status": "FAIL",
        "exception_id": exception_id,
        "record_path": None,
        "record_sha256": None,
        "artifacts": [],
        "post_remote_review": None,
    }


def _is_remote_visual_only_contract(contract: dict[str, Any]) -> bool:
    return isinstance(contract.get("remote_visual_only_exception"), dict)


def _remote_visual_only_route(contract: dict[str, Any]) -> dict[str, Any] | None:
    """Return the one centrally enumerated route for this remote visual contract."""

    kernel_id = contract.get("kernel_id")
    return (
        REMOTE_VISUAL_ONLY_COMPETITION_ROUTES.get(kernel_id)
        if isinstance(kernel_id, str)
        else None
    )


def check_remote_visual_only_competition_route(
    *,
    contract: dict[str, Any],
    metadata: dict[str, Any],
    source: str,
) -> list[str]:
    """Allow exactly the D2K2 direct competition mount, never a broad bypass.

    This check deliberately lives before ordinary private-dataset ownership
    validation.  The later exception-record allowlist remains mandatory, so a
    contract cannot gain authority merely by spelling the competition path.
    """

    pinned = _remote_visual_only_route(contract)
    if pinned is None:
        return ["remote visual-only competition route has an unapproved kernel ID"]
    route = contract.get("remote_visual_input_route")
    if pinned.get("surface") == "d2k5_declared_asset":
        expected_route = {
            "mode": pinned["mode"],
            "competition_source": pinned["competition_source"],
            "competition_mount_aliases": pinned["competition_mount_aliases"],
            "asset_source": pinned["asset_source"],
            "asset_mount_root": pinned["asset_mount_root"],
        }
        expected_dataset_sources = [pinned["asset_source"]]
        expected_mount_root = pinned["competition_mount_aliases"][0]
    else:
        expected_route = {
            "mode": pinned["mode"],
            "competition_source": pinned["competition_source"],
            "mount_root": pinned["mount_root"],
        }
        expected_dataset_sources = []
        expected_mount_root = pinned["mount_root"]
    failures: list[str] = []
    if route != expected_route:
        failures.append("remote visual-only competition route is not the centrally pinned D2K2 route")
    if contract.get("dataset_sources") != expected_dataset_sources:
        failures.append(
            "remote visual-only competition route must declare dataset_sources=[]"
            if not expected_dataset_sources
            else "remote visual-only declared dataset sources drift"
        )
    if contract.get("competition_sources") != [
        pinned["competition_source"]
    ]:
        failures.append("remote visual-only competition source drift")
    if contract.get("expected_mount_root") != expected_mount_root:
        failures.append("remote visual-only competition mount root drift")
    if "exclusive_dataset_source" in contract:
        failures.append("remote visual-only competition route must not declare a private dataset source")
    if metadata != pinned["metadata"]:
        failures.append("remote visual-only competition metadata drift")
    if pinned.get("surface") == "d2k5_declared_asset":
        missing = [
            mount for mount in pinned["competition_mount_aliases"]
            if mount not in source
        ]
        if missing:
            failures.append("code does not contain every pinned D2K5 competition mount alias")
        if pinned["asset_mount_root"] not in source:
            failures.append("code does not contain the pinned D2K5 asset mount root")
    elif pinned["mount_root"] not in source:
        failures.append("code does not contain the pinned D2K2 competition mount root")
    return failures


def check_d2k2_remote_visual_surface(
    *,
    contract: dict[str, Any],
    kernel_dir: Path,
    source: str,
    project_root: Path,
    route: dict[str, Any] | None,
) -> tuple[int | None, list[str]]:
    """Validate the immutable CELLECT alternative to carrier/STEP0 conventions.

    CELLECT D2K2 was already packaged and independently certified before this
    guard amendment.  Its ``step0()`` owns a direct competition-mount
    assignment and its package manifest owns the offline native runtime and
    three frozen weights, so requiring the generic carrier's textual
    ``run_step0``/request/wheel/log markers would rewrite its certified bytes.
    This branch is available only after the exact route check and replaces
    those carrier-specific forms with hash-bound D2K2 package and release
    semantics.  It never applies to another visual or model kernel.
    """

    failures: list[str] = []
    step0_at: int | None = None
    try:
        if route is None:
            raise GuardError("remote visual-only route is not centrally enumerated")
        is_d2k3 = route["kernel_id"] == REMOTE_VISUAL_ONLY_D2K3_COMPETITION_ROUTE["kernel_id"]
        expected_mode = (
            D2K3_INLINE_STEP0_MODE
            if is_d2k3
            else "d2k2_step0_before_native_release.v1"
        )
        if contract.get("remote_preflight_mode") != expected_mode:
            raise GuardError("CELLECT explicit inline step-0 release mode is missing or drifted")
        if "remote_preflight_attestation" in contract:
            raise GuardError(
                "CELLECT remote visual-only launch must not use a synthetic remote preflight attestation"
            )
        package = kernel_dir.resolve()
        assets: dict[str, tuple[Path, str]] = {}
        for field, filename in (
            ("d2k2_package_contract", "D2K2_CONTRACT.json"),
            ("d2k2_package_manifest", "PACKAGE_MANIFEST.json"),
        ):
            path, digest = _bound_reference(
                contract.get(field),
                project_root=project_root,
                label=f"D2K2 {field}",
                suffix=".json",
            )
            if path != package / filename:
                raise GuardError(f"D2K2 {field} must bind kernel_package/{filename}")
            assets[field] = (path, digest)

        package_contract_path, package_contract_sha = assets["d2k2_package_contract"]
        package_manifest_path, _ = assets["d2k2_package_manifest"]
        package_contract = load_json(package_contract_path)
        package_manifest = load_json(package_manifest_path)

        if package_contract.get("schema") != D2K2_PACKAGE_CONTRACT_SCHEMA:
            raise GuardError("D2K2 package contract schema drift")
        if package_contract.get("candidate_id") != D2K2_CANDIDATE_ID:
            raise GuardError("D2K2 package contract candidate drift")
        if package_contract.get("learning_tuning_network") is not False:
            raise GuardError("D2K2 package contract permits learning or tuning")
        if package_contract.get("local_simulator_egress_forbidden") is not True:
            raise GuardError("D2K2 package contract permits simulator egress")
        competition_input = package_contract.get("competition_input")
        if not isinstance(competition_input, dict) or competition_input.get("source") != (
            route["competition_source"]
        ):
            raise GuardError("D2K2 package contract competition source drift")
        arrays = competition_input.get("arrays")
        if not isinstance(arrays, dict) or set(arrays) != {
            identity.split(":", 1)[1] for identity in D2K2_FIXED_IDENTITIES
        }:
            raise GuardError("D2K2 package contract fixed visual arrays drift")
        resource_policy = package_contract.get("resource_policy")
        if not isinstance(resource_policy, dict):
            raise GuardError("D2K2 package resource policy is missing")
        if resource_policy.get("native_child_host_rss_cap_bytes") != D2K2_RESOURCE_POLICY[
            "native_child_host_rss_cap_bytes"
        ]:
            raise GuardError("D2K2 package host resource cap drift")
        if resource_policy.get("cuda_per_process_cap_rule") != D2K2_RESOURCE_POLICY[
            "cuda_per_process_cap_rule"
        ]:
            raise GuardError("D2K2 package CUDA resource-cap rule drift")

        if package_manifest.get("schema") != D2K2_PACKAGE_MANIFEST_SCHEMA:
            raise GuardError("D2K2 package manifest schema drift")
        if package_manifest.get("candidate_id") != D2K2_CANDIDATE_ID:
            raise GuardError("D2K2 package manifest candidate drift")
        if package_manifest.get("entrypoint_unsealed_but_certifier_bound") != "kernel.py":
            raise GuardError("D2K2 package manifest entrypoint drift")
        if package_manifest.get("metadata_unsealed_but_certifier_bound") != "kernel-metadata.json":
            raise GuardError("D2K2 package manifest metadata binding drift")
        records = package_manifest.get("files")
        if not isinstance(records, list) or not records:
            raise GuardError("D2K2 package manifest file list is missing")
        manifest_records: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise GuardError(f"D2K2 package manifest file {index} is malformed")
            raw = record.get("path")
            if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
                raise GuardError(f"D2K2 package manifest file {index} path is unsafe")
            if raw in manifest_records:
                raise GuardError("D2K2 package manifest repeats a file path")
            path = package / raw
            if path.is_symlink() or not path.is_file():
                raise GuardError(f"D2K2 package manifest file is absent or linked: {raw}")
            expected_bytes = record.get("bytes")
            if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
                raise GuardError(f"D2K2 package manifest file bytes are malformed: {raw}")
            if path.stat().st_size != expected_bytes:
                raise GuardError(f"D2K2 package manifest file size drift: {raw}")
            if sha256(path) != _sha_field(record.get("sha256"), f"D2K2 package manifest {raw}"):
                raise GuardError(f"D2K2 package manifest file hash drift: {raw}")
            manifest_records[raw] = record
        contract_record = manifest_records.get("D2K2_CONTRACT.json")
        if contract_record is None or contract_record.get("sha256") != package_contract_sha:
            raise GuardError("D2K2 package manifest does not bind D2K2_CONTRACT.json")

        missing_markers = [marker for marker in D2K2_REQUIRED_SOURCE_MARKERS if marker not in source]
        if missing_markers:
            raise GuardError(
                "D2K2 source lacks required direct-release markers: "
                + ", ".join(missing_markers)
            )
        tree = ast.parse(source, filename=str(package / "kernel.py"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        d2k2_step0 = functions.get("step0")
        main = functions.get("main")
        if d2k2_step0 is None or main is None:
            raise GuardError("D2K2 source must define step0() and main()")
        step0_at = d2k2_step0.lineno
        main_calls = [
            (node.lineno, node.func.id)
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        step0_calls = [line for line, name in main_calls if name == "step0"]
        release_calls = [line for line, name in main_calls if name in {"preflight", "release"}]
        if not step0_calls or not release_calls or min(step0_calls) >= min(release_calls):
            raise GuardError("D2K2 main() does not execute step0 before release work")
        if is_d2k3:
            def has_environment_default(key: str, expected: str | int) -> bool:
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or len(node.args) < 2:
                        continue
                    function = node.func
                    if not (
                        isinstance(function, ast.Attribute)
                        and function.attr == "get"
                        and isinstance(function.value, ast.Attribute)
                        and function.value.attr == "environ"
                        and isinstance(function.value.value, ast.Name)
                        and function.value.value.id == "os"
                    ):
                        continue
                    first, default = node.args[:2]
                    if (
                        isinstance(first, ast.Constant)
                        and first.value == key
                        and isinstance(default, ast.Constant)
                        and default.value == expected
                    ):
                        return True
                return False

            if not has_environment_default("D2K2_MODE", "release"):
                raise GuardError("D2K3 source lacks the unattended release-mode default")
            if not has_environment_default("D2K2_WORK_UNITS", "32"):
                raise GuardError("D2K3 source lacks the unattended 32-unit default")

            def has_release_32_guard(node: ast.AST) -> bool:
                for expression in ast.walk(node):
                    if not isinstance(expression, ast.BoolOp) or not isinstance(expression.op, ast.And):
                        continue
                    comparisons = [part for part in expression.values if isinstance(part, ast.Compare)]
                    has_mode = any(
                        isinstance(compare.left, ast.Name)
                        and compare.left.id == "mode"
                        and len(compare.ops) == 1
                        and isinstance(compare.ops[0], ast.Eq)
                        and len(compare.comparators) == 1
                        and isinstance(compare.comparators[0], ast.Constant)
                        and compare.comparators[0].value == "release"
                        for compare in comparisons
                    )
                    has_units = any(
                        isinstance(compare.left, ast.Name)
                        and compare.left.id == "units"
                        and len(compare.ops) == 1
                        and isinstance(compare.ops[0], ast.NotEq)
                        and len(compare.comparators) == 1
                        and isinstance(compare.comparators[0], ast.Constant)
                        and compare.comparators[0].value == 32
                        for compare in comparisons
                    )
                    if has_mode and has_units:
                        return True
                return False

            if not has_release_32_guard(d2k2_step0):
                raise GuardError("D2K3 source lacks the release-mode 32-unit safety guard")
    except (GuardError, OSError, json.JSONDecodeError, SyntaxError) as exc:
        failures.append(str(exc))
    return step0_at, failures


def check_d2k5_remote_visual_surface(
    *,
    contract: dict[str, Any],
    source: str,
    project_root: Path,
) -> tuple[int | None, list[str]]:
    """Validate the script-only CELLECT kernel backed by the proven Dataset V2 carrier."""

    failures: list[str] = []
    step0_at: int | None = None
    try:
        if contract.get("remote_preflight_mode") != D2K5_INLINE_STEP0_MODE:
            raise GuardError("D2K5 explicit carrier-authenticated step-0 mode is missing or drifted")
        if "remote_preflight_attestation" in contract:
            raise GuardError("D2K5 must not use a synthetic remote preflight attestation")
        if contract.get("asset_manifest_sha256") != D2K5_ASSET_MANIFEST_SHA256:
            raise GuardError("D2K5 asset manifest hash drift")
        roundtrip_path, _ = _bound_reference(
            contract.get("asset_roundtrip_receipt"),
            project_root=project_root,
            label="D2K5 Dataset V2 round-trip receipt",
            suffix=".json",
        )
        roundtrip = load_json(roundtrip_path)
        roundtrip_exact = {
            "schema": "celltrack-alpha.cellect-d2k4-dataset-roundtrip.v1",
            "status": "PASS",
            "dataset_id": "onebitbrain/g02-cellect-native-threehead-assets-v1",
            "dataset_version": 2,
            "expected_mount_root": REMOTE_VISUAL_ONLY_D2K5_COMPETITION_ROUTE[
                "asset_mount_root"
            ],
            "carrier_manifest_sha256": D2K5_ASSET_MANIFEST_SHA256,
            "native_inference": False,
            "competition_input": False,
            "scoring": False,
            "submission": False,
        }
        for key, expected in roundtrip_exact.items():
            if roundtrip.get(key) != expected:
                raise GuardError(f"D2K5 Dataset V2 round-trip drift: {key}")
        comparison = roundtrip.get("comparison")
        if comparison != {
            "same_declared_paths": True,
            "same_declared_bytes": True,
            "same_declared_sha256": True,
            "raw_or_score_payload_present": False,
        }:
            raise GuardError("D2K5 Dataset V2 byte comparison drift")

        required_markers = (
            D2K5_ASSET_MANIFEST_SHA256,
            "verify_asset_carrier",
            "resolve_competition_root",
            "D2K5_STEP0_ASSIGNMENT_COMPLETE",
            "D2K5_HEAVY_WORK_BEGIN",
            "D2K5_VISUAL_RELEASE_COMPLETE",
            "RESOURCE_CAP_ABORT.json",
            "RESOURCE_MONITOR_ABORT.json",
            "D2K2_CUDA_EXIT_RECEIPT",
            "PREDICTIONS.json",
            "extended_sparse",
            "extended_crowded",
            ".mp4",
        )
        missing = [marker for marker in required_markers if marker not in source]
        if missing:
            raise GuardError("D2K5 source lacks required release markers: " + ", ".join(missing))

        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        step0 = functions.get("step0")
        main = functions.get("main")
        if step0 is None or main is None:
            raise GuardError("D2K5 source must define step0() and main()")
        step0_at = step0.lineno
        calls = [
            (node.lineno, node.func.id)
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        step0_calls = [line for line, name in calls if name == "step0"]
        release_calls = [line for line, name in calls if name in {"preflight", "release"}]
        if not step0_calls or not release_calls or min(step0_calls) >= min(release_calls):
            raise GuardError("D2K5 main() does not execute step0 before release work")

        defaults: dict[str, object] = {}
        for node in ast.walk(step0):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr == "get"
                and isinstance(function.value, ast.Attribute)
                and function.value.attr == "environ"
                and isinstance(function.value.value, ast.Name)
                and function.value.value.id == "os"
            ):
                continue
            key, default = node.args[:2]
            if isinstance(key, ast.Constant) and isinstance(default, ast.Constant):
                defaults[str(key.value)] = default.value
        if defaults.get("D2K5_MODE") != "release" or defaults.get("D2K5_WORK_UNITS") != "32":
            raise GuardError("D2K5 unattended release defaults drift")

        release_guard = any(
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.And)
            and "mode == 'release'" in ast.unparse(node)
            and "units != 32" in ast.unparse(node)
            for node in ast.walk(step0)
        )
        if not release_guard:
            raise GuardError("D2K5 source lacks the release-mode 32-unit safety guard")
    except (GuardError, OSError, json.JSONDecodeError, SyntaxError) as exc:
        failures.append(str(exc))
    return step0_at, failures


def _validate_private_carriers(
    value: Any,
    *,
    expected_roles: tuple[str, ...],
    metadata: dict[str, Any],
    source: str,
    project_root: Path,
    linajea_runtime_v2: bool = False,
    canonical_provider_roots: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Bind explicit private carriers and their downloaded provider bytes."""

    failures: list[str] = []
    artifacts: list[dict[str, Any]] = []
    if not isinstance(value, list) or len(value) != len(expected_roles):
        count_words = {2: "two", 3: "three"}
        count = count_words.get(len(expected_roles), str(len(expected_roles)))
        return [], artifacts, [
            f"remote visual successor requires exactly {count} private carriers"
        ]
    carriers: list[dict[str, Any]] = []
    try:
        for index, raw in enumerate(value):
            if not isinstance(raw, dict):
                raise GuardError(f"private carrier {index} is malformed")
            role = raw.get("role")
            if role != expected_roles[index]:
                raise GuardError(f"private carrier role/order drift at index {index}")
            dataset_id = raw.get("dataset_id")
            if (
                not isinstance(dataset_id, str)
                or not dataset_id.startswith("onebitbrain/")
                or dataset_id.count("/") != 1
            ):
                raise GuardError(f"private carrier {role} dataset ID is malformed")
            version = raw.get("dataset_version")
            if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
                raise GuardError(f"private carrier {role} dataset version is malformed")
            if linajea_runtime_v2 and role == "tf1_runtime_opaque" and version != 2:
                raise GuardError("Linajea opaque runtime must be Dataset Version 2")
            mount = raw.get("expected_mount_root")
            if (
                not isinstance(mount, str)
                or not mount.startswith("/kaggle/input/")
                or ".." in Path(mount).parts
            ):
                raise GuardError(f"private carrier {role} mount root is malformed")
            canonical_mount = f"/kaggle/input/datasets/{dataset_id}"
            if canonical_provider_roots and mount != canonical_mount:
                raise GuardError(
                    f"private carrier {role} mount root is not the canonical "
                    "owner-qualified Dataset root derived from metadata"
                )
            mount_name = raw.get("certifier_mount_name")
            if not isinstance(mount_name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", mount_name) is None:
                raise GuardError(f"private carrier {role} certifier mount name is malformed")
            provides_release_imports = raw.get("provides_release_imports")
            if not isinstance(provides_release_imports, bool):
                raise GuardError(
                    f"private carrier {role} release-import provider flag is malformed"
                )
            manifest_sha = _sha_field(
                raw.get("manifest_sha256"), f"private carrier {role} manifest hash"
            )
            if mount not in source or manifest_sha not in source:
                raise GuardError(f"private carrier {role} mount/hash is absent from release code")
            roundtrip_path, roundtrip_sha = _bound_reference(
                raw.get("dataset_roundtrip_receipt"),
                project_root=project_root,
                label=f"private carrier {role} round-trip receipt",
                suffix=".json",
            )
            roundtrip = load_json(roundtrip_path)
            allowed_schemas = {PRIVATE_CARRIER_ROUNDTRIP_SCHEMA}
            if role == "model_assets":
                allowed_schemas.update(
                    {
                        "celltrack-alpha.cellect-d2k4-dataset-roundtrip.v1",
                        "celltrack-alpha.cellect-asset-canonical-roundtrip.v1",
                    }
                )
            if roundtrip.get("schema") not in allowed_schemas:
                raise GuardError(f"private carrier {role} round-trip schema drift")
            exact: dict[str, Any] = {
                "status": "PASS",
                "dataset_id": dataset_id,
                "dataset_version": version,
            }
            if roundtrip.get("schema") == "celltrack-alpha.cellect-asset-canonical-roundtrip.v1":
                original_path, original_sha = _bound_reference(
                    roundtrip.get("original_subdirectory_roundtrip_receipt"),
                    project_root=project_root,
                    label="canonical CELLECT asset original round-trip receipt",
                    suffix=".json",
                )
                original = load_json(original_path)
                if (
                    original.get("schema")
                    != "celltrack-alpha.cellect-d2k4-dataset-roundtrip.v1"
                    or original.get("status") != "PASS"
                    or original.get("dataset_id") != dataset_id
                    or original.get("dataset_version") != version
                    or original.get("carrier_manifest_sha256") != manifest_sha
                ):
                    raise GuardError(
                        "canonical CELLECT asset round-trip does not bind the original "
                        "manifest-authenticated provider download"
                    )
                artifacts.append(
                    {
                        "role": "model_assets_original_dataset_roundtrip",
                        **_reference_dict(
                            original_path,
                            original_sha,
                            project_root,
                            "canonical CELLECT asset original round-trip receipt",
                        ),
                    }
                )
            else:
                exact["carrier_manifest_sha256"] = manifest_sha
            for key, expected in exact.items():
                if roundtrip.get(key) != expected:
                    raise GuardError(f"private carrier {role} round-trip drift: {key}")
            roundtrip_mount = roundtrip.get("expected_mount_root")
            correction_reference = raw.get("mount_layout_correction")
            if roundtrip_mount == mount:
                if correction_reference is not None:
                    raise GuardError(
                        f"private carrier {role} has an unnecessary mount-layout correction"
                    )
            else:
                correction_path, correction_sha = _bound_reference(
                    correction_reference,
                    project_root=project_root,
                    label=f"private carrier {role} mount-layout correction",
                    suffix=".json",
                )
                correction = load_json(correction_path)
                correction_exact = {
                    "schema": "celltrack-alpha.private-carrier-mount-layout-correction.v1",
                    "status": "PASS_SUPERSEDES_MOUNT_ASSERTION_ONLY",
                    "dataset_id": dataset_id,
                    "dataset_version": version,
                    "correct_expected_mount_root": mount,
                    "dataset_bytes_unchanged": True,
                    "dataset_version_unchanged": True,
                    "source_or_science_change": False,
                }
                for key, expected in correction_exact.items():
                    if correction.get(key) != expected:
                        raise GuardError(
                            f"private carrier {role} mount-layout correction drift: {key}"
                        )
                corrected_manifest = correction.get("carrier_manifest")
                # The correction already hash-binds the exact round-trip receipt,
                # whose carrier_manifest_sha256 was checked above.  A duplicated
                # manifest block is optional, but if present it must agree.
                if corrected_manifest is not None and (
                    not isinstance(corrected_manifest, dict)
                    or corrected_manifest.get("sha256") != manifest_sha
                ):
                    raise GuardError(
                        f"private carrier {role} mount-layout correction manifest drift"
                    )
                superseded = correction.get("superseded_roundtrip_receipt")
                if (
                    not isinstance(superseded, dict)
                    or superseded.get("path")
                    != _relative(
                        roundtrip_path,
                        project_root,
                        f"private carrier {role} round-trip receipt",
                    )
                    or superseded.get("sha256") != roundtrip_sha
                    or superseded.get("superseded_field") != "expected_mount_root"
                ):
                    raise GuardError(
                        f"private carrier {role} mount-layout correction does not bind the round-trip"
                    )
                if correction.get("download_root") != roundtrip.get("download_root"):
                    raise GuardError(
                        f"private carrier {role} corrected provider download root drift"
                    )
                artifacts.append(
                    {
                        "role": f"{role}_mount_layout_correction",
                        **_reference_dict(
                            correction_path,
                            correction_sha,
                            project_root,
                            f"private carrier {role} mount-layout correction",
                        ),
                    }
                )
            if roundtrip.get("comparison") != {
                "same_declared_paths": True,
                "same_declared_bytes": True,
                "same_declared_sha256": True,
                "raw_or_score_payload_present": False,
            }:
                raise GuardError(f"private carrier {role} byte comparison drift")
            carriers.append(dict(raw))
            artifacts.append(
                {
                    "role": f"{role}_dataset_roundtrip",
                    **_reference_dict(
                        roundtrip_path,
                        roundtrip_sha,
                        project_root,
                        f"private carrier {role} round-trip receipt",
                    ),
                }
            )

        if len({item["dataset_id"] for item in carriers}) != len(expected_roles):
            raise GuardError("private carrier dataset IDs must be distinct")
        if len({item["expected_mount_root"] for item in carriers}) != len(expected_roles):
            raise GuardError("private carrier mount roots must be distinct")
        if len({item["certifier_mount_name"] for item in carriers}) != len(expected_roles):
            raise GuardError("private carrier certifier mount names must be distinct")
        if not any(item["provides_release_imports"] for item in carriers):
            raise GuardError("private carriers declare no release-import provider")
        if metadata.get("dataset_sources") != [item["dataset_id"] for item in carriers]:
            raise GuardError("kernel metadata does not declare the exact two private carriers in order")
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
    return carriers, artifacts, failures


def _validate_enhanced_ship_certificate(
    reference: Any,
    *,
    contract: dict[str, Any],
    kernel_dir: Path,
    code_sha256: str,
    carriers: list[dict[str, Any]],
    project_root: Path,
    expected_memory_limit_bytes: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Require the post-zarr-failure release-import certificate, not an old receipt."""

    artifacts: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        certificate_path, certificate_sha = _bound_reference(
            reference,
            project_root=project_root,
            label="enhanced ship-kaggle certificate",
            suffix=".json",
        )
        certificate = load_json(certificate_path)
        current_lmt_path, current_lmt_sha, lmt_policy = _lmt_policy(project_root)
        exact = {
            "schema": "ship-kaggle-kernels.certification.v1",
            "status": "ELIGIBLE",
            "eligible": True,
            "autonomous_launch_eligible": True,
            "kernel_id": contract.get("kernel_id"),
            "limits_sha256": current_lmt_sha,
            "artifact_sha256": code_sha256,
            "metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
            "package_inventory_sha256": _ship_package_inventory_sha256(kernel_dir),
        }
        for key, expected in exact.items():
            if certificate.get(key) != expected:
                raise GuardError(f"enhanced ship-kaggle certificate drift: {key}")
        limits_text = certificate.get("limits_file")
        if not isinstance(limits_text, str) or not Path(limits_text).is_absolute():
            raise GuardError("enhanced ship-kaggle certificate limits path is not absolute")
        limits_path = Path(limits_text).resolve()
        if limits_path == Path("/workspace/LMT.md"):
            limits_path = current_lmt_path
        if limits_path != current_lmt_path:
            raise GuardError("enhanced ship-kaggle certificate limits path drift")

        has_mount_layout_correction = any(
            item.get("mount_layout_correction") is not None for item in carriers
        )
        quick_purpose = (
            "exercise the exact Kaggle CLI script/notebook materialization: only the "
            "declared entrypoint is present beside the fresh mounts, so undeclared local "
            "package siblings and provider mount-depth drift must fail before a remote "
            "version is spent"
            if has_mount_layout_correction
            else
            "exercise the exact Kaggle CLI script/notebook materialization: only the "
            "declared entrypoint is present beside the fresh mounts, so undeclared local "
            "package siblings must fail before a remote version is spent"
        )
        quick = certificate.get("quick_debug")
        if quick != {
            "auto_triggered": True,
            "name": "focused_pre_push_release_debug",
            "purpose": quick_purpose,
            "status": "PASS",
            "transport": "kaggle-cli-save-kernel.v1",
        }:
            raise GuardError("enhanced ship-kaggle focused release debug is absent or drifted")
        static = certificate.get("static")
        if not isinstance(static, dict) or static.get("status") != "PASS":
            raise GuardError("enhanced ship-kaggle static checks did not pass")
        repeatability = certificate.get("repeatability")
        if not isinstance(repeatability, dict) or repeatability.get("status") != "PASS":
            raise GuardError("enhanced ship-kaggle repeatability did not pass")
        projection = certificate.get("runtime_projection")
        if not isinstance(projection, dict) or projection.get("status") != "PASS":
            raise GuardError("enhanced ship-kaggle runtime projection did not pass")
        physical_memory = projection.get("physical_memory_limit_bytes")
        projected_peak = projection.get("projected_peak_rss_bytes")
        if (
            not isinstance(physical_memory, (int, float))
            or isinstance(physical_memory, bool)
            or physical_memory <= 0
            or not isinstance(projected_peak, (int, float))
            or isinstance(projected_peak, bool)
            or projected_peak <= 0
            or projected_peak >= physical_memory
        ):
            raise GuardError("enhanced ship-kaggle memory projection does not fit physical capacity")
        if expected_memory_limit_bytes is not None and physical_memory != expected_memory_limit_bytes:
            raise GuardError("enhanced ship-kaggle physical memory capacity drift")

        profile_text = certificate.get("profile_path")
        profile_sha = _sha_field(
            certificate.get("profile_sha256"), "enhanced ship-kaggle profile hash"
        )
        if not isinstance(profile_text, str) or not Path(profile_text).is_absolute():
            raise GuardError("enhanced ship-kaggle profile path is not absolute")
        profile_path = Path(profile_text).resolve()
        workspace_root = Path("/workspace")
        try:
            workspace_relative = profile_path.relative_to(workspace_root)
        except ValueError:
            workspace_relative = None
        if workspace_relative is not None:
            if not workspace_relative.parts or ".." in workspace_relative.parts:
                raise GuardError("enhanced ship-kaggle /workspace profile path is malformed")
            profile_path = (project_root / workspace_relative).resolve()
        try:
            profile_path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise GuardError("enhanced ship-kaggle profile escapes the repository") from exc
        if profile_path.is_symlink() or not profile_path.is_file() or sha256(profile_path) != profile_sha:
            raise GuardError("enhanced ship-kaggle profile hash drift")
        profile = load_json(profile_path)
        profile_mounts = profile.get("mounts")
        canonical_provider_profile = (
            contract.get("execution_profile") == CELLECT_P100_EXECUTION_PROFILE
            or contract.get("kernel_id") in LINAJEA_CANONICAL_PROVIDER_KERNEL_IDS
        )
        corrected_carriers = [
            item for item in carriers if item.get("mount_layout_correction") is not None
        ]
        checked_carriers = carriers if canonical_provider_profile else corrected_carriers
        if checked_carriers:
            if not isinstance(profile_mounts, list):
                raise GuardError("enhanced ship-kaggle profile mounts are missing")
            mounts_by_name = {
                item.get("name"): item
                for item in profile_mounts
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            for carrier in checked_carriers:
                mount_profile = mounts_by_name.get(carrier["certifier_mount_name"])
                if (
                    not isinstance(mount_profile, dict)
                    or mount_profile.get("remote_root") != carrier["expected_mount_root"]
                ):
                    raise GuardError(
                        f"enhanced ship-kaggle provider mount root drift: {carrier['role']}"
                    )
                correction_reference = carrier.get("mount_layout_correction")
                provider_layout = mount_profile.get("provider_layout")
                if (
                    not isinstance(provider_layout, dict)
                    or provider_layout.get("roundtrip_receipt")
                    != carrier["dataset_roundtrip_receipt"]
                    or provider_layout.get("mount_layout_correction")
                    != correction_reference
                ):
                    raise GuardError(
                        f"enhanced ship-kaggle provider topology binding drift: {carrier['role']}"
                    )
                if canonical_provider_profile and (
                    provider_layout.get("source_type") != "dataset"
                    or provider_layout.get("source_id") != carrier["dataset_id"]
                ):
                    raise GuardError(
                        f"enhanced ship-kaggle provider provenance drift: {carrier['role']}"
                    )
            if canonical_provider_profile:
                metadata = load_json(kernel_dir / "kernel-metadata.json")
                profile_kernel = profile.get("kernel")
                if (
                    not isinstance(profile_kernel, dict)
                    or profile_kernel.get("expected_metadata") != metadata
                ):
                    raise GuardError(
                        "fresh successor ship-kaggle profile does not bind the exact "
                        "kernel metadata"
                    )
                dataset_mounts = [
                    item
                    for item in profile_mounts
                    if isinstance(item, dict)
                    and isinstance(item.get("provider_layout"), dict)
                    and item["provider_layout"].get("source_type") == "dataset"
                ]
                if {
                    item["provider_layout"].get("source_id") for item in dataset_mounts
                } != set(metadata.get("dataset_sources", [])) or len(dataset_mounts) != len(
                    metadata.get("dataset_sources", [])
                ):
                    raise GuardError(
                        "enhanced ship-kaggle Dataset provider mapping is not one-to-one "
                        "with kernel metadata"
                    )
                competition_sources = metadata.get("competition_sources")
                competition_mounts = [
                    item
                    for item in profile_mounts
                    if isinstance(item, dict)
                    and isinstance(item.get("provider_layout"), dict)
                    and item["provider_layout"].get("source_type") == "competition"
                ]
                if not isinstance(competition_sources, list) or len(competition_sources) != 1:
                    raise GuardError("fresh successor requires exactly one competition source")
                competition_source = competition_sources[0]
                canonical_competition_root = (
                    f"/kaggle/input/competitions/{competition_source}"
                )
                if (
                    len(competition_mounts) != 1
                    or competition_mounts[0]["provider_layout"].get("source_id")
                    != competition_source
                    or competition_mounts[0].get("remote_root")
                    != canonical_competition_root
                ):
                    raise GuardError(
                        "enhanced ship-kaggle competition provider root/provenance drift"
                    )
        environment = profile.get("environment")
        if not isinstance(environment, dict):
            raise GuardError("enhanced ship-kaggle profile environment is missing")
        if canonical_provider_profile and environment.get("python_version_prefix") != "3.12":
            raise GuardError(
                "fresh successor ship-kaggle profile is not bound to Kaggle CPython 3.12"
            )
        required = environment.get("required_imports")
        carrier_imports = environment.get("carrier_provided_imports")
        declared = environment.get("declared_release_imports")
        if not all(isinstance(value, list) for value in (required, carrier_imports, declared)):
            raise GuardError("enhanced ship-kaggle release import declarations are missing")
        if not all(
            isinstance(item, str) and item
            for values in (required, carrier_imports, declared)
            for item in values
        ):
            raise GuardError("enhanced ship-kaggle release imports contain malformed module names")
        if (
            not carrier_imports
            or any(len(values) != len(set(values)) for values in (required, carrier_imports, declared))
        ):
            raise GuardError("enhanced ship-kaggle carrier release imports are empty or repeated")
        if set(required) & set(carrier_imports) or set(declared) != set(required) | set(carrier_imports):
            raise GuardError("enhanced ship-kaggle release import provider partition drift")
        if "zarr" not in set(declared):
            raise GuardError("enhanced ship-kaggle release imports do not prove zarr")
        if contract.get("execution_profile") in {
            CELLECT_TWO_CARRIER_EXECUTION_PROFILE,
            CELLECT_P100_EXECUTION_PROFILE,
        } and "cv2" not in set(declared):
            raise GuardError("CELLECT enhanced release imports do not prove cv2")
        if (
            contract.get("execution_profile") == CELLECT_P100_EXECUTION_PROFILE
            and "torch" not in set(carrier_imports)
        ):
            raise GuardError("CELLECT P100 enhanced release imports do not prove carrier torch")
        if contract.get("execution_profile") == CELLECT_P100_EXECUTION_PROFILE:
            expected_base = {"numpy", "pandas", "tifffile", "tqdm"}
            if set(required) != expected_base:
                raise GuardError(
                    "CELLECT P100 enhanced release imports do not prove the exact "
                    "native base dependency closure"
                )
        probe = environment.get("release_dependency_probe")
        marker = probe.get("stdout_marker") if isinstance(probe, dict) else None
        expected_mount_names = [
            item["certifier_mount_name"]
            for item in carriers
            if item["provides_release_imports"]
        ]
        if (
            not isinstance(marker, str)
            or not marker
            or probe.get("carrier_mounts") != expected_mount_names
            or contract.get("release_dependency_probe_marker") != marker
        ):
            raise GuardError("enhanced release dependency probe marker/carrier mounts drift")
        scenarios = profile.get("scenarios")
        if not isinstance(scenarios, list) or len(scenarios) != 4:
            raise GuardError("enhanced ship-kaggle profile must contain four scenarios")
        roles = set()
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict) or marker not in scenario.get("stdout_markers", []):
                raise GuardError(f"enhanced dependency probe is absent from scenario {index}")
            roles.add(scenario.get("role"))
        if roles != {"tiny", "stratified", "stress", "expansive"}:
            raise GuardError("enhanced ship-kaggle scenario roles drift")

        runs = certificate.get("simulation_runs")
        if not isinstance(runs, list) or len(runs) < 8:
            raise GuardError("enhanced ship-kaggle certificate lacks eight simulation runs")
        run_roles: list[str] = []
        for index, run in enumerate(runs):
            if not isinstance(run, dict) or run.get("status") != "PASS":
                raise GuardError(f"enhanced ship-kaggle simulation run {index} did not pass")
            if marker not in str(run.get("stdout_tail", "")):
                raise GuardError(f"enhanced dependency probe marker missing from simulation run {index}")
            materialization = run.get("upload_materialization")
            if not isinstance(materialization, dict) or materialization.get("transport") != "kaggle-cli-save-kernel.v1":
                raise GuardError(f"enhanced simulation run {index} transport drift")
            run_roles.append(str(run.get("role")))
        if set(run_roles) != {"tiny", "stratified", "stress", "expansive"}:
            raise GuardError("enhanced ship-kaggle simulation role coverage drift")
        if any(run_roles.count(role) < 2 for role in {"tiny", "stratified", "stress", "expansive"}):
            raise GuardError("enhanced ship-kaggle requires two runs per scenario role")
        adversarial = certificate.get("adversarial_runs")
        if not isinstance(adversarial, list) or len(adversarial) != 4 or {
            item.get("fault") for item in adversarial if isinstance(item, dict)
        } != {"missing_mount", "missing_required_file", "corrupt_required_file", "duplicate_basename"}:
            raise GuardError("enhanced ship-kaggle four-fault coverage drift")
        if any(not isinstance(item, dict) or item.get("status") != "PASS" for item in adversarial):
            raise GuardError("enhanced ship-kaggle adversarial fault did not pass")

        artifacts.extend(
            [
                {
                    "role": "enhanced_ship_kaggle_certificate",
                    **_reference_dict(
                        certificate_path,
                        certificate_sha,
                        project_root,
                        "enhanced ship-kaggle certificate",
                    ),
                },
                {
                    "role": "enhanced_ship_kaggle_profile",
                    **_reference_dict(
                        profile_path,
                        profile_sha,
                        project_root,
                        "enhanced ship-kaggle profile",
                    ),
                },
            ]
        )
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
    return artifacts, failures


def check_successor_remote_visual_surface(
    *,
    contract: dict[str, Any],
    metadata: dict[str, Any],
    source: str,
    project_root: Path,
) -> tuple[int | None, list[str]]:
    """Validate CELLECT's two-carrier repair or Linajea's legacy-GPU lane."""

    profile = contract.get("execution_profile")
    if profile == CELLECT_TWO_CARRIER_EXECUTION_PROFILE:
        approved_kernel_ids = {
            CELLECT_TWO_CARRIER_KERNEL_ID,
            CELLECT_TWO_CARRIER_MOUNT_REPAIR_KERNEL_ID,
        }
        actual_kernel_id = contract.get("kernel_id")
        expected_kernel = (
            actual_kernel_id
            if actual_kernel_id in approved_kernel_ids
            else CELLECT_TWO_CARRIER_KERNEL_ID
        )
        expected_roles = CELLECT_TWO_CARRIER_ROLES
        expected_mode = CELLECT_TWO_CARRIER_STEP0_MODE
        required_markers = (
            "verify_asset_carrier",
            "verify_dependency_carrier",
            "D2K5_STEP0_ASSIGNMENT_COMPLETE",
            "D2K5_HEAVY_WORK_BEGIN",
            "D2K5_VISUAL_RELEASE_COMPLETE",
            "PREDICTIONS.json",
            ".mp4",
        )
        linajea = False
        canonical_provider_roots = False
    elif profile == CELLECT_P100_EXECUTION_PROFILE:
        approved_kernel_ids = CELLECT_P100_KERNEL_IDS
        actual_kernel_id = contract.get("kernel_id")
        expected_kernel = (
            actual_kernel_id
            if actual_kernel_id in approved_kernel_ids
            else CELLECT_P100_KERNEL_ID
        )
        expected_roles = CELLECT_P100_CARRIER_ROLES
        expected_mode = CELLECT_P100_STEP0_MODE
        required_markers = (
            "verify_asset_carrier",
            "verify_dependency_carrier",
            "verify_torch_dependency_carrier",
            "D2K5_STEP0_ASSIGNMENT_COMPLETE",
            "D2K5_HEAVY_WORK_BEGIN",
            "D2K5_VISUAL_RELEASE_COMPLETE",
            (
                "D2K5_V6_NATIVE_SM60_TORCH_BOUND"
                if actual_kernel_id == CELLECT_P100_TYPED_REAP_KERNEL_ID
                else "D2K5_V5_NATIVE_SM60_TORCH_BOUND"
                if actual_kernel_id == CELLECT_P100_REAP_KERNEL_ID
                else "D2K5_V4_NATIVE_SM60_TORCH_BOUND"
            ),
            "PREDICTIONS.json",
            ".mp4",
        )
        linajea = False
        canonical_provider_roots = True
    elif profile == LINAJEA_LEGACY_GPU_EXECUTION_PROFILE:
        approved_kernel_ids = {
            LINAJEA_LEGACY_GPU_KERNEL_ID,
            LINAJEA_LEGACY_GPU_MOUNT_REPAIR_KERNEL_ID,
            *LINAJEA_CANONICAL_PROVIDER_KERNEL_IDS,
        }
        actual_kernel_id = contract.get("kernel_id")
        expected_kernel = (
            actual_kernel_id
            if actual_kernel_id in approved_kernel_ids
            else LINAJEA_LEGACY_GPU_KERNEL_ID
        )
        expected_roles = LINAJEA_TWO_CARRIER_ROLES
        expected_mode = LINAJEA_LEGACY_GPU_STEP0_MODE
        required_markers = (
            "verify_opaque_payload_transport",
            "LINAJEA_V3_STEP0_ASSIGNMENT_COMPLETE",
            "LINAJEA_V3_HEAVY_WORK_BEGIN",
            "LINAJEA_V3_VISUAL_RELEASE_COMPLETE",
            "side_1",
            "side_2",
            "dual_consensus",
            "_tracking.mp4",
        )
        linajea = True
        canonical_provider_roots = (
            actual_kernel_id in LINAJEA_CANONICAL_PROVIDER_KERNEL_IDS
        )
    else:
        return None, ["remote visual successor execution profile is unsupported"]

    failures: list[str] = []
    if contract.get("kernel_id") != expected_kernel:
        failures.append("remote visual successor kernel ID drift")
    if contract.get("remote_preflight_mode") != expected_mode:
        failures.append("remote visual successor inline Step-0 mode drift")
    if "remote_preflight_attestation" in contract:
        failures.append("remote visual successor must not use a synthetic preflight attestation")
    route = contract.get("remote_visual_input_route")
    expected_route = {
        "mode": expected_mode,
        "competition_source": "biohub-cell-tracking-during-development",
        "competition_mount_aliases": (
            ["/kaggle/input/competitions/biohub-cell-tracking-during-development"]
            if canonical_provider_roots
            else [
                "/kaggle/input/biohub-cell-tracking-during-development",
                "/kaggle/input/competitions/biohub-cell-tracking-during-development",
            ]
        ),
    }
    if route != expected_route:
        failures.append("remote visual successor competition route drift")
    carriers, _, carrier_failures = _validate_private_carriers(
        contract.get("private_carriers"),
        expected_roles=expected_roles,
        metadata=metadata,
        source=source,
        project_root=project_root,
        linajea_runtime_v2=linajea,
        canonical_provider_roots=canonical_provider_roots,
    )
    failures.extend(carrier_failures)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [item["dataset_id"] for item in carriers],
        "enable_gpu": True,
        "enable_internet": False,
        "id": expected_kernel,
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "title": expected_kernel.split("/", 1)[1],
    }
    if metadata != expected_metadata:
        failures.append("remote visual successor metadata drift")
    if contract.get("dataset_sources") != expected_metadata["dataset_sources"]:
        failures.append("remote visual successor declared dataset sources drift")
    if contract.get("competition_sources") != expected_metadata["competition_sources"]:
        failures.append("remote visual successor competition source drift")
    if contract.get("kernel_properties") != {"enable_gpu": True, "enable_internet": False}:
        failures.append("remote visual successor offline GPU properties drift")
    marker = contract.get("release_dependency_probe_marker")
    if not isinstance(marker, str) or not marker or marker not in source:
        failures.append("remote visual successor release dependency probe marker is absent from code")
    missing = [item for item in required_markers if item not in source]
    if missing:
        failures.append("remote visual successor release markers missing: " + ", ".join(missing))
    if actual_kernel_id == CELLECT_P100_REAP_KERNEL_ID:
        reap_markers = (
            "reap_after_monitor_disappearance",
            "exit_reaped_within_grace",
            "persist_bounded_log",
            "NATIVE_MONITOR_ABORT.json",
            "RESOURCE_CAP_ABORT.json",
            "publish_decoded_mp4",
            "atomic_publish_after_full_decode",
            "D2K5_V5_MONITOR_PROBE_COMPLETE",
            "D2K5_V5_CHILD_DIAGNOSTICS_PROBE_COMPLETE",
            "D2K5_V5_MEDIA_PROBE_COMPLETE",
        )
        missing_reap = [marker for marker in reap_markers if marker not in source]
        if missing_reap:
            failures.append(
                "fresh CELLECT V5 lacks monitor/reap/output hardening: "
                + ", ".join(missing_reap)
            )
    elif actual_kernel_id == CELLECT_P100_TYPED_REAP_KERNEL_ID:
        typed_reap_markers = (
            "NativeChildExited",
            "exit_observed_before_rss_error",
            "persist_bounded_log",
            "NATIVE_MONITOR_ABORT.json",
            "RESOURCE_CAP_ABORT.json",
            "NATIVE_OUTPUT_ACCEPTANCE_ABORT.json",
            "publish_decoded_mp4",
            "atomic_publish_after_full_decode",
            "D2K5_V6_MONITOR_PROBE_COMPLETE",
            "D2K5_V6_CHILD_DIAGNOSTICS_PROBE_COMPLETE",
            "D2K5_V6_MEDIA_PROBE_COMPLETE",
        )
        missing_typed_reap = [
            marker for marker in typed_reap_markers if marker not in source
        ]
        if missing_typed_reap:
            failures.append(
                "fresh CELLECT V6 lacks typed terminal/reap/output hardening: "
                + ", ".join(missing_typed_reap)
            )
    if canonical_provider_roots:
        legacy_competition_root = (
            "/kaggle/input/biohub-cell-tracking-during-development"
        )
        if legacy_competition_root in source:
            failures.append(
                "fresh successor code retains the legacy unqualified competition root"
            )
        if linajea:
            hardened_release_markers = (
                '"nvidia-smi"',
                '"-i"',
                "LINAJEA_V3_CV2_CODEC_PROBE_PASS",
                "cv2.VideoWriter",
                "cv2.VideoCapture",
                "full_decode_mp4",
                '"full_decode_verified": True',
            )
            missing_hardening = [
                marker for marker in hardened_release_markers if marker not in source
            ]
            if missing_hardening:
                failures.append(
                    "fresh Linajea successor lacks runtime/output hardening: "
                    + ", ".join(missing_hardening)
                )
            if "imageio" in source or "libx264" in source:
                failures.append(
                    "fresh Linajea successor retains an unproved imageio/libx264 MP4 path"
                )
            if actual_kernel_id in {
                LINAJEA_HOST_DRIVER_KERNEL_ID,
                LINAJEA_DEVICE_ROUTING_KERNEL_ID,
            }:
                host_driver_markers = (
                    "HOST_DRIVER_CANDIDATES",
                    "RUNTIME_CARRIER_HOST_DRIVER_SHADOW",
                    "HOST_DRIVER_LIBRARY_UNAVAILABLE",
                    "HOST_DRIVER_PY37_PROBE_FAILED",
                    "loaded_libcuda_provenance_pass",
                    "bound_child_environment_sha256",
                    "inherited_library_path_diagnostics",
                    '"inherited_path_used_for_loader_authority": False',
                    "PROMISED_TRACKING_MP4_OUTPUTS",
                )
                missing_host_driver = [
                    marker for marker in host_driver_markers if marker not in source
                ]
                if missing_host_driver:
                    failures.append(
                        "fresh Linajea host-driver successor lacks provenance hardening: "
                        + ", ".join(missing_host_driver)
                    )
                if actual_kernel_id == LINAJEA_DEVICE_ROUTING_KERNEL_ID:
                    device_routing_markers = (
                        '"allow_soft_placement": False',
                        "ScalarSummary|MergeSummary|SaveV2|RestoreV2|Const[dtype=DT_STRING]->CPU:0;all_other_ops->GPU:0",
                    )
                    missing_device_routing = [
                        marker for marker in device_routing_markers if marker not in source
                    ]
                    if missing_device_routing:
                        failures.append(
                            "fresh Linajea V9 lacks fail-closed device routing: "
                            + ", ".join(missing_device_routing)
                        )
            elif "validated_inherited_library_paths" not in source:
                failures.append(
                    "fresh Linajea successor lacks validated inherited library paths"
                )
    aliases = expected_route["competition_mount_aliases"]
    if any(alias not in source for alias in aliases):
        failures.append("remote visual successor code lacks a pinned competition mount alias")
    step0_at = next(
        (
            index
            for index, line in enumerate(source.splitlines(), start=1)
            if "STEP0_ASSIGNMENT_COMPLETE" in line
        ),
        None,
    )
    if step0_at is None:
        failures.append("remote visual successor Step-0 assignment marker is absent")
    return step0_at, failures


def _linajea_remote_review_lock(
    value: Any,
    *,
    identities: list[str],
    require_full_decode: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GuardError("Linajea six-MP4 post-review lock is missing or malformed")
    expected_artifacts = [
        {
            "identity": identity,
            "view": view,
            "relative_path": identity.split(":", 1)[1] + "/" + view + ".mp4",
        }
        for identity in identities
        for view in LINAJEA_REQUIRED_VIDEO_VIEWS
    ]
    exact = {
        "remote_mp4_retrieval_required": True,
        "local_visual_review_required": True,
        "score_or_promotion_permitted_before_review": False,
        "remote_review_receipt_schema": REMOTE_VISUAL_REVIEW_RECEIPT_SCHEMA,
        "required_remote_mp4_artifacts": expected_artifacts,
    }
    if require_full_decode:
        exact["full_decode_required"] = True
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise GuardError(f"Linajea six-MP4 post-review lock drift: {key}")
    if len(expected_artifacts) != 6:
        raise GuardError("Linajea post-review lock does not enumerate six MP4s")
    return {**exact, "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_REVIEW"}


def _validate_private_dataset_external_action_approval(
    reference: Any,
    *,
    carriers: list[dict[str, Any]],
    linajea: bool,
    project_root: Path,
    required_roles: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind owner approval to the exact newly created private dataset versions."""

    approval_path, approval_sha = _bound_reference(
        reference,
        project_root=project_root,
        label="private dataset external-action approval",
        suffix=".json",
    )
    approval = load_json(approval_path)
    for key, expected in {
        "schema": PRIVATE_DATASET_EXTERNAL_ACTION_APPROVAL_SCHEMA,
        "status": "APPROVED_AND_EXECUTED",
    }.items():
        if approval.get(key) != expected:
            raise GuardError(f"private dataset external-action approval drift: {key}")
    targets = approval.get("approved_targets")
    if not isinstance(targets, list) or not all(isinstance(item, dict) for item in targets):
        raise GuardError("private dataset external-action approved targets are malformed")
    target_tuples = {
        (
            item.get("dataset_id"),
            item.get("dataset_version"),
            item.get("roundtrip_receipt_path"),
            item.get("roundtrip_receipt_sha256"),
        )
        for item in targets
    }
    roles = set(required_roles) if required_roles is not None else {"cp312_release_dependencies"}
    if required_roles is None and linajea:
        roles.add("tf1_runtime_opaque")
    for carrier in carriers:
        if carrier["role"] not in roles:
            continue
        receipt = carrier["dataset_roundtrip_receipt"]
        required = (
            carrier["dataset_id"],
            carrier["dataset_version"],
            receipt["path"],
            receipt["sha256"],
        )
        if required not in target_tuples:
            raise GuardError(
                f"private dataset external-action approval does not bind {carrier['role']}"
            )
    required_exclusions = {
        "competition submission",
        "scoring",
        "training or fine-tuning",
        "raw competition data egress",
    }
    exclusions = approval.get("scope_exclusions")
    if not isinstance(exclusions, list) or set(exclusions) != required_exclusions:
        raise GuardError("private dataset external-action approval scope exclusions drift")
    return approval, [
        {
            "role": "private_dataset_external_action_approval",
            **_reference_dict(
                approval_path,
                approval_sha,
                project_root,
                "private dataset external-action approval",
            ),
        }
    ]


def _validate_cp312_certifier_interpreter_receipt(
    reference: Any,
    *,
    certificate_reference: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]]:
    """Bind a passing ship certificate to its actual CPython-3.12 executor."""

    receipt_path, receipt_sha = _bound_reference(
        reference,
        project_root=project_root,
        label="CPython 3.12 certifier interpreter receipt",
        suffix=".json",
    )
    certificate_path, certificate_sha = _bound_reference(
        certificate_reference,
        project_root=project_root,
        label="enhanced ship-kaggle certificate",
        suffix=".json",
    )
    receipt = load_json(receipt_path)
    certificate = load_json(certificate_path)
    if (
        receipt.get("schema")
        != "celltrack-alpha.local-certifier-interpreter-execution.v1"
        or receipt.get("status") != "PASS"
    ):
        raise GuardError("CPython 3.12 certifier interpreter receipt status/schema drift")
    interpreter = receipt.get("interpreter")
    if (
        not isinstance(interpreter, dict)
        or interpreter.get("implementation") != "CPython"
        or re.fullmatch(r"3\.12(?:\.\d+)?", str(interpreter.get("version", ""))) is None
        or not str(interpreter.get("executable", "")).startswith("/")
        or "Linux" not in str(interpreter.get("platform", ""))
        or interpreter.get("machine") != "x86_64"
    ):
        raise GuardError("certifier interpreter is not authenticated Linux x86_64 CPython 3.12")
    container = receipt.get("container")
    if not isinstance(container, dict) or container.get("platform_requested") != "linux/amd64":
        raise GuardError("certifier interpreter container platform drift")
    image_id = str(container.get("image_id", ""))
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise GuardError("certifier interpreter container image identity is malformed")
    dockerfile_path, dockerfile_sha = _bound_reference(
        {
            "path": container.get("dockerfile_path"),
            "sha256": container.get("dockerfile_sha256"),
        },
        project_root=project_root,
        label="certifier interpreter Dockerfile",
    )

    execution = receipt.get("certification")
    if not isinstance(execution, dict):
        raise GuardError("certifier interpreter certification binding is missing")
    profile_text = certificate.get("profile_path")
    if not isinstance(profile_text, str) or not Path(profile_text).is_absolute():
        raise GuardError("enhanced ship-kaggle certificate profile path is malformed")
    profile_path = Path(profile_text).resolve()
    workspace_root = Path("/workspace")
    try:
        workspace_relative = profile_path.relative_to(workspace_root)
    except ValueError:
        workspace_relative = None
    if workspace_relative is not None:
        profile_path = (project_root / workspace_relative).resolve()
    certificate_content_sha = _sha_field(
        certificate.get("receipt_content_sha256"),
        "enhanced ship-kaggle certificate content hash",
    )
    expected_execution = {
        "profile_path": _relative(
            profile_path,
            project_root,
            "certifier interpreter profile",
        ),
        "profile_sha256": certificate.get("profile_sha256"),
        "receipt_path": _relative(
            certificate_path,
            project_root,
            "certifier interpreter enhanced certificate",
        ),
        "receipt_sha256": certificate_sha,
        "receipt_content_sha256": certificate_content_sha,
        "static_status": "PASS",
        "static_failures": [],
        "simulation_runs_passed": 8,
        "adversarial_faults_passed": 4,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise GuardError(f"certifier interpreter certification binding drift: {key}")
    if certificate.get("static") != {"failures": [], "status": "PASS", "warnings": []}:
        raise GuardError("certifier interpreter does not bind a clean static PASS")
    if len(certificate.get("simulation_runs", [])) != 8 or len(
        certificate.get("adversarial_runs", [])
    ) != 4:
        raise GuardError("certifier interpreter run-count binding drift")
    return [
        {
            "role": "cp312_certifier_interpreter_receipt",
            **_reference_dict(
                receipt_path,
                receipt_sha,
                project_root,
                "CPython 3.12 certifier interpreter receipt",
            ),
        },
        {
            "role": "cp312_certifier_dockerfile",
            **_reference_dict(
                dockerfile_path,
                dockerfile_sha,
                project_root,
                "certifier interpreter Dockerfile",
            ),
        },
    ]


def _validate_cellect_v6_cp312_certifier_receipt(
    reference: Any,
    *,
    certificate_reference: dict[str, Any],
    interrupted_reference: dict[str, Any],
    kernel_dir: Path,
    code_sha256: str,
    project_root: Path,
) -> list[dict[str, Any]]:
    """Validate V6's truthful implicit-linux/amd64 CPython execution evidence."""

    receipt_path, receipt_sha = _bound_reference(
        reference,
        project_root=project_root,
        label="CELLECT V6 CPython 3.12 certifier receipt",
        suffix=".json",
    )
    certificate_path, certificate_sha = _bound_reference(
        certificate_reference,
        project_root=project_root,
        label="CELLECT V6 enhanced certificate",
        suffix=".json",
    )
    interrupted_path, interrupted_sha = _bound_reference(
        interrupted_reference,
        project_root=project_root,
        label="CELLECT V6 interrupted pre-B3 receipt",
        suffix=".json",
    )
    receipt = load_json(receipt_path)
    certificate = load_json(certificate_path)
    interrupted = load_json(interrupted_path)
    if (
        receipt.get("schema")
        != "celltrack-alpha.cellect-v6-local-cp312-certifier-execution.v1"
        or receipt.get("status") != "PASS"
    ):
        raise GuardError("CELLECT V6 certifier execution receipt status/schema drift")
    interpreter = receipt.get("interpreter")
    if (
        not isinstance(interpreter, dict)
        or interpreter.get("implementation") != "CPython"
        or re.fullmatch(r"3\.12(?:\.\d+)?", str(interpreter.get("version", "")))
        is None
        or not str(interpreter.get("executable", "")).startswith("/")
        or "Linux" not in str(interpreter.get("platform", ""))
        or interpreter.get("machine") != "x86_64"
    ):
        raise GuardError("CELLECT V6 certifier is not Linux x86_64 CPython 3.12")
    container = receipt.get("container")
    image_id = str(container.get("image_id", "")) if isinstance(container, dict) else ""
    if (
        not isinstance(container, dict)
        or container.get("image_platform") != "linux/amd64"
        or container.get("platform_requested") is not None
        or container.get("host_detected_platform") != "linux/arm64/v8"
        or container.get("network") != "none"
        or container.get("platform_resolution")
        != (
            "implicit_image_manifest; Docker warned that linux/amd64 differed "
            "from the detected linux/arm64/v8 host"
        )
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
    ):
        raise GuardError("CELLECT V6 truthful implicit platform binding drift")
    dockerfile_path, dockerfile_sha = _bound_reference(
        {
            "path": container.get("dockerfile_path"),
            "sha256": container.get("dockerfile_sha256"),
        },
        project_root=project_root,
        label="CELLECT V6 certifier Dockerfile",
    )
    image_build_path, image_build_sha = _bound_reference(
        {
            "path": container.get("image_build_receipt_path"),
            "sha256": container.get("image_build_receipt_sha256"),
        },
        project_root=project_root,
        label="CELLECT V6 certifier image build receipt",
        suffix=".json",
    )
    execution_evidence = receipt.get("execution_evidence")
    command = (
        execution_evidence.get("command")
        if isinstance(execution_evidence, dict)
        else None
    )
    if (
        not isinstance(execution_evidence, dict)
        or execution_evidence.get("status") != "PASS"
        or execution_evidence.get("returncode") != 0
        or not isinstance(command, list)
        or "--platform" in command
        or execution_evidence.get("resource_guard", {}).get("network") != "none"
    ):
        raise GuardError("CELLECT V6 certifier process/platform evidence drift")
    execution = receipt.get("certification")
    certifier_relative = (
        ".agents/skills/ship-kaggle-kernels/scripts/kaggle_kernel_certifier.py"
    )
    profile_text = certificate.get("profile_path")
    if not isinstance(profile_text, str) or not Path(profile_text).is_absolute():
        raise GuardError("CELLECT V6 certificate profile path is malformed")
    profile_path = Path(profile_text).resolve()
    try:
        workspace_relative = profile_path.relative_to(Path("/workspace"))
    except ValueError:
        workspace_relative = None
    if workspace_relative is not None:
        profile_path = (project_root / workspace_relative).resolve()
    expected_execution = {
        "artifact_sha256": code_sha256,
        "autonomous_launch_eligible": True,
        "certifier_path": certifier_relative,
        # This receipt is immutable historical evidence.  Bind the exact
        # certifier recorded when it was produced; do not retroactively make
        # it depend on today's deliberately mutable skill implementation.
        "certifier_sha256": ARCHIVED_SHIP_CERTIFIER_SHA256,
        "profile_path": _relative(
            profile_path, project_root, "CELLECT V6 certifier profile"
        ),
        "profile_sha256": certificate.get("profile_sha256"),
        "receipt_path": _relative(
            certificate_path, project_root, "CELLECT V6 certificate"
        ),
        "receipt_sha256": certificate_sha,
        "receipt_content_sha256": certificate.get("receipt_content_sha256"),
        "static_status": "PASS",
        "static_failures": [],
        "simulation_runs_passed": 8,
        "adversarial_faults_passed": 4,
        "status": "ELIGIBLE",
    }
    if not isinstance(execution, dict) or any(
        execution.get(key) != expected for key, expected in expected_execution.items()
    ):
        raise GuardError("CELLECT V6 certifier/certificate binding drift")
    validation = receipt.get("receipt_validation")
    expected_validation = {
        "frozen_kernel_sha256": code_sha256,
        "frozen_metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
        "frozen_package_inventory_sha256": _ship_package_inventory_sha256(kernel_dir),
        "frozen_profile_sha256": certificate.get("profile_sha256"),
    }
    if not isinstance(validation, dict) or any(
        validation.get(key) != expected for key, expected in expected_validation.items()
    ):
        raise GuardError("CELLECT V6 frozen receipt validation drift")
    pre_b3 = receipt.get("pre_b3_interrupted_attempt")
    expected_pre_b3 = {
        "eligible": False,
        "external_action": False,
        "path": _relative(
            interrupted_path, project_root, "CELLECT V6 interrupted receipt"
        ),
        "promotion_permitted": False,
        "sha256": interrupted_sha,
        "status": "INELIGIBLE_INTERRUPTED",
    }
    if not isinstance(pre_b3, dict) or any(
        pre_b3.get(key) != expected for key, expected in expected_pre_b3.items()
    ):
        raise GuardError("CELLECT V6 pre-B3 ineligibility binding drift")
    interrupted_exact = {
        "schema": "celltrack-alpha.cellect-d2k5-v6-interrupted-certification.v1",
        "status": "INTERRUPTED_INELIGIBLE_PRESERVED_NO_PROMOTION",
        "certifier_receipt_published": False,
        "external_action": False,
        "partial_result_accepted": False,
        "model_or_competition_raw_executed": False,
        "scoring_or_submission": False,
        "kernel_sha256": code_sha256,
        "profile_sha256": certificate.get("profile_sha256"),
    }
    if any(interrupted.get(key) != expected for key, expected in interrupted_exact.items()):
        raise GuardError("CELLECT V6 interrupted pre-B3 receipt is not ineligible")
    terminal_path, terminal_sha = _bound_reference(
        receipt.get("terminal_summary"),
        project_root=project_root,
        label="CELLECT V6 terminal certification summary",
        suffix=".json",
    )
    terminal = load_json(terminal_path)
    if (
        terminal.get("schema")
        != "celltrack-alpha.cellect-d2k5-v6-certification-terminal-summary.v1"
        or terminal.get("status") != "ELIGIBLE"
        or terminal.get("autonomous_launch_eligible") is not True
        or terminal.get("static") != "PASS"
        or terminal.get("simulation_runs") != {"passed": 8, "total": 8}
        or terminal.get("adversarial_runs") != {"passed": 4, "total": 4}
    ):
        raise GuardError("CELLECT V6 terminal certification summary drift")
    if (
        certificate.get("static") != {"failures": [], "status": "PASS", "warnings": []}
        or len(certificate.get("simulation_runs", [])) != 8
        or len(certificate.get("adversarial_runs", [])) != 4
    ):
        raise GuardError("CELLECT V6 enhanced certificate run-count drift")
    return [
        {
            "role": "cp312_certifier_interpreter_receipt",
            **_reference_dict(
                receipt_path,
                receipt_sha,
                project_root,
                "CELLECT V6 CPython 3.12 certifier receipt",
            ),
        },
        {
            "role": "cp312_certifier_dockerfile",
            **_reference_dict(
                dockerfile_path,
                dockerfile_sha,
                project_root,
                "CELLECT V6 certifier Dockerfile",
            ),
        },
        {
            "role": "cp312_certifier_image_build_receipt",
            **_reference_dict(
                image_build_path,
                image_build_sha,
                project_root,
                "CELLECT V6 image build receipt",
            ),
        },
        {
            "role": "interrupted_pre_b3_certification_ineligible",
            **_reference_dict(
                interrupted_path,
                interrupted_sha,
                project_root,
                "CELLECT V6 interrupted pre-B3 receipt",
            ),
        },
        {
            "role": "certification_terminal_summary",
            **_reference_dict(
                terminal_path,
                terminal_sha,
                project_root,
                "CELLECT V6 terminal certification summary",
            ),
        },
    ]


def check_successor_remote_visual_exception(
    *,
    contract: dict[str, Any],
    code_sha256: str,
    code_path: Path,
    kernel_dir: Path,
    project_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Validate separately governed CELLECT-v2 and Linajea-TF1 exceptions."""

    profile = contract.get("execution_profile")
    linajea = profile == LINAJEA_LEGACY_GPU_EXECUTION_PROFILE
    p100 = profile == CELLECT_P100_EXECUTION_PROFILE
    expected_roles = (
        LINAJEA_TWO_CARRIER_ROLES
        if linajea
        else CELLECT_P100_CARRIER_ROLES
        if p100
        else CELLECT_TWO_CARRIER_ROLES
    )
    canonical_provider_roots = p100 or (
        linajea
        and contract.get("kernel_id") in LINAJEA_CANONICAL_PROVIDER_KERNEL_IDS
    )
    summary = _remote_exception_summary()
    failures: list[str] = []
    nominated = contract.get("remote_visual_only_exception")
    exception_id = nominated.get("exception_id") if isinstance(nominated, dict) else None
    summary["exception_id"] = exception_id if isinstance(exception_id, str) else None
    if not isinstance(nominated, dict) or not isinstance(exception_id, str) or not exception_id:
        failures.append("remote visual successor exception reference is missing or malformed")
        return summary, failures
    central = REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST.get(exception_id)
    required_central = {
        "record_path": nominated.get("record_path"),
        "record_sha256": nominated.get("record_sha256"),
        "kernel_id": contract.get("kernel_id"),
        "execution_profile": profile,
    }
    if central != required_central:
        failures.append("remote visual successor exception is not in the central exact allowlist")
        return summary, failures

    try:
        record_path = _project_file(
            nominated.get("record_path"),
            project_root=project_root,
            label="remote visual successor exception record",
            suffix=".json",
        )
        record_sha = _sha_field(
            nominated.get("record_sha256"),
            "remote visual successor exception record hash",
        )
        if sha256(record_path) != record_sha:
            raise GuardError("remote visual successor exception record hash drift")
        record = load_json(record_path)
        remote = contract.get("remote_visual_only")
        if not isinstance(remote, dict):
            raise GuardError("remote visual successor prelaunch contract is missing")
        identities = _identity_set(
            remote.get("fixed_preregistered_visual_identities"),
            "remote visual successor fixed identities",
        )
        if identities != list(D2K2_FIXED_IDENTITIES):
            raise GuardError("remote visual successor fixed identity set/order drift")
        if contract.get("preregistered_visual_identities") != identities:
            raise GuardError("remote visual successor contract identity drift")
        record_exact = {
            "schema": REMOTE_VISUAL_ONLY_EXCEPTION_SCHEMA,
            "status": "APPROVED",
            "exception_id": exception_id,
            "candidate_id": remote.get("candidate_id"),
            "kernel_id": contract.get("kernel_id"),
            "execution_profile": profile,
            "fixed_preregistered_visual_identities": identities,
        }
        for key, expected in record_exact.items():
            if record.get(key) != expected:
                raise GuardError(f"remote visual successor exception record drift: {key}")
        remote_exact = {
            "schema": REMOTE_VISUAL_ONLY_PRELAUNCH_SCHEMA,
            "candidate_id": record.get("candidate_id"),
            "kernel_id": contract.get("kernel_id"),
            "execution_profile": profile,
            "model_execution": True,
            "candidate_code_path": _relative(code_path, project_root, "successor candidate code"),
            "candidate_code_sha256": code_sha256,
            "kernel_package_tree_sha256": _package_inventory(kernel_dir, project_root)["tree_sha256"],
            "fixed_preregistered_visual_identities": identities,
        }
        for key, expected in remote_exact.items():
            if remote.get(key) != expected:
                raise GuardError(f"remote visual successor prelaunch drift: {key}")

        metadata = load_json(kernel_dir / "kernel-metadata.json")
        carriers, carrier_artifacts, carrier_failures = _validate_private_carriers(
            contract.get("private_carriers"),
            expected_roles=expected_roles,
            metadata=metadata,
            source=code_path.read_text(encoding="utf-8"),
            project_root=project_root,
            linajea_runtime_v2=linajea,
            canonical_provider_roots=canonical_provider_roots,
        )
        if carrier_failures:
            raise GuardError("; ".join(carrier_failures))
        if remote.get("private_carriers") != carriers or record.get("private_carriers") != carriers:
            raise GuardError("remote visual successor private-carrier bindings drift")

        approval_key = "private_dataset_external_action_approval"
        if remote.get(approval_key) != record.get(approval_key):
            raise GuardError(
                "remote visual successor record/prelaunch external-action approval drift"
            )
        _, approval_artifacts = _validate_private_dataset_external_action_approval(
            remote.get(approval_key),
            carriers=carriers,
            linajea=linajea,
            project_root=project_root,
            required_roles={"p100_torch_runtime"} if p100 else None,
        )

        current_lmt_path, current_lmt_sha, lmt_policy = _lmt_policy(project_root)
        lmt_path, lmt_sha = _bound_reference(
            record.get("lmt"),
            project_root=project_root,
            label="remote visual successor current LMT",
        )
        if lmt_path != current_lmt_path or lmt_sha != current_lmt_sha:
            raise GuardError("remote visual successor is not bound to current LMT.md")

        for key in ("gpu_visual_only_contract", "ship_kaggle_certificate"):
            if remote.get(key) != record.get(key):
                raise GuardError(f"remote visual successor record/prelaunch drift: {key}")
        gpu_path, gpu_sha = _bound_reference(
            remote.get("gpu_visual_only_contract"),
            project_root=project_root,
            label="remote successor GPU visual-only contract",
            suffix=".json",
        )
        gpu = load_json(gpu_path)
        gpu_exact = {
            "schema": REMOTE_VISUAL_ONLY_CONTRACT_SCHEMA,
            "status": "FROZEN",
            "candidate_id": record.get("candidate_id"),
            "kernel_id": contract.get("kernel_id"),
            "execution_profile": profile,
            "model_execution": True,
            "device": "gpu",
            "fixed_preregistered_visual_identities": identities,
        }
        for key, expected in gpu_exact.items():
            if gpu.get(key) != expected:
                raise GuardError(f"remote successor GPU visual contract drift: {key}")
        prohibitions = gpu.get("prohibitions")
        forbidden = (
            "labels", "targets", "scorer", "full38", "training", "tuning",
            "calibration", "hpo", "submission",
        )
        if not isinstance(prohibitions, dict) or any(prohibitions.get(key) is not False for key in forbidden):
            raise GuardError("remote successor GPU contract does not prohibit score/training capability")
        remote_memory = gpu.get("remote_memory_limit_bytes")
        if not isinstance(remote_memory, int) or isinstance(remote_memory, bool) or remote_memory <= 0:
            raise GuardError("remote successor GPU contract has no physical memory capacity")

        if linajea:
            review = _linajea_remote_review_lock(
                gpu.get("post_remote_review"),
                identities=identities,
                require_full_decode=(
                    contract.get("kernel_id")
                    in {
                        LINAJEA_LEGACY_GPU_MOUNT_REPAIR_KERNEL_ID,
                        *LINAJEA_CANONICAL_PROVIDER_KERNEL_IDS,
                    }
                ),
            )
        else:
            review = _remote_review_lock(
                gpu.get("post_remote_review"),
                identities=identities,
                label="CELLECT successor post-review lock",
                require_full_decode=p100,
            )
            if contract.get("kernel_id") == CELLECT_P100_TYPED_REAP_KERNEL_ID:
                expected_mp4_artifacts = [
                    {
                        "identity": "train:44b6_d754aa59",
                        "stage": "extended_sparse",
                        "relative_path": "extended_sparse.mp4",
                        "full_decode_receipt": "extended_sparse.decode.json",
                    },
                    {
                        "identity": "train:44b6_18ced818",
                        "stage": "extended_crowded",
                        "relative_path": "extended_crowded.mp4",
                        "full_decode_receipt": "extended_crowded.decode.json",
                    },
                ]
                if (
                    gpu.get("post_remote_review", {}).get(
                        "required_remote_mp4_artifacts"
                    )
                    != expected_mp4_artifacts
                ):
                    raise GuardError("CELLECT V6 exact two-MP4 artifact lock drift")
                review["required_remote_mp4_artifacts"] = expected_mp4_artifacts
        if remote.get("post_remote_review") != gpu.get("post_remote_review"):
            raise GuardError("remote successor prelaunch post-review lock drift")
        if record.get("post_remote_review") != gpu.get("post_remote_review"):
            raise GuardError("remote successor exception post-review lock drift")

        certificate_artifacts, certificate_failures = _validate_enhanced_ship_certificate(
            remote.get("ship_kaggle_certificate"),
            contract=contract,
            kernel_dir=kernel_dir,
            code_sha256=code_sha256,
            carriers=carriers,
            project_root=project_root,
            expected_memory_limit_bytes=remote_memory,
        )
        if certificate_failures:
            raise GuardError("; ".join(certificate_failures))

        interpreter_artifacts: list[dict[str, Any]] = []
        if canonical_provider_roots:
            interpreter_key = "certifier_interpreter_receipt"
            if remote.get(interpreter_key) != record.get(interpreter_key):
                raise GuardError(
                    "fresh successor record/prelaunch certifier interpreter receipt drift"
                )
            if contract.get("kernel_id") == CELLECT_P100_TYPED_REAP_KERNEL_ID:
                interrupted_key = "pre_b3_interrupted_certification"
                if remote.get(interrupted_key) != record.get(interrupted_key):
                    raise GuardError(
                        "CELLECT V6 record/prelaunch interrupted receipt drift"
                    )
                interpreter_artifacts = _validate_cellect_v6_cp312_certifier_receipt(
                    remote.get(interpreter_key),
                    certificate_reference=remote.get("ship_kaggle_certificate"),
                    interrupted_reference=remote.get(interrupted_key),
                    kernel_dir=kernel_dir,
                    code_sha256=code_sha256,
                    project_root=project_root,
                )
            else:
                interpreter_artifacts = _validate_cp312_certifier_interpreter_receipt(
                    remote.get(interpreter_key),
                    certificate_reference=remote.get("ship_kaggle_certificate"),
                    project_root=project_root,
                )

        source_audit_artifacts: list[dict[str, Any]] = []
        if contract.get("kernel_id") in {
            LINAJEA_HOST_DRIVER_KERNEL_ID,
            LINAJEA_DEVICE_ROUTING_KERNEL_ID,
        }:
            audit_key = "independent_source_audit"
            if remote.get(audit_key) != record.get(audit_key):
                raise GuardError(
                    "Linajea successor record/prelaunch independent source-audit drift"
                )
            audit_path, audit_sha = _bound_reference(
                remote.get(audit_key),
                project_root=project_root,
                label="Linajea independent full-source audit",
                suffix=".json",
            )
            audit = load_json(audit_path)
            certificate_path, _ = _bound_reference(
                remote.get("ship_kaggle_certificate"),
                project_root=project_root,
                label="Linajea enhanced certificate",
                suffix=".json",
            )
            certificate = load_json(certificate_path)
            frozen = audit.get("frozen_inputs")
            expected_frozen = {
                "kernel_sha256": code_sha256,
                "metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
                "package_inventory_sha256": _ship_package_inventory_sha256(kernel_dir),
                "profile_sha256": certificate.get("profile_sha256"),
            }
            if contract.get("kernel_id") == LINAJEA_DEVICE_ROUTING_KERNEL_ID:
                frozen = audit.get("frozen_hash_verification")
                expected_frozen = {
                    "kernel.py": code_sha256,
                    "kernel_metadata.json": sha256(kernel_dir / "kernel-metadata.json"),
                    "model_kernel_package_inventory": _ship_package_inventory_sha256(kernel_dir),
                    "preflight_profile.json": certificate.get("profile_sha256"),
                }
                if (
                    audit.get("status") != "PASS_NO_REMAINING_RELEASE_PATH_BLOCKER"
                    or audit.get("disposition")
                    != "PASS_NO_REMAINING_RELEASE_PATH_BLOCKER"
                ):
                    raise GuardError("Linajea V9 independent source audit did not pass")
                if not isinstance(frozen, dict) or any(
                    frozen.get(key) != expected
                    for key, expected in expected_frozen.items()
                ):
                    raise GuardError("Linajea V9 independent source-audit frozen input drift")
                properties = audit.get("required_property_results")
                required_properties = {
                    "all_4581_assigned_nodedefs_exact_kernel_match_per_side",
                    "allow_soft_placement_false",
                    "checkpoint_and_science_identity_unchanged",
                    "checkpoint_graph_variable_map_equal",
                    "critical_163_explicit_gpu0_per_side",
                    "durable_diagnostics_and_monitor_classification",
                    "effective_embedded_runtime_exact",
                    "exact_18_cpu_routed_operations_per_side",
                    "exact_10_dt_string_const_cpu0_per_side",
                    "exact_78_nchw_biasadd_gpu0_per_side",
                    "nondevice_old_v9_graph_equal",
                    "package_exact_two_regular_files_no_symlink_or_pycache",
                    "provider_linker_import_paths",
                    "restore_v2_string_inputs_cpu0",
                    "six_atomic_mp4_full_decode_contract",
                    "workload_distinctness",
                }
                if not isinstance(properties, dict) or any(
                    properties.get(key) is not True for key in required_properties
                ):
                    raise GuardError("Linajea V9 independent source-audit property drift")
                execution = audit.get("independent_execution_evidence")
                replay = (
                    execution.get("auditor_authored_registry_replay")
                    if isinstance(execution, dict)
                    else None
                )
                if (
                    not isinstance(replay, dict)
                    or any(
                        replay.get(side, {}).get("assigned_operations") != 4581
                        or replay.get(side, {}).get("cpu_operations") != 18
                        or replay.get(side, {}).get("gpu_operations") != 4563
                        or replay.get(side, {}).get("exact_kernel_missing") != []
                        or replay.get(side, {}).get("critical_not_gpu0") != []
                        for side in ("side_1", "side_2")
                    )
                ):
                    raise GuardError("Linajea V9 independent device registry replay drift")
                interpreter_path, _ = _bound_reference(
                    remote.get("certifier_interpreter_receipt"),
                    project_root=project_root,
                    label="Linajea V9 CPython 3.12 certifier receipt",
                    suffix=".json",
                )
                interpreter_execution = load_json(interpreter_path).get("certification")
                certifier_relative = (
                    ".agents/skills/ship-kaggle-kernels/scripts/"
                    "kaggle_kernel_certifier.py"
                )
                if (
                    not isinstance(interpreter_execution, dict)
                    or interpreter_execution.get("certifier_path")
                    != certifier_relative
                    or interpreter_execution.get("certifier_sha256")
                    != ARCHIVED_SHIP_CERTIFIER_SHA256
                ):
                    raise GuardError("Linajea V9 archived certifier binding drift")
            else:
                if audit.get("status") != "PASS_NO_REMAINING_RELEASE_PATH_BLOCKER":
                    raise GuardError("Linajea V8 independent source audit did not pass")
                if not isinstance(frozen, dict) or any(
                    frozen.get(key) != expected for key, expected in expected_frozen.items()
                ):
                    raise GuardError("Linajea V8 independent source-audit frozen input drift")
                authority = audit.get("certification_authority")
                if (
                    not isinstance(authority, dict)
                    or authority.get("certifier_sha256")
                    != ARCHIVED_SHIP_CERTIFIER_SHA256
                ):
                    raise GuardError("Linajea V8 independent source-audit certifier drift")
                properties = audit.get("reviewed_release_properties")
                required_properties = {
                    "authenticated_runtime_and_dependency_materialization",
                    "canonical_provider_roots_and_source_mapping",
                    "exact_proved_environment_reused_by_all_model_children",
                    "libcuda_loaded_mapping_provenance_all_confined_one_identity",
                    "production_host_driver_roots_declared_in_profile",
                    "python37_carrier_first_linker_and_target_like_driver_probe",
                    "six_promised_mp4_full_decode_hash_frame_eof_decoder_receipts",
                    "strict_script_only_package_materialization",
                }
                if not isinstance(properties, dict) or any(
                    properties.get(key) != "PASS" for key in required_properties
                ):
                    raise GuardError("Linajea V8 independent source-audit property drift")
            source_audit_artifacts.append(
                {
                    "role": "independent_full_source_audit",
                    **_reference_dict(
                        audit_path,
                        audit_sha,
                        project_root,
                        "Linajea independent full-source audit",
                    ),
                }
            )
        elif contract.get("kernel_id") == CELLECT_P100_TYPED_REAP_KERNEL_ID:
            audit_key = "independent_source_audit"
            if remote.get(audit_key) != record.get(audit_key):
                raise GuardError("CELLECT V6 record/prelaunch B3 audit drift")
            audit_path, audit_sha = _bound_reference(
                remote.get(audit_key),
                project_root=project_root,
                label="CELLECT V6 independent B3 full-source reaudit",
                suffix=".json",
            )
            audit = load_json(audit_path)
            certificate_path, _ = _bound_reference(
                remote.get("ship_kaggle_certificate"),
                project_root=project_root,
                label="CELLECT V6 enhanced certificate",
                suffix=".json",
            )
            certificate = load_json(certificate_path)
            frozen = audit.get("frozen_hash_verification")
            expected_frozen = {
                "kernel.py": code_sha256,
                "kernel_metadata.json": sha256(kernel_dir / "kernel-metadata.json"),
                "package_inventory": _ship_package_inventory_sha256(kernel_dir),
                "preflight_profile.json": certificate.get("profile_sha256"),
                "limits_sha256": sha256(project_root / "LMT.md"),
                "ship_certifier_sha256": ARCHIVED_SHIP_CERTIFIER_SHA256,
            }
            if (
                audit.get("status") != "PASS_NO_REMAINING_RELEASE_PATH_BLOCKER"
                or audit.get("disposition")
                != "PASS_NO_REMAINING_RELEASE_PATH_BLOCKER"
            ):
                raise GuardError("CELLECT V6 independent B3 audit did not pass")
            if not isinstance(frozen, dict) or any(
                frozen.get(key) != expected for key, expected in expected_frozen.items()
            ):
                raise GuardError("CELLECT V6 independent B3 frozen input drift")
            properties = audit.get("required_property_results")
            required_properties = {
                "absolute_parent_and_resolved_outside_paths_reject",
                "captured_hash_bytes_and_65536_bound_exact",
                "exact_stage_specific_diagnostic_paths",
                "exact_ten_key_typed_native_child_exited_semantics",
                "final_and_intermediate_symlinks_reject",
                "finite_nonnegative_reap_timing_and_zero_release_delay",
                "full_equals_captured_hash_when_untruncated",
                "kernel_and_profile_unchanged",
                "package_exact_two_regular_files_no_symlink_or_pycache",
                "prior_terminal_resource_prediction_media_bindings_preserved",
                "truncation_semantics_exact",
            }
            if not isinstance(properties, dict) or any(
                properties.get(key) is not True for key in required_properties
            ):
                raise GuardError("CELLECT V6 independent B3 property drift")
            replay = audit.get("independent_adversarial_replay")
            cases = replay.get("cases") if isinstance(replay, dict) else None
            if (
                not isinstance(replay, dict)
                or replay.get("base_valid_fixture") != "PASS_3_STAGES_2_MEDIA"
                or replay.get(
                    "resource_stage_sha_rebound_for_b3_semantic_mutations"
                )
                is not True
                or not isinstance(cases, dict)
                or len(cases) < 20
                or any(
                    not isinstance(value, str) or not value.startswith("REJECTED:")
                    for value in cases.values()
                )
            ):
                raise GuardError("CELLECT V6 independent B3 adversarial replay drift")
            interrupted = audit.get("interrupted_attempt")
            if interrupted != {
                "certifier_receipt_published": False,
                "external_action": False,
                "partial_result_accepted": False,
                "status": "INTERRUPTED_INELIGIBLE_PRESERVED_NO_PROMOTION",
            }:
                raise GuardError("CELLECT V6 audit did not preserve pre-B3 ineligibility")
            source_audit_artifacts.append(
                {
                    "role": "independent_b3_full_source_reaudit",
                    **_reference_dict(
                        audit_path,
                        audit_sha,
                        project_root,
                        "CELLECT V6 independent B3 full-source reaudit",
                    ),
                }
            )
        elif contract.get("kernel_id") == CELLECT_P100_REAP_KERNEL_ID:
            audit_key = "independent_source_audit"
            diff_key = "independent_profile_diff_audit"
            for key in (audit_key, diff_key):
                if remote.get(key) != record.get(key):
                    raise GuardError(
                        f"CELLECT V5 record/prelaunch independent audit drift: {key}"
                    )
            audit_path, audit_sha = _bound_reference(
                remote.get(audit_key),
                project_root=project_root,
                label="CELLECT V5 independent full-source audit",
                suffix=".json",
            )
            audit = load_json(audit_path)
            frozen = audit.get("frozen_inputs")
            expected_frozen = {
                "kernel_sha256": code_sha256,
                "metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
                "package_inventory_sha256": _ship_package_inventory_sha256(kernel_dir),
            }
            if audit.get("status") != "PASS_NO_REMAINING_RELEASE_PATH_BLOCKER":
                raise GuardError("CELLECT V5 independent source audit did not pass")
            if not isinstance(frozen, dict) or any(
                frozen.get(key) != expected for key, expected in expected_frozen.items()
            ):
                raise GuardError("CELLECT V5 independent source-audit frozen input drift")
            authority = audit.get("certification_authority")
            if (
                not isinstance(authority, dict)
                or authority.get("certifier_sha256")
                != ARCHIVED_SHIP_CERTIFIER_SHA256
            ):
                raise GuardError("CELLECT V5 independent source-audit certifier drift")
            properties = audit.get("reviewed_release_properties")
            required_properties = {
                "dependencies_and_provider_roots",
                "durable_subprocess_and_native_output_failure_diagnostics",
                "environment_and_cuda_architecture_binding",
                "exact_script_only_package_materialization",
                "media_atomic_decode_before_publish_and_full_decode_receipts",
                "native_child_exit_reap_and_resource_classification",
                "observation_distinct_executed_scale_proxy",
                "runtime_safety_profile_probe_linkage",
                "strict_native_output_acceptance_including_raw_unicode_or_os_errors",
            }
            if not isinstance(properties, dict) or any(
                properties.get(key) != "PASS" for key in required_properties
            ):
                raise GuardError("CELLECT V5 independent source-audit property drift")

            diff_path, diff_sha = _bound_reference(
                remote.get(diff_key),
                project_root=project_root,
                label="CELLECT V5 independent profile-only diff audit",
                suffix=".json",
            )
            diff = load_json(diff_path)
            diff_frozen = diff.get("frozen_inputs")
            certificate_path, _ = _bound_reference(
                remote.get("ship_kaggle_certificate"),
                project_root=project_root,
                label="CELLECT V5 enhanced certificate",
                suffix=".json",
            )
            certificate = load_json(certificate_path)
            expected_diff_frozen = {
                "kernel_sha256": code_sha256,
                "metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
                "package_inventory_sha256": _ship_package_inventory_sha256(kernel_dir),
                "profile_sha256": certificate.get("profile_sha256"),
            }
            if diff.get("verdict") != "PASS_NO_REMAINING_PROFILE_DIFF_BLOCKER":
                raise GuardError("CELLECT V5 independent profile diff did not pass")
            if not isinstance(diff_frozen, dict) or any(
                diff_frozen.get(key) != expected
                for key, expected in expected_diff_frozen.items()
            ):
                raise GuardError("CELLECT V5 independent profile-diff frozen input drift")
            diff_proof = diff.get("diff_proof")
            added = diff_proof.get("fields_added_to_each") if isinstance(diff_proof, dict) else None
            if (
                not isinstance(diff_proof, dict)
                or diff_proof.get("reconstruction_matches_previous_frozen_profile") is not True
                or diff_proof.get("changed_output_paths")
                != [
                    "subprocess_diagnostics/native-launch-probe.stdout.tail",
                    "subprocess_diagnostics/native-launch-probe.stderr.tail",
                ]
                or added
                != {
                    "max_bytes": 0,
                    "min_bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                }
            ):
                raise GuardError("CELLECT V5 independent profile-diff scope drift")
            validation = diff.get("validation")
            if (
                not isinstance(validation, dict)
                or validation.get("certifier_sha256")
                != ARCHIVED_SHIP_CERTIFIER_SHA256
                or validation.get("certifier_exact_zero_file_validation") != "PASS"
                or validation.get("repair_is_path_specific") is not True
                or validation.get("all_other_output_contracts_byte_identical_to_previous_profile")
                is not True
            ):
                raise GuardError("CELLECT V5 independent profile-diff validation drift")
            source_audit_artifacts.extend(
                [
                    {
                        "role": "independent_full_source_audit",
                        **_reference_dict(
                            audit_path,
                            audit_sha,
                            project_root,
                            "CELLECT V5 independent full-source audit",
                        ),
                    },
                    {
                        "role": "independent_profile_diff_audit",
                        **_reference_dict(
                            diff_path,
                            diff_sha,
                            project_root,
                            "CELLECT V5 independent profile-only diff audit",
                        ),
                    },
                ]
            )

        extra_artifacts: list[dict[str, Any]] = []
        if linajea:
            for key in ("cpu_layout_failure_receipt", "runtime_transport_manifest"):
                if remote.get(key) != record.get(key):
                    raise GuardError(f"Linajea record/prelaunch reference drift: {key}")
            cpu_path, cpu_sha = _bound_reference(
                remote.get("cpu_layout_failure_receipt"),
                project_root=project_root,
                label="Linajea CPU layout failure receipt",
                suffix=".json",
            )
            cpu = load_json(cpu_path)
            cpu_exact = {
                "schema": "celltrack-alpha.linajea-zf-cpu-v2-terminal.v1",
                "status": "PARK_CPU_LAYOUT_INCOMPATIBLE",
                "builder_exit_code": 1,
            }
            for key, expected in cpu_exact.items():
                if cpu.get(key) != expected:
                    raise GuardError(f"Linajea CPU layout failure receipt drift: {key}")
            signature = str(cpu.get("terminal_signature", ""))
            if "Conv3DBackpropInputOpV2 only supports NDHWC on the CPU" not in signature:
                raise GuardError("Linajea CPU layout failure signature drift")
            interpretation = cpu.get("interpretation")
            if not isinstance(interpretation, str) or "not model-quality evidence" not in interpretation:
                raise GuardError("Linajea CPU failure is not explicitly non-scientific")

            transport_path, transport_sha = _bound_reference(
                remote.get("runtime_transport_manifest"),
                project_root=project_root,
                label="Linajea opaque runtime transport manifest",
                suffix=".json",
            )
            transport = load_json(transport_path)
            runtime_carrier = carriers[0]
            payload = transport.get("payload")
            if (
                transport.get("schema") != "celltrack-alpha.linajea-zf-tf1-gpu-runtime-dataset.v2"
                or transport.get("contains_competition_raw") is not False
                or transport.get("transport") != "opaque_gzip_tar_payload_r3"
                or transport.get("transport_revision") != "R3"
                or transport.get("dataset_files_contract")
                != ["CARRIER_DATASET_MANIFEST.json", "linajea_zf_tf115_gpu_v3_runtime.payload"]
                or not isinstance(payload, dict)
                or payload.get("path") != "linajea_zf_tf115_gpu_v3_runtime.payload"
                or payload.get("encoding") != "gzip+tar"
                or payload.get("archive_name_autodetection_prevented") is not True
                or _sha_field(payload.get("sha256"), "Linajea opaque payload hash")
                not in code_path.read_text(encoding="utf-8")
                or runtime_carrier.get("manifest_sha256") != transport_sha
            ):
                raise GuardError("Linajea opaque Dataset Version 2 transport drift")
            extra_artifacts.extend(
                [
                    {"role": "cpu_layout_failure_receipt", **_reference_dict(cpu_path, cpu_sha, project_root, "Linajea CPU layout failure receipt")},
                    {"role": "opaque_runtime_transport_manifest", **_reference_dict(transport_path, transport_sha, project_root, "Linajea opaque runtime transport manifest")},
                ]
            )
        else:
            if remote.get("resource_stop_receipt") != record.get("resource_stop_receipt"):
                raise GuardError("CELLECT successor record/prelaunch resource-stop drift")
            resource_path, resource_sha = _bound_reference(
                remote.get("resource_stop_receipt"),
                project_root=project_root,
                label="CELLECT local resource-stop receipt",
                suffix=".json",
            )
            resource_receipt = load_json(resource_path)
            for key, expected in {
                "schema": "celltrack-alpha.local-model-execution-abort.v1",
                "status": "ABORT",
                "failure_type": "ResourceStop",
                "pass_receipt_absent": True,
                "owned_process_group_terminated": True,
            }.items():
                if resource_receipt.get(key) != expected:
                    raise GuardError(f"CELLECT local resource-stop receipt drift: {key}")
            resource = resource_receipt.get("resource")
            peak = resource.get("peak_combined_rss_mib") if isinstance(resource, dict) else None
            ceiling = lmt_policy["local_resource_safety"]["maximum_combined_rss_mib"]
            if not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak < ceiling:
                raise GuardError("CELLECT local resource stop does not prove the current LMT ceiling")
            extra_artifacts.append(
                {
                    "role": "local_resource_stop_receipt",
                    **_reference_dict(
                        resource_path,
                        resource_sha,
                        project_root,
                        "CELLECT local resource-stop receipt",
                    ),
                }
            )

        summary.update(
            {
                "status": REMOTE_VISUAL_PENDING_STATUS,
                "candidate_id": record.get("candidate_id"),
                "record_path": _relative(record_path, project_root, "successor exception record"),
                "record_sha256": record_sha,
                "package_tree_sha256": _package_inventory(kernel_dir, project_root)["tree_sha256"],
                "identities": identities,
                "video_count": 0,
                "post_remote_review": review,
                "artifacts": [
                    {"role": "exception_record", **_reference_dict(record_path, record_sha, project_root, "successor exception record")},
                    {"role": "gpu_visual_only_contract", **_reference_dict(gpu_path, gpu_sha, project_root, "successor GPU visual-only contract")},
                    {"role": "current_lmt", **_reference_dict(current_lmt_path, current_lmt_sha, project_root, "current LMT")},
                    *carrier_artifacts,
                    *approval_artifacts,
                    *certificate_artifacts,
                    *interpreter_artifacts,
                    *source_audit_artifacts,
                    *extra_artifacts,
                ],
            }
        )
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
    return summary, failures


def check_remote_visual_only_exception(
    *,
    contract: dict[str, Any],
    code_sha256: str,
    code_path: Path,
    kernel_dir: Path,
    project_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the only model-executing no-local-MP4 exception.

    The contract can nominate an exception record, but cannot authorize itself:
    the exact record path/hash/kernel/profile tuple must first be compiled into
    ``REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST`` by an independent root amendment.
    A successful result remains ``REMOTE_VISUAL_PENDING`` rather than a local
    visual PASS, so it cannot be reused as a scoring/promotion prerequisite.
    """

    if contract.get("execution_profile") in {
        CELLECT_TWO_CARRIER_EXECUTION_PROFILE,
        CELLECT_P100_EXECUTION_PROFILE,
        LINAJEA_LEGACY_GPU_EXECUTION_PROFILE,
    }:
        return check_successor_remote_visual_exception(
            contract=contract,
            code_sha256=code_sha256,
            code_path=code_path,
            kernel_dir=kernel_dir,
            project_root=project_root,
        )

    failures: list[str] = []
    nominated = contract.get("remote_visual_only_exception")
    exception_id = nominated.get("exception_id") if isinstance(nominated, dict) else None
    summary = _remote_exception_summary(
        exception_id=exception_id if isinstance(exception_id, str) else None
    )
    if not isinstance(nominated, dict):
        failures.append("remote visual-only exception reference is missing or malformed")
        return summary, failures
    if not isinstance(exception_id, str) or not exception_id:
        failures.append("remote visual-only exception_id is missing or malformed")
        return summary, failures
    central = REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST.get(exception_id)
    if not isinstance(central, dict):
        failures.append("remote visual-only exception is not centrally allowlisted")
        return summary, failures
    required_central = {
        "record_path": nominated.get("record_path"),
        "record_sha256": nominated.get("record_sha256"),
        "kernel_id": contract.get("kernel_id"),
        "execution_profile": contract.get("execution_profile"),
    }
    if central != required_central:
        failures.append("remote visual-only exception does not match the central exact allowlist")
        return summary, failures
    if contract.get("execution_profile") != REMOTE_VISUAL_ONLY_EXECUTION_PROFILE:
        failures.append("remote visual-only exception has an unsupported execution profile")
    expected_properties = REMOTE_VISUAL_ONLY_COMPETITION_ROUTE[
        "contract_kernel_properties"
    ]
    if contract.get("kernel_properties") != expected_properties:
        failures.append("remote visual-only exception does not declare the exact offline GPU profile")
    try:
        record_path = _project_file(
            nominated.get("record_path"),
            project_root=project_root,
            label="remote visual-only exception record",
            suffix=".json",
        )
        record_sha = _sha_field(
            nominated.get("record_sha256"), "remote visual-only exception record hash"
        )
        if sha256(record_path) != record_sha:
            raise GuardError("remote visual-only exception record hash drift")
        record = load_json(record_path)
        summary["record_path"] = _relative(record_path, project_root, "remote visual-only exception record")
        summary["record_sha256"] = record_sha

        remote = contract.get("remote_visual_only")
        if not isinstance(remote, dict):
            raise GuardError("remote visual-only prelaunch contract is missing or malformed")
        fixed_identities = _identity_set(
            remote.get("fixed_preregistered_visual_identities"),
            "remote visual-only fixed identities",
        )
        if len(fixed_identities) != 2:
            raise GuardError("remote visual-only exception requires exactly two fixed identities")
        contract_identities = _identity_set(
            contract.get("preregistered_visual_identities"),
            "contract preregistered_visual_identities",
        )
        if contract_identities != fixed_identities:
            raise GuardError("remote visual-only identities differ from contract preregistration")
        record_exact = {
            "schema": REMOTE_VISUAL_ONLY_EXCEPTION_SCHEMA,
            "status": "APPROVED",
            "exception_id": exception_id,
            "candidate_id": remote.get("candidate_id"),
            "kernel_id": contract.get("kernel_id"),
            "execution_profile": REMOTE_VISUAL_ONLY_EXECUTION_PROFILE,
            "fixed_preregistered_visual_identities": fixed_identities,
        }
        for key, expected in record_exact.items():
            if record.get(key) != expected:
                raise GuardError(f"remote visual-only exception record drift: {key}")
        remote_exact = {
            "schema": REMOTE_VISUAL_ONLY_PRELAUNCH_SCHEMA,
            "candidate_id": record.get("candidate_id"),
            "kernel_id": contract.get("kernel_id"),
            "execution_profile": REMOTE_VISUAL_ONLY_EXECUTION_PROFILE,
            "model_execution": True,
            "fixed_preregistered_visual_identities": fixed_identities,
        }
        for key, expected in remote_exact.items():
            if remote.get(key) != expected:
                raise GuardError(f"remote visual-only prelaunch contract drift: {key}")
        if remote.get("candidate_code_sha256") != code_sha256:
            raise GuardError("remote visual-only prelaunch candidate code hash drift")
        if remote.get("candidate_code_path") != _relative(code_path, project_root, "candidate code"):
            raise GuardError("remote visual-only prelaunch candidate code path drift")
        if remote.get("kernel_package_tree_sha256") != _package_inventory(kernel_dir, project_root)["tree_sha256"]:
            raise GuardError("remote visual-only prelaunch kernel package hash drift")

        current_lmt_path, current_lmt_sha, lmt_policy = _lmt_policy(project_root)
        record_lmt_path, record_lmt_sha = _bound_reference(
            record.get("lmt"), project_root=project_root, label="remote visual-only current LMT"
        )
        if record_lmt_path != current_lmt_path or record_lmt_sha != current_lmt_sha:
            raise GuardError("remote visual-only exception is not bound to the current LMT.md")

        reference_keys = (
            "resource_stop_receipt",
            "gpu_visual_only_contract",
            "ship_kaggle_certificate",
        )
        resolved: dict[str, tuple[Path, str]] = {}
        for key in reference_keys:
            if remote.get(key) != record.get(key):
                raise GuardError(f"remote visual-only record/prelaunch reference drift: {key}")
            resolved[key] = _bound_reference(
                remote.get(key), project_root=project_root, label=f"remote visual-only {key}", suffix=".json"
            )

        resource_path, resource_sha = resolved["resource_stop_receipt"]
        resource_receipt = load_json(resource_path)
        resource_exact = {
            "schema": "celltrack-alpha.local-model-execution-abort.v1",
            "status": "ABORT",
            "failure_type": "ResourceStop",
            "pass_receipt_absent": True,
            "owned_process_group_terminated": True,
        }
        for key, expected in resource_exact.items():
            if resource_receipt.get(key) != expected:
                raise GuardError(f"remote visual-only resource stop drift: {key}")
        resource = resource_receipt.get("resource")
        ceiling = lmt_policy["local_resource_safety"]["maximum_combined_rss_mib"]
        peak = resource.get("peak_combined_rss_mib") if isinstance(resource, dict) else None
        if not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak < ceiling:
            raise GuardError("remote visual-only resource stop does not prove the current LMT RSS ceiling")

        gpu_contract_path, gpu_contract_sha = resolved["gpu_visual_only_contract"]
        gpu_contract = load_json(gpu_contract_path)
        gpu_exact = {
            "schema": REMOTE_VISUAL_ONLY_CONTRACT_SCHEMA,
            "status": "FROZEN",
            "candidate_id": record.get("candidate_id"),
            "kernel_id": contract.get("kernel_id"),
            "execution_profile": REMOTE_VISUAL_ONLY_EXECUTION_PROFILE,
            "model_execution": True,
            "device": "gpu",
            "fixed_preregistered_visual_identities": fixed_identities,
        }
        for key, expected in gpu_exact.items():
            if gpu_contract.get(key) != expected:
                raise GuardError(f"remote GPU visual-only contract drift: {key}")
        prohibitions = gpu_contract.get("prohibitions")
        forbidden = (
            "labels",
            "targets",
            "scorer",
            "full38",
            "training",
            "tuning",
            "calibration",
            "hpo",
            "submission",
        )
        if not isinstance(prohibitions, dict) or any(prohibitions.get(key) is not False for key in forbidden):
            raise GuardError("remote GPU visual-only contract does not prohibit labels/score/training")
        remote_memory = gpu_contract.get("remote_memory_limit_bytes")
        if not isinstance(remote_memory, int) or isinstance(remote_memory, bool) or remote_memory <= 0:
            raise GuardError("remote GPU visual-only contract has no positive remote memory limit")
        remote_review = _remote_review_lock(
            gpu_contract.get("post_remote_review"),
            identities=fixed_identities,
            label="remote GPU visual-only post-review lock",
        )
        if remote.get("post_remote_review") != gpu_contract.get("post_remote_review"):
            raise GuardError("remote visual-only prelaunch post-review lock drift")
        if record.get("post_remote_review") != gpu_contract.get("post_remote_review"):
            raise GuardError("remote visual-only exception record post-review lock drift")

        certificate_path, certificate_sha = resolved["ship_kaggle_certificate"]
        certificate = load_json(certificate_path)
        certificate_exact = {
            "schema": "ship-kaggle-kernels.certification.v1",
            "status": "ELIGIBLE",
            "eligible": True,
            "autonomous_launch_eligible": True,
            "kernel_id": contract.get("kernel_id"),
            "limits_sha256": current_lmt_sha,
            "artifact_sha256": code_sha256,
            "metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
            "package_inventory_sha256": _ship_package_inventory_sha256(kernel_dir),
        }
        for key, expected in certificate_exact.items():
            if certificate.get(key) != expected:
                raise GuardError(f"remote visual-only ship-kaggle certificate drift: {key}")
        certificate_lmt = Path(str(certificate.get("limits_file", ""))).resolve()
        if certificate_lmt != current_lmt_path:
            raise GuardError("remote visual-only ship-kaggle certificate limits path drift")
        profile_path_text = certificate.get("profile_path")
        profile_sha = _sha_field(certificate.get("profile_sha256"), "remote visual-only ship-kaggle profile hash")
        if not isinstance(profile_path_text, str) or not Path(profile_path_text).is_absolute():
            raise GuardError("remote visual-only ship-kaggle profile path is not absolute")
        profile_path = Path(profile_path_text).resolve()
        try:
            profile_path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise GuardError("remote visual-only ship-kaggle profile path escapes the repository") from exc
        if profile_path.is_symlink() or not profile_path.is_file() or sha256(profile_path) != profile_sha:
            raise GuardError("remote visual-only ship-kaggle profile hash drift")
        projection = certificate.get("runtime_projection")
        if not isinstance(projection, dict) or projection.get("status") != "PASS":
            raise GuardError("remote visual-only ship-kaggle runtime projection is not PASS")
        if projection.get("physical_memory_limit_bytes") != remote_memory:
            raise GuardError("remote visual-only GPU memory limit is not bound by the ship-kaggle certificate")
        projected_peak = projection.get("projected_peak_rss_bytes")
        if not isinstance(projected_peak, (int, float)) or isinstance(projected_peak, bool) or projected_peak >= remote_memory:
            raise GuardError("remote visual-only ship-kaggle projection does not fit the GPU memory limit")

        summary.update(
            {
                "status": REMOTE_VISUAL_PENDING_STATUS,
                "exception_id": exception_id,
                "candidate_id": record.get("candidate_id"),
                "package_tree_sha256": _package_inventory(kernel_dir, project_root)["tree_sha256"],
                "identities": fixed_identities,
                "video_count": 0,
                "post_remote_review": remote_review,
                "artifacts": [
                    {"role": "exception_record", **_reference_dict(record_path, record_sha, project_root, "remote visual-only exception record")},
                    {"role": "resource_stop_receipt", **_reference_dict(resource_path, resource_sha, project_root, "resource stop receipt")},
                    {"role": "gpu_visual_only_contract", **_reference_dict(gpu_contract_path, gpu_contract_sha, project_root, "GPU visual-only contract")},
                    {"role": "ship_kaggle_certificate", **_reference_dict(certificate_path, certificate_sha, project_root, "ship-kaggle certificate")},
                    {"role": "ship_kaggle_profile", **_reference_dict(profile_path, profile_sha, project_root, "ship-kaggle profile")},
                    {"role": "current_lmt", **_reference_dict(current_lmt_path, current_lmt_sha, project_root, "current LMT")},
                ],
            }
        )
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    return summary, failures


def _source_has_model_markers(source: str, tree: ast.AST) -> list[str]:
    markers = sorted(
        {
            root
            for node in ast.walk(tree)
            for root in imported_roots(node)
            if root in MODEL_EXECUTION_IMPORTS
        }
    )
    markers.extend(marker for marker in MODEL_EXECUTION_MARKERS if marker in source)
    return markers


def _literal_assignment(tree: ast.Module, name: str, label: str) -> Any:
    matches: list[ast.AST] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            matches.append(node.value)
    if len(matches) != 1:
        raise GuardError(f"{label} must have exactly one top-level assignment")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError, SyntaxError) as exc:
        raise GuardError(f"{label} must be a frozen literal") from exc


def _repository_directory(raw_path: Any, *, project_root: Path, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise GuardError(f"{label} must be a nonempty repository-relative path")
    root = project_root.resolve()
    unresolved = root / raw_path
    if unresolved.is_symlink():
        raise GuardError(f"{label} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GuardError(f"{label} escapes the repository root") from exc
    if path.is_symlink() or not path.is_dir():
        raise GuardError(f"{label} must be a regular directory")
    return path


def _container_path_matches_repository_file(
    raw_path: Any,
    *,
    expected_path: Path,
    project_root: Path,
    label: str,
) -> None:
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise GuardError(f"{label} must be an absolute path")
    expected = expected_path.resolve()
    root = project_root.resolve()
    relative = expected.relative_to(root)
    accepted = {
        str(expected),
        str(Path("/workspace") / relative),
        str(Path("/repo") / relative),
    }
    if raw_path not in accepted:
        raise GuardError(f"{label} path drift")


def _validate_f2_sweep_ship_certificate(
    *,
    contract: dict[str, Any],
    kernel_dir: Path,
    metadata: dict[str, Any],
    code_sha256: str,
    project_root: Path,
    expected: dict[str, str],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    certificate_path, certificate_sha = _bound_reference(
        contract.get("ship_kaggle_certificate"),
        project_root=project_root,
        label="F2 sweep ship-kaggle certificate",
        suffix=".json",
    )
    profile_path, profile_sha = _bound_reference(
        contract.get("ship_kaggle_profile"),
        project_root=project_root,
        label="F2 sweep ship-kaggle profile",
        suffix=".json",
    )
    certificate = load_json(certificate_path)
    profile = load_json(profile_path)
    current_lmt_path, current_lmt_sha, lmt_policy = _lmt_policy(project_root)
    package_inventory_sha = _ship_package_inventory_sha256(kernel_dir)
    exact = {
        "schema": "ship-kaggle-kernels.certification.v1",
        "status": "ELIGIBLE",
        "eligible": True,
        "autonomous_launch_eligible": True,
        "kernel_id": expected["kernel_id"],
        "limits_sha256": current_lmt_sha,
        "artifact_sha256": code_sha256,
        "metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
        "package_inventory_sha256": package_inventory_sha,
        "profile_sha256": profile_sha,
    }
    for key, value in exact.items():
        if certificate.get(key) != value:
            raise GuardError(f"F2 sweep ship-kaggle certificate drift: {key}")
    _container_path_matches_repository_file(
        certificate.get("limits_file"),
        expected_path=current_lmt_path,
        project_root=project_root,
        label="F2 sweep certificate LMT",
    )
    _container_path_matches_repository_file(
        certificate.get("profile_path"),
        expected_path=profile_path,
        project_root=project_root,
        label="F2 sweep certificate profile",
    )
    quick = certificate.get("quick_debug")
    if not isinstance(quick, dict) or {
        "auto_triggered": quick.get("auto_triggered"),
        "name": quick.get("name"),
        "status": quick.get("status"),
        "transport": quick.get("transport"),
    } != {
        "auto_triggered": True,
        "name": "focused_pre_push_release_debug",
        "status": "PASS",
        "transport": "kaggle-cli-save-kernel.v1",
    } or not isinstance(quick.get("purpose"), str) or not quick["purpose"].strip():
        raise GuardError("F2 sweep focused pre-push release debug is absent or drifted")
    for key in ("static", "repeatability"):
        value = certificate.get(key)
        if not isinstance(value, dict) or value.get("status") != "PASS":
            raise GuardError(f"F2 sweep ship-kaggle {key} did not pass")
    projection = certificate.get("runtime_projection")
    if not isinstance(projection, dict) or projection.get("status") != "PASS":
        raise GuardError("F2 sweep ship-kaggle runtime projection did not pass")
    projected_seconds = projection.get("projected_remote_upper_seconds")
    projected_peak = projection.get("projected_peak_rss_bytes")
    physical_memory = projection.get("physical_memory_limit_bytes")
    kaggle_policy = lmt_policy.get("kaggle")
    cap_hours = (
        kaggle_policy.get("autonomous_projection_strictly_below_hours")
        if isinstance(kaggle_policy, dict)
        else None
    )
    if (
        not isinstance(projected_seconds, (int, float))
        or isinstance(projected_seconds, bool)
        or projected_seconds <= 0
        or not isinstance(cap_hours, (int, float))
        or isinstance(cap_hours, bool)
        or projected_seconds >= float(cap_hours) * 3600.0
    ):
        raise GuardError("F2 sweep projected runtime is outside autonomous LMT scope")
    if (
        not isinstance(projected_peak, (int, float))
        or isinstance(projected_peak, bool)
        or not isinstance(physical_memory, (int, float))
        or isinstance(physical_memory, bool)
        or projected_peak <= 0
        or projected_peak >= physical_memory
    ):
        raise GuardError("F2 sweep projected memory does not fit physical capacity")

    if profile.get("schema") != "ship-kaggle-kernels.profile.v1":
        raise GuardError("F2 sweep ship-kaggle profile schema drift")
    profile_kernel = profile.get("kernel")
    if not isinstance(profile_kernel, dict):
        raise GuardError("F2 sweep ship-kaggle profile kernel section is absent")
    profile_directory = profile_kernel.get("directory")
    if not isinstance(profile_directory, str) or not profile_directory:
        raise GuardError("F2 sweep ship-kaggle profile directory is malformed")
    if (profile_path.parent / profile_directory).resolve() != kernel_dir.resolve():
        raise GuardError("F2 sweep ship-kaggle profile package path drift")
    profile_exact = {
        "entrypoint": expected["code_file"],
        "artifact_sha256": code_sha256,
        "metadata_file": "kernel-metadata.json",
        "metadata_sha256": sha256(kernel_dir / "kernel-metadata.json"),
        "package_inventory_sha256": package_inventory_sha,
        "expected_metadata": metadata,
        "self_contained": True,
        "upload_transport": "kaggle-cli-save-kernel.v1",
        "execution_control_env": ["KAGGLE_PREFLIGHT_MODE", "KAGGLE_PREFLIGHT_UNITS"],
    }
    for key, value in profile_exact.items():
        if profile_kernel.get(key) != value:
            raise GuardError(f"F2 sweep ship-kaggle profile drift: kernel.{key}")
    step0 = profile_kernel.get("step0")
    if not isinstance(step0, dict) or step0.get("assignment_marker") != "SHIP_STEP0 = _v32_ship_step0()" or step0.get("heavy_work_marker") != "V32_HEAVY_WORK_BEGIN":
        raise GuardError("F2 sweep ship-kaggle profile Step 0 contract drift")

    scenarios = profile.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 4:
        raise GuardError("F2 sweep ship-kaggle profile must contain four scenarios")
    if {item.get("role") for item in scenarios if isinstance(item, dict)} != {
        "tiny", "stratified", "stress", "expansive"
    }:
        raise GuardError("F2 sweep ship-kaggle scenario roles drift")
    first_candidate = candidates[0]["id"]
    final_graph_path = (
        f"{expected['output_root']}/{first_candidate}/V32_FINAL_GRAPH_ROWS.csv"
    )
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise GuardError(f"F2 sweep scenario {index} is malformed")
        env = scenario.get("env")
        if not isinstance(env, dict) or set(env) != {
            "KAGGLE_PREFLIGHT_MODE", "KAGGLE_PREFLIGHT_UNITS"
        }:
            raise GuardError(f"F2 sweep scenario {index} has arbitrary execution controls")
        outputs = scenario.get("required_outputs")
        if not isinstance(outputs, list):
            raise GuardError(f"F2 sweep scenario {index} outputs are missing")
        paths = [item.get("path") for item in outputs if isinstance(item, dict)]
        if final_graph_path not in paths:
            raise GuardError(f"F2 sweep scenario {index} lacks final graph rows")
        if any(
            isinstance(path, str) and Path(path).name.lower() == "submission.csv"
            for path in paths
        ):
            raise GuardError(f"F2 sweep scenario {index} declares a submission output")
        completion = scenario.get("completion")
        if not isinstance(completion, dict) or completion.get("path") != "F2_SWEEP_COMPLETION.json" or completion.get("status_pointer") != "/status" or completion.get("units_pointer") != "/completed_units" or completion.get("success_values") != ["PASS"]:
            raise GuardError(f"F2 sweep scenario {index} completion contract drift")
    return (
        {
            "status": "PASS",
            "certificate_path": _relative(certificate_path, project_root, "F2 sweep certificate"),
            "certificate_sha256": certificate_sha,
            "profile_path": _relative(profile_path, project_root, "F2 sweep profile"),
            "profile_sha256": profile_sha,
            "projected_remote_upper_seconds": projected_seconds,
            "projected_peak_rss_bytes": projected_peak,
        },
        [
            {"role": "ship_kaggle_certificate", **_reference_dict(certificate_path, certificate_sha, project_root, "F2 sweep certificate")},
            {"role": "ship_kaggle_profile", **_reference_dict(profile_path, profile_sha, project_root, "F2 sweep profile")},
        ],
    )


def _validate_f2_child_visual_evidence(
    *,
    evidence: Any,
    embedded_child_sha256: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(evidence, dict):
        raise GuardError("F2 sweep child visual evidence is missing or malformed")
    child_kernel_dir = _repository_directory(
        evidence.get("kernel_dir"),
        project_root=project_root,
        label="F2 sweep visual child package",
    )
    child_contract_path, child_contract_sha = _bound_reference(
        evidence.get("contract"),
        project_root=project_root,
        label="F2 sweep child visual contract",
        suffix=".json",
    )
    child_contract = load_json(child_contract_path)
    child_metadata_path = child_kernel_dir / "kernel-metadata.json"
    child_metadata = load_json(child_metadata_path)
    if child_metadata != {
        "id": F2_CHILD_VISUAL_KERNEL_ID,
        "title": "f2-v32-child-visual-v1",
        "code_file": "v32_child.py",
    }:
        raise GuardError("F2 sweep visual child metadata drift")
    child_code_path = child_kernel_dir / str(child_metadata.get("code_file", ""))
    if not child_code_path.is_file() or child_code_path.is_symlink():
        raise GuardError("F2 sweep visual child code is missing or symbolic")
    child_code_sha = sha256(child_code_path)
    if child_code_sha != embedded_child_sha256:
        raise GuardError("F2 sweep embedded child differs from visual-executed child bytes")
    exact_contract = {
        "schema": F2_CHILD_VISUAL_CONTRACT_SCHEMA,
        "execution_profile": F2_CHILD_VISUAL_EXECUTION_PROFILE,
        "kernel_id": F2_CHILD_VISUAL_KERNEL_ID,
        "model_execution": True,
        "code_sha256": embedded_child_sha256,
        "checkpoint_path": F2_CHILD_VISUAL_PRIMARY_CHECKPOINT,
        "checkpoint_sha256": F2_CHILD_VISUAL_PRIMARY_CHECKPOINT_SHA256,
        "preregistered_visual_identities": list(F2_CHILD_VISUAL_IDENTITIES),
    }
    for key, expected_value in exact_contract.items():
        if child_contract.get(key) != expected_value:
            raise GuardError(f"F2 sweep child visual contract drift: {key}")
    if "remote_visual_only_exception" in child_contract or child_contract.get("non_model_execution_allowlist_id") is not None:
        raise GuardError("F2 sweep child visual evidence cannot use an execution exception")
    identities = _identity_set(
        child_contract.get("preregistered_visual_identities"),
        "F2 sweep child visual identities",
    )
    if identities != list(F2_CHILD_VISUAL_IDENTITIES):
        raise GuardError("F2 sweep child visual identities drift")

    input_path, input_sha = _bound_project_asset(
        child_contract,
        path_key="input_manifest_path",
        sha_key="input_manifest_sha256",
        project_root=project_root,
        label="F2 child visual input manifest",
    )
    input_manifest = load_json(input_path)
    input_exact = {
        "schema": VISUAL_INPUT_MANIFEST_SCHEMA,
        "status": "PASS_PREPARED",
        "embedded_child_sha256": embedded_child_sha256,
        "preregistered_visual_identities": list(F2_CHILD_VISUAL_IDENTITIES),
        "frames_per_identity": F2_CHILD_VISUAL_FRAMES_PER_IDENTITY,
        "label_free": True,
        "labels_or_ground_truth_permitted": False,
        "scorer_permitted": False,
        "competition_test_namespace_permitted": False,
        "primary_checkpoint": {
            "path": F2_CHILD_VISUAL_PRIMARY_CHECKPOINT,
            "sha256": F2_CHILD_VISUAL_PRIMARY_CHECKPOINT_SHA256,
        },
        "secondary_checkpoint": {
            "path": F2_CHILD_VISUAL_SECONDARY_CHECKPOINT,
            "sha256": F2_CHILD_VISUAL_SECONDARY_CHECKPOINT_SHA256,
        },
    }
    for key, expected_value in input_exact.items():
        if input_manifest.get(key) != expected_value:
            raise GuardError(f"F2 child visual input manifest drift: {key}")
    records = input_manifest.get("files")
    if not isinstance(records, list) or input_manifest.get("inventory_sha256") != _canonical_sha(records):
        raise GuardError("F2 child visual input inventory digest drift")
    observed_paths: set[str] = set()
    observed_hashes: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise GuardError(f"F2 child visual input inventory record {index} is malformed")
        record_path = _project_file(
            record.get("path"),
            project_root=project_root,
            label=f"F2 child visual input inventory record {index}",
        )
        record_relative = _relative(
            record_path, project_root, f"F2 child visual input inventory record {index}"
        )
        if record_relative in observed_paths:
            raise GuardError("F2 child visual input inventory contains duplicate paths")
        observed_paths.add(record_relative)
        record_sha = _sha_field(
            record.get("sha256"), f"F2 child visual input inventory hash {index}"
        )
        if sha256(record_path) != record_sha or record.get("bytes") != record_path.stat().st_size:
            raise GuardError(f"F2 child visual input inventory record {index} drift")
        observed_hashes[record_relative] = record_sha
    exact_assets = {
        F2_CHILD_VISUAL_PRIMARY_CHECKPOINT: F2_CHILD_VISUAL_PRIMARY_CHECKPOINT_SHA256,
        F2_CHILD_VISUAL_SECONDARY_CHECKPOINT: F2_CHILD_VISUAL_SECONDARY_CHECKPOINT_SHA256,
        F2_CHILD_VISUAL_PRIMARY_MANIFEST: F2_CHILD_VISUAL_PRIMARY_MANIFEST_SHA256,
        F2_CHILD_VISUAL_SECONDARY_MANIFEST: F2_CHILD_VISUAL_SECONDARY_MANIFEST_SHA256,
    }
    if any(observed_hashes.get(path) != expected for path, expected in exact_assets.items()):
        raise GuardError("F2 child visual exact checkpoint/carrier binding drift")
    raw_prefix = F2_CHILD_VISUAL_RAW_ROOT + "/train/"
    raw_records = observed_paths - set(exact_assets)
    if not raw_records or any(not path.startswith(raw_prefix) for path in raw_records):
        raise GuardError("F2 child visual input inventory has an arbitrary asset surface")
    expected_dataset_roots = {
        raw_prefix + identity.split(":", 1)[1] + ".zarr"
        for identity in F2_CHILD_VISUAL_IDENTITIES
    }
    for path in raw_records:
        if not any(path == root + "/zarr.json" or path == root + "/0/zarr.json" or any(
            path.startswith(root + f"/0/c/{frame}/")
            for frame in range(F2_CHILD_VISUAL_FRAMES_PER_IDENTITY)
        ) for root in expected_dataset_roots):
            raise GuardError("F2 child visual input inventory escapes the six-frame prefixes")
    for dataset_root in expected_dataset_roots:
        required = {dataset_root + "/zarr.json", dataset_root + "/0/zarr.json"}
        if not required.issubset(raw_records) or any(
            not any(path.startswith(dataset_root + f"/0/c/{frame}/") for path in raw_records)
            for frame in range(F2_CHILD_VISUAL_FRAMES_PER_IDENTITY)
        ):
            raise GuardError("F2 child visual input inventory lacks a true six-frame prefix")

    environment_path, environment_sha = _bound_project_asset(
        child_contract,
        path_key="execution_environment_path",
        sha_key="execution_environment_sha256",
        project_root=project_root,
        label="F2 child visual execution environment",
    )
    environment = load_json(environment_path)
    environment_exact = {
        "schema": "celltrack-alpha.f2-child-local-environment.v1",
        "status": "PASS_PREPARED",
        "execution_profile": F2_CHILD_VISUAL_EXECUTION_PROFILE,
        "platform": "linux/amd64",
        "python_version_prefix": "3.12",
        "docker_image_id": F2_CHILD_VISUAL_DOCKER_IMAGE_ID,
        "cpu_only": True,
        "network": "none",
        "docker_memory_mib": F2_CHILD_VISUAL_DOCKER_MEMORY_MIB,
        "maximum_combined_rss_mib": F2_CHILD_VISUAL_MAXIMUM_COMBINED_RSS_MIB,
        "preflight_mode": "stratified_both",
        "frames_per_identity": F2_CHILD_VISUAL_FRAMES_PER_IDENTITY,
        "identities": list(F2_CHILD_VISUAL_IDENTITIES),
        "lmt_sha256": sha256(project_root / "LMT.md"),
    }
    for key, expected_value in environment_exact.items():
        if environment.get(key) != expected_value:
            raise GuardError(f"F2 child visual execution environment drift: {key}")
    ffmpeg = environment.get("ffmpeg")
    if not isinstance(ffmpeg, dict) or ffmpeg.get("sha256") != F2_CHILD_VISUAL_FFMPEG_SHA256:
        raise GuardError("F2 child visual deterministic ffmpeg binding drift")
    driver_path = _project_file(
        environment.get("container_driver_path"),
        project_root=project_root,
        label="F2 child visual container driver",
    )
    driver_sha = _sha_field(
        environment.get("container_driver_sha256"),
        "F2 child visual container driver hash",
    )
    if sha256(driver_path) != driver_sha:
        raise GuardError("F2 child visual container driver drift")
    visual_path, visual_sha = _bound_reference(
        child_contract.get("local_visual_preflight_receipt"),
        project_root=project_root,
        label="F2 sweep child local visual receipt",
        suffix=".json",
    )
    visual_receipt = load_json(visual_path)
    for key, expected_value in {
        "schema": VISUAL_RECEIPT_SCHEMA,
        "status": "PASS",
        "model_execution": True,
    }.items():
        if visual_receipt.get(key) != expected_value:
            raise GuardError(f"F2 child visual receipt drift: {key}")
    execution_path, execution_sha = _bound_reference(
        child_contract.get("local_execution_receipt"),
        project_root=project_root,
        label="F2 sweep child local execution receipt",
        suffix=".json",
    )
    execution = load_json(execution_path)
    execution_exact = {
        "schema": "celltrack-alpha.local-model-execution.v1",
        "status": "PASS",
        "execution_origin": "local",
        "exit_code": 0,
        "kernel_id": F2_CHILD_VISUAL_KERNEL_ID,
        "model_id": F2_CHILD_VISUAL_MODEL_ID,
        "local_run_id": F2_CHILD_VISUAL_LOCAL_RUN_ID,
        "candidate_code_sha256": embedded_child_sha256,
        "checkpoint_path": F2_CHILD_VISUAL_PRIMARY_CHECKPOINT,
        "checkpoint_sha256": F2_CHILD_VISUAL_PRIMARY_CHECKPOINT_SHA256,
        "input_manifest_path": _relative(input_path, project_root, "F2 child visual input manifest"),
        "input_manifest_sha256": input_sha,
        "execution_environment_path": _relative(environment_path, project_root, "F2 child visual execution environment"),
        "execution_environment_sha256": environment_sha,
        "preregistered_visual_identities": list(F2_CHILD_VISUAL_IDENTITIES),
        "runner_id": F2_CHILD_VISUAL_RUNNER_ID,
        "runner_path": TRUSTED_LOCAL_RUNNERS[F2_CHILD_VISUAL_RUNNER_ID]["path"],
        "runner_sha256": TRUSTED_LOCAL_RUNNERS[F2_CHILD_VISUAL_RUNNER_ID]["sha256"],
        "execution_profile": F2_CHILD_VISUAL_EXECUTION_PROFILE,
        "docker_image_id": F2_CHILD_VISUAL_DOCKER_IMAGE_ID,
        "cpu_preflight": True,
        "frames_per_identity": F2_CHILD_VISUAL_FRAMES_PER_IDENTITY,
        "completed_frames_by_identity": {
            identity: F2_CHILD_VISUAL_FRAMES_PER_IDENTITY
            for identity in F2_CHILD_VISUAL_IDENTITIES
        },
        "labels_or_ground_truth_accessed": False,
        "scorer_accessed": False,
        "competition_test_namespace_accessed": False,
        "submission_absent": True,
    }
    for key, expected_value in execution_exact.items():
        if execution.get(key) != expected_value:
            raise GuardError(f"F2 child visual execution receipt drift: {key}")
    runner_path = _project_file(
        execution.get("runner_path"), project_root=project_root, label="F2 trusted local runner"
    )
    if sha256(runner_path) != execution_exact["runner_sha256"]:
        raise GuardError("F2 child visual trusted runner byte drift")
    peak_rss = execution.get("peak_rss_bytes")
    if (
        not isinstance(peak_rss, (int, float))
        or isinstance(peak_rss, bool)
        or peak_rss <= 0
        or peak_rss > F2_CHILD_VISUAL_MAXIMUM_COMBINED_RSS_MIB * 1024 * 1024
    ):
        raise GuardError("F2 child visual execution peak RSS is absent or out of policy")
    argv = execution.get("invoked_argv")
    if (
        not isinstance(argv, list)
        or len(argv) != 6
        or argv[:5] != [
            TRUSTED_LOCAL_RUNNERS[F2_CHILD_VISUAL_RUNNER_ID]["path"],
            "--project-root",
            str(project_root.resolve()),
            "run",
            "--memory-headroom-receipt",
        ]
    ):
        raise GuardError("F2 child visual runner command surface drift")
    memory_path = _project_file(
        argv[5], project_root=project_root, label="F2 child visual memory-headroom receipt"
    )
    memory_sha = _sha_field(
        execution.get("memory_headroom_receipt_sha256"),
        "F2 child visual memory-headroom receipt hash",
    )
    if sha256(memory_path) != memory_sha:
        raise GuardError("F2 child visual memory-headroom receipt drift")
    memory_receipt = load_json(memory_path)
    for key, expected_value in {
        "schema": "celltrack-alpha.local-memory-headroom.v1",
        "status": "PASS",
        "maximum_combined_rss_mib": F2_CHILD_VISUAL_MAXIMUM_COMBINED_RSS_MIB,
        "authorized_runner_sha256": execution_exact["runner_sha256"],
        "active_conflicting_jobs": [],
    }.items():
        if memory_receipt.get(key) != expected_value:
            raise GuardError(f"F2 child visual memory-headroom receipt drift: {key}")
    available = memory_receipt.get("available_for_new_run_mib")
    if not isinstance(available, (int, float)) or isinstance(available, bool) or available < F2_CHILD_VISUAL_DOCKER_MEMORY_MIB:
        raise GuardError("F2 child visual memory-headroom receipt is insufficient")
    memory_issued = _parse_time(
        memory_receipt.get("issued_at"), "F2 child visual memory-headroom issued_at"
    )
    execution_started = _parse_time(
        execution.get("started_at"), "F2 child visual execution started_at"
    )
    if memory_issued > execution_started or (execution_started - memory_issued).total_seconds() > 900:
        raise GuardError("F2 child visual memory-headroom receipt was not fresh at start")
    summary, failures = check_local_visual_preflight(
        contract=child_contract,
        code_sha256=child_code_sha,
        code_path=child_code_path,
        kernel_dir=child_kernel_dir,
        source=child_code_path.read_text(encoding="utf-8"),
        project_root=project_root,
    )
    if failures or summary.get("status") != "PASS":
        raise GuardError(
            "F2 sweep child local visual evidence failed: "
            + "; ".join(failures or [str(summary.get("status"))])
        )
    return (
        {
            "status": "PASS",
            "child_source_sha256": child_code_sha,
            "child_package_tree_sha256": summary.get("package_tree_sha256"),
            "receipt_path": _relative(visual_path, project_root, "F2 sweep child visual receipt"),
            "receipt_sha256": visual_sha,
            "execution_receipt_path": _relative(execution_path, project_root, "F2 sweep child execution receipt"),
            "execution_receipt_sha256": execution_sha,
            "video_count": summary.get("video_count"),
            "identities": summary.get("identities"),
            "artifacts": summary.get("artifacts", []),
        },
        [
            {"role": "child_visual_contract", **_reference_dict(child_contract_path, child_contract_sha, project_root, "F2 sweep child visual contract")},
            {"role": "child_visual_receipt", **_reference_dict(visual_path, visual_sha, project_root, "F2 sweep child visual receipt")},
            {"role": "child_execution_receipt", **_reference_dict(execution_path, execution_sha, project_root, "F2 sweep child execution receipt")},
        ],
    )


def _f2_d_v6_evidence_file(
    root: Path, raw_path: Any, *, label: str
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise GuardError(f"{label} path is absent or malformed")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in raw_path:
        raise GuardError(f"{label} path escapes its evidence root")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise GuardError(f"{label} path escapes its evidence root") from exc
    if path.is_symlink() or not path.is_file():
        raise GuardError(f"{label} is missing or symbolic")
    return path


def _validate_f2_d_v6_download_zip(
    *,
    archive_path: Path,
    evidence_root: Path,
    required_members: set[str],
    project_root: Path,
    manifest_schema: str = "celltrack-alpha.f2-d-v6-native-visual-download.v1",
) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members: dict[str, zipfile.ZipInfo] = {}
            total_uncompressed = 0
            for info in archive.infolist():
                name = info.filename
                relative = PurePosixPath(name)
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in name
                    or not relative.parts
                    or name in members
                    or mode & 0o170000 == 0o120000
                    or info.flag_bits & 0x1
                ):
                    raise GuardError("F2 D-v6 downloaded ZIP member surface drift")
                members[name] = info
                total_uncompressed += info.file_size
            if total_uncompressed <= 0 or total_uncompressed > 1_000_000_000:
                raise GuardError("F2 D-v6 downloaded ZIP size is outside the visual-only bound")
            if "DOWNLOAD_MANIFEST.json" not in members:
                raise GuardError("F2 D-v6 downloaded ZIP manifest is absent")
            manifest = json.loads(archive.read("DOWNLOAD_MANIFEST.json"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema") != manifest_schema
            ):
                raise GuardError("F2 D-v6 downloaded ZIP manifest schema drift")
            records = manifest.get("files")
            if not isinstance(records, list):
                raise GuardError("F2 D-v6 downloaded ZIP manifest records are absent")
            by_path: dict[str, dict[str, Any]] = {}
            for record in records:
                if not isinstance(record, dict):
                    raise GuardError("F2 D-v6 downloaded ZIP record is malformed")
                name = record.get("path")
                if not isinstance(name, str) or name in by_path:
                    raise GuardError("F2 D-v6 downloaded ZIP record path drift")
                by_path[name] = record
            if set(by_path) != set(members) - {"DOWNLOAD_MANIFEST.json"}:
                raise GuardError("F2 D-v6 downloaded ZIP manifest/member set drift")
            if not required_members.issubset(members):
                raise GuardError("F2 D-v6 downloaded ZIP lacks required evidence members")
            for name, record in by_path.items():
                content = archive.read(name)
                info = members[name]
                extracted = _f2_d_v6_evidence_file(
                    evidence_root, name, label=f"F2 D-v6 extracted ZIP member {name}"
                )
                expected_sha = _sha_field(
                    record.get("sha256"), f"F2 D-v6 ZIP member hash {name}"
                )
                if (
                    record.get("bytes") != len(content)
                    or info.file_size != len(content)
                    or hashlib.sha256(content).hexdigest() != expected_sha
                    or extracted.stat().st_size != len(content)
                    or sha256(extracted) != expected_sha
                ):
                    raise GuardError(f"F2 D-v6 downloaded ZIP member drift: {name}")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise GuardError("F2 D-v6 downloaded ZIP is unreadable") from exc
    return {
        "status": "PASS",
        "path": _relative(archive_path, project_root, "F2 D-v6 downloaded ZIP"),
        "sha256": sha256(archive_path),
        "member_count": len(members),
        "total_uncompressed_bytes": total_uncompressed,
    }


def _require_f2_d_native_route_enabled(
    route_key: str, *, project_root: Path
) -> dict[str, Any]:
    """Return an enabled native route or fail on its exact frozen blocker."""

    route = F2_D_NATIVE_VISUAL_ROUTES.get(route_key)
    if route is None:
        raise GuardError("unsupported F2 D native visual route")
    if not route["production_enabled"]:
        version = str(route["version"])
        terminal_reference = route.get("terminal_rejected_certificate")
        if terminal_reference is not None:
            terminal_path = _project_file(
                terminal_reference["path"],
                project_root=project_root,
                label=f"F2 D-{version} terminal native certificate",
                suffix=".json",
            )
            terminal = load_json(terminal_path)
            if (
                sha256(terminal_path) != terminal_reference["sha256"]
                or terminal.get("status") != "REJECTED"
                or terminal.get("eligible") is not False
                or terminal.get("autonomous_launch_eligible") is not False
                or terminal.get("kernel_id") != route["bindings"]["kernel_id"]
            ):
                raise GuardError(
                    f"F2 D-{version} terminal REJECTED certificate binding drift"
                )
            raise GuardError(
                f"F2 D-{version} native certificate is terminal REJECTED; "
                "production route is disabled"
            )
        if route.get("block_reason") == "pending_exact_native_eligible_certificate":
            raise GuardError(
                f"F2 D-{version} production route is disabled pending a "
                "downloaded exact ELIGIBLE native certificate"
            )
        raise GuardError("F2 D native production route is disabled")
    return route


def check_f2_d_v6_native_production_contract(
    *,
    production_contract_reference: Any,
    downloaded_zip_reference: Any,
    kernel_dir: Path,
    code_path: Path,
    metadata_path: Path,
    embedded_child_sha256: str,
    project_root: Path,
    _route_key: str = "v6",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate an exact additive native-T4/downloaded-visual production route."""

    route = _require_f2_d_native_route_enabled(
        _route_key, project_root=project_root
    )
    bindings = route["bindings"]
    execution_profile = route["execution_profile"]
    production_schema = route["production_contract_schema"]
    runner_id = route["runner_id"]
    requirements_record = route["requirements"]
    sealer_record = route["sealer"]
    identities = tuple(bindings["identities"])
    frames_per_identity = int(bindings["frames_per_identity"])
    envelope = bindings["transport_execution_envelope"]

    production_path, production_sha = _bound_reference(
        production_contract_reference,
        project_root=project_root,
        label="F2 D-v6 native production contract",
        suffix=".json",
    )
    archive_path, archive_sha = _bound_reference(
        downloaded_zip_reference,
        project_root=project_root,
        label="F2 D-v6 downloaded visual ZIP",
        suffix=".zip",
    )
    production = load_json(production_path)
    runner = TRUSTED_REMOTE_VISUAL_RUNNERS[runner_id]
    exact_contract = {
        "schema": production_schema,
        "status": "PASS",
        "execution_profile": execution_profile,
        "kernel_id": bindings["kernel_id"],
        "model_execution": True,
        "exact_bindings": bindings,
        "trusted_runner": {
            "id": runner_id,
            "path": runner["path"],
            "sha256": runner["sha256"],
        },
        "science_and_ilp_bytes_unchanged": True,
        "final_graph_rows_required": True,
        "submission_output_permitted": False,
        "labels_ground_truth_scorer_test_training_tuning_permitted": False,
    }
    for key, expected in exact_contract.items():
        if production.get(key) != expected:
            raise GuardError(f"F2 D-v6 production contract drift: {key}")

    exact_files = {
        bindings["code_relative"]: bindings["code_sha256"],
        requirements_record["path"]: requirements_record["sha256"],
        sealer_record["path"]: sealer_record["sha256"],
        runner["path"]: runner["sha256"],
        bindings["profile_relative"]: bindings["profile_sha256"],
        **route["native_exact_files"],
        "LMT.md": bindings["lmt_sha256"],
    }
    for relative, expected_sha in exact_files.items():
        path = _project_file(
            relative, project_root=project_root, label=f"F2 D-v6 exact file {relative}"
        )
        if sha256(path) != expected_sha:
            raise GuardError(f"F2 D-v6 exact file hash drift: {relative}")
    if (
        code_path.resolve() != (project_root / bindings["code_relative"]).resolve()
        or sha256(code_path) != bindings["code_sha256"]
        or sha256(metadata_path) != bindings["metadata_sha256"]
        or _ship_package_inventory_sha256(kernel_dir)
        != bindings["package_inventory_sha256"]
        or sha256(kernel_dir / "F2_SWEEP_SOURCE_FREEZE.json")
        != bindings["source_freeze_sha256"]
        or embedded_child_sha256 != bindings["embedded_child_sha256"]
    ):
        raise GuardError("F2 D-v6 immutable package/science/ILP binding drift")
    requirements = load_json(project_root / requirements_record["path"])
    if (
        requirements.get("execution_profile") != execution_profile
        or requirements.get("production_contract_schema") != production_schema
        or requirements.get("trusted_runner_id") != runner_id
        or requirements.get("trusted_runner_path") != runner["path"]
        or requirements.get("trusted_runner_sha256") != runner["sha256"]
        or requirements.get("exact_bindings") != bindings
    ):
        raise GuardError("F2 D-v6 central guard requirements drift")
    if route["requirements_bind_sealer"] and (
        requirements.get("trusted_sealer_path") != sealer_record["path"]
        or requirements.get("trusted_sealer_sha256") != sealer_record["sha256"]
    ):
        raise GuardError("F2 D-v6 central guard sealer binding drift")

    certificate_path, certificate_sha = _bound_reference(
        production.get("native_ship_certificate"),
        project_root=project_root,
        label="F2 D-v6 native ship certificate",
        suffix=".json",
    )
    certificate = load_json(certificate_path)
    certificate_exact = {
        "schema": "ship-kaggle-kernels.certification.v1",
        "status": "ELIGIBLE",
        "eligible": True,
        "autonomous_launch_eligible": True,
        "kernel_id": bindings["kernel_id"],
        "profile_sha256": bindings["profile_sha256"],
        "limits_sha256": bindings["lmt_sha256"],
        "artifact_sha256": bindings["code_sha256"],
        "metadata_sha256": bindings["metadata_sha256"],
        "package_inventory_sha256": bindings["package_inventory_sha256"],
    }
    for key, expected in certificate_exact.items():
        if certificate.get(key) != expected:
            raise GuardError(f"F2 D-v6 native certificate drift: {key}")
    _container_path_matches_repository_file(
        certificate.get("limits_file"),
        expected_path=project_root / "LMT.md",
        project_root=project_root,
        label="F2 D-v6 native certificate limits",
    )
    _container_path_matches_repository_file(
        certificate.get("profile_path"),
        expected_path=project_root / bindings["profile_relative"],
        project_root=project_root,
        label="F2 D-v6 native certificate profile",
    )

    execution_path, execution_sha = _bound_reference(
        production.get("native_t4_visual_execution"),
        project_root=project_root,
        label="F2 D-v6 native T4 visual execution",
        suffix=".json",
    )
    execution = load_json(execution_path)
    execution_exact = {
        "schema": route["execution_schema"],
        "status": "PASS_PENDING_DOWNLOADED_REVIEW",
        "execution_profile": execution_profile,
        "model_execution": True,
        "bindings": bindings,
        "transport_execution_envelope": envelope,
        "application_cuda_visible_devices": "0",
        "exit_code": 0,
        "completed_frames_by_identity": {
            identity: frames_per_identity for identity in identities
        },
        "labels_or_ground_truth_accessed": False,
        "scorer_accessed": False,
        "competition_test_namespace_accessed": False,
        "submission_absent": True,
    }
    for key, expected in execution_exact.items():
        if execution.get(key) != expected:
            raise GuardError(f"F2 D-v6 native visual execution drift: {key}")
    host = execution.get("host")
    physical = host.get("physical_gpus") if isinstance(host, dict) else None
    if (
        not isinstance(physical, list)
        or len(physical) != 2
        or any("T4" not in str(item) or "7.5" not in str(item) for item in physical)
        or host.get("application_cuda_visible_devices") != "0"
    ):
        raise GuardError("F2 D-v6 native visual host is not T4x2/application GPU 0")
    evidence_root = execution_path.parent.resolve()
    if (
        execution.get("native_certificate_sha256") != certificate_sha
        or Path(str(execution.get("native_certificate_path", ""))).name
        != certificate_path.name
    ):
        raise GuardError("F2 D-v6 execution/native certificate binding drift")
    native_result_path = _f2_d_v6_evidence_file(
        evidence_root,
        execution.get("native_result_path"),
        label="F2 D-v6 native interactive result",
    )
    if sha256(native_result_path) != _sha_field(
        execution.get("native_result_sha256"), "F2 D-v6 native result hash"
    ):
        raise GuardError("F2 D-v6 native interactive result hash drift")
    native_result = load_json(native_result_path)
    native_result_bindings = {
        "certifier_sha256": bindings.get(
            "certifier_sha256",
            "49d8ea390e67c0aa8a084f119e143ce878197119de81c4e72202ffdbb83d955f",
        ),
        "kernel_artifact_sha256": bindings["code_sha256"],
        "kernel_id": bindings["kernel_id"],
        "kernel_metadata_sha256": bindings["metadata_sha256"],
        "kernel_package_inventory_sha256": bindings["package_inventory_sha256"],
        "lmt_sha256": bindings["lmt_sha256"],
        "profile_sha256": bindings["profile_sha256"],
        "python_version_prefix": "3.12",
    }
    native_certificate = native_result.get("certification_receipt")
    if (
        native_result.get("schema") != route["native_result_schema"]
        or native_result.get("status") != "PASS_EXACT_ELIGIBLE"
        or native_result.get("no_push_performed") is not True
        or native_result.get("exact_bindings") != native_result_bindings
        or native_result.get("runtime_bundle_sha256")
        != bindings["native_bundle_sha256"]
        or native_result.get("host") != host
        or native_result.get("transport_execution_envelope") != envelope
        or native_result.get("exit_code") != 0
        or native_result.get("binding_failures") != []
        or not isinstance(native_certificate, dict)
        or native_certificate.get("sha256") != certificate_sha
        or Path(str(native_certificate.get("path", ""))).name
        != certificate_path.name
    ):
        raise GuardError("F2 D-v6 native interactive result drift")
    graph_path = _f2_d_v6_evidence_file(
        evidence_root,
        execution.get("final_graph_rows_path"),
        label="F2 D-v6 final graph rows",
    )
    if sha256(graph_path) != _sha_field(
        execution.get("final_graph_rows_sha256"), "F2 D-v6 final graph rows hash"
    ):
        raise GuardError("F2 D-v6 final graph rows hash drift")
    for name, field in (
        ("V32_SHIP_PREFLIGHT_COMPLETION.json", "child_completion_sha256"),
        ("PUBLIC_TEMPORALUNET_STEP0.json", "child_step0_sha256"),
    ):
        path = _f2_d_v6_evidence_file(
            evidence_root, f"child_outputs/{name}", label=f"F2 D-v6 {name}"
        )
        if sha256(path) != _sha_field(execution.get(field), f"F2 D-v6 {field}"):
            raise GuardError(f"F2 D-v6 child receipt hash drift: {name}")

    video_rows = execution.get("videos")
    if not isinstance(video_rows, list) or len(video_rows) != 2:
        raise GuardError("F2 D-v6 native visual video coverage drift")
    video_by_identity: dict[str, dict[str, Any]] = {}
    required_zip_members = {
        execution_path.name,
        certificate_path.name,
        native_result_path.name,
        str(execution.get("final_graph_rows_path")),
        "child_outputs/V32_SHIP_PREFLIGHT_COMPLETION.json",
        "child_outputs/PUBLIC_TEMPORALUNET_STEP0.json",
    }
    for row in video_rows:
        if not isinstance(row, dict):
            raise GuardError("F2 D-v6 native visual video row is malformed")
        identity = row.get("identity")
        if identity not in identities or identity in video_by_identity:
            raise GuardError("F2 D-v6 native visual video identity drift")
        video = _f2_d_v6_evidence_file(
            evidence_root, row.get("video_path"), label=f"F2 D-v6 video {identity}"
        )
        video_sha = _sha_field(row.get("video_sha256"), f"F2 D-v6 video hash {identity}")
        if sha256(video) != video_sha:
            raise GuardError("F2 D-v6 native visual video hash drift")
        remote_decode = row.get("remote_full_decode")
        if (
            not isinstance(remote_decode, dict)
            or remote_decode.get("decoded_frames") != frames_per_identity
            or remote_decode.get("eof_reached") is not True
            or not str(remote_decode.get("decoder", "")).startswith("ffmpeg:")
        ):
            raise GuardError("F2 D-v6 remote full-decode drift")
        required_zip_members.add(str(row.get("video_path")))
        video_by_identity[str(identity)] = {
            "path": video,
            "sha256": video_sha,
            "remote_full_decode": remote_decode,
        }
    if set(video_by_identity) != set(identities):
        raise GuardError("F2 D-v6 native visual video identity set drift")

    decode_path, decode_sha = _bound_reference(
        production.get("downloaded_full_decode"),
        project_root=project_root,
        label="F2 D-v6 downloaded full-decode receipt",
        suffix=".json",
    )
    decode = load_json(decode_path)
    if (
        decode.get("schema") != route["download_decode_schema"]
        or decode.get("status") != "PASS"
        or decode.get("frames_per_identity") != frames_per_identity
        or decode.get("all_eof_reached") is not True
    ):
        raise GuardError("F2 D-v6 downloaded full-decode receipt drift")
    decoded_rows = decode.get("videos")
    if not isinstance(decoded_rows, list) or len(decoded_rows) != 2:
        raise GuardError("F2 D-v6 downloaded full-decode coverage drift")
    decoded_identities: set[str] = set()
    for row in decoded_rows:
        if not isinstance(row, dict):
            raise GuardError("F2 D-v6 downloaded decode row is malformed")
        identity = row.get("identity")
        if identity not in video_by_identity or identity in decoded_identities:
            raise GuardError("F2 D-v6 downloaded decode identity drift")
        decoded_identities.add(str(identity))
        local_decode = row.get("local_full_decode")
        downloaded = _project_file(
            row.get("downloaded_video_path"),
            project_root=project_root,
            label=f"F2 D-v6 downloaded MP4 {identity}",
            suffix=".mp4",
        )
        if (
            row.get("video_sha256") != video_by_identity[str(identity)]["sha256"]
            or sha256(downloaded) != video_by_identity[str(identity)]["sha256"]
            or row.get("remote_full_decode")
            != video_by_identity[str(identity)]["remote_full_decode"]
            or not isinstance(local_decode, dict)
            or local_decode.get("decoded_frames") != frames_per_identity
            or local_decode.get("eof_reached") is not True
            or not str(local_decode.get("decoder", "")).startswith("ffmpeg:")
        ):
            raise GuardError("F2 D-v6 downloaded local full-decode drift")
    if decoded_identities != set(identities):
        raise GuardError("F2 D-v6 downloaded decode identity set drift")

    review_path, review_sha = _bound_reference(
        production.get("human_visual_review"),
        project_root=project_root,
        label="F2 D-v6 human visual review",
        suffix=".json",
    )
    review = load_json(review_path)
    if (
        review.get("schema") != route["human_review_schema"]
        or review.get("status") != "PASS"
    ):
        raise GuardError("F2 D-v6 human visual review schema/status drift")
    reviews = review.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise GuardError("F2 D-v6 human visual review coverage drift")
    taxonomy = VISUAL_FAILURE_TAXONOMY
    reviewed: set[str] = set()
    for row in reviews:
        if not isinstance(row, dict):
            raise GuardError("F2 D-v6 human visual review row is malformed")
        identity = row.get("identity")
        failures = row.get("failure_taxonomy")
        if (
            identity not in video_by_identity
            or identity in reviewed
            or row.get("video_sha256") != video_by_identity[str(identity)]["sha256"]
            or row.get("visual_analysis_performed") is not True
            or row.get("metrics_only") is not False
            or row.get("verdict") != "PASS"
            or not isinstance(row.get("reviewer"), str)
            or not row["reviewer"].strip()
            or not isinstance(row.get("promotion_rationale"), str)
            or not row["promotion_rationale"].strip()
            or not isinstance(row.get("observations"), list)
            or not row["observations"]
            or not isinstance(failures, dict)
            or set(failures) != taxonomy
            or any(not isinstance(value, str) or not value.strip() for value in failures.values())
        ):
            raise GuardError("F2 D-v6 human visual review is not an exact hash-bound PASS")
        _parse_time(row.get("reviewed_at"), "F2 D-v6 human review reviewed_at")
        reviewed.add(str(identity))
    if reviewed != set(identities):
        raise GuardError("F2 D-v6 human visual review identity set drift")
    expected_video_hashes = {
        identity: video_by_identity[identity]["sha256"]
        for identity in identities
    }
    if (
        production.get("reviewed_video_sha256_by_identity") != expected_video_hashes
        or production.get("human_reviews_sha256") != _canonical_sha(reviews)
    ):
        raise GuardError("F2 D-v6 production human review digest drift")

    archive = _validate_f2_d_v6_download_zip(
        archive_path=archive_path,
        evidence_root=evidence_root,
        required_members=required_zip_members,
        project_root=project_root,
        manifest_schema=route["download_manifest_schema"],
    )
    return (
        {
            "required": True,
            "status": "PASS",
            "execution_profile": execution_profile,
            "video_count": 2,
            "identities": list(identities),
            "production_contract_path": _relative(
                production_path, project_root, "F2 D-v6 production contract"
            ),
            "production_contract_sha256": production_sha,
            "downloaded_zip": archive,
        },
        [
            {"role": "native_t4_visual_production_contract", **_reference_dict(production_path, production_sha, project_root, "F2 D-v6 production contract")},
            {"role": "downloaded_native_visual_zip", **_reference_dict(archive_path, archive_sha, project_root, "F2 D-v6 downloaded ZIP")},
            {"role": "native_ship_certificate", **_reference_dict(certificate_path, certificate_sha, project_root, "F2 D-v6 native certificate")},
            {"role": "native_t4_visual_execution", **_reference_dict(execution_path, execution_sha, project_root, "F2 D-v6 native visual execution")},
            {"role": "downloaded_full_decode", **_reference_dict(decode_path, decode_sha, project_root, "F2 D-v6 downloaded decode")},
            {"role": "human_visual_review", **_reference_dict(review_path, review_sha, project_root, "F2 D-v6 human review")},
        ],
    )


def check_f2_d_v7_native_production_contract(
    *,
    production_contract_reference: Any,
    downloaded_zip_reference: Any,
    kernel_dir: Path,
    code_path: Path,
    metadata_path: Path,
    embedded_child_sha256: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate only the exact additive D-v7 native-T4 visual route."""

    return check_f2_d_v6_native_production_contract(
        production_contract_reference=production_contract_reference,
        downloaded_zip_reference=downloaded_zip_reference,
        kernel_dir=kernel_dir,
        code_path=code_path,
        metadata_path=metadata_path,
        embedded_child_sha256=embedded_child_sha256,
        project_root=project_root,
        _route_key="v7",
    )


def check_f2_d_v8_native_production_contract(
    *,
    production_contract_reference: Any,
    downloaded_zip_reference: Any,
    kernel_dir: Path,
    code_path: Path,
    metadata_path: Path,
    embedded_child_sha256: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate only the exact additive D-v8 native-T4 visual route."""

    return check_f2_d_v6_native_production_contract(
        production_contract_reference=production_contract_reference,
        downloaded_zip_reference=downloaded_zip_reference,
        kernel_dir=kernel_dir,
        code_path=code_path,
        metadata_path=metadata_path,
        embedded_child_sha256=embedded_child_sha256,
        project_root=project_root,
        _route_key="v8",
    )


def check_f2_d_v9_native_production_contract(
    *,
    production_contract_reference: Any,
    downloaded_zip_reference: Any,
    kernel_dir: Path,
    code_path: Path,
    metadata_path: Path,
    embedded_child_sha256: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate only the exact additive D-v9 native-T4 visual route."""

    return check_f2_d_v6_native_production_contract(
        production_contract_reference=production_contract_reference,
        downloaded_zip_reference=downloaded_zip_reference,
        kernel_dir=kernel_dir,
        code_path=code_path,
        metadata_path=metadata_path,
        embedded_child_sha256=embedded_child_sha256,
        project_root=project_root,
        _route_key="v9",
    )


def check_f2_d_v10_native_production_contract(
    *,
    production_contract_reference: Any,
    downloaded_zip_reference: Any,
    kernel_dir: Path,
    code_path: Path,
    metadata_path: Path,
    embedded_child_sha256: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate only the exact additive D-v10 native-T4 visual route."""

    return check_f2_d_v6_native_production_contract(
        production_contract_reference=production_contract_reference,
        downloaded_zip_reference=downloaded_zip_reference,
        kernel_dir=kernel_dir,
        code_path=code_path,
        metadata_path=metadata_path,
        embedded_child_sha256=embedded_child_sha256,
        project_root=project_root,
        _route_key="v10",
    )


def check_m7r6_native_t4_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate the one exact M7R6 certification-only native-T4 package."""

    failures: list[str] = []
    route = M7R6_NATIVE_T4_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "onebitbrain/dinov2-vits14-official-code-model-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "M7R6 native T4 whole pipeline cert v1",
    }
    if metadata != expected_metadata:
        failures.append("M7R6 exact script-kernel metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m7r6-native-t4-central-prelaunch.v1",
        "execution_profile": M7R6_NATIVE_T4_CERT_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M7R6_NATIVE_T4_CERTIFICATION_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "m7r5_source_sha256": route["m7r5_source_sha256"],
        "m7r5_terminal_sha256": route["m7r5_terminal_sha256"],
        "m7r5_rejected_receipt_sha256": route["m7r5_rejected_receipt_sha256"],
        "authenticated_child_sha256": route["authenticated_child_sha256"],
        "ship_profile_sha256": route["ship_profile_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "certification_only": True,
        "production_output_permitted": False,
        "submission_output_permitted": False,
        "scoring_permitted": False,
        "downloaded_evidence_required_for_any_successor": True,
        "whole_run_hard_bound_seconds": 10620,
        "repeat_count": 2,
        "one_shot": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M7R6 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M7R6 central contract has an arbitrary surface")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M7R6 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M7R6 exact metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M7R6 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M7R6 package inventory drift")
    exact_markers = (
        route["m7r5_source_sha256"],
        route["m7r5_terminal_sha256"],
        route["m7r5_rejected_receipt_sha256"],
        route["authenticated_child_sha256"],
        'WHOLE_RUN_HARD_BOUND_SECONDS = 10620',
        'CHILD_BUDGET_SECONDS = 10500',
        '"torch": "2.7.1+cu126"',
        '"torch_cuda": "12.6"',
        '"physical_gpu_name": "Tesla T4"',
        '"compute_capability": "7.5"',
        '"application_cuda_visible_devices": "0"',
        '"network_enabled": False',
        '"production_output": False',
        '"submission_output": False',
        '"scoring": False',
        "M7R6_NATIVE_T4_CERT_OUTPUT.zip",
        "M7R6_NATIVE_T4_CERTIFICATION_ELIGIBLE",
    )
    for marker in exact_markers:
        if marker not in source:
            failures.append(f"M7R6 required source marker absent: {marker}")
    forbidden = (
        "submission.csv",
        "/kaggle/input/competitions/biohub-cell-tracking-during-development/test",
        "44b6_6bba",
    )
    for marker in forbidden:
        if marker in source:
            failures.append(f"M7R6 forbidden source surface present: {marker}")
    main_functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    ]
    if len(main_functions) != 1:
        failures.append("M7R6 main function cardinality drift")
        step0_line = None
    else:
        calls = [
            node for node in ast.walk(main_functions[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        step0_calls = [node for node in calls if node.func.id == "_step0"]
        repeat_calls = [node for node in calls if node.func.id == "_run_repeat"]
        if len(step0_calls) != 1 or len(repeat_calls) != 1 or step0_calls[0].lineno >= repeat_calls[0].lineno:
            failures.append("M7R6 Step0 is not the unique pre-model boundary")
        step0_line = step0_calls[0].lineno if len(step0_calls) == 1 else None
    if phase == "launch":
        failures.append(
            "M7R6 central guard static profile cannot authorize launch without the separate current-LMT/quota/fresh-slug/serial-lock coordinator"
        )
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m7r6-native-t4-whole-pipeline-cert-v1",
        "record_sha256": route["m7r5_terminal_sha256"],
        "video_count": 0,
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [
            {
                "path": (
                    "dag_prompts/2026-08-23-model-clade-renewal-v1/"
                    "m7r5_embedded_child_derived_hash_transport/"
                    "M7R5_MATERIAL_TERMINAL.json"
                ),
                "sha256": route["m7r5_terminal_sha256"],
            }
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m7r7_native_runtime_diagnostic(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate the exact no-science M7R7 native runtime diagnostic."""

    failures: list[str] = []
    route = M7R7_NATIVE_RUNTIME_DIAGNOSTIC_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": ["onebitbrain/g02-cellect-zarr-cv2-runtime-deps-v1"],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "M7R7 native runtime tuple diagnostic v1",
    }
    if metadata != expected_metadata:
        failures.append("M7R7 exact script-kernel metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m7r7-native-runtime-central-prelaunch.v1",
        "execution_profile": M7R7_NATIVE_RUNTIME_DIAGNOSTIC_EXECUTION_PROFILE,
        "model_execution": False,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M7R7_NATIVE_RUNTIME_DIAGNOSTIC_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "environment_sha256": route["environment_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "local_env_probe_sha256": route["local_env_probe_sha256"],
        "ship_profile_sha256": route["ship_profile_sha256"],
        "ship_certificate_sha256": route["ship_certificate_sha256"],
        "ship_certificate_content_sha256": route[
            "ship_certificate_content_sha256"
        ],
        "downloaded_runtime_sealer_sha256": route[
            "downloaded_runtime_sealer_sha256"
        ],
        "m7r6_terminal_sha256": route["m7r6_terminal_sha256"],
        "m7r6_status_sha256": route["m7r6_status_sha256"],
        "sentinel_roundtrip_sha256": route["sentinel_roundtrip_sha256"],
        "sentinel_mount_correction_sha256": route[
            "sentinel_mount_correction_sha256"
        ],
        "diagnostic_only": True,
        "production_output_permitted": False,
        "submission_output_permitted": False,
        "scoring_permitted": False,
        "competition_or_raw_input_permitted": False,
        "model_or_science_execution_permitted": False,
        "downloaded_runtime_seal_required": True,
        "whole_run_hard_bound_seconds": 540,
        "max_workload_units": 8,
        "one_shot": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M7R7 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M7R7 central contract has an arbitrary surface")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M7R7 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M7R7 exact metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M7R7 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M7R7 package inventory drift")

    exact_artifacts = (
        ("M7R7 ship certificate", route["ship_certificate_path"], route["ship_certificate_sha256"]),
        ("M7R6 terminal", route["m7r6_terminal_path"], route["m7r6_terminal_sha256"]),
        (
            "M7R7 sentinel roundtrip",
            route["sentinel_roundtrip_path"],
            route["sentinel_roundtrip_sha256"],
        ),
        (
            "M7R7 sentinel mount correction",
            route["sentinel_mount_correction_path"],
            route["sentinel_mount_correction_sha256"],
        ),
    )
    artifact_paths: dict[str, Path] = {}
    for label, relative, expected_sha in exact_artifacts:
        path = project_root / relative
        artifact_paths[label] = path
        if not path.is_file() or path.is_symlink() or sha256(path) != expected_sha:
            failures.append(f"{label} exact receipt drift")
    certificate_path = artifact_paths["M7R7 ship certificate"]
    if certificate_path.is_file() and not certificate_path.is_symlink():
        certificate = load_json(certificate_path)
        certificate_exact = {
            "status": "ELIGIBLE",
            "eligible": True,
            "autonomous_launch_eligible": True,
            "kernel_id": route["kernel_id"],
            "metadata_sha256": route["metadata_sha256"],
            "package_inventory_sha256": route["package_inventory_sha256"],
            "profile_sha256": route["ship_profile_sha256"],
            "receipt_content_sha256": route["ship_certificate_content_sha256"],
        }
        for key, expected in certificate_exact.items():
            if certificate.get(key) != expected:
                failures.append(f"M7R7 ship certificate drift: {key}")

    exact_markers = (
        route["m7r6_terminal_sha256"],
        route["m7r6_status_sha256"],
        'HARD_BOUND_SECONDS = 540',
        'OUTPUT = WORKING / "M7R7_RUNTIME_TUPLE.json"',
        'EXPECTED_TENSOR_SHA256 = "a4f2ce00a8f1b4d4b95c51bb596312ac69ce47e1a19817ca8a0481b16beba820"',
        '"competition_or_raw_input_opened": False',
        '"model_or_science_executed": False',
        '"production_enabled": False',
        '"scoring": False',
        '"submission_output": False',
        "M7R7_DIAGNOSTIC_COMPLETE",
    )
    for marker in exact_markers:
        if marker not in source:
            failures.append(f"M7R7 required source marker absent: {marker}")
    forbidden = (
        "/kaggle/input/competitions",
        "44b6_6bba",
        "submission.csv",
        "tracksdata",
        "dinov2",
    )
    for marker in forbidden:
        if marker in source.lower():
            failures.append(f"M7R7 forbidden source surface present: {marker}")

    imports = [
        node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    import_roots = {
        alias.name.split(".", 1)[0]
        for node in imports
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=node.module or "")])
    }
    expected_import_roots = {
        "__future__",
        "hashlib",
        "json",
        "os",
        "platform",
        "signal",
        "site",
        "struct",
        "sys",
        "pathlib",
        "typing",
        "torch",
    }
    if import_roots != expected_import_roots:
        failures.append("M7R7 import surface drift")
    main_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    ]
    torch_imports = [
        node
        for node in imports
        if isinstance(node, ast.Import)
        and any(alias.name == "torch" for alias in node.names)
    ]
    if len(main_functions) != 1 or len(torch_imports) != 1:
        failures.append("M7R7 main/torch import cardinality drift")
    elif torch_imports[0] not in list(ast.walk(main_functions[0])):
        failures.append("M7R7 torch import moved before the Step0 boundary")
    try:
        observed_step0_line = step0_line(tree)
    except GuardError as exc:
        failures.append(f"M7R7 Step0 boundary drift: {exc}")
        observed_step0_line = None
    if (
        main_functions
        and observed_step0_line is not None
        and observed_step0_line >= main_functions[0].lineno
    ):
        failures.append("M7R7 Step0 boundary no longer precedes main")

    sentinel = (
        "/kaggle/input/datasets/onebitbrain/"
        "g02-cellect-zarr-cv2-runtime-deps-v1"
    )
    sentinel_name = "UNUSED_DEPENDENCY_SENTINEL_ROOT"
    if source.count(sentinel) != 1:
        failures.append("M7R7 dependency-only sentinel literal drift")
    sentinel_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == sentinel_name
        and isinstance(node.ctx, ast.Load)
    ]
    sentinel_stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == sentinel_name
        and isinstance(node.ctx, ast.Store)
    ]
    if sentinel_loads or len(sentinel_stores) != 1:
        failures.append("M7R7 dependency-only sentinel became executable")
    if phase == "launch":
        failures.append(
            "M7R7 central guard cannot authorize launch without the separate current-LMT/quota/fresh-slug/serial-lock coordinator"
        )
    visual_summary = {
        "required": False,
        "status": "NOT_APPLICABLE",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "diagnostic_only": True,
        "exception_id": None,
        "record_sha256": route["m7r6_terminal_sha256"],
        "video_count": 0,
        "identities": [],
        "post_remote_certification": {
            "downloaded_output_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted": False,
            "status": "PENDING_REMOTE_RUNTIME_TUPLE",
        },
        "artifacts": [
            {
                "path": route["ship_certificate_path"],
                "sha256": route["ship_certificate_sha256"],
            },
            {
                "path": route["m7r6_terminal_path"],
                "sha256": route["m7r6_terminal_sha256"],
            },
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=observed_step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m12r4_no_target_synthetic_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact frozen M12R4 no-target synthetic route."""

    failures: list[str] = []
    route = M12R4_NO_TARGET_SYNTHETIC_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": ["onebitbrain/g02-cellect-zarr-cv2-runtime-deps-v1"],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "M12R3 native T4 forward backward v1",
    }
    if metadata != expected_metadata:
        failures.append("M12R4 exact no-target script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m12r4-static-central-prelaunch.v1",
        "execution_profile": M12R4_NO_TARGET_SYNTHETIC_CERT_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M12R4_NO_TARGET_SYNTHETIC_MODEL_CERT_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        **{
            key: route[key]
            for key in route
            if key.startswith(("m12r", "m7r7_"))
        },
        "one_shot_sha256": route["one_shot_sha256"],
        "monitor_sha256": route["monitor_sha256"],
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "target_input_permitted": False,
        "production_output_permitted": False,
        "submission_output_permitted": False,
        "scoring_permitted": False,
        "downloaded_exact_evidence_required": True,
        "compatibility_claim_permitted_before_seal": False,
        "local_host_target_compatible": False,
        "local_host_python": "3.14.6",
        "local_host_platform": "darwin-arm64",
        "provider_role_control_bound": False,
        "material_ship_certificate_present": False,
        "max_workload_units": 32,
        "one_shot": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M12R4 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M12R4 central contract has an arbitrary surface")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M12R4 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M12R4 exact metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M12R4 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M12R4 package inventory drift")

    node = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    m12r4 = node / "m12r4_static_central_route"
    exact_artifacts = (
        (m12r4 / "M12R4_CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "candidate inventory"),
        (m12r4 / "M12R4_CENTRAL_GUARD_CONTRACT.json", route["contract_sha256"], "central contract"),
        (m12r4 / "M12R4_GUARD_REGISTRY.json", route["minimal_registry_sha256"], "minimal registry"),
        (m12r4 / "M12R4_PACKAGE_EQUIVALENCE.json", route["package_equivalence_sha256"], "package equivalence"),
        (m12r4 / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "experiment plan"),
        (m12r4 / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "experiment validation"),
        (m12r4 / "one_shot_m12r4.py", route["one_shot_sha256"], "one-shot code"),
        (m12r4 / "monitor_m12r4.py", route["monitor_sha256"], "monitor code"),
        (m12r4 / "seal_downloaded_m12r4.py", route["downloaded_evidence_sealer_sha256"], "download sealer"),
        (node / "m12r1_static_implementation/M12R1_TERMINAL.json", route["m12r1_terminal_sha256"], "M12R1 terminal"),
        (node / "m12r1_static_implementation/m12r1_cleanroom.py", route["m12r1_source_sha256"], "M12R1 source"),
        (node / "m12r1_static_implementation/M12R1_FROZEN_CONFIG.json", route["m12r1_config_sha256"], "M12R1 config"),
        (node / "m12r1_static_implementation/SYNTHETIC_CONTROL_RECEIPT.json", route["m12r1_synthetic_sha256"], "M12R1 synthetic receipt"),
        (node / "m12r2_static_package/M12R2_TERMINAL.json", route["m12r2_terminal_sha256"], "M12R2 terminal"),
        (node / "m12r2_static_package/M12R2_SHIP_STATIC_REJECTION.json", route["m12r2_rejection_sha256"], "M12R2 rejection"),
        (node / "m12r3_static_sentinel_transport/M12R3_TERMINAL.json", route["m12r3_terminal_sha256"], "M12R3 terminal"),
        (node / "m12r3_static_sentinel_transport/preflight-profile.json", route["m12r3_profile_sha256"], "M12R3 ship profile"),
        (node / "m12r3_static_sentinel_transport/M12R3_SENTINEL_AUDIT.json", route["m12r3_sentinel_audit_sha256"], "M12R3 sentinel audit"),
        (node / "m12r3_static_sentinel_transport/M12R3_ENVIRONMENT.json", route["m12r3_environment_sha256"], "M12R3 environment"),
        (node / "m12r3_static_sentinel_transport/M12R3_SHIP_STATIC_REJECTION.json", route["m12r3_ship_rejection_sha256"], "M12R3 ship rejection"),
        (node / "m7r7_native_runtime_tuple_diagnostic/M7R7_NATIVE_T4_V1_TERMINAL.json", route["m7r7_terminal_sha256"], "M7R7 terminal"),
        (node / "m7r7_native_runtime_tuple_diagnostic/M7R7_DOWNLOADED_RUNTIME_SEAL.json", route["m7r7_runtime_seal_sha256"], "M7R7 runtime seal"),
        (node / "m7r7_native_runtime_tuple_diagnostic/downloaded_native_t4_v1/M7R7_RUNTIME_TUPLE.json", route["m7r7_runtime_receipt_sha256"], "M7R7 runtime receipt"),
    )
    for path, expected, label in exact_artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M12R4 exact artifact drift: {label}")

    required_source = (
        'SEED = 104729',
        'STEP0 = run_step0()',
        'role = os.environ["M12R2_ROLE"]',
        'torch.__version__ != "2.10.0+cu128"',
        'torch.version.cuda != "12.8"',
        'M12R2_CALIBRATION_COMPLETE',
    )
    for marker in required_source:
        if marker not in source:
            failures.append(f"M12R4 required source marker absent: {marker}")
    forbidden = ("/kaggle/input", "competition_test", "6bba", "submission.csv")
    if any(marker in source.lower() for marker in forbidden):
        failures.append("M12R4 target/scoring source firewall drift")
    read_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "open")
            or (isinstance(node.func, ast.Attribute) and node.func.attr in {"read_bytes", "read_text", "glob", "rglob"})
        )
    ]
    if read_calls:
        failures.append("M12R4 frozen no-target kernel gained an input read surface")
    step0_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "STEP0" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "run_step0"
    ]
    if len(step0_assignments) != 1:
        failures.append("M12R4 exact in-main Step0 boundary drift")
        observed_step0_line = None
    else:
        observed_step0_line = step0_assignments[0].lineno
    profile_path = node / "m12r3_static_sentinel_transport/preflight-profile.json"
    if profile_path.is_file() and not profile_path.is_symlink():
        profile = load_json(profile_path)
        scenarios = profile.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios or any(
            scenario.get("env", {}).get("M12R2_ROLE") != scenario.get("role")
            for scenario in scenarios
        ):
            failures.append("M12R4 local profile role controls drift")
        command_text = " ".join(profile.get("kernel", {}).get("command", []))
        if "M12R2_ROLE" in command_text:
            failures.append("M12R4 profile falsely claims a provider-bound role")
    rejection_path = node / "m12r3_static_sentinel_transport/M12R3_SHIP_STATIC_REJECTION.json"
    if rejection_path.is_file() and not rejection_path.is_symlink():
        rejection = load_json(rejection_path)
        if rejection.get("static_result", {}).get("failures") != [
            "local Python 3.14 does not match required 3.12"
        ]:
            failures.append("M12R4 truthful local host mismatch drift")
    if phase in {"inline", "launch"}:
        failures.append(
            "M12R4 frozen package has no provider-bound M12R2_ROLE; local scenario injection is not Kaggle launch transport"
        )
    if phase == "launch":
        failures.append(
            "M12R4 central guard cannot authorize launch without exact material ship certification, resolved M7R14 coordination, current-LMT/quota/fresh-slug/serial-lock gates"
        )
    visual_summary = {
        "required": False,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "target_input": False,
        "score": False,
        "compatibility_claim": False,
        "local_host_target_compatible": False,
        "post_remote_certification": {
            "downloaded_output_required": True,
            "exact_local_seal_required": True,
            "compatibility_claim_permitted_before_seal": False,
            "production_or_score_permitted": False,
            "status": "PENDING_EXACT_DOWNLOADED_EVIDENCE",
        },
        "artifacts": [
            {"path": str(profile_path.relative_to(project_root)), "sha256": route["m12r3_profile_sha256"]},
            {"path": str(rejection_path.relative_to(project_root)), "sha256": route["m12r3_ship_rejection_sha256"]},
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=observed_step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m12r5_no_target_synthetic_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact self-orchestrating M12R5 synthetic route."""

    failures: list[str] = []
    route = M12R5_NO_TARGET_SYNTHETIC_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": ["onebitbrain/g02-cellect-zarr-cv2-runtime-deps-v1"],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "M12R5 self orchestrating no target cert v1",
    }
    if metadata != expected_metadata:
        failures.append("M12R5 exact no-target script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m12r5-static-central-prelaunch.v1",
        "execution_profile": M12R5_NO_TARGET_SYNTHETIC_CERT_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M12R5_NO_TARGET_SYNTHETIC_CERT_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "provider_profile_sha256": route["provider_profile_sha256"],
        "transport_equivalence_sha256": route["transport_equivalence_sha256"],
        "static_certification_sha256": route["static_certification_sha256"],
        "no_target_contract_sha256": route["no_target_contract_sha256"],
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "m12r1_terminal_sha256": route["m12r1_terminal_sha256"],
        "m12r1_source_sha256": route["m12r1_source_sha256"],
        "m12r1_config_sha256": route["m12r1_config_sha256"],
        "m12r1_synthetic_sha256": route["m12r1_synthetic_sha256"],
        "m12r2_terminal_sha256": route["m12r2_terminal_sha256"],
        "m12r3_terminal_sha256": route["m12r3_terminal_sha256"],
        "m12r3_sentinel_sha256": route["m12r3_sentinel_sha256"],
        "m12r4_terminal_sha256": route["m12r4_terminal_sha256"],
        "m7r7_terminal_sha256": route["m7r7_terminal_sha256"],
        "m7r7_seal_sha256": route["m7r7_seal_sha256"],
        "m7r7_receipt_sha256": route["m7r7_receipt_sha256"],
        "m7r7_runtime_tuple_sha256": route["m7r7_runtime_tuple_sha256"],
        "lmt_sha256": route["lmt_sha256"],
        "normal_invocation_requires_external_role_env_or_argv": False,
        "target_input_permitted": False,
        "dependency_only_unused_sentinel_count": 1,
        "production_output_permitted": False,
        "submission_output_permitted": False,
        "scoring_permitted": False,
        "downloaded_exact_evidence_required": True,
        "compatibility_claim_permitted_before_seal": False,
        "local_host_target_compatible": False,
        "role_count": 4,
        "repeat_count": 2,
        "total_completed_units": 88,
        "whole_run_hard_bound_seconds": 11520,
        "material_ship_certificate_present": False,
        "one_shot": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M12R5 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M12R5 central contract has an arbitrary surface")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M12R5 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M12R5 exact metadata hash drift")
    freeze = kernel_dir / "M12R5_PACKAGE_FREEZE.json"
    if not freeze.is_file() or freeze.is_symlink() or sha256(freeze) != route["package_freeze_sha256"]:
        failures.append("M12R5 package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M12R5 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M12R5 package inventory drift")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != "M12R5_NO_TARGET_SYNTHETIC_CERT_ONLY":
        failures.append("M12R5 exact registry ownership drift")
    if registry.get("dataset_ownership", {}).get("onebitbrain/g02-cellect-zarr-cv2-runtime-deps-v1") != "DEPENDENCY_ONLY_UNUSED_SENTINEL":
        failures.append("M12R5 sentinel registry ownership drift")

    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m12r5_self_orchestrating_package"
    exact_artifacts = (
        (node / "M12R5_CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "candidate inventory"),
        (node / "M12R5_PROVIDER_PROFILE.json", route["provider_profile_sha256"], "provider profile"),
        (node / "M12R5_TRANSPORT_EQUIVALENCE.json", route["transport_equivalence_sha256"], "transport equivalence"),
        (node / "M12R5_STATIC_CERTIFICATION.json", route["static_certification_sha256"], "static certification"),
        (node / "M12R5_NO_TARGET_INPUT_CONTRACT.json", route["no_target_contract_sha256"], "no-target contract"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "experiment validation"),
        (node / "seal_downloaded_m12r5_evidence.py", route["downloaded_evidence_sealer_sha256"], "downloaded evidence sealer"),
        (node / "M12R5_CENTRAL_GUARD_CONTRACT.json", route["contract_sha256"], "central contract"),
        (node / "M12R5_GUARD_REGISTRY.json", route["minimal_registry_sha256"], "minimal registry"),
        (campaign / "m12r1_static_implementation/M12R1_TERMINAL.json", route["m12r1_terminal_sha256"], "M12R1 terminal"),
        (campaign / "m12r1_static_implementation/m12r1_cleanroom.py", route["m12r1_source_sha256"], "M12R1 source"),
        (campaign / "m12r1_static_implementation/M12R1_FROZEN_CONFIG.json", route["m12r1_config_sha256"], "M12R1 config"),
        (campaign / "m12r1_static_implementation/SYNTHETIC_CONTROL_RECEIPT.json", route["m12r1_synthetic_sha256"], "M12R1 synthetic receipt"),
        (campaign / "m12r2_static_package/M12R2_TERMINAL.json", route["m12r2_terminal_sha256"], "M12R2 terminal"),
        (campaign / "m12r3_static_sentinel_transport/M12R3_TERMINAL.json", route["m12r3_terminal_sha256"], "M12R3 terminal"),
        (campaign / "m12r3_static_sentinel_transport/M12R3_SENTINEL_AUDIT.json", route["m12r3_sentinel_sha256"], "M12R3 sentinel audit"),
        (campaign / "m12r4_static_central_route/M12R4_TERMINAL.json", route["m12r4_terminal_sha256"], "M12R4 rejection"),
        (campaign / "m7r7_native_runtime_tuple_diagnostic/M7R7_NATIVE_T4_V1_TERMINAL.json", route["m7r7_terminal_sha256"], "M7R7 terminal"),
        (campaign / "m7r7_native_runtime_tuple_diagnostic/M7R7_DOWNLOADED_RUNTIME_SEAL.json", route["m7r7_seal_sha256"], "M7R7 runtime seal"),
        (campaign / "m7r7_native_runtime_tuple_diagnostic/downloaded_native_t4_v1/M7R7_RUNTIME_TUPLE.json", route["m7r7_receipt_sha256"], "M7R7 runtime receipt"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    for path, expected, label in exact_artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M12R5 exact artifact drift: {label}")
    try:
        child = _literal_assignment(tree, "M12R3_CHILD_SOURCE", "M12R5 embedded M12R3 child")
        child_path = campaign / "m12r3_static_sentinel_transport/package_v1/kernel.py"
        if not isinstance(child, str) or hashlib.sha256(child.encode()).hexdigest() != route["m12r3_child_sha256"]:
            raise GuardError("M12R5 embedded child hash drift")
        if child.encode() != child_path.read_bytes():
            raise GuardError("M12R5 embedded child bytes differ from M12R3")
        if _literal_assignment(tree, "ROLE_ORDER", "M12R5 role order") != ("tiny", "stratified", "stress", "expansive"):
            raise GuardError("M12R5 role order drift")
        if _literal_assignment(tree, "REPEAT_COUNT", "M12R5 repeat count") != 2:
            raise GuardError("M12R5 repeat count drift")
        if _literal_assignment(tree, "WHOLE_RUN_HARD_BOUND_SECONDS", "M12R5 whole-run bound") != 11520:
            raise GuardError("M12R5 whole-run bound drift")
        external_reads = [
            item.lineno
            for item in ast.walk(tree)
            if (
                isinstance(item, ast.Subscript)
                and isinstance(item.ctx, ast.Load)
                and isinstance(item.value, ast.Attribute)
                and isinstance(item.value.value, ast.Name)
                and item.value.value.id == "os"
                and item.value.attr == "environ"
            )
            or (
                isinstance(item, ast.Attribute)
                and isinstance(item.ctx, ast.Load)
                and isinstance(item.value, ast.Name)
                and item.value.id == "sys"
                and item.attr == "argv"
            )
        ]
        if external_reads:
            raise GuardError("M12R5 normal invocation acquired external env/argv reads")
        markers = (
            'os.environ["M12R2_ROLE"] = role',
            'sys.argv = ["<embedded-m12r3-child>"]',
            "for role in ROLE_ORDER:",
            "for repeat in range(1, REPEAT_COUNT + 1):",
            "signal.setitimer(signal.ITIMER_REAL, timeout_seconds)",
            "M12R5_SYNTHETIC_CERT_EVIDENCE.zip",
            "M12R5_SYNTHETIC_CERT_FAILURE.zip",
            '"production_enabled": False',
            '"target_input_opened": False',
            '"score_or_submission": False',
        )
        if any(marker not in source for marker in markers):
            raise GuardError("M12R5 self-orchestration/output marker drift")
        if any(marker in source.lower() for marker in ("/kaggle/input", "competition_test", "6bba", "submission.csv", "leaderboard")):
            raise GuardError("M12R5 forbidden target/score surface present")
    except (GuardError, SyntaxError, ValueError) as exc:
        failures.append(str(exc))
    if phase == "launch":
        failures.append(
            "M12R5 launch requires exact material ship certification, resolved M7R14 coordination, current-LMT/quota/fresh-slug/serial-lock gates"
        )
    visual_summary = {
        "required": False,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "target_input": False,
        "score": False,
        "compatibility_claim": False,
        "local_host_target_compatible": False,
        "role_transport": "SELF_ORCHESTRATING_EXACT_CHILD_4_ROLES_X2",
        "post_remote_certification": {
            "downloaded_output_required": True,
            "exact_local_seal_required": True,
            "compatibility_claim_permitted_before_seal": False,
            "production_or_score_permitted": False,
            "status": "PENDING_EXACT_DOWNLOADED_EVIDENCE",
        },
        "artifacts": [
            {"path": _relative(path, project_root, label), "sha256": expected}
            for path, expected, label in exact_artifacts
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m12r6_remote_certification_exception(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact pending M12R6 remote-certification exception."""

    failures: list[str] = []
    route = M12R6_REMOTE_CERTIFICATION_ROUTE
    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m12r6_remote_certification_exception"
    parent = campaign / "m12r5_self_orchestrating_package"
    parent_contract_path = parent / "M12R5_CENTRAL_GUARD_CONTRACT.json"
    parent_registry_path = parent / "M12R5_GUARD_REGISTRY.json"
    if (
        not parent_contract_path.is_file()
        or parent_contract_path.is_symlink()
        or sha256(parent_contract_path) != M12R5_NO_TARGET_SYNTHETIC_CERT_ROUTE["contract_sha256"]
        or not parent_registry_path.is_file()
        or parent_registry_path.is_symlink()
        or sha256(parent_registry_path) != M12R5_NO_TARGET_SYNTHETIC_CERT_ROUTE["minimal_registry_sha256"]
    ):
        failures.append("M12R6 exact M12R5 central parent drift")
    else:
        parent_result = check_m12r5_no_target_synthetic_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=load_json(parent_contract_path),
            registry=load_json(parent_registry_path),
            phase="inline",
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=parent_contract_path,
            registry_path=parent_registry_path,
        )
        if parent_result.get("status") != "PASS" or parent_result.get("eligible") is not True:
            failures.append("M12R6 exact M12R5 parent inline predicate is not PASS")

    expected_contract = {
        "schema": "celltrack-alpha.m12r6-static-central-prelaunch.v1",
        "execution_profile": M12R6_REMOTE_CERTIFICATION_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M12R6_REMOTE_CERTIFICATION_EXCEPTION_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "exception_sha256": route["exception_sha256"],
        "registry_sha256": route["registry_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "one_shot_sha256": route["one_shot_sha256"],
        "monitor_sha256": route["monitor_sha256"],
        "reconcile_sha256": route["reconcile_sha256"],
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "m12r5_terminal_sha256": route["m12r5_terminal_sha256"],
        "m12r5_static_freeze_sha256": route["m12r5_static_freeze_sha256"],
        "m12r5_inline_sha256": route["m12r5_inline_sha256"],
        "m12r5_profile_sha256": route["m12r5_profile_sha256"],
        "m12r5_no_target_sha256": route["m12r5_no_target_sha256"],
        "m12r5_equivalence_sha256": route["m12r5_equivalence_sha256"],
        "m12r4_terminal_sha256": route["m12r4_terminal_sha256"],
        "m12r3_terminal_sha256": route["m12r3_terminal_sha256"],
        "m12r3_sentinel_sha256": route["m12r3_sentinel_sha256"],
        "m7r7_terminal_sha256": route["m7r7_terminal_sha256"],
        "m7r7_runtime_seal_sha256": route["m7r7_runtime_seal_sha256"],
        "m7r7_runtime_receipt_sha256": route["m7r7_runtime_receipt_sha256"],
        "m7r7_runtime_tuple_sha256": route["m7r7_runtime_tuple_sha256"],
        "lmt_sha256": route["lmt_sha256"],
        "remote_certification_pending": True,
        "local_native_material_certificate_present": False,
        "root_authorization_present": False,
        "m7r15_coordination_released": False,
        "target_input_permitted": False,
        "dependency_only_unused_sentinel_count": 1,
        "production_output_permitted": False,
        "submission_output_permitted": False,
        "scoring_permitted": False,
        "downloaded_exact_evidence_required": True,
        "compatibility_claim_permitted_before_seal": False,
        "local_host_target_compatible": False,
        "role_count": 4,
        "repeat_count": 2,
        "total_completed_units": 88,
        "whole_run_hard_bound_seconds": 11520,
        "under_eight_hours": True,
        "one_shot": True,
    }
    if contract != expected_contract:
        failures.append("M12R6 exact remote-certification contract drift")
    if registry.get("kernel_ownership") != {
        route["kernel_id"]: "M12R6_REMOTE_CERTIFICATION_EXCEPTION_ONLY"
    }:
        failures.append("M12R6 exact kernel ownership drift")
    if registry.get("dataset_ownership") != {
        "onebitbrain/g02-cellect-zarr-cv2-runtime-deps-v1": "DEPENDENCY_ONLY_UNUSED_SENTINEL"
    }:
        failures.append("M12R6 exact unused sentinel ownership drift")
    if registry.get("remote_certification_exception_allowlist") != M12R6_REMOTE_CERTIFICATION_EXCEPTION_ALLOWLIST:
        failures.append("M12R6 exact remote-certification exception allowlist drift")
    if set(registry) != {
        "schema",
        "kernel_ownership",
        "dataset_ownership",
        "remote_certification_exception_allowlist",
    } or registry.get("schema") != "celltrack-alpha.m12r6-minimal-guard-registry.v1":
        failures.append("M12R6 registry has an arbitrary surface")

    package = _package_inventory(kernel_dir, project_root)
    if sha256(code_path) != route["code_sha256"] or sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M12R6 exact unchanged M12R5 package drift")
    freeze = kernel_dir / "M12R5_PACKAGE_FREEZE.json"
    if not freeze.is_file() or freeze.is_symlink() or sha256(freeze) != route["package_freeze_sha256"]:
        failures.append("M12R6 exact M12R5 package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M12R6 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M12R6 exact package inventory drift")

    exact_artifacts = (
        (node / "M12R6_REMOTE_CERTIFICATION_EXCEPTION.json", route["exception_sha256"], "exception"),
        (node / "M12R6_GUARD_REGISTRY.json", route["registry_sha256"], "registry"),
        (node / "M12R6_CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "candidate inventory"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "experiment validation"),
        (node / "one_shot_m12r6.py", route["one_shot_sha256"], "one-shot"),
        (node / "monitor_m12r6.py", route["monitor_sha256"], "monitor"),
        (node / "reconcile_m12r6.py", route["reconcile_sha256"], "reconcile"),
        (node / "seal_downloaded_m12r6_evidence.py", route["downloaded_evidence_sealer_sha256"], "downloaded evidence sealer"),
        (parent / "M12R5_TERMINAL.json", route["m12r5_terminal_sha256"], "M12R5 terminal"),
        (parent / "M12R5_STATIC_FREEZE.json", route["m12r5_static_freeze_sha256"], "M12R5 static freeze"),
        (parent / "M12R5_INLINE_PREDICATE.json", route["m12r5_inline_sha256"], "M12R5 inline"),
        (parent / "M12R5_PROVIDER_PROFILE.json", route["m12r5_profile_sha256"], "M12R5 profile"),
        (parent / "M12R5_NO_TARGET_INPUT_CONTRACT.json", route["m12r5_no_target_sha256"], "M12R5 no-target contract"),
        (parent / "M12R5_TRANSPORT_EQUIVALENCE.json", route["m12r5_equivalence_sha256"], "M12R5 equivalence"),
        (campaign / "m12r4_static_central_route/M12R4_TERMINAL.json", route["m12r4_terminal_sha256"], "M12R4 terminal"),
        (campaign / "m12r3_static_sentinel_transport/M12R3_TERMINAL.json", route["m12r3_terminal_sha256"], "M12R3 terminal"),
        (campaign / "m12r3_static_sentinel_transport/M12R3_SENTINEL_AUDIT.json", route["m12r3_sentinel_sha256"], "M12R3 sentinel"),
        (campaign / "m7r7_native_runtime_tuple_diagnostic/M7R7_NATIVE_T4_V1_TERMINAL.json", route["m7r7_terminal_sha256"], "M7R7 terminal"),
        (campaign / "m7r7_native_runtime_tuple_diagnostic/M7R7_DOWNLOADED_RUNTIME_SEAL.json", route["m7r7_runtime_seal_sha256"], "M7R7 runtime seal"),
        (campaign / "m7r7_native_runtime_tuple_diagnostic/downloaded_native_t4_v1/M7R7_RUNTIME_TUPLE.json", route["m7r7_runtime_receipt_sha256"], "M7R7 runtime receipt"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    for path, expected, label in exact_artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M12R6 exact artifact drift: {label}")
    if phase == "launch":
        failures.append(
            "M12R6 launch requires explicit root authorization, released M7R15 coordination, and current LMT/quota/fresh-slug/serial-lock gates"
        )
    visual_summary = {
        "required": False,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "target_input": False,
        "score": False,
        "compatibility_claim": False,
        "local_host_target_compatible": False,
        "local_native_material_certificate_present": False,
        "role_transport": "SELF_ORCHESTRATING_EXACT_CHILD_4_ROLES_X2",
        "whole_run_hard_bound_seconds": 11520,
        "downloaded_output_required": True,
        "exact_local_seal_required": True,
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m12r7_canonical_registry_transport(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only M12R7's canonical-registry transport normalization."""

    failures: list[str] = []
    route = M12R7_CANONICAL_REGISTRY_ROUTE
    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m12r7_canonical_registry_transport"
    m12r6 = campaign / "m12r6_remote_certification_exception"
    parent_contract_path = m12r6 / "M12R6_CENTRAL_GUARD_CONTRACT.json"
    parent_registry_path = m12r6 / "M12R6_GUARD_REGISTRY.json"
    if (
        not parent_contract_path.is_file()
        or parent_contract_path.is_symlink()
        or sha256(parent_contract_path) != M12R6_REMOTE_CERTIFICATION_ROUTE["contract_sha256"]
        or not parent_registry_path.is_file()
        or parent_registry_path.is_symlink()
        or sha256(parent_registry_path) != M12R6_REMOTE_CERTIFICATION_ROUTE["registry_sha256"]
    ):
        failures.append("M12R7 exact M12R6 parent central artifacts drift")
    else:
        parent_result = check_m12r6_remote_certification_exception(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=load_json(parent_contract_path),
            registry=load_json(parent_registry_path),
            phase="inline",
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=parent_contract_path,
            registry_path=parent_registry_path,
        )
        if parent_result.get("status") != "PASS" or parent_result.get("eligible") is not True:
            failures.append("M12R7 exact M12R6 parent inline predicate is not PASS")

    expected_contract = {
        "schema": "celltrack-alpha.m12r7-static-central-prelaunch.v1",
        "execution_profile": M12R7_CANONICAL_REGISTRY_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M12R7_REMOTE_CERTIFICATION_CANONICAL_REGISTRY_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "registry_sha256": route["registry_sha256"],
        "canonical_registry_required": True,
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "transport_equivalence_sha256": route["transport_equivalence_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "m12r6_blocker_sha256": route["m12r6_blocker_sha256"],
        "m12r6_terminal_sha256": route["m12r6_terminal_sha256"],
        "remote_certification_pending": True,
        "local_native_material_certificate_present": False,
        "target_input_permitted": False,
        "dependency_only_unused_sentinel_count": 1,
        "production_output_permitted": False,
        "submission_output_permitted": False,
        "scoring_permitted": False,
        "downloaded_exact_evidence_required": True,
        "compatibility_claim_permitted_before_seal": False,
        "local_host_target_compatible": False,
        "role_count": 4,
        "repeat_count": 2,
        "total_completed_units": 88,
        "whole_run_hard_bound_seconds": 11520,
        "under_eight_hours": True,
        "one_shot": True,
    }
    if contract != expected_contract:
        failures.append("M12R7 exact canonical-registry contract drift")
    canonical_registry = (
        project_root
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json"
    ).resolve()
    if registry_path is None or registry_path.resolve() != canonical_registry:
        failures.append("M12R7 registry is not the canonical repository registry")
    elif registry_path.is_symlink() or sha256(registry_path) != route["registry_sha256"]:
        failures.append("M12R7 canonical registry hash drift")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != (
        "M12R7_REMOTE_CERTIFICATION_CANONICAL_REGISTRY_ONLY"
    ):
        failures.append("M12R7 canonical kernel ownership drift")
    if registry.get("dataset_ownership", {}).get(
        "onebitbrain/g02-cellect-zarr-cv2-runtime-deps-v1"
    ) != "M12R7_DEPENDENCY_ONLY_UNUSED_SENTINEL":
        failures.append("M12R7 canonical sentinel ownership drift")

    package = _package_inventory(kernel_dir, project_root)
    if sha256(code_path) != route["code_sha256"] or sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M12R7 exact unchanged M12R5 package drift")
    freeze = kernel_dir / "M12R5_PACKAGE_FREEZE.json"
    if not freeze.is_file() or freeze.is_symlink() or sha256(freeze) != route["package_freeze_sha256"]:
        failures.append("M12R7 exact package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M12R7 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M12R7 exact package inventory drift")
    exact_artifacts = (
        (node / "M12R7_CENTRAL_GUARD_CONTRACT.json", route["contract_sha256"], "contract"),
        (node / "M12R7_CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "candidate inventory"),
        (node / "M12R7_TRANSPORT_EQUIVALENCE.json", route["transport_equivalence_sha256"], "transport equivalence"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "experiment validation"),
        (m12r6 / "M12R6_DYNAMIC_LAUNCH_BLOCKED_CANONICAL_REGISTRY.json", route["m12r6_blocker_sha256"], "M12R6 blocker"),
        (m12r6 / "M12R6_TERMINAL.json", route["m12r6_terminal_sha256"], "M12R6 terminal"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    for path, expected, label in exact_artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M12R7 exact artifact drift: {label}")
    visual_summary = {
        "required": False,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "target_input": False,
        "score": False,
        "compatibility_claim": False,
        "local_host_target_compatible": False,
        "local_native_material_certificate_present": False,
        "role_transport": "SELF_ORCHESTRATING_EXACT_CHILD_4_ROLES_X2",
        "whole_run_hard_bound_seconds": 11520,
        "downloaded_output_required": True,
        "exact_local_seal_required": True,
        "canonical_registry": True,
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m12r8_synthetic_visual_preflight(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate M12R8's no-target synthetic-control visual preflight only."""

    del source, tree
    failures: list[str] = []
    route = M12R8_SYNTHETIC_VISUAL_ROUTE
    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m12r8_synthetic_visual_preflight"
    m12r7 = campaign / "m12r7_canonical_registry_transport"
    m12r1 = campaign / "m12r1_static_implementation"
    expected_contract = {
        "canonical_registry_required": True,
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "code_sha256": route["code_sha256"],
        "compatibility_claim_permitted_before_seal": False,
        "dependency_only_unused_sentinel_count": 1,
        "downloaded_exact_evidence_required": True,
        "execution_profile": M12R8_SYNTHETIC_VISUAL_EXECUTION_PROFILE,
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "kernel_id": route["kernel_id"],
        "lane_owner": "M12R8_SYNTHETIC_VISUAL_PREFLIGHT_ONLY",
        "local_host_target_compatible": False,
        "local_native_material_certificate_present": False,
        "m12r1_config_sha256": route["m12r1_config_sha256"],
        "m12r1_runner_sha256": route["m12r1_runner_sha256"],
        "m12r1_source_sha256": route["m12r1_source_sha256"],
        "m12r1_synthetic_receipt_sha256": route["m12r1_synthetic_receipt_sha256"],
        "m12r7_blocker_sha256": route["m12r7_blocker_sha256"],
        "m12r7_terminal_sha256": route["m12r7_terminal_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "model_execution": True,
        "one_shot": True,
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "production_output_permitted": False,
        "registry_sha256": route["registry_sha256"],
        "remote_certification_pending": True,
        "remote_preflight_kernel_id": route["kernel_id"],
        "repeat_count": 2,
        "role_count": 4,
        "schema": "celltrack-alpha.m12r8-static-central-prelaunch.v1",
        "scoring_permitted": False,
        "submission_output_permitted": False,
        "synthetic_control_visual_binding_sha256": route["synthetic_control_visual_binding_sha256"],
        "target_input_permitted": False,
        "total_completed_units": 88,
        "under_eight_hours": True,
        "whole_run_hard_bound_seconds": 11520,
    }
    if contract != expected_contract:
        failures.append("M12R8 exact synthetic-visual contract drift")
    canonical_registry = (
        project_root
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json"
    ).resolve()
    if registry_path is None or registry_path.resolve() != canonical_registry:
        failures.append("M12R8 registry is not the canonical repository registry")
    elif registry_path.is_symlink() or sha256(registry_path) != route["registry_sha256"]:
        failures.append("M12R8 canonical registry hash drift")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != (
        "M12R8_SYNTHETIC_VISUAL_PREFLIGHT_ONLY"
    ):
        failures.append("M12R8 canonical kernel ownership drift")
    if registry.get("dataset_ownership", {}).get(
        "onebitbrain/g02-cellect-zarr-cv2-runtime-deps-v1"
    ) != "M12R8_DEPENDENCY_ONLY_UNUSED_SENTINEL":
        failures.append("M12R8 canonical sentinel ownership drift")

    package = _package_inventory(kernel_dir, project_root)
    if sha256(code_path) != route["code_sha256"] or sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M12R8 exact unchanged M12R5 package drift")
    freeze = kernel_dir / "M12R5_PACKAGE_FREEZE.json"
    if not freeze.is_file() or freeze.is_symlink() or sha256(freeze) != route["package_freeze_sha256"]:
        failures.append("M12R8 exact package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M12R8 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M12R8 exact package inventory drift")

    exact_artifacts = (
        (node / "M12R8_CENTRAL_GUARD_CONTRACT.json", route["contract_sha256"], "contract"),
        (node / "M12R8_CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "candidate inventory"),
        (node / "M12R8_SYNTHETIC_CONTROL_VISUAL_BINDING.json", route["synthetic_control_visual_binding_sha256"], "synthetic visual binding"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "experiment validation"),
        (m12r7 / "M12R7_STATIC_BLOCKER.json", route["m12r7_blocker_sha256"], "M12R7 blocker"),
        (m12r7 / "M12R7_TERMINAL.json", route["m12r7_terminal_sha256"], "M12R7 terminal"),
        (m12r1 / "m12r1_cleanroom.py", route["m12r1_source_sha256"], "M12R1 source"),
        (m12r1 / "M12R1_FROZEN_CONFIG.json", route["m12r1_config_sha256"], "M12R1 config"),
        (m12r1 / "run_synthetic_controls.py", route["m12r1_runner_sha256"], "M12R1 runner"),
        (m12r1 / "SYNTHETIC_CONTROL_RECEIPT.json", route["m12r1_synthetic_receipt_sha256"], "M12R1 synthetic receipt"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    visual_artifacts: list[dict[str, str]] = []
    for path, expected, label in exact_artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M12R8 exact artifact drift: {label}")
        visual_artifacts.append(
            {"path": path.relative_to(project_root).as_posix(), "sha256": expected}
        )

    visual_binding_path = node / "M12R8_SYNTHETIC_CONTROL_VISUAL_BINDING.json"
    if visual_binding_path.is_file() and not visual_binding_path.is_symlink():
        visual_binding = load_json(visual_binding_path)
        expected_visual_binding = {
            "status": "PASS_HASH_AUTHENTICATED_SYNTHETIC_CONTROL_REPLAY",
            "model_execution": True,
            "target_input": False,
            "production": False,
            "score_or_submission": False,
            "video_count": 0,
            "identities": [],
            "source_sha256": route["m12r1_source_sha256"],
            "config_sha256": route["m12r1_config_sha256"],
            "runner_sha256": route["m12r1_runner_sha256"],
            "synthetic_receipt_sha256": route["m12r1_synthetic_receipt_sha256"],
            "negative_controls_all_pass": True,
        }
        for key, expected in expected_visual_binding.items():
            if visual_binding.get(key) != expected:
                failures.append(f"M12R8 synthetic visual binding drift: {key}")

    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "target_input": False,
        "score": False,
        "compatibility_claim": False,
        "local_host_target_compatible": False,
        "local_native_material_certificate_present": False,
        "role_transport": "SELF_ORCHESTRATING_EXACT_CHILD_4_ROLES_X2",
        "whole_run_hard_bound_seconds": 11520,
        "downloaded_output_required": True,
        "exact_local_seal_required": True,
        "canonical_registry": True,
        "exception_id": "m12r8-synthetic-control-visual-preflight-v1",
        "record_sha256": route["m12r7_terminal_sha256"],
        "identities": [],
        "video_count": 0,
        "synthetic_control_only": True,
        "synthetic_control_binding_sha256": route["synthetic_control_visual_binding_sha256"],
        "artifacts": visual_artifacts,
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m7r8_native_t4_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate M7R8 as the exact runtime-tuple-only successor of M7R6."""

    failures: list[str] = []
    route = M7R8_NATIVE_T4_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "onebitbrain/dinov2-vits14-official-code-model-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "M7R8 native T4 whole pipeline cert v1",
    }
    if metadata != expected_metadata:
        failures.append("M7R8 exact script-kernel metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m7r8-native-t4-central-prelaunch.v1",
        "execution_profile": M7R8_NATIVE_T4_CERT_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M7R8_NATIVE_T4_CERTIFICATION_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "freeze_sha256": route["freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "parent_m7r6_code_sha256": route["parent_m7r6_code_sha256"],
        "m7r6_terminal_sha256": route["m7r6_terminal_sha256"],
        "m7r7_terminal_sha256": route["m7r7_terminal_sha256"],
        "m7r7_runtime_seal_sha256": route["m7r7_runtime_seal_sha256"],
        "m7r7_runtime_receipt_sha256": route["m7r7_runtime_receipt_sha256"],
        "m7r7_runtime_tuple_sha256": route["m7r7_runtime_tuple_sha256"],
        "runtime_tuple_proof_sha256": route["runtime_tuple_proof_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "environment_sha256": route["environment_sha256"],
        "dependency_audit_sha256": route["dependency_audit_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "ship_profile_sha256": route["ship_profile_sha256"],
        "downloaded_evidence_sealer_sha256": route[
            "downloaded_evidence_sealer_sha256"
        ],
        "certification_only": True,
        "production_output_permitted": False,
        "submission_output_permitted": False,
        "scoring_permitted": False,
        "downloaded_evidence_required_for_any_successor": True,
        "whole_run_hard_bound_seconds": 10620,
        "repeat_count": 2,
        "one_shot": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M7R8 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M7R8 central contract has an arbitrary surface")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M7R8 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M7R8 exact metadata hash drift")
    freeze = kernel_dir / "M7R8_PACKAGE_FREEZE.json"
    if not freeze.is_file() or freeze.is_symlink() or sha256(freeze) != route["freeze_sha256"]:
        failures.append("M7R8 package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M7R8 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M7R8 package inventory drift")

    node = (
        project_root
        / "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r8_native_t4_whole_pipeline_cert"
    )
    exact_artifacts = (
        (project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/m7r6_native_t4_whole_pipeline_cert/package_v1/kernel.py", route["parent_m7r6_code_sha256"], "M7R6 parent code"),
        (project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/m7r6_native_t4_whole_pipeline_cert/M7R6_NATIVE_T4_V1_TERMINAL.json", route["m7r6_terminal_sha256"], "M7R6 terminal"),
        (project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/m7r7_native_runtime_tuple_diagnostic/M7R7_NATIVE_T4_V1_TERMINAL.json", route["m7r7_terminal_sha256"], "M7R7 terminal"),
        (project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/m7r7_native_runtime_tuple_diagnostic/M7R7_DOWNLOADED_RUNTIME_SEAL.json", route["m7r7_runtime_seal_sha256"], "M7R7 runtime seal"),
        (project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/m7r7_native_runtime_tuple_diagnostic/downloaded_native_t4_v1/M7R7_RUNTIME_TUPLE.json", route["m7r7_runtime_receipt_sha256"], "M7R7 runtime receipt"),
        (node / "M7R8_RUNTIME_TUPLE_REWRITE_PROOF.json", route["runtime_tuple_proof_sha256"], "M7R8 tuple rewrite proof"),
        (node / "M7R8_CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "M7R8 candidate inventory"),
        (node / "M7R8_ENVIRONMENT.json", route["environment_sha256"], "M7R8 environment"),
        (node / "M7R8_OFFLINE_DEPENDENCY_AUDIT.json", route["dependency_audit_sha256"], "M7R8 dependency audit"),
        (node / "M7R8_EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M7R8 experiment plan"),
        (node / "M7R8_EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "M7R8 experiment validation"),
        (node / "M7R8_SHIP_PROFILE.json", route["ship_profile_sha256"], "M7R8 ship profile"),
        (node / "seal_downloaded_m7r8_evidence.py", route["downloaded_evidence_sealer_sha256"], "M7R8 evidence sealer"),
    )
    for path, expected, label in exact_artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"{label} exact artifact drift")

    rewrites = (
        ('"python": "3.12"', '"python": "3.12.13"'),
        ('"torch": "2.7.1+cu126"', '"torch": "2.10.0+cu128"'),
        ('"torch_cuda": "12.6"', '"torch_cuda": "12.8"'),
        ("sys.version_info[:2] != (3, 12)", "sys.version_info[:3] != (3, 12, 13)"),
        ('torch.__version__ != "2.7.1+cu126"', 'torch.__version__ != "2.10.0+cu128"'),
        ('torch.version.cuda != "12.6"', 'torch.version.cuda != "12.8"'),
    )
    normalized = source
    for old, new in reversed(rewrites):
        if normalized.count(new) != 1:
            failures.append(f"M7R8 exact runtime tuple source-node drift: {new}")
            continue
        normalized = normalized.replace(new, old, 1)
    parent_path = exact_artifacts[0][0]
    if parent_path.is_file() and normalized != parent_path.read_text(encoding="utf-8"):
        failures.append("M7R8 executable bytes drift beyond the runtime tuple")
    if hashlib.sha256(normalized.encode()).hexdigest() != route["parent_m7r6_code_sha256"]:
        failures.append("M7R8 reverse-normalized parent hash drift")
    exact_markers = (
        'WHOLE_RUN_HARD_BOUND_SECONDS = 10620',
        'CHILD_BUDGET_SECONDS = 10500',
        '"python": "3.12.13"',
        '"torch": "2.10.0+cu128"',
        '"torch_cuda": "12.8"',
        '"physical_gpu_name": "Tesla T4"',
        '"compute_capability": "7.5"',
        '"application_cuda_visible_devices": "0"',
        '"network_enabled": False',
        '"production_output": False',
        '"submission_output": False',
        '"scoring": False',
        "M7R6_NATIVE_T4_CERT_OUTPUT.zip",
        "M7R6_NATIVE_T4_CERTIFICATION_ELIGIBLE",
    )
    for marker in exact_markers:
        if marker not in source:
            failures.append(f"M7R8 required source marker absent: {marker}")
    for marker in ("submission.csv", "44b6_6bba"):
        if marker in source:
            failures.append(f"M7R8 forbidden source surface present: {marker}")
    main_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    ]
    if len(main_functions) != 1:
        failures.append("M7R8 main function cardinality drift")
        observed_step0_line = None
    else:
        calls = [
            node
            for node in ast.walk(main_functions[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        step0_calls = [node for node in calls if node.func.id == "_step0"]
        repeat_calls = [node for node in calls if node.func.id == "_run_repeat"]
        if (
            len(step0_calls) != 1
            or len(repeat_calls) != 1
            or step0_calls[0].lineno >= repeat_calls[0].lineno
        ):
            failures.append("M7R8 Step0 is not the unique pre-model boundary")
        observed_step0_line = step0_calls[0].lineno if len(step0_calls) == 1 else None
    if phase == "launch":
        failures.append(
            "M7R8 central guard cannot authorize launch without the separate current-LMT/quota/fresh-slug/serial-lock coordinator"
        )
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m7r8-native-t4-whole-pipeline-cert-v1",
        "record_sha256": route["m7r7_runtime_seal_sha256"],
        "video_count": 0,
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [
            {
                "path": "dag_prompts/2026-08-23-model-clade-renewal-v1/m7r7_native_runtime_tuple_diagnostic/M7R7_DOWNLOADED_RUNTIME_SEAL.json",
                "sha256": route["m7r7_runtime_seal_sha256"],
            }
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=observed_step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m7r13_provider_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate the exact train-only M7R13 full-21 provider certificate."""

    failures: list[str] = []
    route = M7R13_PROVIDER_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "M7R13 full21 V32 M6 provider cert v1",
    }
    if metadata != expected_metadata:
        failures.append("M7R13 exact saved-script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m7r13-central-prelaunch.v1",
        "execution_profile": M7R13_PROVIDER_CERT_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M7R13_FULL21_PROVIDER_CERTIFICATION_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "m7r12_code_sha256": route["m7r12_code_sha256"],
        "m7r12_freeze_sha256": route["m7r12_freeze_sha256"],
        "m7r12_canary_sha256": route["m7r12_canary_sha256"],
        "m7r12_full_profile_sha256": route["m7r12_full_profile_sha256"],
        "fit_manifest_sha256": route["fit_manifest_sha256"],
        "m7r7_runtime_seal_sha256": route["m7r7_runtime_seal_sha256"],
        "m7r7_runtime_tuple_sha256": route["m7r7_runtime_tuple_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "transport_equivalence_sha256": route["transport_equivalence_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "ship_profile_sha256": route["ship_profile_sha256"],
        "downloaded_evidence_sealer_sha256": route[
            "downloaded_evidence_sealer_sha256"
        ],
        "lmt_sha256": route["lmt_sha256"],
        "provider_prefix_sha256": route["provider_prefix_sha256"],
        "selected_fit_identity_count": 21,
        "role_count": 4,
        "whole_run_hard_bound_seconds": 5400,
        "expected_version": 1,
        "one_shot": True,
        "provider_certification_only": True,
        "full_census_permitted": False,
        "action_supply_evaluation_permitted": False,
        "training_permitted": False,
        "scoring_permitted": False,
        "submission_output_permitted": False,
        "production_output_permitted": False,
        "downloaded_evidence_required_for_any_successor": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M7R13 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M7R13 central contract has an arbitrary surface")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M7R13 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M7R13 exact metadata hash drift")
    freeze = kernel_dir / "M7R13_PACKAGE_FREEZE.json"
    if (
        not freeze.is_file()
        or freeze.is_symlink()
        or sha256(freeze) != route["package_freeze_sha256"]
    ):
        failures.append("M7R13 package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M7R13 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M7R13 package inventory drift")

    node = (
        project_root
        / "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r13_full21_provider_cert"
    )
    artifacts = (
        (node.parent / "m7r12_postilp_fit_census/package_v1/m7r12_v32_postilp_fit_census.py", route["m7r12_code_sha256"], "M7R12 code"),
        (node.parent / "m7r12_postilp_fit_census/package_v1/PACKAGE_FREEZE.json", route["m7r12_freeze_sha256"], "M7R12 freeze"),
        (node.parent / "m7r12_postilp_fit_census/SELECTED_TWO_CANARY_TERMINAL.json", route["m7r12_canary_sha256"], "M7R12 canary"),
        (node.parent / "m7r12_postilp_fit_census/FULL_21_CENSUS_PROFILE.json", route["m7r12_full_profile_sha256"], "M7R12 full profile"),
        (project_root / "dag_prompts/2026-08-20-alpha-funnel-v1/runs/manifests/fit-groups.json", route["fit_manifest_sha256"], "selected-fit manifest"),
        (node.parent / "m7r7_native_runtime_tuple_diagnostic/M7R7_DOWNLOADED_RUNTIME_SEAL.json", route["m7r7_runtime_seal_sha256"], "M7R7 runtime seal"),
        (node / "CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "M7R13 candidate inventory"),
        (node / "M7R13_TRANSPORT_EQUIVALENCE.json", route["transport_equivalence_sha256"], "M7R13 transport proof"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M7R13 experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "M7R13 plan validation"),
        (node / "M7R13_SHIP_PROFILE.json", route["ship_profile_sha256"], "M7R13 ship profile"),
        (node / "seal_downloaded_m7r13_evidence.py", route["downloaded_evidence_sealer_sha256"], "M7R13 evidence sealer"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"{label} exact artifact drift")
    observed_step0_line = None
    try:
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PROVIDER_PREFIX_B64"
                for target in node.targets
            )
        )
        encoded = ast.literal_eval(assignment.value)
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        parent_path = artifacts[0][0]
        parent = parent_path.read_text(encoding="utf-8")
        boundary = "# ---- M7R12 POST-ILP FIT CENSUS TERMINAL ----\n"
        if parent.count(boundary) != 1 or decoded != parent.split(boundary, 1)[0]:
            raise GuardError("M7R13 embedded provider prefix is not exact M7R12 bytes")
        if hashlib.sha256(decoded.encode()).hexdigest() != route["provider_prefix_sha256"]:
            raise GuardError("M7R13 embedded provider-prefix hash drift")
        if boundary in decoded or "M7R12_CENSUS_COMPLETE" in decoded:
            raise GuardError("M7R13 embedded child crosses the census boundary")
        required = (
            'TRAIN_ROOT = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")',
            'ROLE_ORDER = ("tiny", "stratified", "stress", "expansive")',
            "WHOLE_RUN_HARD_BOUND_SECONDS = 5400",
            '"python": "3.12.13"',
            '"torch": "2.10.0+cu128"',
            '"torch_cuda": "12.8"',
            '"physical_gpu_count": 2',
            '"application_cuda_visible_devices": "0"',
            '"full_census_executed": False',
            '"action_supply_evaluated": False',
            '"training": False',
            '"scoring": False',
            '"submission_output": False',
            "M7R13_PROVIDER_CERT_EVIDENCE.zip",
        )
        if any(marker not in source for marker in required):
            raise GuardError("M7R13 required runtime/firewall marker drift")
        if any(marker in source for marker in ("44b6_6bba", "competition_test")):
            raise GuardError("M7R13 forbidden identity surface present")
        main_functions = [
            value
            for value in tree.body
            if isinstance(value, ast.FunctionDef) and value.name == "main"
        ]
        if len(main_functions) != 1:
            raise GuardError("M7R13 main function cardinality drift")
        calls = [
            value
            for value in ast.walk(main_functions[0])
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
        ]
        step0_calls = [value for value in calls if value.func.id == "_runtime_step0"]
        auth_calls = [value for value in calls if value.func.id == "_authenticate_full_population"]
        role_calls = [value for value in calls if value.func.id == "_run_role"]
        if (
            len(step0_calls) != 1
            or len(auth_calls) != 1
            or len(role_calls) != 1
            or not step0_calls[0].lineno < auth_calls[0].lineno < role_calls[0].lineno
        ):
            raise GuardError("M7R13 Step0/full-population/model boundary order drift")
        observed_step0_line = step0_calls[0].lineno
    except (GuardError, SyntaxError, ValueError, StopIteration) as exc:
        failures.append(str(exc))
    if phase == "launch":
        failures.append(
            "M7R13 central guard requires the separate current-LMT/quota/fresh-slug/serial-lock coordinator"
        )
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m7r13-full21-provider-cert-v1",
        "record_sha256": route["m7r12_canary_sha256"],
        "video_count": 0,
        "identities": [],
        "full21_path_identity_count": 21,
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [
            {"path": _relative(path, project_root, label), "sha256": expected}
            for path, expected, label in artifacts
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=observed_step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m7r14_raw_metadata_transport(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate the exact M7R14 raw-metadata-only transport successor."""

    failures: list[str] = []
    coordination_released = (
        contract.get("execution_profile")
        == M7R14_RAW_METADATA_LAUNCH_EXECUTION_PROFILE
    )
    route = (
        M7R14_RAW_METADATA_LAUNCH_ROUTE
        if coordination_released
        else M7R14_RAW_METADATA_ROUTE
    )
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "M7R14 full21 V32 M6 raw metadata repair cert v1",
    }
    if metadata != expected_metadata:
        failures.append("M7R14 exact saved-script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m7r14-central-prelaunch.v1",
        "execution_profile": (
            M7R14_RAW_METADATA_LAUNCH_EXECUTION_PROFILE
            if coordination_released
            else M7R14_RAW_METADATA_EXECUTION_PROFILE
        ),
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M7R14_FULL21_RAW_METADATA_TRANSPORT_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "parent_m7r13_code_sha256": route["parent_m7r13_code_sha256"],
        "parent_m7r13_terminal_sha256": route["parent_m7r13_terminal_sha256"],
        "parent_m7r13_input_auth_sha256": route["parent_m7r13_input_auth_sha256"],
        "fit_manifest_sha256": route["fit_manifest_sha256"],
        "dataset_registry_sha256": route["dataset_registry_sha256"],
        "raw_metadata_manifest_sha256": route["raw_metadata_manifest_sha256"],
        "causal_audit_sha256": route["causal_audit_sha256"],
        "transport_equivalence_sha256": route["transport_equivalence_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "ship_profile_sha256": route["ship_profile_sha256"],
        "provider_prefix_sha256": route["provider_prefix_sha256"],
        "lmt_sha256": route["lmt_sha256"],
        "selected_fit_identity_count": 21,
        "raw_metadata_record_count": 42,
        "changed_child_assignment_count": 1,
        "role_count": 4,
        "whole_run_hard_bound_seconds": 5400,
        "expected_version": 1,
        "one_shot": True,
        "provider_certification_only": True,
        "full_census_permitted": False,
        "action_supply_evaluation_permitted": False,
        "training_permitted": False,
        "scoring_permitted": False,
        "submission_output_permitted": False,
        "production_output_permitted": False,
        "coordination_status": (
            "PASS_M10R5_TERMINAL_RELEASED"
            if coordination_released
            else "BLOCKED_ACTIVE_M10R5"
        ),
        "launch_permitted": coordination_released,
    }
    if coordination_released:
        exact_contract.update(
            {
                "m10r5_terminal_seal_sha256": route[
                    "m10r5_terminal_seal_sha256"
                ],
                "m10r5_terminal_classification_sha256": route[
                    "m10r5_terminal_classification_sha256"
                ],
            }
        )
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M7R14 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M7R14 central contract has an arbitrary surface")
    if (
        registry.get("kernel_ownership", {}).get(route["kernel_id"])
        != "M7R14_FULL21_RAW_METADATA_TRANSPORT_ONLY"
    ):
        failures.append("M7R14 kernel ownership is not exclusively registered")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M7R14 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M7R14 exact metadata hash drift")
    freeze = kernel_dir / "M7R14_PACKAGE_FREEZE.json"
    if (
        not freeze.is_file()
        or freeze.is_symlink()
        or sha256(freeze) != route["package_freeze_sha256"]
    ):
        failures.append("M7R14 package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M7R14 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M7R14 package inventory drift")

    node = (
        project_root
        / "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r14_full21_raw_metadata_transport"
    )
    parent = node.parent / "m7r13_full21_provider_cert"
    artifacts = [
        (parent / "package_v1/kernel.py", route["parent_m7r13_code_sha256"], "M7R13 parent code"),
        (parent / "M7R13_TERMINAL_SEAL.json", route["parent_m7r13_terminal_sha256"], "M7R13 terminal"),
        (parent / "provider_output_v1/m7r13_provider_cert/M7R13_FULL21_INPUT_AUTH.json", route["parent_m7r13_input_auth_sha256"], "M7R13 input auth"),
        (project_root / "dag_prompts/2026-08-20-alpha-funnel-v1/runs/manifests/fit-groups.json", route["fit_manifest_sha256"], "selected-fit manifest"),
        (project_root / "reports/g02/dataset_registry_v1.json", route["dataset_registry_sha256"], "dataset registry"),
        (node / "M7R14_FULL21_RAW_METADATA.json", route["raw_metadata_manifest_sha256"], "M7R14 raw metadata"),
        (node / "M7R14_CAUSAL_AUDIT.json", route["causal_audit_sha256"], "M7R14 causal audit"),
        (node / "M7R14_TRANSPORT_EQUIVALENCE.json", route["transport_equivalence_sha256"], "M7R14 transport proof"),
        (node / "CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "M7R14 candidate inventory"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M7R14 experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "M7R14 plan validation"),
        (node / "M7R14_SHIP_PROFILE.json", route["ship_profile_sha256"], "M7R14 ship profile"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    ]
    if coordination_released:
        m10r5 = node.parent / "m10r5_native_t4_provider_cert"
        artifacts.extend(
            [
                (
                    m10r5 / "M10R5_TERMINAL_SEAL.json",
                    route["m10r5_terminal_seal_sha256"],
                    "M10R5 terminal release seal",
                ),
                (
                    m10r5 / "M10R5_TERMINAL_CLASSIFICATION.json",
                    route["m10r5_terminal_classification_sha256"],
                    "M10R5 terminal classification",
                ),
            ]
        )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"{label} exact artifact drift")
    if coordination_released:
        try:
            terminal = load_json(artifacts[-2][0])
            classification = load_json(artifacts[-1][0])
            if (
                terminal.get("schema")
                != "celltrack-alpha.m10r5-terminal-seal.v1"
                or terminal.get("provider_status") != "COMPLETE"
                or terminal.get("reservation_reconciliation", {}).get(
                    "terminal_status"
                )
                != "COMPLETE"
                or terminal.get("reservation_reconciliation", {}).get(
                    "released_max_gpu_h"
                )
                != 5.3
            ):
                raise GuardError("M7R14 M10R5 terminal release semantics drift")
            if (
                classification.get("status")
                != "PASS_NATIVE_T4_SYNTHETIC_RUNTIME_CERT_ONLY"
                or classification.get("quota_reconciliation", {}).get(
                    "active_reservations_after"
                )
                != 0
                or classification.get("quota_reconciliation", {}).get(
                    "authoritative_remaining_hours"
                )
                != 23.22
            ):
                raise GuardError("M7R14 M10R5 coordination release drift")
        except (GuardError, OSError, json.JSONDecodeError, TypeError) as exc:
            failures.append(str(exc))

    observed_step0_line = None
    try:
        def assignment(module: ast.Module, name: str) -> ast.Assign:
            rows = [
                value
                for value in ast.walk(module)
                if isinstance(value, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in value.targets
                )
            ]
            if len(rows) != 1:
                raise GuardError(f"M7R14 assignment cardinality drift: {name}")
            return rows[0]

        def embedded(value: str) -> str:
            module = ast.parse(value)
            encoded = ast.literal_eval(assignment(module, "PROVIDER_PREFIX_B64").value)
            return base64.b64decode(encoded, validate=True).decode("utf-8")

        def raw_assignment(module: ast.Module) -> ast.Assign:
            functions = [
                value
                for value in module.body
                if isinstance(value, ast.FunctionDef)
                and value.name == "_v32_ship_step0"
            ]
            if len(functions) != 1:
                raise GuardError("M7R14 child Step0 cardinality drift")
            rows = [
                value
                for value in ast.walk(functions[0])
                if isinstance(value, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "raw_metadata"
                    for target in value.targets
                )
            ]
            if len(rows) != 1:
                raise GuardError("M7R14 child raw_metadata cardinality drift")
            return rows[0]

        def normalized_child(value: str) -> str:
            module = ast.parse(value)
            raw_assignment(module).value = ast.Constant(
                value="__RAW_METADATA_TRANSPORT_SURFACE__"
            )
            return ast.dump(module, include_attributes=False)

        child = embedded(source)
        parent_source = artifacts[0][0].read_text(encoding="utf-8")
        parent_child = embedded(parent_source)
        if hashlib.sha256(child.encode()).hexdigest() != route["provider_prefix_sha256"]:
            raise GuardError("M7R14 embedded provider-prefix hash drift")
        if normalized_child(child) != normalized_child(parent_child):
            raise GuardError("M7R14 child drift outside exact raw_metadata assignment")
        raw_manifest = json.loads(artifacts[5][0].read_text(encoding="utf-8"))
        records = raw_manifest.get("records")
        if (
            raw_manifest.get("status") != "PASS_EXACT_SELECTED_FIT_DUAL_METADATA"
            or raw_manifest.get("identity_count") != 21
            or raw_manifest.get("record_count") != 21
            or raw_manifest.get("metadata_files_per_identity") != 2
            or not isinstance(records, list)
            or len(records) != 21
        ):
            raise GuardError("M7R14 full-21 raw metadata manifest drift")
        expected_table = {
            row["dataset"]: (row["group"]["sha256"], row["array"]["sha256"])
            for row in records
        }
        observed_table = ast.literal_eval(raw_assignment(ast.parse(child)).value)
        if observed_table != expected_table or len(observed_table) != 21:
            raise GuardError("M7R14 embedded full-21 raw metadata table drift")
        parent_table = ast.literal_eval(raw_assignment(ast.parse(parent_child)).value)
        if set(parent_table) != {"44b6_d754aa59", "44b6_18ced818"}:
            raise GuardError("M7R14 parent legacy metadata surface drift")
        if set(parent_table) & set(observed_table):
            raise GuardError("M7R14 legacy metadata leaked into selected-fit repair")
        required = (
            'ROLE_ORDER = ("tiny", "stratified", "stress", "expansive")',
            "WHOLE_RUN_HARD_BOUND_SECONDS = 5400",
            '"full_census_executed": False',
            '"action_supply_evaluated": False',
            '"training": False',
            '"scoring": False',
            '"submission_output": False',
            "M7R13_PROVIDER_CERT_EVIDENCE.zip",
        )
        if any(marker not in source for marker in required):
            raise GuardError("M7R14 preserved role/output/firewall marker drift")
        if any(marker in source for marker in ("44b6_6bba", "competition_test")):
            raise GuardError("M7R14 forbidden identity surface present")
        main_functions = [
            value
            for value in tree.body
            if isinstance(value, ast.FunctionDef) and value.name == "main"
        ]
        if len(main_functions) != 1:
            raise GuardError("M7R14 main function cardinality drift")
        calls = [
            value
            for value in ast.walk(main_functions[0])
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
        ]
        step0_calls = [value for value in calls if value.func.id == "_runtime_step0"]
        auth_calls = [
            value for value in calls if value.func.id == "_authenticate_full_population"
        ]
        role_calls = [value for value in calls if value.func.id == "_run_role"]
        if (
            len(step0_calls) != 1
            or len(auth_calls) != 1
            or len(role_calls) != 1
            or not step0_calls[0].lineno < auth_calls[0].lineno < role_calls[0].lineno
        ):
            raise GuardError("M7R14 Step0/full-population/model boundary order drift")
        observed_step0_line = step0_calls[0].lineno
    except (GuardError, KeyError, SyntaxError, TypeError, ValueError) as exc:
        failures.append(str(exc))
    if phase == "launch" and not coordination_released:
        failures.append("M7R14 launch blocked while M10R5 has the active remote reservation")
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m7r14-full21-raw-metadata-transport-v1",
        "record_sha256": route["parent_m7r13_terminal_sha256"],
        "video_count": 0,
        "identities": [],
        "full21_path_identity_count": 21,
        "raw_metadata_record_count": 42,
        "coordination_status": (
            "PASS_M10R5_TERMINAL_RELEASED"
            if coordination_released
            else "BLOCKED_ACTIVE_M10R5"
        ),
        "launch_permitted": coordination_released,
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [
            {"path": _relative(path, project_root, label), "sha256": expected}
            for path, expected, label in artifacts
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=observed_step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m7r15_action_supply_census(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate the exact one-look M7R15 full-fit action-supply census."""

    failures: list[str] = []
    route = M7R15_ACTION_SUPPLY_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "m7r15-full21-action-supply-census-v1",
    }
    if metadata != expected_metadata:
        failures.append("M7R15 exact saved-script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m7r15-central-prelaunch.v1",
        "execution_profile": M7R15_ACTION_SUPPLY_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M7R15_FULL21_ONE_LOOK_ACTION_SUPPLY_CENSUS_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "child_sha256": route["child_sha256"],
        "source_equivalence_sha256": route["source_equivalence_sha256"],
        "environment_sha256": route["environment_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "ship_profile_sha256": route["ship_profile_sha256"],
        "downloaded_evidence_sealer_sha256": route[
            "downloaded_evidence_sealer_sha256"
        ],
        "noconflict_report_sha256": route["noconflict_report_sha256"],
        "local_analysis_probe_sha256": route["local_analysis_probe_sha256"],
        "m7r14_terminal_sha256": route["m7r14_terminal_sha256"],
        "m7r14_local_seal_sha256": route["m7r14_local_seal_sha256"],
        "m7r14_alias_sha256": route["m7r14_alias_sha256"],
        "m7r14_evidence_zip_sha256": route["m7r14_evidence_zip_sha256"],
        "m7r12_code_sha256": route["m7r12_code_sha256"],
        "m7r12_schema_sha256": route["m7r12_schema_sha256"],
        "m7r12_profile_sha256": route["m7r12_profile_sha256"],
        "m7r11_boundary_sha256": route["m7r11_boundary_sha256"],
        "fit_manifest_sha256": route["fit_manifest_sha256"],
        "lmt_sha256": route["lmt_sha256"],
        "selected_fit_identity_count": 21,
        "configurations": 1,
        "seed": 104729,
        "hpo": False,
        "minimum_legal_alternatives": 16,
        "threshold_looks": 1,
        "whole_run_hard_bound_seconds": 25_200,
        "expected_version": 1,
        "one_shot": True,
        "provider_certification_complete": True,
        "full_census_permitted": True,
        "action_supply_evaluation_permitted": True,
        "graph_efficacy_permitted": False,
        "training_permitted": False,
        "scoring_permitted": False,
        "outcome_access_permitted": False,
        "submission_output_permitted": False,
        "production_output_permitted": False,
        "launch_permitted": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M7R15 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M7R15 central contract has an arbitrary surface")
    if (
        registry.get("kernel_ownership", {}).get(route["kernel_id"])
        != "M7R15_FULL21_ONE_LOOK_ACTION_SUPPLY_CENSUS_ONLY"
    ):
        failures.append("M7R15 kernel ownership is not exclusively registered")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M7R15 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M7R15 exact metadata hash drift")
    freeze = kernel_dir / "M7R15_PACKAGE_FREEZE.json"
    if (
        not freeze.is_file()
        or freeze.is_symlink()
        or sha256(freeze) != route["package_freeze_sha256"]
    ):
        failures.append("M7R15 package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M7R15 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M7R15 package inventory drift")

    node = (
        project_root
        / "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r15_full21_action_supply_census"
    )
    m7r14 = node.parent / "m7r14_full21_raw_metadata_transport"
    m7r12 = node.parent / "m7r12_postilp_fit_census"
    m7r11 = node.parent / "m7r11_v32_prepost_inventory_order_transport"
    fit = (
        project_root
        / "dag_prompts/2026-08-20-alpha-funnel-v1/runs/manifests/fit-groups.json"
    )
    artifacts = [
        (node / "M7R15_SOURCE_EQUIVALENCE.json", route["source_equivalence_sha256"], "M7R15 source proof"),
        (node / "M7R15_ENVIRONMENT.json", route["environment_sha256"], "M7R15 environment"),
        (node / "CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "M7R15 candidate inventory"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M7R15 experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "M7R15 plan validation"),
        (node / "M7R15_SHIP_PROFILE.json", route["ship_profile_sha256"], "M7R15 ship profile"),
        (node / "seal_downloaded_m7r15_evidence.py", route["downloaded_evidence_sealer_sha256"], "M7R15 evidence sealer"),
        (node / "M7R15_NOCONFLICT_REPORT.json", route["noconflict_report_sha256"], "M7R15 no-conflict report"),
        (node / "M7R15_LOCAL_ANALYSIS_ENV_PROBE.json", route["local_analysis_probe_sha256"], "M7R15 local analysis probe"),
        (m7r14 / "M7R14_TERMINAL_SEAL.json", route["m7r14_terminal_sha256"], "M7R14 terminal seal"),
        (m7r14 / "M7R14_DOWNLOADED_EVIDENCE_SEAL.json", route["m7r14_local_seal_sha256"], "M7R14 local evidence seal"),
        (m7r14 / "M7R14_ALIAS_RECONCILIATION.json", route["m7r14_alias_sha256"], "M7R14 alias reconciliation"),
        (m7r14 / "provider_output_v1/M7R13_PROVIDER_CERT_EVIDENCE.zip", route["m7r14_evidence_zip_sha256"], "M7R14 downloaded evidence ZIP"),
        (m7r12 / "package_v1/m7r12_v32_postilp_fit_census.py", route["m7r12_code_sha256"], "M7R12 full-census source"),
        (m7r12 / "EXPORT_SCHEMA.json", route["m7r12_schema_sha256"], "M7R12 export schema"),
        (m7r12 / "FULL_21_CENSUS_PROFILE.json", route["m7r12_profile_sha256"], "M7R12 full-21 profile"),
        (m7r11 / "INTERPRETATION_ADDENDUM.json", route["m7r11_boundary_sha256"], "M7R11 boundary"),
        (fit, route["fit_manifest_sha256"], "selected-fit manifest"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    ]
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"{label} exact artifact drift")

    observed_step0_line = None
    try:
        def assignment(module: ast.Module, name: str) -> ast.Assign:
            rows = [
                value
                for value in ast.walk(module)
                if isinstance(value, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in value.targets
                )
            ]
            if len(rows) != 1:
                raise GuardError(f"M7R15 assignment cardinality drift: {name}")
            return rows[0]

        encoded = ast.literal_eval(assignment(tree, "CHILD_SOURCE_B64").value)
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        declared_child_sha = ast.literal_eval(
            assignment(tree, "CHILD_SOURCE_SHA256").value
        )
        if (
            hashlib.sha256(decoded.encode()).hexdigest() != route["child_sha256"]
            or declared_child_sha != route["child_sha256"]
        ):
            raise GuardError("M7R15 embedded child hash drift")
        m7r14_source = (m7r14 / "package_v1/kernel.py").read_text(encoding="utf-8")
        m7r14_tree = ast.parse(m7r14_source)
        provider_encoded = ast.literal_eval(
            assignment(m7r14_tree, "PROVIDER_PREFIX_B64").value
        )
        provider_prefix = base64.b64decode(provider_encoded, validate=True).decode(
            "utf-8"
        )
        m7r12_source = artifacts[13][0].read_text(encoding="utf-8")
        boundary = "# ---- M7R12 POST-ILP FIT CENSUS TERMINAL ----\n"
        if m7r12_source.count(boundary) != 1:
            raise GuardError("M7R15 M7R12 census boundary cardinality drift")
        expected_child = provider_prefix + boundary + m7r12_source.split(boundary, 1)[1]
        if decoded != expected_child:
            raise GuardError("M7R15 child is not exact M7R14 provider plus M7R12 census")
        identities = ast.literal_eval(assignment(tree, "FIT_IDENTITIES").value)
        fit_identities = json.loads(fit.read_text(encoding="utf-8"))["groups"]
        if identities != fit_identities or len(identities) != 21:
            raise GuardError("M7R15 exact selected-fit population drift")
        expected_assignments = {
            "KERNEL_ID": route["kernel_id"],
            "CANDIDATE_ID": "M7R15-FULL21-POSTILP-ACTION-SUPPLY-CENSUS-v1",
            "SEED": 104729,
            "WHOLE_RUN_HARD_BOUND_SECONDS": 25_200,
            "MINIMUM_LEGAL_ALTERNATIVES": 16,
        }
        for name, expected in expected_assignments.items():
            if ast.literal_eval(assignment(tree, name).value) != expected:
                raise GuardError(f"M7R15 frozen assignment drift: {name}")
        required = (
            'environment.pop("KAGGLE_PREFLIGHT_MODE", None)',
            'environment["CUDA_VISIBLE_DEVICES"] = "0"',
            '"threshold_looks": 1',
            '"full_census_executed": True',
            '"action_supply_evaluated": True',
            '"training": False',
            '"graph_efficacy_evaluated": False',
            '"scoring": False',
            '"outcomes_opened": False',
            '"submission_output": False',
            '"production_enabled": False',
            "M7R15_ACTION_SUPPLY_EVIDENCE.zip",
        )
        if any(marker not in source for marker in required):
            raise GuardError("M7R15 runtime/science/output firewall marker drift")
        if any(
            marker in (source + decoded).lower()
            for marker in ("44b6_6bba", "/test/")
        ):
            raise GuardError("M7R15 forbidden namespace/output surface present")
        subprocess_calls = [
            value
            for value in ast.walk(tree)
            if isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "subprocess"
            and value.func.attr == "run"
        ]
        if len(subprocess_calls) != 1:
            raise GuardError("M7R15 subprocess execution cardinality drift")
        keywords = {value.arg: value.value for value in subprocess_calls[0].keywords}
        if (
            not isinstance(keywords.get("timeout"), ast.Name)
            or keywords["timeout"].id != "WHOLE_RUN_HARD_BOUND_SECONDS"
            or ast.literal_eval(keywords["check"]) is not False
        ):
            raise GuardError("M7R15 subprocess hard-bound drift")
        main_functions = [
            value
            for value in tree.body
            if isinstance(value, ast.FunctionDef) and value.name == "main"
        ]
        if len(main_functions) != 1:
            raise GuardError("M7R15 main function cardinality drift")
        calls = [
            value
            for value in ast.walk(main_functions[0])
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
        ]
        step0_calls = [value for value in calls if value.func.id == "runtime_step0"]
        materialize_calls = [
            value for value in calls if value.func.id == "materialize_child"
        ]
        if (
            len(step0_calls) != 1
            or len(materialize_calls) != 1
            or not step0_calls[0].lineno < materialize_calls[0].lineno
        ):
            raise GuardError("M7R15 Step0/materialization order drift")
        observed_step0_line = step0_calls[0].lineno
        terminal = load_json(artifacts[9][0])
        local_seal = load_json(artifacts[10][0])
        alias = load_json(artifacts[11][0])
        no_conflict = load_json(artifacts[7][0])
        probe = load_json(artifacts[8][0])
        if (
            terminal.get("provider_status") != "COMPLETE"
            or terminal.get("scientific_evidence_status")
            != "PASS_EXACT_FULL21_PROVIDER_CERTIFICATION_ONLY"
            or terminal.get("full_census_executed") is not False
            or terminal.get("action_supply_evaluated") is not False
            or local_seal.get("status")
            != "PASS_EXACT_FULL21_PROVIDER_CERTIFICATION_ONLY"
            or local_seal.get("archive_sha256") != route["m7r14_evidence_zip_sha256"]
            or alias.get("status")
            != "PASS_EXACT_M7R14_ONLY_ALIAS_MONITORING_ROUTE"
            or no_conflict.get("status")
            != "PASS_NO_DEPENDENCY_OR_RUNTIME_TUPLE_CHANGE"
            or no_conflict.get("target_status")
            != "PASS_INHERITED_EXACT_TERMINAL_M7R14_NATIVE_TUPLE"
            or probe.get("status") != "PASS"
            or probe.get("platform", {}).get("machine") != "arm64"
            or probe.get("target_python", {}).get("version") != "3.14.6"
        ):
            raise GuardError("M7R15 provider/environment evidence semantics drift")
    except (GuardError, KeyError, SyntaxError, TypeError, ValueError) as exc:
        failures.append(str(exc))

    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m7r15-full21-one-look-action-supply-census-v1",
        "record_sha256": route["m7r14_terminal_sha256"],
        "video_count": 0,
        "identities": [],
        "full_selected_fit_identity_count": 21,
        "configurations": 1,
        "seed": 104729,
        "hpo": False,
        "minimum_legal_alternatives": 16,
        "threshold_looks": 1,
        "whole_run_hard_bound_seconds": 25_200,
        "provider_certification_complete": True,
        "full_census_pending": True,
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "post_remote_census": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "scientific_disposition": "PENDING_ONE_LOOK_CENSUS",
            "production_or_score_permitted": False,
        },
        "artifacts": [
            {"path": _relative(path, project_root, label), "sha256": expected}
            for path, expected, label in artifacts
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=observed_step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m8r13_native_t4_exporter_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate the exact M8R12-science, provider-kernelized M8R13 certificate."""

    failures: list[str] = []
    route = M8R13_NATIVE_T4_EXPORTER_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "onebitbrain/m8r12-native-t4-exporter-carrier-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "m8r13-native-t4-exporter-cert-v1",
    }
    if metadata != expected_metadata:
        failures.append("M8R13 exact private saved-script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m8r13-central-prelaunch.v1",
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "carrier_action_execution_sha256": route["carrier_action_execution_sha256"],
        "carrier_authorization_sha256": route["carrier_authorization_sha256"],
        "carrier_roundtrip_sha256": route["carrier_roundtrip_sha256"],
        "certification_only": True,
        "code_sha256": route["code_sha256"],
        "completed_units": 200,
        "configuration_count": 1,
        "downloaded_evidence_required_for_any_successor": True,
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "evidence_contract_sha256": route["evidence_contract_sha256"],
        "execution_profile": M8R13_NATIVE_T4_EXPORTER_EXECUTION_PROFILE,
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "frames_per_identity": 100,
        "hpo": False,
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
        "kernel_id": route["kernel_id"],
        "lane_owner": "M8R13_SELECTED_TWO_EXPORTER_CERTIFICATION_ONLY",
        "lmt_sha256": route["lmt_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "model_execution": True,
        "noconflict_report_sha256": route["noconflict_report_sha256"],
        "one_shot": True,
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "parent_kernel_sha256": route["parent_kernel_sha256"],
        "parent_package_freeze_sha256": route["parent_package_freeze_sha256"],
        "production_output_permitted": False,
        "provider_profile_sha256": route["provider_profile_sha256"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "scoring_permitted": False,
        "seed": 104729,
        "static_readiness_sha256": route["static_readiness_sha256"],
        "submission_output_permitted": False,
        "target_input": True,
        "training_permitted": False,
        "transport_equivalence_sha256": route["transport_equivalence_sha256"],
        "whole_run_hard_bound_seconds": 27900,
    }
    if contract != exact_contract:
        failures.append("M8R13 exact central contract drift")
    canonical_registry = (
        project_root
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json"
    ).resolve()
    if registry_path is None or registry_path.resolve() != canonical_registry:
        failures.append("M8R13 registry is not the canonical repository registry")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != (
        "M8R13_SELECTED_TWO_EXPORTER_CERTIFICATION_ONLY"
    ):
        failures.append("M8R13 exact kernel ownership drift")
    if registry.get("dataset_ownership", {}).get(
        "onebitbrain/m8r12-native-t4-exporter-carrier-v1"
    ) != "M8R13_SELECTED_TWO_EXPORTER_CERTIFICATION_ONLY":
        failures.append("M8R13 exact private-carrier ownership drift")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M8R13 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M8R13 exact metadata hash drift")
    freeze = kernel_dir / "M8R13_PACKAGE_FREEZE.json"
    if (
        not freeze.is_file()
        or freeze.is_symlink()
        or sha256(freeze) != route["package_freeze_sha256"]
    ):
        failures.append("M8R13 exact package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M8R13 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M8R13 exact ship package inventory drift")

    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m8r13_native_t4_exporter_kernelized"
    parent = campaign / "m8r12_native_t4_exporter_cert"
    artifacts = (
        (parent / "package_v1/kernel.py", route["parent_kernel_sha256"], "M8R12 parent kernel"),
        (parent / "package_v1/M8R12_PACKAGE_FREEZE.json", route["parent_package_freeze_sha256"], "M8R12 parent freeze"),
        (parent / "EVIDENCE_ZIP_CONTRACT.json", route["evidence_contract_sha256"], "M8R12 evidence contract"),
        (parent / "M8R12_PROVIDER_ROUNDTRIP.json", route["carrier_roundtrip_sha256"], "private carrier roundtrip"),
        (parent / "M8R12_PROVIDER_ACTION_EXECUTION.json", route["carrier_action_execution_sha256"], "private carrier execution"),
        (parent / "M8R12_PROVIDER_ACTION_AUTHORIZATION.json", route["carrier_authorization_sha256"], "private carrier authorization"),
        (node / "M8R13_TRANSPORT_EQUIVALENCE.json", route["transport_equivalence_sha256"], "M8R13 transport equivalence"),
        (node / "M8R13_NOCONFLICT_REPORT.json", route["noconflict_report_sha256"], "M8R13 dependency closure"),
        (node / "M8R13_PROVIDER_PROFILE.json", route["provider_profile_sha256"], "M8R13 provider profile"),
        (node / "M8R13_CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "M8R13 candidate inventory"),
        (node / "M8R13_STATIC_READINESS.json", route["static_readiness_sha256"], "M8R13 static readiness"),
        (parent / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M8R12 frozen experiment plan"),
        (parent / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "M8R12 plan validation"),
        (node / "seal_downloaded_m8r13_evidence.py", route["downloaded_evidence_sealer_sha256"], "M8R13 evidence sealer"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    visual_artifacts: list[dict[str, str]] = []
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M8R13 exact artifact drift: {label}")
        visual_artifacts.append(
            {"path": _relative(path, project_root, label), "sha256": expected}
        )

    try:
        embedded = _literal_assignment(tree, "PARENT_SOURCE", "M8R13 embedded parent")
        if not isinstance(embedded, str):
            raise GuardError("M8R13 embedded parent is not a literal string")
        if hashlib.sha256(embedded.encode("utf-8")).hexdigest() != route["parent_kernel_sha256"]:
            raise GuardError("M8R13 embedded parent hash drift")
        parent_path = parent / "package_v1/kernel.py"
        if embedded.encode("utf-8") != parent_path.read_bytes():
            raise GuardError("M8R13 embedded parent bytes differ from M8R12")
        if _literal_assignment(tree, "PARENT_KERNEL_SOURCE_SHA256", "M8R13 parent hash") != route["parent_kernel_sha256"]:
            raise GuardError("M8R13 parent hash literal drift")
        if _literal_assignment(tree, "WHOLE_RUN_HARD_BOUND_SECONDS", "M8R13 bound") != 27900:
            raise GuardError("M8R13 whole-run bound drift")
        mode_at = source.index('os.environ["KAGGLE_PREFLIGHT_MODE"] = "expansive_both"')
        units_at = source.index('os.environ["KAGGLE_PREFLIGHT_UNITS"] = "100"')
        exec_at = source.index("exec(  # noqa: S102")
        if not (mode_at < exec_at and units_at < exec_at):
            raise GuardError("M8R13 internal fixed role is not established before parent execution")
        required = (
            'sys.argv = ["kernel.py"]',
            "signal.setitimer(signal.ITIMER_REAL, float(WHOLE_RUN_HARD_BOUND_SECONDS))",
            '"status": "PASS_EXACT_M8R12_SCIENCE_PROVIDER_KERNELIZED"',
            '"completed_units": 200',
            '"frames_per_identity": 100',
            '"graph_efficacy_test_performed": False',
            '"scoring": False',
            '"training": False',
            '"submission_output": False',
            '"protected_data_opened": False',
            '"promotion_claim": False',
        )
        if any(marker not in source for marker in required):
            raise GuardError("M8R13 execution-control/output marker drift")
        if "kaggle competitions submit" in source or "competition_submit" in source:
            raise GuardError("M8R13 forbidden submission action surface")
    except (GuardError, SyntaxError, ValueError) as exc:
        failures.append(str(exc))

    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m8r13-native-t4-exporter-cert-v1",
        "record_sha256": route["transport_equivalence_sha256"],
        "video_count": 0,
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
        "target_input": True,
        "frames_per_identity": 100,
        "completed_units": 200,
        "whole_run_hard_bound_seconds": 27900,
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": visual_artifacts,
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m8r14_postilp_target_feature_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the fresh post-ILP-bound M8R14 provider certificate."""

    failures: list[str] = []
    route = M8R14_POSTILP_TARGET_FEATURE_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "onebitbrain/m8r12-native-t4-exporter-carrier-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "m8r14-postilp-target-feature-cert-v1",
    }
    if metadata != expected_metadata:
        failures.append("M8R14 exact private saved-script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m8r14-central-prelaunch.v1",
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "certification_only": True,
        "code_sha256": route["code_sha256"],
        "completed_units": 200,
        "configuration_count": 1,
        "downloaded_evidence_required_for_any_successor": True,
        "downloaded_evidence_sealer_sha256": route[
            "downloaded_evidence_sealer_sha256"
        ],
        "evidence_contract_sha256": route["evidence_contract_sha256"],
        "execution_profile": M8R14_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE,
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "frames_per_identity": 100,
        "hpo": False,
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
        "kernel_id": route["kernel_id"],
        "lane_owner": "M8R14_POSTILP_TARGET_FEATURE_CERTIFICATION_ONLY",
        "lmt_sha256": route["lmt_sha256"],
        "local_falsifier_sha256": route["local_falsifier_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "model_execution": True,
        "noconflict_report_sha256": route["noconflict_report_sha256"],
        "one_shot": True,
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "postilp_contract_sha256": route["postilp_contract_sha256"],
        "production_output_permitted": False,
        "provider_profile_sha256": route["provider_profile_sha256"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "scoring_permitted": False,
        "seed": 104729,
        "source_equivalence_sha256": route["source_equivalence_sha256"],
        "static_readiness_sha256": route["static_readiness_sha256"],
        "submission_output_permitted": False,
        "target_input": True,
        "training_permitted": False,
        "whole_run_hard_bound_seconds": 27900,
    }
    if contract != exact_contract:
        failures.append("M8R14 exact central contract drift")
    canonical_registry = (
        project_root
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json"
    ).resolve()
    if registry_path is None or registry_path.resolve() != canonical_registry:
        failures.append("M8R14 registry is not the canonical repository registry")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != (
        "M8R14_POSTILP_TARGET_FEATURE_CERTIFICATION_ONLY"
    ):
        failures.append("M8R14 exact kernel ownership drift")
    if registry.get("dataset_ownership", {}).get(
        "onebitbrain/m8r12-native-t4-exporter-carrier-v1"
    ) != "M8R13_SELECTED_TWO_EXPORTER_CERTIFICATION_ONLY":
        failures.append("M8R14 immutable predecessor-carrier ownership drift")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M8R14 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M8R14 exact metadata hash drift")
    freeze = kernel_dir / "M8R14_PACKAGE_FREEZE.json"
    if (
        not freeze.is_file()
        or freeze.is_symlink()
        or sha256(freeze) != route["package_freeze_sha256"]
    ):
        failures.append("M8R14 exact package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M8R14 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M8R14 exact ship package inventory drift")

    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m8r14_postilp_target_feature_export"
    m7r12 = campaign / "m7r12_postilp_fit_census"
    m8r12 = campaign / "m8r12_native_t4_exporter_cert"
    m8r13 = campaign / "m8r13_native_t4_exporter_kernelized"
    artifacts = (
        (
            node / "CANDIDATE_INVENTORY.json",
            route["candidate_inventory_sha256"],
            "M8R14 candidate inventory",
        ),
        (
            node / "SOURCE_EQUIVALENCE.json",
            route["source_equivalence_sha256"],
            "M8R14 source equivalence",
        ),
        (
            node / "LOCAL_POSTILP_FALSIFIER.json",
            route["local_falsifier_sha256"],
            "M8R14 local post-ILP falsifier",
        ),
        (
            node / "NOCONFLICT_REPORT.json",
            route["noconflict_report_sha256"],
            "M8R14 dependency closure",
        ),
        (
            node / "PROVIDER_PROFILE.json",
            route["provider_profile_sha256"],
            "M8R14 provider profile",
        ),
        (
            node / "STATIC_READINESS.json",
            route["static_readiness_sha256"],
            "M8R14 static readiness",
        ),
        (
            node / "EVIDENCE_ZIP_CONTRACT.json",
            route["evidence_contract_sha256"],
            "M8R14 evidence contract",
        ),
        (
            node / "EXPERIMENT_PLAN.json",
            route["experiment_plan_sha256"],
            "M8R14 frozen experiment plan",
        ),
        (
            node / "EXPERIMENT_PLAN_VALIDATION.json",
            route["experiment_validation_sha256"],
            "M8R14 plan validation",
        ),
        (
            node / "seal_downloaded_m8r14_evidence.py",
            route["downloaded_evidence_sealer_sha256"],
            "M8R14 evidence sealer",
        ),
        (
            m7r12 / "m7r12_exporter_contract.py",
            route["postilp_contract_sha256"],
            "exact M7R12 post-ILP contract",
        ),
        (
            m8r13 / "M8R13_TERMINAL_ADJUDICATION.json",
            route["r13_terminal_sha256"],
            "M8R13 causal terminal",
        ),
        (
            m8r12 / "provider_roundtrip_download/M8R12_CARRIER_MANIFEST.json",
            route["carrier_manifest_sha256"],
            "exact private carrier manifest",
        ),
        (
            m8r12 / "provider_roundtrip_download/M8R6_WEIGHTS.pt",
            route["checkpoint_sha256"],
            "exact M8 checkpoint",
        ),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    visual_artifacts: list[dict[str, str]] = []
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M8R14 exact artifact drift: {label}")
        visual_artifacts.append(
            {"path": _relative(path, project_root, label), "sha256": expected}
        )

    try:
        embedded = _literal_assignment(tree, "COMBINED_SOURCE", "M8R14 combined source")
        if not isinstance(embedded, str):
            raise GuardError("M8R14 embedded combined source is not a literal string")
        if hashlib.sha256(embedded.encode("utf-8")).hexdigest() != route[
            "combined_source_sha256"
        ]:
            raise GuardError("M8R14 embedded combined source hash drift")
        if "# ---- M6 AUDIT-ONLY TERMINAL EXPORT ----" in embedded:
            raise GuardError("M8R14 retained the dead M6 final-vocabulary tail")
        if "# ---- M8R14 FRESH POST-ILP TARGET-FEATURE EXPORT ----" not in embedded:
            raise GuardError("M8R14 fresh post-ILP tail is absent")
        if _literal_assignment(
            tree, "COMBINED_SOURCE_SHA256", "M8R14 combined source hash"
        ) != route["combined_source_sha256"]:
            raise GuardError("M8R14 combined source hash literal drift")
        if _literal_assignment(
            tree, "WHOLE_RUN_HARD_BOUND_SECONDS", "M8R14 bound"
        ) != 27900:
            raise GuardError("M8R14 whole-run bound drift")
        mode_at = source.index('os.environ["KAGGLE_PREFLIGHT_MODE"] = "expansive_both"')
        units_at = source.index('os.environ["KAGGLE_PREFLIGHT_UNITS"] = "100"')
        exec_at = source.index("exec(  # noqa: S102")
        if not (mode_at < exec_at and units_at < exec_at):
            raise GuardError("M8R14 fixed role is not established before execution")
        required = (
            'sys.argv = ["kernel.py"]',
            "M8R14_WHOLE_STARTED_MONOTONIC",
            '"status": "PASS_POSTILP_TARGET_FEATURE_CERT_ONLY"',
            '"completed_units": 200',
            '"legal_distinct_alternatives": int(terminal["legal_distinct_alternatives"])',
            '"graph_efficacy_test_performed": False',
            '"scoring": False',
            '"training": False',
            '"submission_output": False',
            '"protected_data_opened": False',
            '"promotion_claim": False',
        )
        if any(marker not in source for marker in required):
            raise GuardError("M8R14 execution-control/output marker drift")
        if "kaggle competitions submit" in source or "competition_submit" in source:
            raise GuardError("M8R14 forbidden submission action surface")
    except (GuardError, SyntaxError, ValueError) as exc:
        failures.append(str(exc))

    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m8r14-postilp-target-feature-cert-v1",
        "record_sha256": route["local_falsifier_sha256"],
        "video_count": 0,
        "identities": ["train:44b6_cf2536e8", "train:44b6_3bb3690f"],
        "target_input": True,
        "frames_per_identity": 100,
        "completed_units": 200,
        "whole_run_hard_bound_seconds": 27900,
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": visual_artifacts,
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m8r15_postilp_target_feature_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate the registered-module M8R15 provider certificate only."""

    failures: list[str] = []
    route = M8R15_POSTILP_TARGET_FEATURE_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "onebitbrain/m8r12-native-t4-exporter-carrier-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "m8r15-postilp-target-feature-cert-v1",
    }
    if metadata != expected_metadata:
        failures.append("M8R15 exact private saved-script metadata drift")
    if contract != route:
        failures.append("M8R15 exact central contract drift")
    canonical_registry = (
        project_root
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
        "KAGGLE_FAILURE_REGISTRY.json"
    ).resolve()
    if registry_path is None or registry_path.resolve() != canonical_registry:
        failures.append("M8R15 registry is not the canonical repository registry")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != route[
        "lane_owner"
    ]:
        failures.append("M8R15 exact kernel ownership drift")
    if registry.get("dataset_ownership", {}).get(
        "onebitbrain/m8r12-native-t4-exporter-carrier-v1"
    ) != "M8R13_SELECTED_TWO_EXPORTER_CERTIFICATION_ONLY":
        failures.append("M8R15 immutable carrier ownership drift")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M8R15 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M8R15 exact metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M8R15 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route[
        "package_inventory_sha256"
    ]:
        failures.append("M8R15 exact ship package inventory drift")

    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m8r15_module_registration_transport"
    m7r12 = campaign / "m7r12_postilp_fit_census"
    m8r12 = campaign / "m8r12_native_t4_exporter_cert"
    m8r14 = campaign / "m8r14_postilp_target_feature_export"
    artifacts = (
        (
            node / "M8R15_PACKAGE_FREEZE.json",
            route["package_freeze_sha256"],
            "M8R15 package freeze",
        ),
        (
            node / "SOURCE_EQUIVALENCE.json",
            route["source_equivalence_sha256"],
            "M8R15 source equivalence",
        ),
        (
            node / "SHIP_CERTIFICATION.json",
            route["ship_certification_sha256"],
            "M8R15 exact ship certification",
        ),
        (
            node / "NOCONFLICT_REPORT.json",
            route["noconflict_report_sha256"],
            "M8R15 dependency closure",
        ),
        (
            node / "PROVIDER_PROFILE.json",
            route["provider_profile_sha256"],
            "M8R15 provider profile",
        ),
        (
            node / "STATIC_READINESS.json",
            route["static_readiness_sha256"],
            "M8R15 static readiness",
        ),
        (
            node / "EVIDENCE_ZIP_CONTRACT.json",
            route["evidence_contract_sha256"],
            "M8R15 evidence contract",
        ),
        (
            node / "EXPERIMENT_PLAN.json",
            route["experiment_plan_sha256"],
            "M8R15 frozen experiment plan",
        ),
        (
            node / "EXPERIMENT_PLAN_VALIDATION.json",
            route["experiment_validation_sha256"],
            "M8R15 plan validation",
        ),
        (
            node / "seal_downloaded_m8r15_evidence.py",
            route["downloaded_evidence_sealer_sha256"],
            "M8R15 downloaded-evidence sealer",
        ),
        (
            m7r12 / "m7r12_exporter_contract.py",
            route["postilp_contract_sha256"],
            "exact M7R12 post-ILP contract",
        ),
        (
            m8r12 / "provider_roundtrip_download/M8R6_WEIGHTS.pt",
            route["checkpoint_sha256"],
            "exact M8 checkpoint",
        ),
        (
            m8r14 / "TERMINAL_PROVIDER.json",
            route["parent_terminal_sha256"],
            "M8R14 causal provider terminal",
        ),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    visual_artifacts: list[dict[str, str]] = []
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M8R15 exact artifact drift: {label}")
        visual_artifacts.append(
            {"path": _relative(path, project_root, label), "sha256": expected}
        )

    ship_path = node / "SHIP_CERTIFICATION.json"
    try:
        ship = load_json(ship_path)
        if not (
            ship.get("status") == "ELIGIBLE"
            and ship.get("eligible") is True
            and ship.get("autonomous_launch_eligible") is True
            and ship.get("receipt_content_sha256")
            == route["ship_receipt_content_sha256"]
            and ship.get("runtime_projection", {}).get("status") == "PASS"
        ):
            raise GuardError("M8R15 exact ship certificate semantics drift")
        embedded = _literal_assignment(tree, "COMBINED_SOURCE", "M8R15 combined source")
        if not isinstance(embedded, str):
            raise GuardError("M8R15 embedded combined source is not literal")
        if hashlib.sha256(embedded.encode("utf-8")).hexdigest() != route[
            "combined_source_sha256"
        ]:
            raise GuardError("M8R15 embedded combined source byte drift")
        if _literal_assignment(
            tree, "COMBINED_SOURCE_SHA256", "M8R15 combined source digest"
        ) != route["combined_source_sha256"]:
            raise GuardError("M8R15 embedded combined source digest drift")
        for literal_name in ("POSTILP_CONTRACT_SOURCE", "TARGET_ADAPTER_SOURCE"):
            value = _literal_assignment(tree, literal_name, literal_name)
            declared = _literal_assignment(tree, f"{literal_name}_SHA256", literal_name)
            if not isinstance(value, str) or hashlib.sha256(
                value.encode("utf-8")
            ).hexdigest() != declared:
                raise GuardError(f"M8R15 embedded source/hash drift: {literal_name}")
        if _literal_assignment(
            tree, "WHOLE_RUN_HARD_BOUND_SECONDS", "M8R15 bound"
        ) != route["whole_run_hard_bound_seconds"]:
            raise GuardError("M8R15 whole-run hard bound drift")
        required = (
            "M8R15_PREFLIGHT_STEP0 = run_preflight_step0()",
            'types.ModuleType("m8r15_combined_exporter")',
            "sys.modules[combined_module.__name__] = combined_module",
            'types.ModuleType("m8r14_m7r12_contract")',
            'types.ModuleType("m8r14_adapter")',
            'os.environ["KAGGLE_PREFLIGHT_MODE"] = "expansive_both"',
            'os.environ["KAGGLE_PREFLIGHT_UNITS"] = "100"',
            '"candidate_id": "M8R15-M8-POSTILP-TARGET-FEATURE-EXPORT-v1"',
            '"status": "PASS_POSTILP_TARGET_FEATURE_CERT_ONLY"',
            '"graph_efficacy_test_performed": False',
            '"scoring": False',
            '"training": False',
            '"submission_output": False',
            '"protected_data_opened": False',
            '"promotion_claim": False',
        )
        if any(marker not in source for marker in required):
            raise GuardError("M8R15 execution-control/output marker drift")
        mode_at = source.index('os.environ["KAGGLE_PREFLIGHT_MODE"] = "expansive_both"')
        units_at = source.index('os.environ["KAGGLE_PREFLIGHT_UNITS"] = "100"')
        exec_at = source.rindex("exec(  # noqa: S102")
        if not (mode_at < exec_at and units_at < exec_at):
            raise GuardError("M8R15 full provider role is not fixed before execution")
        if "kaggle competitions submit" in source or "competition_submit" in source:
            raise GuardError("M8R15 forbidden submission action surface")
    except (GuardError, OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m8r15-postilp-target-feature-cert-v1",
        "record_sha256": route["ship_certification_sha256"],
        "video_count": 0,
        "identities": route["identities"],
        "target_input": True,
        "frames_per_identity": route["frames_per_identity"],
        "completed_units": route["completed_units"],
        "whole_run_hard_bound_seconds": route["whole_run_hard_bound_seconds"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": visual_artifacts,
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m8r16_postilp_target_feature_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate M8R16's saved-script dependency re-entry certificate only."""

    failures: list[str] = []
    route = M8R16_POSTILP_TARGET_FEATURE_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "onebitbrain/m8r12-native-t4-exporter-carrier-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "m8r16-postilp-target-feature-cert-v1",
    }
    if metadata != expected_metadata:
        failures.append("M8R16 exact private saved-script metadata drift")
    if contract != route:
        failures.append("M8R16 exact central contract drift")
    canonical_registry = (
        project_root
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
        "KAGGLE_FAILURE_REGISTRY.json"
    ).resolve()
    if registry_path is None or registry_path.resolve() != canonical_registry:
        failures.append("M8R16 registry is not the canonical repository registry")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != route[
        "lane_owner"
    ]:
        failures.append("M8R16 exact kernel ownership drift")
    if registry.get("dataset_ownership", {}).get(
        "onebitbrain/m8r12-native-t4-exporter-carrier-v1"
    ) != "M8R13_SELECTED_TWO_EXPORTER_CERTIFICATION_ONLY":
        failures.append("M8R16 immutable carrier ownership drift")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M8R16 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M8R16 exact metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M8R16 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route[
        "package_inventory_sha256"
    ]:
        failures.append("M8R16 exact ship package inventory drift")

    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m8r16_saved_script_reexec_transport"
    m7r12 = campaign / "m7r12_postilp_fit_census"
    m8r12 = campaign / "m8r12_native_t4_exporter_cert"
    m8r14 = campaign / "m8r14_postilp_target_feature_export"
    m8r15 = campaign / "m8r15_module_registration_transport"
    artifacts = (
        (node / "M8R16_PACKAGE_FREEZE.json", route["package_freeze_sha256"], "M8R16 package freeze"),
        (node / "SOURCE_EQUIVALENCE.json", route["source_equivalence_sha256"], "M8R16 source equivalence"),
        (node / "SHIP_CERTIFICATION.json", route["ship_certification_sha256"], "M8R16 exact ship certification"),
        (node / "NOCONFLICT_REPORT.json", route["noconflict_report_sha256"], "M8R16 dependency closure"),
        (node / "PROVIDER_PROFILE.json", route["provider_profile_sha256"], "M8R16 provider profile"),
        (node / "STATIC_READINESS.json", route["static_readiness_sha256"], "M8R16 static readiness"),
        (node / "EVIDENCE_ZIP_CONTRACT.json", route["evidence_contract_sha256"], "M8R16 evidence contract"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M8R16 frozen experiment plan"),
        (node / "EXPERIMENT_PLAN_VALIDATION.json", route["experiment_validation_sha256"], "M8R16 plan validation"),
        (node / "seal_downloaded_m8r16_evidence.py", route["downloaded_evidence_sealer_sha256"], "M8R16 downloaded-evidence sealer"),
        (node / "M8R15_PROVIDER_FAILURE_LOG.txt", route["causal_failure_log_sha256"], "M8R15 causal failure log"),
        (m7r12 / "m7r12_exporter_contract.py", route["postilp_contract_sha256"], "exact M7R12 post-ILP contract"),
        (m8r12 / "provider_roundtrip_download/M8R6_WEIGHTS.pt", route["checkpoint_sha256"], "exact M8 checkpoint"),
        (m8r14 / "TERMINAL_PROVIDER.json", route["parent_terminal_sha256"], "M8R14 scientific parent terminal"),
        (m8r15 / "TERMINAL_PROVIDER.json", route["causal_terminal_sha256"], "M8R15 causal provider terminal"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    visual_artifacts: list[dict[str, str]] = []
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M8R16 exact artifact drift: {label}")
        visual_artifacts.append(
            {"path": _relative(path, project_root, label), "sha256": expected}
        )

    ship_path = node / "SHIP_CERTIFICATION.json"
    try:
        ship = load_json(ship_path)
        if not (
            ship.get("status") == "ELIGIBLE"
            and ship.get("eligible") is True
            and ship.get("autonomous_launch_eligible") is True
            and ship.get("receipt_content_sha256")
            == route["ship_receipt_content_sha256"]
            and ship.get("runtime_projection", {}).get("status") == "PASS"
        ):
            raise GuardError("M8R16 exact ship certificate semantics drift")
        lifecycle = next(
            (
                row
                for row in ship.get("runtime_safety_runs", [])
                if row.get("scenario")
                == "runtime_safety::outer_saved_script_reentry"
            ),
            None,
        )
        lifecycle_evidence = (
            lifecycle.get("runtime_safety", {}).get(
                "dependency_reexec_lifecycle", {}
            )
            if isinstance(lifecycle, dict)
            else {}
        )
        lifecycle_keys = (
            "initial_file_is_regular",
            "dynamic_module_file_is_outer_entrypoint",
            "dynamic_module_file_is_regular",
            "reexec_target_is_outer_entrypoint",
            "wrapper_reentered",
            "marker_authenticated",
            "completion_path_reachable",
            "second_restart_refused",
        )
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("status") != "PASS"
            or any(lifecycle_evidence.get(key) is not True for key in lifecycle_keys)
        ):
            raise GuardError("M8R16 dependency re-exec lifecycle evidence drift")
        embedded = _literal_assignment(tree, "COMBINED_SOURCE", "M8R16 combined source")
        if not isinstance(embedded, str):
            raise GuardError("M8R16 embedded combined source is not literal")
        if hashlib.sha256(embedded.encode("utf-8")).hexdigest() != route[
            "combined_source_sha256"
        ]:
            raise GuardError("M8R16 embedded combined source byte drift")
        if _literal_assignment(
            tree, "COMBINED_SOURCE_SHA256", "M8R16 combined source digest"
        ) != route["combined_source_sha256"]:
            raise GuardError("M8R16 embedded combined source digest drift")
        for literal_name in ("POSTILP_CONTRACT_SOURCE", "TARGET_ADAPTER_SOURCE"):
            value = _literal_assignment(tree, literal_name, literal_name)
            declared = _literal_assignment(tree, f"{literal_name}_SHA256", literal_name)
            if not isinstance(value, str) or hashlib.sha256(
                value.encode("utf-8")
            ).hexdigest() != declared:
                raise GuardError(f"M8R16 embedded source/hash drift: {literal_name}")
        if _literal_assignment(
            tree, "WHOLE_RUN_HARD_BOUND_SECONDS", "M8R16 bound"
        ) != route["whole_run_hard_bound_seconds"]:
            raise GuardError("M8R16 whole-run hard bound drift")
        required = (
            "M8R16_PREFLIGHT_STEP0 = run_preflight_step0()",
            'types.ModuleType("m8r16_combined_exporter")',
            "combined_module.__file__ = str(Path(__file__).resolve())",
            "sys.modules[combined_module.__name__] = combined_module",
            'types.ModuleType("m8r14_m7r12_contract")',
            'types.ModuleType("m8r14_adapter")',
            "M8R16_WRAPPER_REEXEC_LIFECYCLE_PASS",
            'os.environ["KAGGLE_PREFLIGHT_MODE"] = "expansive_both"',
            'os.environ["KAGGLE_PREFLIGHT_UNITS"] = "100"',
            '"candidate_id": "M8R16-M8-POSTILP-TARGET-FEATURE-EXPORT-v1"',
            '"status": "PASS_POSTILP_TARGET_FEATURE_CERT_ONLY"',
            '"graph_efficacy_test_performed": False',
            '"scoring": False',
            '"training": False',
            '"submission_output": False',
            '"protected_data_opened": False',
            '"promotion_claim": False',
        )
        if any(marker not in source for marker in required):
            raise GuardError("M8R16 execution-control/output marker drift")
        mode_at = source.index('os.environ["KAGGLE_PREFLIGHT_MODE"] = "expansive_both"')
        units_at = source.index('os.environ["KAGGLE_PREFLIGHT_UNITS"] = "100"')
        exec_at = source.rindex("exec(  # noqa: S102")
        if not (mode_at < exec_at and units_at < exec_at):
            raise GuardError("M8R16 full provider role is not fixed before execution")
        if "kaggle competitions submit" in source or "competition_submit" in source:
            raise GuardError("M8R16 forbidden submission action surface")
    except (GuardError, OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m8r16-postilp-target-feature-cert-v1",
        "record_sha256": route["ship_certification_sha256"],
        "video_count": 0,
        "identities": route["identities"],
        "target_input": True,
        "frames_per_identity": route["frames_per_identity"],
        "completed_units": route["completed_units"],
        "whole_run_hard_bound_seconds": route["whole_run_hard_bound_seconds"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": visual_artifacts,
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m8r17_postilp_target_feature_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate M8R17's outer atomic-helper transport repair only."""

    failures: list[str] = []
    route = M8R17_POSTILP_TARGET_FEATURE_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "onebitbrain/m8r12-native-t4-exporter-carrier-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "m8r17-postilp-target-feature-cert-v1",
    }
    if metadata != expected_metadata:
        failures.append("M8R17 exact private saved-script metadata drift")
    if contract != route:
        failures.append("M8R17 exact central contract drift")
    canonical_registry = (
        project_root
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
        "KAGGLE_FAILURE_REGISTRY.json"
    ).resolve()
    if registry_path is None or registry_path.resolve() != canonical_registry:
        failures.append("M8R17 registry is not the canonical repository registry")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != route[
        "lane_owner"
    ]:
        failures.append("M8R17 exact kernel ownership drift")
    if registry.get("dataset_ownership", {}).get(
        "onebitbrain/m8r12-native-t4-exporter-carrier-v1"
    ) != "M8R13_SELECTED_TWO_EXPORTER_CERTIFICATION_ONLY":
        failures.append("M8R17 immutable carrier ownership drift")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M8R17 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M8R17 exact metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M8R17 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route[
        "package_inventory_sha256"
    ]:
        failures.append("M8R17 exact ship package inventory drift")

    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m8r17_outer_atomic_transport"
    m7r12 = campaign / "m7r12_postilp_fit_census"
    m8r12 = campaign / "m8r12_native_t4_exporter_cert"
    m8r16 = campaign / "m8r16_saved_script_reexec_transport"
    artifacts = (
        (node / "M8R17_PACKAGE_FREEZE.json", route["package_freeze_sha256"], "M8R17 package freeze"),
        (node / "SOURCE_EQUIVALENCE.json", route["source_equivalence_sha256"], "M8R17 source equivalence"),
        (node / "LATE_TAIL_PREFLIGHT.json", route["late_tail_preflight_sha256"], "M8R17 late-tail preflight"),
        (node / "SHIP_CERTIFICATION.json", route["ship_certification_sha256"], "M8R17 exact ship certification"),
        (node / "NOCONFLICT_REPORT.json", route["noconflict_report_sha256"], "M8R17 dependency closure"),
        (node / "PROVIDER_PROFILE.json", route["provider_profile_sha256"], "M8R17 provider profile"),
        (node / "STATIC_READINESS.json", route["static_readiness_sha256"], "M8R17 static readiness"),
        (node / "EVIDENCE_ZIP_CONTRACT.json", route["evidence_contract_sha256"], "M8R17 evidence contract"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M8R17 frozen experiment plan"),
        (node / "EXPERIMENT_PLAN_VALIDATION.json", route["experiment_validation_sha256"], "M8R17 plan validation"),
        (node / "seal_downloaded_m8r17_evidence.py", route["downloaded_evidence_sealer_sha256"], "M8R17 downloaded-evidence sealer"),
        (m7r12 / "m7r12_exporter_contract.py", route["postilp_contract_sha256"], "exact M7R12 post-ILP contract"),
        (m8r12 / "provider_roundtrip_download/M8R6_WEIGHTS.pt", route["checkpoint_sha256"], "exact M8 checkpoint"),
        (m8r16 / "TERMINAL_PROVIDER.json", route["causal_terminal_sha256"], "M8R16 causal provider terminal"),
        (m8r16 / "remote_output_v1/M8R16_CERT_FAILURE.json", route["causal_failure_sha256"], "M8R16 causal failure"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    visual_artifacts: list[dict[str, str]] = []
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M8R17 exact artifact drift: {label}")
        visual_artifacts.append(
            {"path": _relative(path, project_root, label), "sha256": expected}
        )

    try:
        ship = load_json(node / "SHIP_CERTIFICATION.json")
        if not (
            ship.get("status") == "ELIGIBLE"
            and ship.get("eligible") is True
            and ship.get("autonomous_launch_eligible") is True
            and ship.get("receipt_content_sha256")
            == route["ship_receipt_content_sha256"]
            and ship.get("runtime_projection", {}).get("status") == "PASS"
            and ship.get("quick_debug", {}).get("status") == "PASS"
        ):
            raise GuardError("M8R17 exact ship certificate semantics drift")
        lifecycle = next(
            (
                row
                for row in ship.get("runtime_safety_runs", [])
                if row.get("scenario")
                == "runtime_safety::outer_saved_script_reentry"
            ),
            None,
        )
        lifecycle_evidence = (
            lifecycle.get("runtime_safety", {}).get(
                "dependency_reexec_lifecycle", {}
            )
            if isinstance(lifecycle, dict)
            else {}
        )
        lifecycle_keys = (
            "initial_file_is_regular",
            "dynamic_module_file_is_outer_entrypoint",
            "dynamic_module_file_is_regular",
            "reexec_target_is_outer_entrypoint",
            "wrapper_reentered",
            "marker_authenticated",
            "completion_path_reachable",
            "second_restart_refused",
        )
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("status") != "PASS"
            or any(lifecycle_evidence.get(key) is not True for key in lifecycle_keys)
        ):
            raise GuardError("M8R17 dependency re-exec lifecycle evidence drift")
        late_tail = load_json(node / "LATE_TAIL_PREFLIGHT.json")
        if not (
            late_tail.get("status") == "PASS_TARGET_PYTHON_OUTER_ATOMIC_TRANSPORT"
            and late_tail.get("python_version") == "3.12.13"
            and late_tail.get("repaired_top_level_atomic_helper_definitions") == 2
            and late_tail.get("parent_top_level_atomic_helper_definitions") == 0
            and late_tail.get("npy_write_and_hash_pass") is True
            and late_tail.get("json_write_and_hash_pass") is True
            and late_tail.get("overwrite_rejected") is True
        ):
            raise GuardError("M8R17 late-tail outer atomic-helper preflight drift")
        embedded = _literal_assignment(tree, "COMBINED_SOURCE", "M8R17 combined source")
        if not isinstance(embedded, str):
            raise GuardError("M8R17 embedded combined source is not literal")
        if hashlib.sha256(embedded.encode("utf-8")).hexdigest() != route[
            "combined_source_sha256"
        ]:
            raise GuardError("M8R17 embedded combined source byte drift")
        if _literal_assignment(
            tree, "COMBINED_SOURCE_SHA256", "M8R17 combined source digest"
        ) != route["combined_source_sha256"]:
            raise GuardError("M8R17 embedded combined source digest drift")
        for literal_name in ("POSTILP_CONTRACT_SOURCE", "TARGET_ADAPTER_SOURCE"):
            value = _literal_assignment(tree, literal_name, literal_name)
            declared = _literal_assignment(tree, f"{literal_name}_SHA256", literal_name)
            if not isinstance(value, str) or hashlib.sha256(
                value.encode("utf-8")
            ).hexdigest() != declared:
                raise GuardError(f"M8R17 embedded source/hash drift: {literal_name}")
        if _literal_assignment(
            tree, "WHOLE_RUN_HARD_BOUND_SECONDS", "M8R17 bound"
        ) != route["whole_run_hard_bound_seconds"]:
            raise GuardError("M8R17 whole-run hard bound drift")
        required = (
            "M8R17_PREFLIGHT_STEP0 = run_preflight_step0()",
            'types.ModuleType("m8r17_combined_exporter")',
            "combined_module.__file__ = str(Path(__file__).resolve())",
            "sys.modules[combined_module.__name__] = combined_module",
            "def _v32_atomic_npy(path, value):",
            "def _v32_atomic_json(path, value):",
            "M8R17_WRAPPER_REEXEC_LIFECYCLE_PASS",
            'os.environ["KAGGLE_PREFLIGHT_MODE"] = "expansive_both"',
            'os.environ["KAGGLE_PREFLIGHT_UNITS"] = "100"',
            '"candidate_id": "M8R17-M8-POSTILP-TARGET-FEATURE-EXPORT-v1"',
            '"status": "PASS_POSTILP_TARGET_FEATURE_CERT_ONLY"',
            '"graph_efficacy_test_performed": False',
            '"scoring": False',
            '"training": False',
            '"submission_output": False',
            '"protected_data_opened": False',
            '"promotion_claim": False',
        )
        if any(marker not in source for marker in required):
            raise GuardError("M8R17 execution-control/output marker drift")
        combined_tree = ast.parse(embedded)
        combined_top_level_helpers = [
            node.name
            for node in combined_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"_v32_atomic_npy", "_v32_atomic_json"}
        ]
        if sorted(combined_top_level_helpers) != [
            "_v32_atomic_json",
            "_v32_atomic_npy",
        ]:
            raise GuardError("M8R17 outer atomic-helper definition count drift")
        mode_at = source.index('os.environ["KAGGLE_PREFLIGHT_MODE"] = "expansive_both"')
        units_at = source.index('os.environ["KAGGLE_PREFLIGHT_UNITS"] = "100"')
        exec_at = source.rindex("exec(  # noqa: S102")
        if not (mode_at < exec_at and units_at < exec_at):
            raise GuardError("M8R17 full provider role is not fixed before execution")
        if "kaggle competitions submit" in source or "competition_submit" in source:
            raise GuardError("M8R17 forbidden submission action surface")
    except (GuardError, OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m8r17-postilp-target-feature-cert-v1",
        "record_sha256": route["ship_certification_sha256"],
        "video_count": 0,
        "identities": route["identities"],
        "target_input": True,
        "frames_per_identity": route["frames_per_identity"],
        "completed_units": route["completed_units"],
        "whole_run_hard_bound_seconds": route["whole_run_hard_bound_seconds"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": visual_artifacts,
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m8r20_full21_target_feature_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate the exact M8R20 full-21 target-feature provider certificate."""

    failures: list[str] = []
    route = M8R20_FULL21_TARGET_FEATURE_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "onebitbrain/m8r12-native-t4-exporter-carrier-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "m8r20-full21-target-feature-export-v1",
    }
    if metadata != expected_metadata:
        failures.append("M8R20 exact private saved-script metadata drift")
    if contract != route:
        failures.append("M8R20 exact central contract drift")
    canonical_registry = (
        project_root
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
        "KAGGLE_FAILURE_REGISTRY.json"
    ).resolve()
    if registry_path is None or registry_path.resolve() != canonical_registry:
        failures.append("M8R20 registry is not the canonical repository registry")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != route[
        "lane_owner"
    ]:
        failures.append("M8R20 exact kernel ownership drift")
    if registry.get("dataset_ownership", {}).get(
        "onebitbrain/m8r12-native-t4-exporter-carrier-v1"
    ) != "M8R13_SELECTED_TWO_EXPORTER_CERTIFICATION_ONLY":
        failures.append("M8R20 immutable carrier ownership drift")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M8R20 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M8R20 exact metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M8R20 exact package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route[
        "package_inventory_sha256"
    ]:
        failures.append("M8R20 exact ship package inventory drift")

    campaign = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1"
    node = campaign / "m8r20_dual_competition_root_canonicalization"
    m8r12 = campaign / "m8r12_native_t4_exporter_cert"
    m8r19 = campaign / "m8r19_preflight_canonicalization_transport"
    m7r14 = campaign / "m7r14_full21_raw_metadata_transport"
    artifacts = (
        (node / "M8R20_PACKAGE_FREEZE.json", route["package_freeze_sha256"], "M8R20 package freeze"),
        (node / "TRANSPORT_EQUIVALENCE.json", route["transport_equivalence_sha256"], "M8R20 transport equivalence"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M8R20 experiment plan"),
        (node / "EXPERIMENT_PLAN_VALIDATION.json", route["experiment_validation_sha256"], "M8R20 plan validation"),
        (node / "PREFLIGHT_PROFILE.json", route["ship_profile_sha256"], "M8R20 ship profile"),
        (node / "SHIP_CERTIFICATION.json", route["ship_certification_sha256"], "M8R20 ship certificate"),
        (node / "LOCAL_MATERIAL_RESOURCE_RECEIPT.json", route["local_material_resource_receipt_sha256"], "M8R20 local resource receipt"),
        (node / "STATIC_READINESS.json", route["static_readiness_sha256"], "M8R20 static readiness"),
        (node / "seal_downloaded_m8r20_evidence.py", route["downloaded_evidence_sealer_sha256"], "M8R20 evidence sealer"),
        (node / "EVIDENCE_SEALER_REPAIR.json", route["evidence_sealer_repair_sha256"], "M8R20 evidence sealer repair"),
        (m8r19 / "TERMINAL_SHIP_REJECTION.json", route["parent_terminal_sha256"], "M8R19 parent rejection"),
        (m7r14 / "M7R14_FULL21_RAW_METADATA.json", route["full21_raw_metadata_sha256"], "full-21 raw metadata"),
        (m8r12 / "provider_roundtrip_download/M8R6_WEIGHTS.pt", route["checkpoint_sha256"], "exact M8 checkpoint"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    evidence: list[dict[str, str]] = []
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M8R20 exact artifact drift: {label}")
        evidence.append({"path": _relative(path, project_root, label), "sha256": expected})

    try:
        ship = load_json(node / "SHIP_CERTIFICATION.json")
        projection = ship.get("runtime_projection", {})
        if not (
            ship.get("status") == "ELIGIBLE"
            and ship.get("eligible") is True
            and ship.get("autonomous_launch_eligible") is True
            and ship.get("receipt_content_sha256")
            == route["ship_receipt_content_sha256"]
            and ship.get("artifact_sha256") == route["code_sha256"]
            and ship.get("metadata_sha256") == route["metadata_sha256"]
            and ship.get("package_inventory_sha256")
            == route["package_inventory_sha256"]
            and ship.get("profile_sha256") == route["ship_profile_sha256"]
            and ship.get("quick_debug", {}).get("status") == "PASS"
            and ship.get("repeatability", {}).get("status") == "PASS"
            and projection.get("status") == "PASS"
            and projection.get("successful_runs") == 8
            and projection.get("projected_remote_upper_seconds")
            == route["projected_remote_upper_seconds"]
            and projection.get("projected_remote_upper_seconds", 28_800) < 28_800
        ):
            raise GuardError("M8R20 exact ship certificate semantics drift")
        resource = load_json(node / "LOCAL_MATERIAL_RESOURCE_RECEIPT.json")
        if not (
            resource.get("status") == "PASS_GUARDS"
            and resource.get("returncode") == 0
            and resource.get("killed_reason") is None
            and resource.get("ship_receipt_exists") is True
            and resource.get("ship_receipt_sha256")
            == route["ship_certification_sha256"]
            and resource.get("external_action_performed") is False
        ):
            raise GuardError("M8R20 local LMT resource receipt drift")
        readiness = load_json(node / "STATIC_READINESS.json")
        if not (
            readiness.get("status")
            == "PASS_STATIC_AND_MATERIAL_SHIP_ELIGIBLE_PENDING_CENTRAL_AND_DYNAMIC_GATES"
            and readiness.get("identity_count") == 21
            and readiness.get("workload_units") == 2100
            and readiness.get("configuration_count") == 1
            and readiness.get("seed") == 104729
            and readiness.get("hpo") is False
            and readiness.get("external_action_performed") is False
        ):
            raise GuardError("M8R20 static readiness semantics drift")
        embedded = _literal_assignment(tree, "COMBINED_SOURCE", "M8R20 combined source")
        adapter = _literal_assignment(tree, "TARGET_ADAPTER_SOURCE", "M8R20 target adapter")
        if not isinstance(embedded, str) or not isinstance(adapter, str):
            raise GuardError("M8R20 embedded sources are not literal strings")
        if hashlib.sha256(embedded.encode()).hexdigest() != _literal_assignment(
            tree, "COMBINED_SOURCE_SHA256", "M8R20 combined source digest"
        ):
            raise GuardError("M8R20 combined source/hash drift")
        if hashlib.sha256(adapter.encode()).hexdigest() != _literal_assignment(
            tree, "TARGET_ADAPTER_SOURCE_SHA256", "M8R20 adapter digest"
        ):
            raise GuardError("M8R20 adapter source/hash drift")
        if _literal_assignment(tree, "IDENTITIES", "M8R20 identities") != tuple(
            row["dataset"]
            for row in load_json(m7r14 / "M7R14_FULL21_RAW_METADATA.json")["records"]
        ):
            raise GuardError("M8R20 exact full-21 identity/order drift")
        if _literal_assignment(
            tree, "WHOLE_RUN_HARD_BOUND_SECONDS", "M8R20 bound"
        ) != route["whole_run_hard_bound_seconds"]:
            raise GuardError("M8R20 whole-run bound drift")
        required = (
            "M8R18_PREFLIGHT_STEP0 = run_preflight_step0()",
            'simulated_competition.rsplit("/", 1)[0]',
            '"/" + "kaggle/input/" + "competitions"',
            "M8R18_PREFLIGHT_HEAVY_WORK_BEGIN",
            'rows != ["Tesla T4, 7.5", "Tesla T4, 7.5"]',
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
            '"candidate_id": "M8R20-EXACT-DUAL-COMPETITION-ROOT-CANONICALIZATION-v1"',
            '"status": "PASS_FULL21_POSTILP_TARGET_FEATURE_EXPORT"',
            '"completed_units": 2100',
            '"graph_efficacy_test_performed": False',
            '"scoring": False',
            '"training": False',
            '"submission_output": False',
            '"protected_data_opened": False',
            '"promotion_claim": False',
        )
        if any(marker not in source for marker in required):
            raise GuardError("M8R20 execution-control/output marker drift")
        if "kaggle competitions submit" in source or "competition_submit" in source:
            raise GuardError("M8R20 forbidden submission action surface")
        if "/test/" in source or "/tests/" in source:
            raise GuardError("M8R20 forbidden data namespace surface")
    except (GuardError, OSError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m8r20-full21-target-feature-provider-cert-v1",
        "record_sha256": route["ship_certification_sha256"],
        "video_count": 0,
        "identities": [
            f"train:{row['dataset']}"
            for row in load_json(m7r14 / "M7R14_FULL21_RAW_METADATA.json")["records"]
        ],
        "identity_count": 21,
        "target_input": True,
        "completed_units": route["completed_units"],
        "whole_run_hard_bound_seconds": route["whole_run_hard_bound_seconds"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": evidence,
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m10r3_provider_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact no-input M10R3 synthetic provider certificate."""

    failures: list[str] = []
    route = M10R3_PROVIDER_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "M10R3 aniso3D dual-slot provider cert v1",
    }
    if metadata != expected_metadata:
        failures.append("M10R3 exact source-free script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m10r3-central-prelaunch.v1",
        "execution_profile": M10R3_PROVIDER_CERT_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M10R3_SYNTHETIC_PROVIDER_CERTIFICATION_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "m10r2_child_sha256": route["m10r2_child_sha256"],
        "m10r2_static_freeze_sha256": route["m10r2_static_freeze_sha256"],
        "m7r7_runtime_seal_sha256": route["m7r7_runtime_seal_sha256"],
        "m7r7_runtime_tuple_sha256": route["m7r7_runtime_tuple_sha256"],
        "candidate_inventory_sha256": route["candidate_inventory_sha256"],
        "provider_profile_sha256": route["provider_profile_sha256"],
        "transport_equivalence_sha256": route["transport_equivalence_sha256"],
        "static_certification_sha256": route["static_certification_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "lmt_sha256": route["lmt_sha256"],
        "certification_only": True,
        "external_model_artifact": False,
        "target_input": False,
        "production_output_permitted": False,
        "submission_output_permitted": False,
        "scoring_permitted": False,
        "downloaded_evidence_required_for_any_successor": True,
        "whole_run_hard_bound_seconds": 12540,
        "repeat_count": 2,
        "one_shot": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            failures.append(f"M10R3 central contract drift: {key}")
    if set(contract) != set(exact_contract):
        failures.append("M10R3 central contract has an arbitrary surface")
    if sha256(code_path) != route["code_sha256"]:
        failures.append("M10R3 exact kernel hash drift")
    if sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M10R3 exact metadata hash drift")
    freeze = kernel_dir / "M10R3_PACKAGE_FREEZE.json"
    if not freeze.is_file() or freeze.is_symlink() or sha256(freeze) != route["package_freeze_sha256"]:
        failures.append("M10R3 package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M10R3 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M10R3 ship package inventory drift")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != "M10R3_SYNTHETIC_PROVIDER_CERTIFICATION_ONLY":
        failures.append("M10R3 exact registry ownership drift")

    node = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/m10r3_self_orchestrating_provider_cert"
    m10r2_node = node.parent / "m10r2_static_t4_package"
    m7r7_node = node.parent / "m7r7_native_runtime_tuple_diagnostic"
    artifacts = (
        (m10r2_node / "package_v1/kernel.py", route["m10r2_child_sha256"], "M10R2 exact child"),
        (m10r2_node / "M10R2_ARTIFACT_FREEZE.json", route["m10r2_static_freeze_sha256"], "M10R2 static freeze"),
        (m7r7_node / "M7R7_DOWNLOADED_RUNTIME_SEAL.json", route["m7r7_runtime_seal_sha256"], "M7R7 runtime seal"),
        (node / "CANDIDATE_INVENTORY.json", route["candidate_inventory_sha256"], "M10R3 candidate inventory"),
        (node / "M10R3_PROVIDER_PROFILE.json", route["provider_profile_sha256"], "M10R3 provider profile"),
        (node / "M10R3_TRANSPORT_EQUIVALENCE.json", route["transport_equivalence_sha256"], "M10R3 transport equivalence"),
        (node / "M10R3_STATIC_CERTIFICATION.json", route["static_certification_sha256"], "M10R3 static certificate"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "M10R3 experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "M10R3 plan validation"),
        (node / "seal_downloaded_m10r3_evidence.py", route["downloaded_evidence_sealer_sha256"], "M10R3 evidence sealer"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"{label} exact artifact drift")
    try:
        child = _literal_assignment(tree, "M10R2_CHILD_SOURCE", "M10R3 embedded M10R2 child")
        if not isinstance(child, str) or hashlib.sha256(child.encode()).hexdigest() != route["m10r2_child_sha256"]:
            raise GuardError("M10R3 embedded child hash drift")
        child_path = artifacts[0][0]
        if child_path.is_file() and child.encode() != child_path.read_bytes():
            raise GuardError("M10R3 embedded child bytes differ from M10R2")
        if _literal_assignment(tree, "ROLE_ORDER", "M10R3 role order") != ("tiny", "stratified", "stress", "expansive"):
            raise GuardError("M10R3 role order drift")
        if _literal_assignment(tree, "REPEAT_COUNT", "M10R3 repeats") != 2:
            raise GuardError("M10R3 repeat count drift")
        if _literal_assignment(tree, "WHOLE_RUN_HARD_BOUND_SECONDS", "M10R3 bound") != 12540:
            raise GuardError("M10R3 whole-run bound drift")
        external_controls = [
            item.lineno
            for item in ast.walk(tree)
            if (
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == "get"
                and isinstance(item.func.value, ast.Attribute)
                and isinstance(item.func.value.value, ast.Name)
                and item.func.value.value.id == "os"
                and item.func.value.attr == "environ"
            )
            or (
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == "sys"
                and item.attr == "argv"
            )
        ]
        if external_controls:
            raise GuardError("M10R3 normal invocation acquired external controls")
        markers = (
            'os.environ["M10R2_CERTIFY_MODE"] = "synthetic_runtime_cert"',
            "for role in ROLE_ORDER:",
            "for repeat in range(1, REPEAT_COUNT + 1):",
            "signal.setitimer(signal.ITIMER_REAL, float(spec[\"timeout_seconds\"]))",
            "M10R3_PROVIDER_CERT_EVIDENCE.zip",
            "M10R3_PROVIDER_CERT_FAILURE.zip",
            '"production_enabled": False',
            '"score_or_submission": False',
        )
        if any(marker not in source for marker in markers):
            raise GuardError("M10R3 self-orchestration/output marker drift")
        if any(marker in source for marker in ("submission.csv", "44b6_6bba", "/kaggle/input")):
            raise GuardError("M10R3 forbidden target/score surface present")
    except (GuardError, SyntaxError, ValueError) as exc:
        failures.append(str(exc))
    if phase == "launch":
        failures.append("M10R3 central guard requires the separate current-LMT/quota/fresh-slug/serial-lock coordinator")
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "m10r3-self-orchestrating-provider-cert-v1",
        "record_sha256": route["m10r2_static_freeze_sha256"],
        "video_count": 0,
        "identities": [],
        "target_input": False,
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [
            {"path": _relative(path, project_root, label), "sha256": expected}
            for path, expected, label in artifacts
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_m10r5_provider_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact no-input M10R5 gather provider certificate."""

    failures: list[str] = []
    route = M10R5_PROVIDER_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py", "competition_sources": [], "dataset_sources": [],
        "enable_gpu": True, "enable_internet": False, "enable_tpu": False,
        "id": route["kernel_id"], "is_private": True, "kernel_sources": [],
        "kernel_type": "script", "language": "python", "machine_shape": "NvidiaTeslaT4",
        "model_sources": [], "title": "m10r5-aniso3d-dualslot-provider-cert-v1",
    }
    if metadata != expected_metadata:
        failures.append("M10R5 exact source-free script metadata drift")
    exact_contract = {
        "schema": "celltrack-alpha.m10r5-central-prelaunch.v1",
        "execution_profile": M10R5_PROVIDER_CERT_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": route["kernel_id"],
        "remote_preflight_kernel_id": route["kernel_id"],
        "lane_owner": "M10R5_GATHER_NATIVE_T4_CERTIFICATION_ONLY",
        "code_sha256": route["code_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "child_sha256": route["child_sha256"],
        "equivalent_model_source_sha256": route["equivalent_model_source_sha256"],
        "equivalence_binding_sha256": route["equivalence_binding_sha256"],
        "provider_profile_sha256": route["provider_profile_sha256"],
        "static_certification_sha256": route["static_certification_sha256"],
        "experiment_plan_sha256": route["experiment_plan_sha256"],
        "experiment_validation_sha256": route["experiment_validation_sha256"],
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "m10r4_static_freeze_sha256": route["m10r4_static_freeze_sha256"],
        "m10r3_terminal_failure_sha256": route["m10r3_terminal_failure_sha256"],
        "m10r3_post_reconciliation_sha256": route["m10r3_post_reconciliation_sha256"],
        "lmt_sha256": route["lmt_sha256"],
        "certification_only": True, "external_model_artifact": False, "target_input": False,
        "production_output_permitted": False, "submission_output_permitted": False,
        "scoring_permitted": False, "downloaded_evidence_required_for_any_successor": True,
        "whole_run_hard_bound_seconds": 18720, "warmup_count_per_role": 1,
        "measured_repeat_count": 2, "one_shot": True,
    }
    if contract != exact_contract:
        failures.append("M10R5 central contract drift")
    if sha256(code_path) != route["code_sha256"] or sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("M10R5 code/metadata hash drift")
    freeze = kernel_dir / "M10R5_PACKAGE_FREEZE.json"
    if not freeze.is_file() or freeze.is_symlink() or sha256(freeze) != route["package_freeze_sha256"]:
        failures.append("M10R5 package freeze drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("M10R5 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("M10R5 ship package inventory drift")
    if registry.get("kernel_ownership", {}).get(route["kernel_id"]) != "M10R5_GATHER_NATIVE_T4_CERTIFICATION_ONLY":
        failures.append("M10R5 exact registry ownership drift")
    node = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/m10r5_native_t4_provider_cert"
    campaign = node.parent
    artifacts = (
        (node / "M10R5_EQUIVALENCE_BINDING.json", route["equivalence_binding_sha256"], "equivalence binding"),
        (node / "M10R5_PROVIDER_PROFILE.json", route["provider_profile_sha256"], "provider profile"),
        (node / "M10R5_STATIC_CERTIFICATION.json", route["static_certification_sha256"], "static certificate"),
        (node / "EXPERIMENT_PLAN.json", route["experiment_plan_sha256"], "experiment plan"),
        (node / "EXPERIMENT_PLAN.validation.json", route["experiment_validation_sha256"], "plan validation"),
        (node / "seal_downloaded_m10r5_evidence.py", route["downloaded_evidence_sealer_sha256"], "evidence sealer"),
        (campaign / "m10r4_deterministic_sampler_seam/M10R4_STATIC_FREEZE.json", route["m10r4_static_freeze_sha256"], "M10R4 freeze"),
        (campaign / "m10r3_self_orchestrating_provider_cert/M10R3_TERMINAL_FAILURE_AUDIT.json", route["m10r3_terminal_failure_sha256"], "M10R3 failure"),
        (campaign / "m10r3_self_orchestrating_provider_cert/M10R3_POST_RECONCILIATION_SEAL.json", route["m10r3_post_reconciliation_sha256"], "M10R3 reconciliation"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"M10R5 {label} exact artifact drift")
    try:
        child = _literal_assignment(tree, "M10R2_CHILD_SOURCE", "M10R5 child")
        if not isinstance(child, str) or hashlib.sha256(child.encode()).hexdigest() != route["child_sha256"]:
            raise GuardError("M10R5 embedded child drift")
        child_tree = ast.parse(child)
        model = _literal_assignment(child_tree, "MODEL_SOURCE", "M10R5 model")
        if not isinstance(model, str) or hashlib.sha256(model.encode()).hexdigest() != route["equivalent_model_source_sha256"]:
            raise GuardError("M10R5 equivalent model source drift")
        if "F.grid_sample(" in model or "torch.gather" not in model:
            raise GuardError("M10R5 sampler seam drift")
        for name, expected in (("ROLE_ORDER", ("tiny", "stratified", "stress", "expansive")), ("WARMUP_COUNT", 1), ("REPEAT_COUNT", 2), ("FIXED_SEED", 104729), ("WHOLE_RUN_HARD_BOUND_SECONDS", 18720)):
            if _literal_assignment(tree, name, f"M10R5 {name}") != expected:
                raise GuardError(f"M10R5 {name} drift")
        external = [item.lineno for item in ast.walk(tree) if (
            isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
            and item.func.attr == "get" and isinstance(item.func.value, ast.Attribute)
            and isinstance(item.func.value.value, ast.Name) and item.func.value.value.id == "os"
            and item.func.value.attr == "environ"
        ) or (isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id == "sys" and item.attr == "argv")]
        if external:
            raise GuardError("M10R5 normal invocation acquired external controls")
        markers = ("M10R5_ROLE_WARMUP", "for repeat in range(1, REPEAT_COUNT + 1):", "M10R5_PROVIDER_CERT_EVIDENCE.zip", "M10R5_PROVIDER_CERT_FAILURE.zip")
        if any(marker not in source for marker in markers) or any(marker in source for marker in ("/kaggle/input", "submission.csv", "44b6_6bba")):
            raise GuardError("M10R5 orchestration/firewall marker drift")
    except (GuardError, SyntaxError, ValueError) as exc:
        failures.append(str(exc))
    if phase == "launch":
        failures.append("M10R5 requires separate current-LMT/quota/fresh-slug/serial-lock coordinator")
    visual_summary = {
        "required": True, "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"], "production_enabled": False,
        "certification_only": True, "exception_id": "m10r5-gather-native-t4-provider-cert-v1",
        "record_sha256": route["m10r4_static_freeze_sha256"], "video_count": 0,
        "identities": [], "target_input": False,
        "post_remote_certification": {"downloaded_output_zip_required": True, "exact_local_seal_required": True, "production_or_score_permitted_before_seal": False, "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL"},
        "artifacts": [{"path": _relative(path, project_root, label), "sha256": expected} for path, expected, label in artifacts],
    }
    return _result(
        kernel_dir=kernel_dir, metadata_path=metadata_path, code_path=code_path,
        metadata=metadata, contract=contract, registry=registry, phase=phase,
        failures=failures, step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary, project_root=project_root,
        contract_path=contract_path, registry_path=registry_path,
    )


def check_d10_current_policy_native_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact saved-script D-v10 native T4 certificate."""

    failures: list[str] = []
    route = D10_CURRENT_POLICY_NATIVE_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "f2-d-v10-native-cert-current-v1",
    }
    if metadata != expected_metadata:
        failures.append("D10 current-policy exact private/offline/T4 script metadata drift")
    exact_contract = {
        "candidate_id": "D-V10-NATIVE-CERT-CURRENT-POLICY-v1",
        "certification_only": True,
        "code_sha256": route["code_sha256"],
        "completed_units": 4200,
        "configuration_count": 1,
        "current_certifier_sha256": route["current_certifier_sha256"],
        "current_ship_skill_sha256": route["current_ship_skill_sha256"],
        "downloaded_evidence_required_for_any_successor": True,
        "downloaded_evidence_sealer_sha256": route[
            "downloaded_evidence_sealer_sha256"
        ],
        "execution_profile": D10_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE,
        "expected_version": 1,
        "external_action_performed": False,
        "fresh_slug_required": True,
        "identity_count": 6,
        "kernel_id": route["kernel_id"],
        "lane_owner": "D10_CURRENT_POLICY_NATIVE_CERTIFICATION_ONLY",
        "launch_permitted": False,
        "lmt_sha256": route["lmt_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "model_execution": True,
        "one_shot": True,
        "outcome_access_permitted": False,
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "production_output_permitted": False,
        "profile_sha256": route["profile_sha256"],
        "projected_remote_upper_seconds": 26697.288724191087,
        "remote_preflight_kernel_id": route["kernel_id"],
        "reservation_hours": 7.75,
        "schema": "celltrack-alpha.d10-current-policy-central-prelaunch.v1",
        "scoring_permitted": False,
        "selected_identities": route["identities"],
        "source_freeze_sha256": route["source_freeze_sha256"],
        "static_readiness_sha256": route["static_readiness_sha256"],
        "ship_skill_forward_test_receipt_sha256": route[
            "ship_skill_forward_test_receipt_sha256"
        ],
        "ship_skill_forward_test_receipt_content_sha256": route[
            "ship_skill_forward_test_receipt_content_sha256"
        ],
        "submission_output_permitted": False,
        "target_input": True,
        "training_permitted": False,
        "whole_run_hard_bound_seconds": 27900,
    }
    if contract != exact_contract:
        failures.append("D10 current-policy central contract drift")
    if (
        sha256(code_path) != route["code_sha256"]
        or sha256(metadata_path) != route["metadata_sha256"]
    ):
        failures.append("D10 current-policy code/metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("D10 current-policy package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("D10 current-policy ship package inventory drift")
    if (
        registry.get("kernel_ownership", {}).get(route["kernel_id"])
        != "D10_CURRENT_POLICY_NATIVE_CERTIFICATION_ONLY"
    ):
        failures.append("D10 current-policy exact registry ownership drift")

    node = (
        project_root
        / "dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1"
        / "d_v10_current_policy_cert_v1"
    )
    old_run = (
        project_root
        / "dag_prompts/2026-08-20-alpha-funnel-v1/runs"
        / "kernel_f2_direct_d_sweep_v10"
    )
    artifacts = (
        (
            node / "D10_CURRENT_POLICY_PACKAGE_FREEZE.json",
            route["package_freeze_sha256"],
            "package freeze",
        ),
        (node / "PREFLIGHT_PROFILE.json", route["profile_sha256"], "ship profile"),
        (
            node / "STATIC_READINESS.json",
            route["static_readiness_sha256"],
            "static readiness",
        ),
        (
            node / "seal_downloaded_evidence.py",
            route["downloaded_evidence_sealer_sha256"],
            "downloaded evidence sealer",
        ),
        (
            old_run / "f2_direct_d_sweep_kernel.py",
            route["source_kernel_sha256"],
            "frozen scientific source",
        ),
        (
            old_run / "F2_SWEEP_SOURCE_FREEZE.json",
            route["source_freeze_sha256"],
            "source freeze",
        ),
        (
            old_run / "kernel-metadata.json",
            route["source_metadata_sha256"],
            "source metadata",
        ),
        (
            old_run.parent / "F2_DIRECT_D_SWEEP_V10_PREFLIGHT_PROFILE.json",
            route["parent_profile_sha256"],
            "parent ship profile",
        ),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
        (
            project_root
            / "dag_prompts/2026-08-23-model-clade-renewal-v1/"
            "kernel_skill_upgrade_v1/FORWARD_TEST_RECEIPT.json",
            route["ship_skill_forward_test_receipt_sha256"],
            "strengthened ship-skill forward test",
        ),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"D10 current-policy {label} exact artifact drift")
    ship_skill = project_root / ".agents/skills/ship-kaggle-kernels/SKILL.md"
    if not ship_skill.is_file() or sha256(ship_skill) != route["current_ship_skill_sha256"]:
        failures.append("D10 current-policy current ship skill exact artifact drift")
    certifier = (
        project_root
        / ".agents/skills/ship-kaggle-kernels/scripts/kaggle_kernel_certifier.py"
    )
    if (
        not certifier.is_file()
        or certifier.is_symlink()
        or sha256(certifier) != route["current_certifier_sha256"]
    ):
        failures.append("D10 current-policy current certifier exact artifact drift")

    try:
        encoded = _literal_assignment(tree, "PAYLOAD_Z85", "D10 current-policy payload")
        if not isinstance(encoded, str):
            raise GuardError("D10 current-policy payload is not a literal string")
        payload = json.loads(zlib.decompress(base64.b85decode(encoded)))
        if not isinstance(payload, dict):
            raise GuardError("D10 current-policy embedded payload is not an object")
        expected_bindings = {
            "certifier_sha256": route["current_certifier_sha256"],
            "kernel_artifact_sha256": route["target_entrypoint_sha256"],
            "kernel_id": "onebitbrain/f2-d-v10-current-policy-target-v1",
            "kernel_metadata_sha256": route["target_metadata_sha256"],
            "kernel_package_inventory_sha256": route[
                "target_package_inventory_sha256"
            ],
            "lmt_sha256": route["lmt_sha256"],
            "profile_sha256": route["profile_sha256"],
            "python_version_prefix": "3.12",
        }
        if payload.get("bindings") != expected_bindings:
            raise GuardError("D10 current-policy embedded target/certifier bindings drift")
        if payload.get("certifier_relative") != "tools/kaggle_kernel_certifier.py":
            raise GuardError("D10 current-policy embedded certifier path drift")
        if payload.get("output_root") != (
            "/kaggle/working/f2_d_v10_current_policy_native_cert"
        ):
            raise GuardError("D10 current-policy evidence output root drift")
        small_files = payload.get("small_files")
        if not isinstance(small_files, dict):
            raise GuardError("D10 current-policy embedded small-file inventory is absent")
        embedded = {
            "LMT.md": route["lmt_sha256"],
            "tools/kaggle_kernel_certifier.py": route["current_certifier_sha256"],
            (
                "dag_prompts/2026-08-23-model-clade-renewal-v1/"
                "portfolio_parallel_nonm_v1/d_v10_current_policy_cert_v1/"
                "PREFLIGHT_PROFILE.json"
            ): route["profile_sha256"],
            (
                "dag_prompts/2026-08-23-model-clade-renewal-v1/"
                "portfolio_parallel_nonm_v1/d_v10_current_policy_cert_v1/"
                "package_v1/kernel.py"
            ): route["target_entrypoint_sha256"],
            (
                "dag_prompts/2026-08-23-model-clade-renewal-v1/"
                "portfolio_parallel_nonm_v1/d_v10_current_policy_cert_v1/"
                "package_v1/kernel-metadata.json"
            ): route["target_metadata_sha256"],
        }
        for relative, expected in embedded.items():
            value = small_files.get(relative)
            if not isinstance(value, str) or hashlib.sha256(
                base64.b64decode(value)
            ).hexdigest() != expected:
                raise GuardError(f"D10 current-policy embedded artifact drift: {relative}")
        markers = (
            '"certify"',
            "timeout=27900",
            "F2_DIRECT_D_SWEEP_V10_NATIVE_CERTIFICATION.json",
            "NATIVE_INTERACTIVE_RESULT.json",
        )
        if any(marker not in source for marker in markers):
            raise GuardError("D10 current-policy self-orchestration marker drift")
        if "sys.argv" in source or any(
            marker in source for marker in ("submission.csv", "44b6_6bba")
        ):
            raise GuardError("D10 current-policy control/outcome firewall drift")
    except (GuardError, SyntaxError, ValueError, TypeError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    if phase == "launch":
        failures.append(
            "D10 current-policy requires separate current-LMT/quota/fresh-slug/serial-lock coordinator"
        )
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "d10-current-policy-native-cert-v1",
        "record_sha256": route["package_freeze_sha256"],
        "video_count": 0,
        "identities": route["identities"],
        "target_input": True,
        "current_certifier_sha256": route["current_certifier_sha256"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [
            {
                "path": _relative(path, project_root, label),
                "sha256": expected,
            }
            for path, expected, label in artifacts
        ],
    }
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_d11_current_policy_native_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact D11 v3 transport successor of frozen D10 science."""

    failures: list[str] = []
    route = D11_CURRENT_POLICY_NATIVE_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "f2-d-v11-native-cert-current-v1",
    }
    if metadata != expected_metadata:
        failures.append("D11 exact private/offline/T4 script metadata drift")
    exact_contract = {
        "candidate_id": "D-V11-RUNTIME-INTERPRETER-TRANSPORT-v1",
        "certification_only": True,
        "code_sha256": route["code_sha256"],
        "completed_units": 4200,
        "configuration_count": 1,
        "current_certifier_sha256": route["current_certifier_sha256"],
        "current_ship_skill_sha256": route["current_ship_skill_sha256"],
        "downloaded_evidence_required_for_any_successor": True,
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "execution_profile": D11_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE,
        "expected_version": 1,
        "external_action_performed": False,
        "fresh_slug_required": True,
        "identity_count": 6,
        "kernel_id": route["kernel_id"],
        "lane_owner": "D11_RUNTIME_INTERPRETER_TRANSPORT_CERTIFICATION_ONLY",
        "launch_permitted": False,
        "lmt_sha256": route["lmt_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "model_execution": True,
        "one_shot": True,
        "outcome_access_permitted": False,
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "production_output_permitted": False,
        "profile_sha256": route["profile_sha256"],
        "projected_remote_upper_seconds": 26697.288724191087,
        "remote_preflight_kernel_id": route["kernel_id"],
        "reservation_hours": 7.75,
        "schema": "celltrack-alpha.d11-current-policy-central-prelaunch.v1",
        "scoring_permitted": False,
        "selected_identities": route["identities"],
        "source_freeze_sha256": route["source_freeze_sha256"],
        "static_readiness_sha256": route["static_readiness_sha256"],
        "submission_output_permitted": False,
        "target_input": True,
        "training_permitted": False,
        "v3_installation_receipt_sha256": route["v3_installation_receipt_sha256"],
        "v3_static_check_sha256": route["v3_static_check_sha256"],
        "whole_run_hard_bound_seconds": 27900,
    }
    if contract != exact_contract:
        failures.append("D11 central contract drift")
    if sha256(code_path) != route["code_sha256"] or sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("D11 code/metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("D11 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("D11 ship package inventory drift")
    if registry.get("kernel_ownership", {}).get(D10_CURRENT_POLICY_NATIVE_CERT_ROUTE["kernel_id"]) != "D10_CURRENT_POLICY_NATIVE_CERTIFICATION_ONLY":
        failures.append("D11 inherited D10 canonical ownership drift")

    node = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d_v11_runtime_interpreter_transport_v1"
    artifacts = (
        (node / "D11_RUNTIME_INTERPRETER_TRANSPORT_FREEZE.json", route["package_freeze_sha256"], "package freeze"),
        (node / "PREFLIGHT_PROFILE.json", route["profile_sha256"], "ship profile"),
        (node / "STATIC_READINESS.json", route["static_readiness_sha256"], "static readiness"),
        (node / "V3_STATIC_CHECK.json", route["v3_static_check_sha256"], "v3 static check"),
        (node / "seal_downloaded_evidence.py", route["downloaded_evidence_sealer_sha256"], "sealer"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
        (project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/kernel_skill_upgrade_v3/INSTALLATION_RECEIPT.json", route["v3_installation_receipt_sha256"], "v3 installation"),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"D11 {label} exact artifact drift")
    for path, expected, label in (
        (project_root / ".agents/skills/ship-kaggle-kernels/SKILL.md", route["current_ship_skill_sha256"], "ship skill"),
        (project_root / ".agents/skills/ship-kaggle-kernels/scripts/kaggle_kernel_certifier.py", route["current_certifier_sha256"], "certifier"),
    ):
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"D11 current {label} drift")

    try:
        encoded = _literal_assignment(tree, "PAYLOAD_Z85", "D11 payload")
        if not isinstance(encoded, str):
            raise GuardError("D11 payload is not a literal string")
        payload = json.loads(zlib.decompress(base64.b85decode(encoded)))
        bindings = payload.get("bindings", {})
        required_bindings = {
            "certifier_sha256": route["current_certifier_sha256"],
            "kernel_artifact_sha256": route["target_entrypoint_sha256"],
            "kernel_id": "onebitbrain/f2-d-v10-current-policy-target-v1",
            "kernel_metadata_sha256": route["target_metadata_sha256"],
            "kernel_package_inventory_sha256": route["target_package_inventory_sha256"],
            "lmt_sha256": route["lmt_sha256"],
            "profile_sha256": route["profile_sha256"],
            "ship_skill_v3_installation_sha256": route["v3_installation_receipt_sha256"],
            "successor_candidate_id": "D-V11-RUNTIME-INTERPRETER-TRANSPORT-v1",
        }
        if any(bindings.get(key) != value for key, value in required_bindings.items()):
            raise GuardError("D11 embedded v3 target bindings drift")
        small_files = payload.get("small_files", {})
        base = "dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d_v11_runtime_interpreter_transport_v1"
        embedded = {
            "LMT.md": route["lmt_sha256"],
            "tools/kaggle_kernel_certifier.py": route["current_certifier_sha256"],
            f"{base}/PREFLIGHT_PROFILE.json": route["profile_sha256"],
            f"{base}/package_v1/kernel.py": route["target_entrypoint_sha256"],
            f"{base}/package_v1/kernel-metadata.json": route["target_metadata_sha256"],
        }
        for relative, expected in embedded.items():
            value = small_files.get(relative)
            if not isinstance(value, str) or hashlib.sha256(base64.b64decode(value)).hexdigest() != expected:
                raise GuardError(f"D11 embedded artifact drift: {relative}")
        target_source = base64.b64decode(small_files[f"{base}/package_v1/kernel.py"]).decode()
        target_tree = ast.parse(target_source)
        if _literal_assignment(target_tree, "FROZEN_SOURCE_SHA256", "D11 scientific source") != route["scientific_source_sha256"]:
            raise GuardError("D11 frozen scientific source identity drift")
        if "D11_STEP0 = run_d11_step0()" not in target_source or 'HEAVY_WORK_MARKER = "V32_HEAVY_WORK_BEGIN"' not in target_source:
            raise GuardError("D11 v3 Step0/transport marker drift")
        if any(marker not in source for marker in ('"certify"', "timeout=27900", "F2_DIRECT_D_SWEEP_V11_NATIVE_CERTIFICATION.json", "NATIVE_INTERACTIVE_RESULT.json")):
            raise GuardError("D11 self-orchestration marker drift")
        if "sys.argv" in source or any(marker in source for marker in ("submission.csv", "44b6_6bba")):
            raise GuardError("D11 control/outcome firewall drift")
    except (GuardError, SyntaxError, ValueError, TypeError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    if phase == "launch":
        failures.append("D11 requires separate current-LMT/quota/fresh-slug/serial-lock coordinator")
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "d11-current-policy-native-cert-v1",
        "record_sha256": route["package_freeze_sha256"],
        "video_count": 0,
        "identities": route["identities"],
        "target_input": True,
        "current_certifier_sha256": route["current_certifier_sha256"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [{"path": _relative(path, project_root, label), "sha256": expected} for path, expected, label in artifacts],
    }
    return _result(
        kernel_dir=kernel_dir, metadata_path=metadata_path, code_path=code_path,
        metadata=metadata, contract=contract, registry=registry, phase=phase,
        failures=failures, step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary, project_root=project_root,
        contract_path=contract_path, registry_path=registry_path,
    )


def check_d16_current_policy_native_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact D16 v3 transport successor of frozen D10 science."""

    failures: list[str] = []
    route = D16_CURRENT_POLICY_NATIVE_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "f2-d-v16-native-cert-current-v1",
    }
    if metadata != expected_metadata:
        failures.append("D16 exact private/offline/T4 script metadata drift")
    exact_contract = {
        "candidate_id": "D-V16-DUAL-DAG-FINALIZATION-v1",
        "central_breaker_sha256": route["v3_static_check_sha256"],
        "certification_only": True,
        "code_sha256": route["code_sha256"],
        "completed_units": 1509,
        "configuration_count": 7,
        "current_certifier_sha256": route["current_certifier_sha256"],
        "current_ship_skill_sha256": route["current_ship_skill_sha256"],
        "downloaded_evidence_required_for_any_successor": True,
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "execution_profile": D16_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE,
        "expected_version": 1,
        "external_action_performed": False,
        "fresh_slug_required": True,
        "identity_count": 6,
        "kernel_id": route["kernel_id"],
        "lane_owner": "D16_DUAL_DAG_FINALIZATION_CERTIFICATION_ONLY",
        "launch_permitted": False,
        "lmt_sha256": route["lmt_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "model_execution": True,
        "one_shot": True,
        "outcome_access_permitted": False,
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "production_output_permitted": False,
        "profile_sha256": route["profile_sha256"],
        "projected_remote_upper_seconds": 26697.288724191087,
        "remote_preflight_kernel_id": route["kernel_id"],
        "reservation_hours": 7.75,
        "schema": "celltrack-alpha.d16-current-policy-central-prelaunch.v1",
        "scoring_permitted": False,
        "selected_identities": route["identities"],
        "source_freeze_sha256": route["source_freeze_sha256"],
        "static_readiness_sha256": route["static_readiness_sha256"],
        "submission_output_permitted": False,
        "target_input": True,
        "training_permitted": False,
        "v3_installation_receipt_sha256": route["v3_installation_receipt_sha256"],
        "v3_static_check_sha256": route["v3_static_check_sha256"],
        "whole_run_hard_bound_seconds": 27900,
    }
    if contract != exact_contract:
        failures.append("D16 central contract drift")
    if sha256(code_path) != route["code_sha256"] or sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("D16 code/metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("D16 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("D16 ship package inventory drift")
    if registry.get("kernel_ownership", {}).get(D10_CURRENT_POLICY_NATIVE_CERT_ROUTE["kernel_id"]) != "D10_CURRENT_POLICY_NATIVE_CERTIFICATION_ONLY":
        failures.append("D16 inherited D10 canonical ownership drift")

    node = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d_v16_dual_dag_finalization_v1"
    artifacts = (
        (node / "D16_BUILTIN_NUMERIC_CONTRACT_FREEZE.json", route["package_freeze_sha256"], "package freeze"),
        (node / "PREFLIGHT_PROFILE.json", route["profile_sha256"], "ship profile"),
        (node / "STATIC_READINESS.json", route["static_readiness_sha256"], "static readiness"),
        (node / "adversarial_breaker_v1/BREAKER_TERMINAL.json", route["v3_static_check_sha256"], "v3 static check"),
        (node / "seal_downloaded_evidence.py", route["downloaded_evidence_sealer_sha256"], "sealer"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
        (project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/kernel_skill_upgrade_v3/INSTALLATION_RECEIPT.json", route["v3_installation_receipt_sha256"], "v3 installation"),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"D16 {label} exact artifact drift")
    for path, expected, label in (
        (project_root / ".agents/skills/ship-kaggle-kernels/SKILL.md", route["current_ship_skill_sha256"], "ship skill"),
        (project_root / ".agents/skills/ship-kaggle-kernels/scripts/kaggle_kernel_certifier.py", route["current_certifier_sha256"], "certifier"),
    ):
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"D16 current {label} drift")

    try:
        encoded = _literal_assignment(tree, "PAYLOAD_Z85", "D16 payload")
        if not isinstance(encoded, str):
            raise GuardError("D16 payload is not a literal string")
        payload = json.loads(zlib.decompress(base64.b85decode(encoded)))
        bindings = payload.get("bindings", {})
        required_bindings = {
            "certifier_sha256": route["current_certifier_sha256"],
            "kernel_artifact_sha256": route["target_entrypoint_sha256"],
            "kernel_id": "onebitbrain/f2-d-v14-builtin-numeric-contract-v1",
            "kernel_metadata_sha256": route["target_metadata_sha256"],
            "kernel_package_inventory_sha256": route["target_package_inventory_sha256"],
            "lmt_sha256": route["lmt_sha256"],
            "profile_sha256": route["profile_sha256"],
            "successor_candidate_id": "D-V14-BUILTIN-NUMERIC-CONTRACT-v1",
        }
        if any(bindings.get(key) != value for key, value in required_bindings.items()):
            raise GuardError("D16 embedded v3 target bindings drift")
        small_files = payload.get("small_files", {})
        base = "dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d_v14_builtin_numeric_contract_v1"
        embedded = {
            "LMT.md": route["lmt_sha256"],
            "tools/kaggle_kernel_certifier.py": route["current_certifier_sha256"],
            f"{base}/PREFLIGHT_PROFILE.json": route["profile_sha256"],
            f"{base}/package_v1/kernel.py": route["target_entrypoint_sha256"],
            f"{base}/package_v1/kernel-metadata.json": route["target_metadata_sha256"],
        }
        for relative, expected in embedded.items():
            value = small_files.get(relative)
            if not isinstance(value, str) or hashlib.sha256(base64.b64decode(value)).hexdigest() != expected:
                raise GuardError(f"D16 embedded artifact drift: {relative}")
        target_source = base64.b64decode(small_files[f"{base}/package_v1/kernel.py"]).decode()
        target_tree = ast.parse(target_source)
        if _literal_assignment(target_tree, "FROZEN_SOURCE_SHA256", "D16 scientific source") != route["scientific_source_sha256"]:
            raise GuardError("D16 frozen scientific source identity drift")
        if "D14_STEP0 = run_d14_step0()" not in target_source or 'HEAVY_WORK_MARKER = "V32_HEAVY_WORK_BEGIN"' not in target_source:
            raise GuardError("D16 v3 Step0/transport marker drift")
        if any(marker not in source for marker in ('"certify"', "timeout=27900", "F2_DIRECT_D_SWEEP_V14_NATIVE_CERTIFICATION.json", "NATIVE_INTERACTIVE_RESULT.json")):
            raise GuardError("D16 self-orchestration marker drift")
        if "sys.argv" in source or any(marker in source for marker in ("submission.csv", "44b6_6bba")):
            raise GuardError("D16 control/outcome firewall drift")
    except (GuardError, SyntaxError, ValueError, TypeError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    if phase == "launch":
        failures.append("D16 requires separate current-LMT/quota/fresh-slug/serial-lock coordinator")
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "d16-current-policy-native-cert-v1",
        "record_sha256": route["package_freeze_sha256"],
        "video_count": 0,
        "identities": route["identities"],
        "target_input": True,
        "current_certifier_sha256": route["current_certifier_sha256"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [{"path": _relative(path, project_root, label), "sha256": expected} for path, expected, label in artifacts],
    }
    return _result(
        kernel_dir=kernel_dir, metadata_path=metadata_path, code_path=code_path,
        metadata=metadata, contract=contract, registry=registry, phase=phase,
        failures=failures, step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary, project_root=project_root,
        contract_path=contract_path, registry_path=registry_path,
    )


def _d28_typed_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected): return False
    if isinstance(expected, dict): return actual.keys() == expected.keys() and all(_d28_typed_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list): return len(actual) == len(expected) and all(_d28_typed_equal(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def check_d28_closed_schema_native_certification(
    *, kernel_dir: Path, metadata_path: Path, metadata: dict[str, Any], code_path: Path,
    source: str, tree: ast.Module, contract: dict[str, Any], registry: dict[str, Any],
    phase: str, expected_initial_kernel_version: int | None, project_root: Path,
    contract_path: Path | None, registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the frozen D28 closed-schema native-certification package."""
    failures: list[str] = []
    route = D28_CLOSED_SCHEMA_NATIVE_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_contract = {'schema': 'celltrack-alpha.d28-central-prelaunch.v1',
 'candidate_id': 'D28-FRESH-RECONCILIATION-TIME-CENTRAL-RELEASE-v1',
 'execution_profile': 'd28_fresh_reconciliation_time_native_t4_certification.v1',
 'kernel_id': 'onebitbrain/f2-d28-native-cert-fresh-reconciliation-time-v1',
 'remote_preflight_kernel_id': 'onebitbrain/f2-d28-native-cert-fresh-reconciliation-time-v1',
 'lane_owner': 'D28_CLOSED_SCHEMA_NATIVE_CERTIFICATION_ONLY',
 'expected_version': 1,
 'fresh_slug_required': True,
 'one_shot': True,
 'certification_only': True,
 'model_execution': True,
 'target_input': True,
 'training_permitted': False,
 'scoring_permitted': False,
 'submission_output_permitted': False,
 'outcome_access_permitted': False,
 'external_action_performed': False,
 'launch_permitted': False,
 'projected_remote_upper_seconds': 23060.206632093435,
 'whole_run_hard_bound_seconds': 27900,
 'reservation_hours': 7.75,
 'code_sha256': '1504f1656594e59788ad55357039cfd725247f84971a06afc38bcd263455223c',
 'metadata_sha256': '9949c051c255ccefa634c327980f4943d706435a06c77be623823aaac0cac0de',
 'package_tree_sha256': '3995593a56d67614fb0fa8fbbd883b454056d8928153691a3f328869ba366e6e',
 'package_inventory_sha256': '68f91649ef3f8cd894273681003dad76e2a39d698e4f55bc55675c732b543c0f',
 'package_freeze_sha256': '821537b6ae65af0452c696da3cbea3ec05b0e9b959a71dd4fadf4dd5cba61ca7',
 'd24_profile_sha256': 'd5db744102748a07109584649a35825c74cbc53545f4ea7b21125680e99eddcf',
 'd24_static_terminal_sha256': '7615f31dce76c625b0eccc0b158bbc83eb5eba10b1bca79902dac96d0cebd99d',
 'd24_independent_breaker_sha256': '2bd27b250f76a75edec1c62184b031aaa8168e74b305a71191d8f33a720e22dc',
 'd24_central_nogo_sha256': '35ff73e0f0bc54ecd195b9c87f18dec27da3af6186a4a662eabbf94a369ba35f',
 'd25_independent_nogo_sha256': '8594556d7fa8dfdade5f142cbb7349ba0bad6c7408c5fcf9c1743511691ce691',
 'd26_independent_nogo_sha256': '35a0f0253f4101b07c62d3676396901eb534e9fab41a1f8e5fd738275f525d7c',
 'd27_independent_nogo_sha256': '3ef1196ebacd69a5fe01d6b384677b07bfb84d9b7e80a9c3f96f4a13d3719977',
 'reconciliation_time_policy_sha256': '7c79fa3d8bcba922916c99f1be64ee9c32a15067fc1a36ebed8746155019eee4',
 'd24_source_freeze_sha256': 'a4074a78772fa3a557c256c2cffa3c29435e3c0bd96b3c55ad6cf01e295516a6',
 'd24_runtime_contract_sha256': '01702a2fb5a0f165cde1380aa40e845e1d4d6c029bf8cad197f9fdb23eca6fba',
 'd24_certifier_copy_sha256': '1cc8408c691c4ab1c7a1f4e9710bbe5c6c8d8a7e7ebd6547f17f92d7cab89466',
 'd24_executable_sha256': '1d7235a0f6989a3304311925eb8fe46be1e6a3bf835df030fa147fe6ebac4e21',
 'lmt_sha256': '35376648ab432e2d7c901b89e5efeb3542c154af5c1dff8fdc3dd681949fb478',
 'd28_priority_release_after_fresh_reconciliation': True,
 'required_lane_compatibility': ['m_class_frontier', 'v_class_frontier'],
 'retry_or_version2_authority': False}
    if metadata != route["expected_metadata"]: failures.append("D28 exact private/offline/T4 metadata drift")
    if not _d28_typed_equal(contract, expected_contract): failures.append("D28 central contract drift")
    if sha256(code_path) != route["code_sha256"] or sha256(metadata_path) != route["metadata_sha256"]: failures.append("D28 code/metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]: failures.append("D28 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]: failures.append("D28 ship package inventory drift")
    node = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d28_fresh_reconciliation_time_central_release_v1"
    d24 = node.parent / "d24_closed_schema_runtime_certifier_v1"
    artifacts = (
        (node / "D28_PACKAGE_FREEZE.json", route["package_freeze_sha256"], "D28 package freeze"),
        (node / "native_cert_package_v1/d24_kernel.py", route["d24_executable_sha256"], "exact D24 executable"),
        (d24 / "PREFLIGHT_PROFILE.json", route["d24_profile_sha256"], "D24 profile"),
        (d24 / "STATIC_TERMINAL.json", route["d24_static_terminal_sha256"], "D24 static terminal"),
        (d24 / "independent_breaker_v1/BREAKER_TERMINAL.json", route["d24_breaker_sha256"], "D24 independent breaker"),
        (d24 / "central_integration_v1/independent_breaker_v1/BREAKER_TERMINAL.json", route["d24_central_nogo_sha256"], "D24 central NO_GO"),
        (node.parent / "d25_closed_central_release_v1/independent_breaker_v1/BREAKER_TERMINAL.json", route["d25_independent_nogo_sha256"], "D25 independent NO_GO"),
        (node.parent / "d26_exact_typed_claim_central_release_v1/central_integration_v1/independent_breaker_v1/BREAKER_TERMINAL.json", route["d26_independent_nogo_sha256"], "D26 independent NO_GO"),
        (node.parent / "d27_closed_reconciliation_central_release_v1/independent_breaker_v1/BREAKER_TERMINAL.json", route["d27_independent_nogo_sha256"], "D27 independent NO_GO"),
        (node / "RECONCILIATION_TIME_POLICY.json", route["reconciliation_time_policy_sha256"], "D28 reconciliation-time policy"),
        (d24 / "SOURCE_PROOF_FREEZE.json", route["d24_source_freeze_sha256"], "D24 source freeze"),
        (d24 / "runtime_envelope_contract.py", route["d24_runtime_contract_sha256"], "D24 runtime contract"),
        (d24 / "certifier_v5/kaggle_kernel_certifier.py", route["d24_certifier_copy_sha256"], "D24 certifier copy"),
        (project_root / "LMT.md", route["lmt_sha256"], "LMT"),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected: failures.append(f"D28 {label} drift")
    try:
        d24_source = (node / "native_cert_package_v1/d24_kernel.py").read_text()
        d24_tree = ast.parse(d24_source)
        encoded = _literal_assignment(d24_tree, "PAYLOAD_Z85", "D24 payload")
        payload = json.loads(zlib.decompress(base64.b85decode(encoded)))
        bindings = payload.get("bindings", {})
        required = {
            "certifier_sha256": route["d24_certifier_copy_sha256"], "runtime_contract_sha256": route["d24_runtime_contract_sha256"],
            "source_proof_freeze_sha256": route["d24_source_freeze_sha256"], "kernel_artifact_sha256": route["target_entrypoint_sha256"],
            "kernel_id": "onebitbrain/f2-d24-closed-schema-runtime-v1", "kernel_metadata_sha256": route["target_metadata_sha256"],
            "kernel_package_inventory_sha256": route["target_package_inventory_sha256"], "lmt_sha256": route["lmt_sha256"],
            "profile_sha256": route["d24_profile_sha256"], "successor_candidate_id": "D24-CLOSED-SCHEMA-RUNTIME-CERTIFIER-v1", "retroactive_d19_eligibility": False,
        }
        if any(type(bindings.get(k)) is not type(v) or bindings.get(k) != v for k, v in required.items()): raise GuardError("D28 embedded binding drift")
        if any(marker not in d24_source for marker in ('"certify"', "timeout=27900", "F2_DIRECT_D_SWEEP_V14_NATIVE_CERTIFICATION.json", "NATIVE_INTERACTIVE_RESULT.json")): raise GuardError("D28 embedded D24 self-orchestration marker drift")
        if "sys.argv" in d24_source or any(marker in d24_source + source for marker in ("submission.csv", "44b6_6bba")): raise GuardError("D28 control/outcome firewall drift")
    except (GuardError, SyntaxError, ValueError, TypeError, json.JSONDecodeError) as exc: failures.append(str(exc))
    if phase == "launch": failures.append("D28 requires separate current-LMT/quota/fresh-slug/serial-lock coordinator")
    visual = {"required": True, "status": "REMOTE_CERTIFICATION_PENDING", "package_tree_sha256": package["tree_sha256"], "production_enabled": False, "certification_only": True, "exception_id": "d28-fresh-reconciliation-time-central-release-v1", "record_sha256": route["d24_central_nogo_sha256"], "video_count": 0, "identities": [], "target_input": True, "d28_priority_release_pending_terminal": True, "post_remote_certification": {"downloaded_output_zip_required": True, "exact_local_seal_required": True, "production_or_score_permitted_before_seal": False, "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL"}, "artifacts": [{"path": _relative(p, project_root, label), "sha256": expected} for p, expected, label in artifacts]}
    return _result(kernel_dir=kernel_dir, metadata_path=metadata_path, code_path=code_path, metadata=metadata, contract=contract, registry=registry, phase=phase, failures=failures, step0=None, expected_initial_kernel_version=expected_initial_kernel_version, local_visual_preflight=visual, project_root=project_root, contract_path=contract_path, registry_path=registry_path)


def check_d19_current_policy_native_certification(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate only the exact D19 v3 transport successor of frozen D10 science."""

    failures: list[str] = []
    route = D19_CURRENT_POLICY_NATIVE_CERT_ROUTE
    package = _package_inventory(kernel_dir, project_root)
    expected_metadata = {
        "code_file": "kernel.py",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": route["kernel_id"],
        "is_private": True,
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "f2-d-v19-native-cert-current-v1",
    }
    if metadata != expected_metadata:
        failures.append("D19 exact private/offline/T4 script metadata drift")
    exact_contract = {
        "candidate_id": "D-V19-EXACT-ZIP-MEMBER-MULTISET-v1",
        "central_breaker_sha256": route["v3_static_check_sha256"],
        "certification_only": True,
        "code_sha256": route["code_sha256"],
        "completed_units": 1509,
        "configuration_count": 7,
        "current_certifier_sha256": route["current_certifier_sha256"],
        "current_ship_skill_sha256": route["current_ship_skill_sha256"],
        "downloaded_evidence_required_for_any_successor": True,
        "downloaded_evidence_sealer_sha256": route["downloaded_evidence_sealer_sha256"],
        "execution_profile": D19_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE,
        "expected_version": 1,
        "external_action_performed": False,
        "fresh_slug_required": True,
        "identity_count": 6,
        "kernel_id": route["kernel_id"],
        "lane_owner": "D19_EXACT_ZIP_MEMBER_MULTISET_CERTIFICATION_ONLY",
        "launch_permitted": False,
        "lmt_sha256": route["lmt_sha256"],
        "metadata_sha256": route["metadata_sha256"],
        "model_execution": True,
        "one_shot": True,
        "outcome_access_permitted": False,
        "package_freeze_sha256": route["package_freeze_sha256"],
        "package_inventory_sha256": route["package_inventory_sha256"],
        "package_tree_sha256": route["package_tree_sha256"],
        "production_output_permitted": False,
        "profile_sha256": route["profile_sha256"],
        "projected_remote_upper_seconds": 26697.288724191087,
        "remote_preflight_kernel_id": route["kernel_id"],
        "reservation_hours": 7.75,
        "schema": "celltrack-alpha.d19-current-policy-central-prelaunch.v1",
        "scoring_permitted": False,
        "selected_identities": route["identities"],
        "source_freeze_sha256": route["source_freeze_sha256"],
        "static_readiness_sha256": route["static_readiness_sha256"],
        "submission_output_permitted": False,
        "target_input": True,
        "training_permitted": False,
        "v3_installation_receipt_sha256": route["v3_installation_receipt_sha256"],
        "v3_static_check_sha256": route["v3_static_check_sha256"],
        "whole_run_hard_bound_seconds": 27900,
    }
    if contract != exact_contract:
        failures.append("D19 central contract drift")
    if sha256(code_path) != route["code_sha256"] or sha256(metadata_path) != route["metadata_sha256"]:
        failures.append("D19 code/metadata hash drift")
    if package["tree_sha256"] != route["package_tree_sha256"]:
        failures.append("D19 package tree drift")
    if _ship_package_inventory_sha256(kernel_dir) != route["package_inventory_sha256"]:
        failures.append("D19 ship package inventory drift")
    if registry.get("kernel_ownership", {}).get(D10_CURRENT_POLICY_NATIVE_CERT_ROUTE["kernel_id"]) != "D10_CURRENT_POLICY_NATIVE_CERTIFICATION_ONLY":
        failures.append("D19 inherited D10 canonical ownership drift")

    node = project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d_v19_exact_zip_member_multiset_v1"
    artifacts = (
        (node / "D19_EXACT_ZIP_MEMBER_MULTISET_FREEZE.json", route["package_freeze_sha256"], "package freeze"),
        (node / "PREFLIGHT_PROFILE.json", route["profile_sha256"], "ship profile"),
        (node / "STATIC_READINESS.json", route["static_readiness_sha256"], "static readiness"),
        (node / "adversarial_breaker_v1/BREAKER_TERMINAL.json", route["v3_static_check_sha256"], "v3 static check"),
        (node / "seal_downloaded_evidence.py", route["downloaded_evidence_sealer_sha256"], "sealer"),
        (project_root / "LMT.md", route["lmt_sha256"], "current LMT"),
        (project_root / "dag_prompts/2026-08-23-model-clade-renewal-v1/kernel_skill_upgrade_v3/INSTALLATION_RECEIPT.json", route["v3_installation_receipt_sha256"], "v3 installation"),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            failures.append(f"D19 {label} exact artifact drift")
    for path, expected, label in (
        (project_root / ".agents/skills/ship-kaggle-kernels/SKILL.md", route["current_ship_skill_sha256"], "ship skill"),
        (project_root / ".agents/skills/ship-kaggle-kernels/scripts/kaggle_kernel_certifier.py", route["current_certifier_sha256"], "certifier"),
    ):
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"D19 current {label} drift")

    try:
        encoded = _literal_assignment(tree, "PAYLOAD_Z85", "D19 payload")
        if not isinstance(encoded, str):
            raise GuardError("D19 payload is not a literal string")
        payload = json.loads(zlib.decompress(base64.b85decode(encoded)))
        bindings = payload.get("bindings", {})
        required_bindings = {
            "certifier_sha256": route["current_certifier_sha256"],
            "kernel_artifact_sha256": route["target_entrypoint_sha256"],
            "kernel_id": "onebitbrain/f2-d-v19-exact-zip-member-multiset-v1",
            "kernel_metadata_sha256": route["target_metadata_sha256"],
            "kernel_package_inventory_sha256": route["target_package_inventory_sha256"],
            "lmt_sha256": route["lmt_sha256"],
            "profile_sha256": route["profile_sha256"],
            "successor_candidate_id": "D-V19-EXACT-ZIP-MEMBER-MULTISET-v1",
        }
        if any(bindings.get(key) != value for key, value in required_bindings.items()):
            raise GuardError("D19 embedded v3 target bindings drift")
        small_files = payload.get("small_files", {})
        base = "dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d_v19_exact_zip_member_multiset_v1"
        embedded = {
            "LMT.md": route["lmt_sha256"],
            "tools/kaggle_kernel_certifier.py": route["current_certifier_sha256"],
            f"{base}/PREFLIGHT_PROFILE.json": route["profile_sha256"],
            f"{base}/package_v1/kernel.py": route["target_entrypoint_sha256"],
            f"{base}/package_v1/kernel-metadata.json": route["target_metadata_sha256"],
        }
        for relative, expected in embedded.items():
            value = small_files.get(relative)
            if not isinstance(value, str) or hashlib.sha256(base64.b64decode(value)).hexdigest() != expected:
                raise GuardError(f"D19 embedded artifact drift: {relative}")
        target_source = base64.b64decode(small_files[f"{base}/package_v1/kernel.py"]).decode()
        target_tree = ast.parse(target_source)
        if _literal_assignment(target_tree, "FROZEN_SOURCE_SHA256", "D19 scientific source") != route["scientific_source_sha256"]:
            raise GuardError("D19 frozen scientific source identity drift")
        if "D14_STEP0 = run_d14_step0()" not in target_source or 'HEAVY_WORK_MARKER = "V32_HEAVY_WORK_BEGIN"' not in target_source:
            raise GuardError("D19 v3 Step0/transport marker drift")
        if any(marker not in source for marker in ('"certify"', "timeout=27900", "F2_DIRECT_D_SWEEP_V14_NATIVE_CERTIFICATION.json", "NATIVE_INTERACTIVE_RESULT.json")):
            raise GuardError("D19 self-orchestration marker drift")
        if "sys.argv" in source or any(marker in source for marker in ("submission.csv", "44b6_6bba")):
            raise GuardError("D19 control/outcome firewall drift")
    except (GuardError, SyntaxError, ValueError, TypeError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    if phase == "launch":
        failures.append("D19 requires separate current-LMT/quota/fresh-slug/serial-lock coordinator")
    visual_summary = {
        "required": True,
        "status": "REMOTE_CERTIFICATION_PENDING",
        "package_tree_sha256": package["tree_sha256"],
        "production_enabled": False,
        "certification_only": True,
        "exception_id": "d19-current-policy-native-cert-v1",
        "record_sha256": route["package_freeze_sha256"],
        "video_count": 0,
        "identities": route["identities"],
        "target_input": True,
        "current_certifier_sha256": route["current_certifier_sha256"],
        "post_remote_certification": {
            "downloaded_output_zip_required": True,
            "exact_local_seal_required": True,
            "production_or_score_permitted_before_seal": False,
            "status": "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_SEAL",
        },
        "artifacts": [{"path": _relative(path, project_root, label), "sha256": expected} for path, expected, label in artifacts],
    }
    return _result(
        kernel_dir=kernel_dir, metadata_path=metadata_path, code_path=code_path,
        metadata=metadata, contract=contract, registry=registry, phase=phase,
        failures=failures, step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary, project_root=project_root,
        contract_path=contract_path, registry_path=registry_path,
    )


def check_certified_sweep_wrapper(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate a frozen D/E wrapper plus the exact visual-executed child it embeds."""

    failures: list[str] = []
    package = _package_inventory(kernel_dir, project_root)
    visual_summary: dict[str, Any] = {
        "required": True,
        "status": "FAIL",
        "receipt_path": None,
        "receipt_sha256": None,
        "video_count": 0,
        "identities": [],
        "artifacts": [],
    }
    ship_summary: dict[str, Any] | None = None
    family = contract.get("family")
    native_matches = [
        (key, route)
        for key, route in F2_D_NATIVE_VISUAL_ROUTES.items()
        if contract.get("visual_evidence_profile") == route["execution_profile"]
    ]
    native_route_key = native_matches[0][0] if len(native_matches) == 1 else None
    native_route_record = native_matches[0][1] if len(native_matches) == 1 else None
    native_route = native_route_record is not None
    native_fields_present = any(
        key in contract
        for key in (
            "visual_evidence_profile",
            "native_t4_visual_production_contract",
            "downloaded_native_visual_zip",
        )
    )
    if native_route_record is not None:
        if family == "D":
            expected = dict(F2_SWEEP_V6_KERNELS["D"])
            expected["kernel_id"] = native_route_record["bindings"]["kernel_id"]
        else:
            expected = None
    else:
        expected = F2_SWEEP_KERNELS.get(str(family))
    expected_candidates = [dict(item) for item in F2_SWEEP_CANDIDATES.get(str(family), ())]
    if expected is None:
        failures.append(
            "certified sweep wrapper family/profile must be exact v5 D/E or one registered native D route"
        )
        expected = {"kernel_id": "", "code_file": "", "lane_owner": "", "output_root": ""}
    if native_fields_present and not native_route:
        failures.append("certified sweep wrapper has an incomplete native visual route")
    exact_contract = {
        "schema": F2_CERTIFIED_SWEEP_CONTRACT_SCHEMA,
        "execution_profile": CERTIFIED_SWEEP_WRAPPER_EXECUTION_PROFILE,
        "model_execution": True,
        "kernel_id": expected["kernel_id"],
        "remote_preflight_kernel_id": expected["kernel_id"],
        "lane_owner": expected["lane_owner"],
        "preregistered_visual_identities": list(F2_SWEEP_IDENTITIES),
        "same_run_control_id": "CONTROL_V32",
        "submission_output_permitted": False,
        "final_graph_rows_required": True,
        "arbitrary_candidate_or_config_surface_permitted": False,
    }
    for key, value in exact_contract.items():
        if contract.get(key) != value:
            failures.append(f"certified sweep wrapper contract drift: {key}")
    if metadata.get("id") != expected["kernel_id"]:
        failures.append("certified sweep wrapper kernel ID drift")
    expected_metadata = {
        "code_file": expected["code_file"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": expected["kernel_id"],
        "is_private": True,
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": expected["kernel_id"].split("/", 1)[-1],
    }
    if metadata != expected_metadata:
        failures.append("certified sweep wrapper metadata drift")
    code_sha = sha256(code_path)
    if contract.get("code_sha256") != code_sha:
        failures.append("certified sweep wrapper code hash drift")
    if contract.get("metadata_sha256") != sha256(metadata_path):
        failures.append("certified sweep wrapper metadata hash drift")
    if contract.get("package_tree_sha256") != package["tree_sha256"]:
        failures.append("certified sweep wrapper package-tree hash drift")
    if contract.get("package_inventory_sha256") != _ship_package_inventory_sha256(kernel_dir):
        failures.append("certified sweep wrapper ship package-inventory hash drift")

    child_source = ""
    try:
        if _literal_assignment(tree, "FAMILY", "F2 sweep FAMILY") != family:
            raise GuardError("F2 sweep FAMILY literal drift")
        child_source = _literal_assignment(tree, "CHILD_SOURCE", "F2 sweep CHILD_SOURCE")
        if not isinstance(child_source, str) or not child_source:
            raise GuardError("F2 sweep CHILD_SOURCE must be a nonempty string")
        candidates = _literal_assignment(tree, "CANDIDATES", "F2 sweep CANDIDATES")
        if candidates != expected_candidates:
            raise GuardError("F2 sweep candidate list differs from the central frozen family")
        moved = _literal_assignment(tree, "MOVED_OUTPUTS", "F2 sweep MOVED_OUTPUTS")
        if moved != (
            "v32_preilp_cache_v1",
            "V32_FINAL_GRAPH_ROWS.csv",
            "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json",
            "PUBLIC_TEMPORALUNET_STEP0.json",
            "V32_SHIP_PREFLIGHT_COMPLETION.json",
            "RUN_LOG.jsonl",
            "run_stats.csv",
        ):
            raise GuardError("F2 sweep moved-output allowlist drift")
        imported = {
            root for node in ast.walk(tree) for root in imported_roots(node)
        }
        unexpected_imports = imported - {
            "hashlib", "json", "os", "pathlib", "shutil", "subprocess", "sys", "time"
        }
        if unexpected_imports:
            raise GuardError(
                "F2 sweep wrapper has arbitrary imports: " + ", ".join(sorted(unexpected_imports))
            )
        dangerous_calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"compile", "eval", "exec", "input"}
        ]
        if dangerous_calls:
            raise GuardError("F2 sweep wrapper has an arbitrary code surface")
        env_gets = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                node.func.attr == "get"
                and isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "os"
                and owner.attr == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                env_gets.append(node.args[0].value)
        if set(env_gets) != {"KAGGLE_PREFLIGHT_MODE", "V32_RUNTIME_SAFETY_PROBE"}:
            raise GuardError("F2 sweep wrapper environment-control surface drift")
        required_outer_markers = (
            'environment["F2_SWEEP_CANDIDATE_ID"] = candidate["id"]',
            'if (WORKING / "submission.csv").exists():',
            '"submission_absent": not (WORKING / "submission.csv").exists()',
            '"V32_FINAL_GRAPH_ROWS.csv"',
            'print("F2_DIRECT_SWEEP_COMPLETE", flush=True)',
        )
        if any(marker not in source for marker in required_outer_markers):
            raise GuardError("F2 sweep wrapper output/logging sentinel drift")

        child_sha = hashlib.sha256(child_source.encode("utf-8")).hexdigest()
        child_tree = ast.parse(child_source, filename="<embedded F2 V32 child>")
        selected = _literal_assignment(
            child_tree, "SELECTED_DATASETS", "embedded F2 child SELECTED_DATASETS"
        )
        if selected != [identity.split(":", 1)[1] for identity in F2_SWEEP_IDENTITIES]:
            raise GuardError("embedded F2 child identity population drift")
        child_markers = (
            "RUN_LOG.jsonl", "inventory_input", "log_lookup", "stage_end",
            "preprocess_counts", "_run_failure", "SHIP_STEP0 = _v32_ship_step0()",
            "STEP0 = run_step0()", "V32_HEAVY_WORK_BEGIN",
            "TEST_DIR = SELECTED_TRAIN", "V32_FINAL_GRAPH_ROWS_PRESERVED",
            "V32_CACHE_NO_SUBMISSION_CONFIRMED",
        )
        if any(marker not in child_source for marker in child_markers):
            raise GuardError("embedded F2 child Step0/logging/output contract drift")
        test_assignments = [
            node
            for node in child_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "TEST_DIR"
                for target in node.targets
            )
        ]
        safe_test_assignments: list[int] = []
        for node in test_assignments:
            rendered = ast.get_source_segment(child_source, node) or ""
            if "COMP_DIR" in rendered or re.search(r"[/'\"]test[\"']?", rendered):
                raise GuardError(
                    "embedded F2 child TEST_DIR assignment reaches competition test"
                )
            if (
                "public_temporalunet_selected_train" in rendered
                or (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "SELECTED_TRAIN"
                )
            ):
                safe_test_assignments.append(node.lineno)
        if len(safe_test_assignments) != len(test_assignments) or not safe_test_assignments:
            raise GuardError("embedded F2 child selected-train TEST_DIR firewall is missing")
        unsafe_test_loads = sorted(
            node.lineno
            for node in ast.walk(child_tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == "TEST_DIR"
            and not any(line < node.lineno for line in safe_test_assignments)
        )
        if unsafe_test_loads:
            raise GuardError(
                "embedded F2 child visits TEST_DIR before selected-train firewall: "
                + ",".join(str(line) for line in unsafe_test_loads)
            )
        if contract.get("embedded_child_sha256") != child_sha:
            raise GuardError("certified sweep wrapper embedded child hash drift")

        freeze_path, freeze_sha = _bound_reference(
            contract.get("source_freeze"),
            project_root=project_root,
            label="F2 sweep source freeze",
            suffix=".json",
        )
        freeze = load_json(freeze_path)
        freeze_exact = {
            "schema": "celltrack-alpha.f2-direct-de-sweep-source-freeze.v1",
            "status": "PASS_BUILD_ONLY",
            "family": family,
            "kernel_id": expected["kernel_id"],
            "candidate_ids": [item["id"] for item in expected_candidates],
            "frozen_experimental_candidate_ids": [
                item["id"] for item in expected_candidates if item["id"] != "CONTROL_V32"
            ],
            "candidates": expected_candidates,
            "identities": list(F2_SWEEP_IDENTITIES),
            "same_run_control_id": "CONTROL_V32",
            "evaluate_in_kernel": False,
            "label_free_kernel": True,
            "submission_written": False,
            "child_source_sha256": child_sha,
        }
        for key, value in freeze_exact.items():
            if freeze.get(key) != value:
                raise GuardError(f"F2 sweep source-freeze drift: {key}")
        if freeze.get("files") != {
            expected["code_file"]: code_sha,
            "kernel-metadata.json": sha256(metadata_path),
        }:
            raise GuardError("F2 sweep source-freeze file hashes drift")
        if contract.get("frozen_candidates_sha256") != _canonical_sha(expected_candidates):
            raise GuardError("F2 sweep contract candidate digest drift")
        if native_route:
            if "child_visual_evidence" in contract:
                raise GuardError(
                    "native visual route cannot substitute or mix the v5 CPU-local evidence"
                )
            production_path, _ = _bound_reference(
                contract.get("native_t4_visual_production_contract"),
                project_root=project_root,
                label="F2 D native production visual contract",
                suffix=".json",
            )
            if load_json(production_path).get("native_ship_certificate") != contract.get(
                "ship_kaggle_certificate"
            ):
                raise GuardError(
                    "F2 D production/wrapper native certificate reference drift"
                )
            visual_summary, visual_artifacts = check_f2_d_v6_native_production_contract(
                production_contract_reference=contract.get(
                    "native_t4_visual_production_contract"
                ),
                downloaded_zip_reference=contract.get(
                    "downloaded_native_visual_zip"
                ),
                kernel_dir=kernel_dir,
                code_path=code_path,
                metadata_path=metadata_path,
                embedded_child_sha256=child_sha,
                project_root=project_root,
                _route_key=str(native_route_key),
            )
        else:
            visual_summary, visual_artifacts = _validate_f2_child_visual_evidence(
                evidence=contract.get("child_visual_evidence"),
                embedded_child_sha256=child_sha,
                project_root=project_root,
            )
        visual_summary["required"] = True
        visual_summary["package_tree_sha256"] = package["tree_sha256"]
        visual_summary["artifacts"] = visual_summary.get("artifacts", []) + [
            {"role": "source_freeze", **_reference_dict(freeze_path, freeze_sha, project_root, "F2 sweep source freeze")},
            *visual_artifacts,
        ]
        ship_summary, ship_artifacts = _validate_f2_sweep_ship_certificate(
            contract=contract,
            kernel_dir=kernel_dir,
            metadata=metadata,
            code_sha256=code_sha,
            project_root=project_root,
            expected=expected,
            candidates=expected_candidates,
        )
        visual_summary["artifacts"].extend(ship_artifacts)
        visual_summary["ship_kaggle_certificate"] = ship_summary
    except (GuardError, OSError, json.JSONDecodeError, SyntaxError) as exc:
        failures.append(str(exc))

    if expected_initial_kernel_version not in {None, 1}:
        failures.append("expected initial kernel version must be exactly 1")
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_summary,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_local_visual_preflight(
    *,
    contract: dict[str, Any],
    code_sha256: str,
    code_path: Path,
    kernel_dir: Path,
    source: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Validate exact-candidate, local, human-readable tracking evidence."""

    failures: list[str] = []
    summary: dict[str, Any] = {
        "required": None,
        "status": "FAIL",
        "receipt_path": None,
        "receipt_sha256": None,
        "video_count": 0,
        "identities": [],
        "artifacts": [],
    }
    model_execution = contract.get("model_execution")
    if not isinstance(model_execution, bool):
        failures.append("model_execution classification is missing or not boolean")
        return summary, failures
    summary["required"] = model_execution
    if not model_execution:
        allowlist_id = contract.get("non_model_execution_allowlist_id")
        expected = NON_MODEL_EXECUTION_ALLOWLIST.get(str(allowlist_id))
        actual = (
            contract.get("kernel_id"),
            contract.get("execution_profile"),
            code_sha256,
        )
        justification = contract.get("non_model_execution_justification")
        if expected is None or expected != actual:
            failures.append("non-model exemption is not an exact centrally allowlisted kernel/profile/code tuple")
        if not isinstance(justification, str) or len(justification.strip()) < 30:
            failures.append("non-model exemption lacks a specific justification")
        try:
            tree = ast.parse(source, filename=str(code_path))
            markers = _source_has_model_markers(source, tree)
            if markers:
                failures.append(
                    "non-model exemption source contains model-execution markers: "
                    + ", ".join(markers)
                )
        except SyntaxError as exc:
            failures.append(f"non-model exemption source is not statically inspectable: {exc}")
        if failures:
            return summary, failures
        summary["status"] = "NOT_APPLICABLE"
        summary["allowlist_id"] = allowlist_id
        return summary, failures

    # The normal model path requires a locally executed prediction and reviewed
    # MP4s.  The only alternate path is the centrally pinned GPU visual-only
    # resource-stop exception below; its result is deliberately pending, never
    # a local visual PASS.
    if "remote_visual_only_exception" in contract:
        return check_remote_visual_only_exception(
            contract=contract,
            code_sha256=code_sha256,
            code_path=code_path,
            kernel_dir=kernel_dir,
            project_root=project_root,
        )

    try:
        package = _package_inventory(kernel_dir, project_root)
        checkpoint_path, checkpoint_sha256 = _bound_project_asset(
            contract,
            path_key="checkpoint_path",
            sha_key="checkpoint_sha256",
            project_root=project_root,
            label="checkpoint",
        )
        input_path, input_sha256 = _bound_project_asset(
            contract,
            path_key="input_manifest_path",
            sha_key="input_manifest_sha256",
            project_root=project_root,
            label="input manifest",
        )
        environment_path, environment_sha256 = _bound_project_asset(
            contract,
            path_key="execution_environment_path",
            sha_key="execution_environment_sha256",
            project_root=project_root,
            label="execution environment",
        )
        input_manifest = load_json(input_path)
        if input_manifest.get("schema") != VISUAL_INPUT_MANIFEST_SCHEMA:
            raise GuardError("bound input manifest schema is missing or unsupported")
        manifest_identities = _identity_set(
            input_manifest.get("preregistered_visual_identities"),
            "bound input manifest preregistered_visual_identities",
        )
        contract_identities = _identity_set(
            contract.get("preregistered_visual_identities"),
            "contract preregistered_visual_identities",
        )
        if set(contract_identities) != set(manifest_identities):
            raise GuardError(
                "contract preregistered identity set differs from bound input manifest"
            )
    except (GuardError, OSError) as exc:
        failures.append(str(exc))
        return summary, failures

    reference = contract.get("local_visual_preflight_receipt")
    if not isinstance(reference, dict):
        failures.append("local visual preflight receipt reference is missing")
        return summary, failures
    try:
        receipt_path = _project_file(
            reference.get("path"),
            project_root=project_root,
            label="local visual preflight receipt",
        )
        expected_receipt_sha256 = _sha_field(
            reference.get("sha256"), "local visual preflight receipt hash"
        )
        observed_receipt_sha256 = sha256(receipt_path)
        if observed_receipt_sha256 != expected_receipt_sha256:
            raise GuardError("local visual preflight receipt hash drift")
        receipt = load_json(receipt_path)
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
        return summary, failures
    summary["receipt_path"] = _relative(
        receipt_path, project_root, "local visual preflight receipt"
    )
    summary["receipt_sha256"] = observed_receipt_sha256

    exact = {
        "schema": VISUAL_RECEIPT_SCHEMA,
        "status": "PASS",
        "model_execution": True,
        "candidate_code_sha256": code_sha256,
        "kernel_id": contract.get("kernel_id"),
        "package_tree_sha256": package["tree_sha256"],
        "candidate_code_path": _relative(code_path, project_root, "candidate code"),
        "checkpoint_path": _relative(checkpoint_path, project_root, "checkpoint"),
        "checkpoint_sha256": checkpoint_sha256,
        "input_manifest_path": _relative(input_path, project_root, "input manifest"),
        "input_manifest_sha256": input_sha256,
        "execution_environment_path": _relative(environment_path, project_root, "execution environment"),
        "execution_environment_sha256": environment_sha256,
        "preregistered_visual_identities": contract_identities,
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            failures.append(f"local visual preflight drift: {key}")
    if not isinstance(receipt.get("local_run_id"), str) or not receipt["local_run_id"].strip():
        failures.append("local visual preflight lacks a local_run_id")
    if not isinstance(receipt.get("model_id"), str) or not receipt["model_id"].strip():
        failures.append("local visual preflight lacks a model_id")

    execution_reference = contract.get("local_execution_receipt")
    execution: dict[str, Any] | None = None
    execution_path: Path | None = None
    try:
        if not isinstance(execution_reference, dict):
            raise GuardError("local execution receipt reference is missing")
        execution_path = _project_file(
            execution_reference.get("path"),
            project_root=project_root,
            label="local execution receipt",
        )
        execution_sha256 = _sha_field(
            execution_reference.get("sha256"), "local execution receipt hash"
        )
        if sha256(execution_path) != execution_sha256:
            raise GuardError("local execution receipt hash drift")
        execution = load_json(execution_path)
        execution_exact = {
            "schema": "celltrack-alpha.local-model-execution.v1",
            "status": "PASS",
            "execution_origin": "local",
            "exit_code": 0,
            "kernel_id": contract.get("kernel_id"),
            "model_id": receipt.get("model_id"),
            "local_run_id": receipt.get("local_run_id"),
            "package_path": package["path"],
            "package_tree_sha256": package["tree_sha256"],
            "candidate_code_path": _relative(code_path, project_root, "candidate code"),
            "candidate_code_sha256": code_sha256,
            "checkpoint_path": _relative(checkpoint_path, project_root, "checkpoint"),
            "checkpoint_sha256": checkpoint_sha256,
            "input_manifest_path": _relative(input_path, project_root, "input manifest"),
            "input_manifest_sha256": input_sha256,
            "execution_environment_path": _relative(environment_path, project_root, "execution environment"),
            "execution_environment_sha256": environment_sha256,
            "preregistered_visual_identities": contract_identities,
        }
        for key, expected_value in execution_exact.items():
            if execution.get(key) != expected_value:
                raise GuardError(f"local execution receipt drift: {key}")
        runner_id = execution.get("runner_id")
        trusted_runner = TRUSTED_LOCAL_RUNNERS.get(str(runner_id))
        if not isinstance(trusted_runner, dict):
            raise GuardError(
                "local execution receipt runner is not centrally trusted"
            )
        runner_path = _project_file(
            execution.get("runner_path"),
            project_root=project_root,
            label="trusted local runner",
        )
        runner_sha256 = _sha_field(
            execution.get("runner_sha256"), "trusted local runner hash"
        )
        trusted_exact = {
            "path": _relative(runner_path, project_root, "trusted local runner"),
            "sha256": runner_sha256,
        }
        if trusted_runner != trusted_exact or sha256(runner_path) != runner_sha256:
            raise GuardError("trusted local runner path/hash drift")
        argv = execution.get("invoked_argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(
                isinstance(item, str)
                and item
                and not any(ord(char) < 32 or ord(char) == 127 for char in item)
                for item in argv
            )
            or argv[0] != trusted_exact["path"]
        ):
            raise GuardError("trusted local runner invoked_argv is malformed")
        command_sha256 = _sha_field(
            execution.get("invoked_argv_sha256"),
            "trusted local runner invoked argv hash",
        )
        if command_sha256 != _canonical_sha(argv):
            raise GuardError("trusted local runner invoked argv hash drift")
        if contract.get("local_execution_command_sha256") != command_sha256:
            raise GuardError("contract/local execution command binding drift")
        execution_started = _parse_time(execution.get("started_at"), "local execution started_at")
        execution_ended = _parse_time(execution.get("ended_at"), "local execution ended_at")
        if execution_started > execution_ended:
            raise GuardError("local execution timestamps are out of order")
        process_receipt = execution.get("process_output_receipt")
        if not isinstance(process_receipt, dict):
            raise GuardError("trusted local runner process/output receipt is missing")
        for key, expected_value in {
            "schema": "celltrack-alpha.local-process-output.v1",
            "exit_code": 0,
            "started_at": execution.get("started_at"),
            "ended_at": execution.get("ended_at"),
        }.items():
            if process_receipt.get(key) != expected_value:
                raise GuardError(f"trusted local runner process receipt drift: {key}")
        for role in ("stdout", "stderr"):
            output_path = _project_file(
                process_receipt.get(f"{role}_path"),
                project_root=project_root,
                label=f"trusted local runner {role}",
                allow_empty=True,
            )
            output_sha = _sha_field(
                process_receipt.get(f"{role}_sha256"),
                f"trusted local runner {role} hash",
            )
            if sha256(output_path) != output_sha:
                raise GuardError(f"trusted local runner {role} hash drift")
        if receipt.get("execution_receipt_sha256") != execution_sha256:
            raise GuardError("local visual preflight does not bind the execution receipt hash")
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
        execution_started = execution_ended = None

    video_records = receipt.get("videos")
    if not isinstance(video_records, list) or not video_records:
        failures.append("local visual preflight must bind at least one reviewed MP4")
        return summary, failures
    identities: list[str] = []
    video_bindings: list[tuple[str, str]] = []
    prediction_bindings: list[dict[str, str]] = []
    sidecar_identities: list[str] = []
    rendered_times: dict[tuple[str, str], datetime] = {}
    for index, record in enumerate(video_records):
        if not isinstance(record, dict):
            failures.append(f"local visual video record {index} is not an object")
            continue
        try:
            identity = _identity(
                record.get("identity"), f"local visual video record {index} identity"
            )
        except GuardError as exc:
            failures.append(str(exc))
            continue
        identities.append(identity)
        try:
            prediction_path = _project_file(
                record.get("prediction_artifact_path"),
                project_root=project_root,
                label=f"local prediction artifact {index}",
            )
            prediction_sha256 = _sha_field(
                record.get("prediction_artifact_sha256"),
                f"local prediction artifact hash {index}",
            )
            if sha256(prediction_path) != prediction_sha256:
                raise GuardError(f"local prediction artifact {index} hash drift")
            video_path = _project_file(
                record.get("video_path"),
                project_root=project_root,
                label=f"local review video {index}",
                suffix=".mp4",
            )
            video_sha256 = _sha_field(
                record.get("video_sha256"), f"local review video hash {index}"
            )
            identity_stem = identity.replace(":", "__")
            if video_path.stem != identity_stem:
                raise GuardError(
                    f"local review video {index} filename is inconsistent with identity"
                )
            if sha256(video_path) != video_sha256:
                raise GuardError(f"local review video {index} hash drift")
            media = _validate_mp4_container(video_path, f"local review video {index}")
            sidecar_path = _project_file(
                record.get("sidecar_path"),
                project_root=project_root,
                label=f"local review sidecar {index}",
            )
            sidecar_sha256 = _sha_field(
                record.get("sidecar_sha256"), f"local review sidecar hash {index}"
            )
            if sidecar_path.stem != identity_stem:
                raise GuardError(
                    f"local review sidecar {index} filename is inconsistent with identity"
                )
            if sha256(sidecar_path) != sidecar_sha256:
                raise GuardError(f"local review sidecar {index} hash drift")
            sidecar = load_json(sidecar_path)
            sidecar_exact = {
                "schema": VISUAL_SIDECAR_SCHEMA,
                "identity": identity,
                "kernel_id": contract.get("kernel_id"),
                "model_id": receipt.get("model_id"),
                "local_run_id": receipt.get("local_run_id"),
                "package_tree_sha256": package["tree_sha256"],
                "candidate_code_path": _relative(code_path, project_root, "candidate code"),
                "candidate_code_sha256": code_sha256,
                "checkpoint_path": _relative(checkpoint_path, project_root, "checkpoint"),
                "checkpoint_sha256": checkpoint_sha256,
                "input_manifest_path": _relative(input_path, project_root, "input manifest"),
                "input_manifest_sha256": input_sha256,
                "execution_environment_path": _relative(environment_path, project_root, "execution environment"),
                "execution_environment_sha256": environment_sha256,
                "prediction_artifact_path": _relative(prediction_path, project_root, "prediction artifact"),
                "prediction_artifact_sha256": prediction_sha256,
                "video_path": _relative(video_path, project_root, "review video"),
                "video_sha256": video_sha256,
                "execution_receipt_sha256": execution_reference.get("sha256") if isinstance(execution_reference, dict) else None,
            }
            for key, expected in sidecar_exact.items():
                if sidecar.get(key) != expected:
                    raise GuardError(f"local review sidecar {index} drift: {key}")
            sidecar_identities.append(
                _identity(sidecar.get("identity"), f"local review sidecar {index} identity")
            )
            rendered_at = _parse_time(sidecar.get("rendered_at"), f"local review sidecar {index} rendered_at")
            if execution_ended is not None and rendered_at < execution_ended:
                raise GuardError(f"local review sidecar {index} predates execution completion")
            video_bindings.append((identity, video_sha256))
            rendered_times[(identity, video_sha256)] = rendered_at
            prediction_bindings.append(
                {
                    "identity": identity,
                    "path": _relative(prediction_path, project_root, "prediction artifact"),
                    "sha256": prediction_sha256,
                }
            )
            summary["artifacts"].extend(
                [
                    {"role": "prediction", "path": prediction_bindings[-1]["path"], "sha256": prediction_sha256},
                    {"role": "video", "path": _relative(video_path, project_root, "review video"), "sha256": video_sha256, **media},
                    {"role": "sidecar", "path": _relative(sidecar_path, project_root, "review sidecar"), "sha256": sidecar_sha256},
                ]
            )
        except (GuardError, OSError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
    if len(identities) != len(set(identities)):
        failures.append("local visual preflight contains duplicate identities")
    expected_identity_set = set(contract_identities)
    if set(identities) != expected_identity_set:
        failures.append("local visual video identity set differs from preregistration")
    if set(sidecar_identities) != expected_identity_set:
        failures.append("local visual sidecar identity set differs from preregistration")
    if isinstance(execution, dict):
        execution_predictions = execution.get("prediction_artifacts")
        process_predictions = (
            execution.get("process_output_receipt", {}).get("prediction_artifacts")
            if isinstance(execution.get("process_output_receipt"), dict)
            else None
        )
        expected_predictions = sorted(
            (item["identity"], item["path"], item["sha256"])
            for item in prediction_bindings
        )
        for label, value in (
            ("local execution receipt", execution_predictions),
            ("trusted local runner process receipt", process_predictions),
        ):
            try:
                if not isinstance(value, list):
                    raise GuardError(f"{label} prediction artifacts are missing")
                observed_predictions = []
                for index, item in enumerate(value):
                    if not isinstance(item, dict):
                        raise GuardError(f"{label} prediction artifact {index} is malformed")
                    prediction_path_value = item.get("path")
                    if (
                        not isinstance(prediction_path_value, str)
                        or not prediction_path_value
                        or Path(prediction_path_value).is_absolute()
                    ):
                        raise GuardError(
                            f"{label} prediction path {index} is not repository-relative"
                        )
                    observed_predictions.append(
                        (
                            _identity(item.get("identity"), f"{label} prediction identity {index}"),
                            prediction_path_value,
                            _sha_field(item.get("sha256"), f"{label} prediction hash {index}"),
                        )
                    )
                if sorted(observed_predictions) != expected_predictions:
                    raise GuardError(f"{label} prediction artifact set drift")
                if {item[0] for item in observed_predictions} != expected_identity_set:
                    raise GuardError(f"{label} prediction identity set differs from preregistration")
            except GuardError as exc:
                failures.append(str(exc))
    reviews = receipt.get("reviews")
    review_keys: list[tuple[str, str]] = []
    if not isinstance(reviews, list) or not reviews:
        failures.append("local visual preflight lacks per-video reviews")
    else:
        for index, review in enumerate(reviews):
            if not isinstance(review, dict):
                failures.append(f"local visual review {index} is not an object")
                continue
            key = (str(review.get("identity", "")), str(review.get("video_sha256", "")))
            review_keys.append(key)
            for field, expected_value in {
                "visual_analysis_performed": True,
                "metrics_only": False,
                "verdict": "PASS",
            }.items():
                if review.get(field) != expected_value:
                    failures.append(f"local visual review {index} drift: {field}")
            for field in ("reviewer", "promotion_rationale"):
                if not isinstance(review.get(field), str) or not review[field].strip():
                    failures.append(f"local visual review {index} lacks {field}")
            observations = review.get("observations")
            if not isinstance(observations, list) or not observations or not all(
                isinstance(item, str) and item.strip() for item in observations
            ):
                failures.append(f"local visual review {index} observations are malformed")
            taxonomy = review.get("failure_taxonomy")
            if not isinstance(taxonomy, dict) or set(taxonomy) != VISUAL_FAILURE_TAXONOMY:
                failures.append(f"local visual review {index} taxonomy is absent or incomplete")
            elif not all(isinstance(value, str) and value.strip() for value in taxonomy.values()):
                failures.append(f"local visual review {index} taxonomy entries are malformed")
            try:
                reviewed_at = _parse_time(review.get("reviewed_at"), f"local visual review {index} reviewed_at")
                rendered_at = rendered_times.get(key)
                if rendered_at is not None and reviewed_at < rendered_at:
                    failures.append(f"local visual review {index} predates video rendering")
            except GuardError as exc:
                failures.append(str(exc))
    if sorted(review_keys) != sorted(video_bindings):
        failures.append("per-video review coverage does not exactly match identity+video hash bindings")
    if {identity for identity, _ in review_keys} != expected_identity_set:
        failures.append("local visual review identity set differs from preregistration")
    try:
        sealed_at = _parse_time(receipt.get("sealed_at"), "local visual preflight sealed_at")
        latest_review = max(
            (_parse_time(review.get("reviewed_at"), "reviewed_at") for review in reviews if isinstance(review, dict)),
            default=None,
        ) if isinstance(reviews, list) else None
        if latest_review is not None and sealed_at < latest_review:
            failures.append("local visual preflight was sealed before review completion")
    except GuardError as exc:
        failures.append(str(exc))
    summary["artifacts"].extend(
        [
            {"role": "execution_receipt", "path": _relative(execution_path, project_root, "execution receipt") if execution_path else None, "sha256": execution_reference.get("sha256") if isinstance(execution_reference, dict) else None},
            {"role": "visual_receipt", "path": _relative(receipt_path, project_root, "visual receipt"), "sha256": observed_receipt_sha256},
            {"role": "checkpoint", "path": _relative(checkpoint_path, project_root, "checkpoint"), "sha256": checkpoint_sha256},
            {"role": "input_manifest", "path": _relative(input_path, project_root, "input manifest"), "sha256": input_sha256},
            {"role": "environment", "path": _relative(environment_path, project_root, "environment"), "sha256": environment_sha256},
        ]
    )
    summary["package_tree_sha256"] = package["tree_sha256"]
    summary["model_id"] = receipt.get("model_id")
    summary["local_run_id"] = receipt.get("local_run_id")
    summary["video_count"] = len(video_records)
    summary["identities"] = identities
    if not failures:
        summary["status"] = "PASS"
    return summary, failures


def imported_roots(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module:
        return {node.module.split(".", 1)[0]}
    return set()


def step0_line(tree: ast.Module) -> int:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "STEP0" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Call):
            name = value.func.id if isinstance(value.func, ast.Name) else ""
            if name in {"run_step0", "run_cellect_step0"}:
                return node.lineno
    raise GuardError("top-level STEP0 assignment calling run_step0 is missing")


def _result(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    code_path: Path,
    metadata: dict[str, Any],
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    failures: list[str],
    step0: int | None,
    expected_initial_kernel_version: int | None,
    local_visual_preflight: dict[str, Any],
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Return the common, CMUX-compatible predicate payload."""

    package = _package_inventory(kernel_dir, project_root)
    if (
        local_visual_preflight.get("status")
        in {"PASS", REMOTE_VISUAL_PENDING_STATUS}
        and local_visual_preflight.get("package_tree_sha256") != package["tree_sha256"]
    ):
        failures.append("kernel package changed during local visual guard validation")
    passed = not failures
    result = {
        "schema": "celltrack-alpha.kaggle-prelaunch-guard-result.v1",
        "status": "PASS" if passed else "FAIL",
        "eligible": passed and phase in {"inline", "launch"},
        "phase": phase,
        "lane_owner": contract["lane_owner"],
        "kernel_id": metadata.get("id"),
        "remote_preflight_kernel_id": contract.get("remote_preflight_kernel_id"),
        "kernel_dir": str(kernel_dir.resolve()),
        "project_root": str(project_root.resolve()),
        "package": package,
        "package_tree_sha256": package["tree_sha256"],
        "metadata_sha256": sha256(metadata_path),
        "code_file": str(code_path.resolve()),
        "code_sha256": sha256(code_path),
        "contract_path": (
            _relative(contract_path, project_root, "contract") if contract_path else None
        ),
        "contract_sha256": (
            sha256(contract_path)
            if contract_path
            else hashlib.sha256(
                json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        ),
        "registry_path": (
            _relative(registry_path, project_root, "registry") if registry_path else None
        ),
        "registry_sha256": (
            sha256(registry_path)
            if registry_path
            else hashlib.sha256(
                json.dumps(registry, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        ),
        "dataset_sources": metadata.get("dataset_sources"),
        "kernel_sources": metadata.get("kernel_sources", []),
        "exclusive_dataset_source": contract.get("exclusive_dataset_source"),
        "expected_mount_root": contract.get("expected_mount_root"),
        "step0_line": step0,
        "model_execution": contract.get("model_execution"),
        "local_visual_preflight": local_visual_preflight,
        "failures": failures,
    }
    if expected_initial_kernel_version is not None:
        result["expected_initial_kernel_version"] = expected_initial_kernel_version
    return result


def check_exact_pristine_control(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Validate a source-byte-exact public-notebook control.

    This is deliberately narrower than the ordinary script guard.  Its only
    admissible source is the pinned public notebook, so adding instrumentation
    to satisfy the normal logging contract would invalidate the control.
    """

    failures: list[str] = []
    expected_kernel_id = contract.get("kernel_id")
    if not isinstance(expected_kernel_id, str) or metadata.get("id") != expected_kernel_id:
        failures.append("kernel ID drift")
    if not isinstance(expected_kernel_id, str) or title_slug(str(metadata.get("title", ""))) != expected_kernel_id.split("/", 1)[-1]:
        failures.append("kernel title does not resolve to the declared ID")
    if metadata.get("kernel_type") != "notebook":
        failures.append("pristine control must use a Kaggle notebook")
    if code_path.suffix != ".ipynb":
        failures.append("pristine control code file must remain .ipynb")
    if contract.get("pristine_notebook_sha256") != PRISTINE_PUBLIC_V32_NOTEBOOK_SHA256:
        failures.append("pristine control contract does not pin the known public V32 notebook")
    if sha256(code_path) != PRISTINE_PUBLIC_V32_NOTEBOOK_SHA256:
        failures.append("pristine control notebook bytes differ from the pinned public V32 source")
    if contract.get("code_sha256") != sha256(code_path):
        failures.append("pristine control code SHA drift")
    if metadata.get("dataset_sources") != contract.get("dataset_sources"):
        failures.append("dataset_sources drift")
    for key, expected in contract.get("kernel_properties", {}).items():
        if metadata.get(key) != expected:
            failures.append(f"kernel property drift: {key}")
    if expected_initial_kernel_version not in {None, 1}:
        failures.append("expected initial kernel version must be exactly 1")
    visual_preflight, visual_failures = check_local_visual_preflight(
        contract=contract,
        code_sha256=sha256(code_path),
        code_path=code_path,
        kernel_dir=kernel_dir,
        source=code_path.read_text(encoding="utf-8"),
        project_root=project_root,
    )
    failures.extend(visual_failures)
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_preflight,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_asset_transport_only(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    source: str,
    tree: ast.Module,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
    project_root: Path,
    contract_path: Path | None,
    registry_path: Path | None,
) -> dict[str, Any]:
    """Guard the narrow, non-model CELLECT asset transport canary.

    This is intentionally distinct from a remote visual/model exception: it
    may prove only that a declared source-complete asset carrier survives the
    Kaggle mount and exact Step-0 authentication.  It cannot deserialize a
    checkpoint, inspect competition inputs, score, or emit a tracking result.
    """
    failures: list[str] = []
    kernel_id = contract.get("kernel_id")
    if not isinstance(kernel_id, str) or not kernel_id:
        failures.append("asset transport kernel ID is missing")
        kernel_id = ""
    if metadata.get("id") != kernel_id:
        failures.append("kernel ID drift")
    if not kernel_id or title_slug(str(metadata.get("title", ""))) != kernel_id.split("/", 1)[-1]:
        failures.append("kernel title does not resolve to the declared ID")

    sources = contract.get("dataset_sources")
    if not isinstance(sources, list) or len(sources) != 1 or not all(isinstance(item, str) for item in sources):
        failures.append("asset transport requires exactly one declared dataset source")
        source_id = ""
    else:
        source_id = sources[0]
    if metadata.get("dataset_sources") != sources:
        failures.append("dataset_sources drift")
    if metadata.get("kernel_sources", []) != contract.get("kernel_sources", []):
        failures.append("kernel_sources drift")
    expected_properties = {"enable_gpu": False, "enable_internet": False}
    if contract.get("kernel_properties") != expected_properties:
        failures.append("asset transport kernel properties are not CPU/offline only")
    for key, expected in expected_properties.items():
        if metadata.get(key) != expected:
            failures.append(f"kernel property drift: {key}")
    if metadata.get("competition_sources", []) != []:
        failures.append("asset transport may not declare a competition source")

    if contract.get("exclusive_dataset_source") != source_id:
        failures.append("asset transport exclusive dataset source drift")
    if not isinstance(contract.get("dataset_version"), int) or contract["dataset_version"] <= 0:
        failures.append("asset transport dataset version is malformed")
    owner = registry.get("dataset_ownership", {}).get(source_id)
    if owner != contract.get("lane_owner"):
        failures.append("dataset source is not exclusively registered to this lane")
    try:
        dataset_owner, dataset_slug = source_id.split("/", 1)
        dataset_root = registry["policy"]["private_dataset_mount_template"].format(
            owner=dataset_owner, slug=dataset_slug
        )
        # Zip-mode Kaggle dataset transport strips the local upload root but
        # retains this one certified wrapper.  The canary never discovers
        # paths: it consumes only this exact post-transport location.
        expected_root = dataset_root + "/cellect_assets"
    except (AttributeError, KeyError, ValueError):
        expected_root = ""
        failures.append("asset transport dataset source is malformed")
    if contract.get("expected_mount_root") != expected_root:
        failures.append("contract mount root does not match the registry template")
    if expected_root and expected_root not in source:
        failures.append("code does not contain the exact owner-qualified mount root")

    code_sha = sha256(code_path)
    if contract.get("code_sha256") != code_sha:
        failures.append("code SHA drift")
    manifest_sha = contract.get("asset_carrier_manifest_sha256")
    if not isinstance(manifest_sha, str) or re.fullmatch(r"[0-9a-f]{64}", manifest_sha) is None:
        failures.append("asset carrier manifest hash is malformed")
    elif manifest_sha not in source:
        failures.append("pinned asset carrier manifest SHA absent from code")

    if contract.get("model_execution") is not False:
        failures.append("asset transport must classify itself as non-model")
    exact_scope = {
        "native_inference": False,
        "checkpoint_deserialization": False,
        "competition_input": False,
        "scoring": False,
        "submission": False,
    }
    if contract.get("transport_scope") != exact_scope:
        failures.append("asset transport scope is malformed or broadened")
    required_markers = (
        "D2K4_STEP0_ASSIGNMENT_COMPLETE",
        "D2K4_TRANSPORT_COMPLETE",
        "D2K4_FAIL_CLOSED",
        "units, manifest, checked = step0()",
    )
    for marker in required_markers:
        if marker not in source:
            failures.append(f"asset transport code lacks required Step-0 marker: {marker}")
    forbidden_imports: set[str] = set()
    for node in ast.walk(tree):
        forbidden_imports.update(imported_roots(node) & (HEAVY_IMPORTS | MODEL_EXECUTION_IMPORTS))
    if forbidden_imports:
        failures.append("asset transport imports model/heavy code: " + ", ".join(sorted(forbidden_imports)))
    for snippet in FORBIDDEN_DISCOVERY_SNIPPETS:
        if snippet in source:
            failures.append(f"forbidden broad input discovery: {snippet}")

    carrier_reference = contract.get("asset_carrier_manifest")
    try:
        if not isinstance(carrier_reference, dict):
            raise GuardError("asset carrier manifest reference is missing")
        carrier_path = _project_file(
            carrier_reference.get("path"), project_root=project_root, label="asset carrier manifest"
        )
        if sha256(carrier_path) != manifest_sha:
            raise GuardError("asset carrier manifest hash drift")
        carrier = load_json(carrier_path)
        expected_carrier = {
            "schema": "celltrack-alpha.cellect-d2k4-asset-carrier.v1",
            "carrier_id": source_id,
            "external_upload_permitted": True,
            "native_inference_permitted": False,
            "score_or_submission_permitted": False,
        }
        if not isinstance(carrier, dict) or any(carrier.get(key) != value for key, value in expected_carrier.items()):
            raise GuardError("asset carrier transport-only scope drift")
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    certificate_reference = contract.get("ship_kaggle_certificate")
    try:
        if not isinstance(certificate_reference, dict):
            raise GuardError("ship-kaggle certificate reference is missing")
        certificate_path = _project_file(
            certificate_reference.get("path"), project_root=project_root, label="ship-kaggle certificate"
        )
        certificate_sha = _sha_field(certificate_reference.get("sha256"), "ship-kaggle certificate hash")
        if sha256(certificate_path) != certificate_sha:
            raise GuardError("ship-kaggle certificate hash drift")
        certificate = load_json(certificate_path)
        if not isinstance(certificate, dict) or certificate.get("status") != "ELIGIBLE":
            raise GuardError("ship-kaggle certificate is not eligible")
        if certificate.get("artifact_sha256") != code_sha or certificate.get("metadata_sha256") != sha256(metadata_path):
            raise GuardError("ship-kaggle certificate package binding drift")
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    roundtrip_reference = contract.get("dataset_roundtrip_receipt")
    try:
        if not isinstance(roundtrip_reference, dict):
            raise GuardError("asset transport dataset round-trip receipt is missing")
        roundtrip_path = _project_file(
            roundtrip_reference.get("path"), project_root=project_root, label="asset transport dataset round-trip receipt"
        )
        roundtrip_sha = _sha_field(roundtrip_reference.get("sha256"), "asset transport dataset round-trip receipt hash")
        if sha256(roundtrip_path) != roundtrip_sha:
            raise GuardError("asset transport dataset round-trip receipt hash drift")
        roundtrip = load_json(roundtrip_path)
        exact_roundtrip = {
            "schema": "celltrack-alpha.cellect-d2k4-dataset-roundtrip.v1",
            "status": "PASS",
            "dataset_id": source_id,
            "dataset_version": contract.get("dataset_version"),
            "expected_mount_root": contract.get("expected_mount_root"),
            "carrier_manifest_sha256": manifest_sha,
        }
        if not isinstance(roundtrip, dict) or any(roundtrip.get(key) != value for key, value in exact_roundtrip.items()):
            raise GuardError("asset transport dataset round-trip receipt binding drift")
    except (GuardError, OSError, json.JSONDecodeError) as exc:
        failures.append(str(exc))

    if phase == "inline":
        if contract.get("remote_preflight_mode") != "inline_asset_transport_step0.v1":
            failures.append("asset transport inline Step-0 mode is not declared")
    elif phase not in {"static", "launch"}:
        failures.append("asset transport guard phase is unsupported")
    if phase == "launch" and contract.get("remote_preflight_kernel_id") != kernel_id:
        failures.append("asset transport preflight is not bound to the launch kernel ID")
    if expected_initial_kernel_version not in {None, 1}:
        failures.append("expected initial kernel version must be exactly 1")

    visual, visual_failures = check_local_visual_preflight(
        contract=contract,
        code_sha256=code_sha,
        code_path=code_path,
        kernel_dir=kernel_dir,
        source=source,
        project_root=project_root,
    )
    failures.extend(visual_failures)
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=next((index for index, line in enumerate(source.splitlines(), start=1) if "units, manifest, checked = step0()" in line), None),
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def check_kernel(
    kernel_dir: Path,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None = None,
    project_root: Path = PROJECT_ROOT,
    contract_path: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise GuardError("kernel-metadata.json missing")
    metadata = load_json(metadata_path)
    code_path = kernel_dir / str(metadata.get("code_file", ""))
    if not code_path.is_file():
        raise GuardError("declared code_file missing")
    if contract.get("execution_profile") == "exact_pristine_control":
        return check_exact_pristine_control(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )
    source = code_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(code_path))

    if contract.get("execution_profile") == M7R6_NATIVE_T4_CERT_EXECUTION_PROFILE:
        return check_m7r6_native_t4_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M7R7_NATIVE_RUNTIME_DIAGNOSTIC_EXECUTION_PROFILE
    ):
        return check_m7r7_native_runtime_diagnostic(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M12R4_NO_TARGET_SYNTHETIC_CERT_EXECUTION_PROFILE
    ):
        return check_m12r4_no_target_synthetic_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M12R5_NO_TARGET_SYNTHETIC_CERT_EXECUTION_PROFILE
    ):
        return check_m12r5_no_target_synthetic_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M12R6_REMOTE_CERTIFICATION_EXECUTION_PROFILE
    ):
        return check_m12r6_remote_certification_exception(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M12R7_CANONICAL_REGISTRY_EXECUTION_PROFILE
    ):
        return check_m12r7_canonical_registry_transport(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M12R8_SYNTHETIC_VISUAL_EXECUTION_PROFILE
    ):
        return check_m12r8_synthetic_visual_preflight(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if contract.get("execution_profile") == M7R8_NATIVE_T4_CERT_EXECUTION_PROFILE:
        return check_m7r8_native_t4_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M8R13_NATIVE_T4_EXPORTER_EXECUTION_PROFILE
    ):
        return check_m8r13_native_t4_exporter_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M8R15_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE
    ):
        return check_m8r15_postilp_target_feature_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M8R16_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE
    ):
        return check_m8r16_postilp_target_feature_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M8R17_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE
    ):
        return check_m8r17_postilp_target_feature_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M8R20_FULL21_TARGET_FEATURE_EXECUTION_PROFILE
    ):
        return check_m8r20_full21_target_feature_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == M8R14_POSTILP_TARGET_FEATURE_EXECUTION_PROFILE
    ):
        return check_m8r14_postilp_target_feature_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if contract.get("execution_profile") == M7R13_PROVIDER_CERT_EXECUTION_PROFILE:
        return check_m7r13_provider_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if contract.get("execution_profile") in {
        M7R14_RAW_METADATA_EXECUTION_PROFILE,
        M7R14_RAW_METADATA_LAUNCH_EXECUTION_PROFILE,
    }:
        return check_m7r14_raw_metadata_transport(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if contract.get("execution_profile") == M7R15_ACTION_SUPPLY_EXECUTION_PROFILE:
        return check_m7r15_action_supply_census(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if contract.get("execution_profile") == M10R3_PROVIDER_CERT_EXECUTION_PROFILE:
        return check_m10r3_provider_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if contract.get("execution_profile") == M10R5_PROVIDER_CERT_EXECUTION_PROFILE:
        return check_m10r5_provider_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == D10_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE
    ):
        return check_d10_current_policy_native_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == D11_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE
    ):
        return check_d11_current_policy_native_certification(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if contract.get("execution_profile") == D16_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE:
        return check_d16_current_policy_native_certification(
            kernel_dir=kernel_dir, metadata_path=metadata_path, metadata=metadata, code_path=code_path,
            source=source, tree=tree, contract=contract, registry=registry, phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version, project_root=project_root,
            contract_path=contract_path, registry_path=registry_path,
        )

    if contract.get("execution_profile") == D28_CLOSED_SCHEMA_NATIVE_CERT_EXECUTION_PROFILE:
        return check_d28_closed_schema_native_certification(
            kernel_dir=kernel_dir, metadata_path=metadata_path, metadata=metadata, code_path=code_path,
            source=source, tree=tree, contract=contract, registry=registry, phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version, project_root=project_root,
            contract_path=contract_path, registry_path=registry_path,
        )

    if contract.get("execution_profile") == D19_CURRENT_POLICY_NATIVE_CERT_EXECUTION_PROFILE:
        return check_d19_current_policy_native_certification(
            kernel_dir=kernel_dir, metadata_path=metadata_path, metadata=metadata, code_path=code_path,
            source=source, tree=tree, contract=contract, registry=registry, phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version, project_root=project_root,
            contract_path=contract_path, registry_path=registry_path,
        )

    if contract.get("execution_profile") == ASSET_TRANSPORT_EXECUTION_PROFILE:
        return check_asset_transport_only(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    if (
        contract.get("execution_profile")
        == CERTIFIED_SWEEP_WRAPPER_EXECUTION_PROFILE
    ):
        return check_certified_sweep_wrapper(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            source=source,
            tree=tree,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    # A malformed generic contract is a failed prelaunch predicate, never a
    # Python exception.  This prevents a new lane from bypassing receipt
    # creation merely by omitting a field expected by the legacy guard.
    generic_required = (
        "kernel_id",
        "dataset_sources",
        "kernel_properties",
        "exclusive_dataset_source",
        "lane_owner",
        "expected_mount_root",
        "code_sha256",
        "input_manifest_sha256",
        "checkpoint_sha256",
        "population_manifest_sha256",
        "request_count",
        "request_file_count",
        "wheel_count",
        "model_execution",
    )
    missing_generic = [key for key in generic_required if key not in contract]
    if missing_generic and not _is_remote_visual_only_contract(contract):
        return _result(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            code_path=code_path,
            metadata=metadata,
            contract={**contract, "lane_owner": contract.get("lane_owner", "MALFORMED_CONTRACT")},
            registry=registry,
            phase=phase,
            failures=["generic contract missing required fields: " + ", ".join(missing_generic)],
            step0=None,
            expected_initial_kernel_version=expected_initial_kernel_version,
            local_visual_preflight={
                "required": None,
                "status": "FAIL",
                "receipt_path": None,
                "receipt_sha256": None,
                "video_count": 0,
                "identities": [],
                "artifacts": [],
            },
            project_root=project_root,
            contract_path=contract_path,
            registry_path=registry_path,
        )

    expected_kernel_id = contract["kernel_id"]
    remote_visual_only = _is_remote_visual_only_contract(contract)
    remote_route = _remote_visual_only_route(contract) if remote_visual_only else None
    if metadata.get("id") != expected_kernel_id:
        failures.append("kernel ID drift")
    expected_slug = expected_kernel_id.split("/", 1)[1]
    if title_slug(str(metadata.get("title", ""))) != expected_slug:
        failures.append("kernel title does not resolve to the declared ID")
    if remote_visual_only:
        if contract.get("execution_profile") in {
            CELLECT_TWO_CARRIER_EXECUTION_PROFILE,
            CELLECT_P100_EXECUTION_PROFILE,
            LINAJEA_LEGACY_GPU_EXECUTION_PROFILE,
        }:
            d2k2_step0_line, d2k2_failures = check_successor_remote_visual_surface(
                contract=contract,
                metadata=metadata,
                source=source,
                project_root=project_root,
            )
            input_source = "biohub-cell-tracking-during-development"
            expected_root = "/kaggle/input/biohub-cell-tracking-during-development"
        else:
            failures.extend(
                check_remote_visual_only_competition_route(
                    contract=contract,
                    metadata=metadata,
                    source=source,
                )
            )
            if remote_route and remote_route.get("surface") == "d2k5_declared_asset":
                d2k2_step0_line, d2k2_failures = check_d2k5_remote_visual_surface(
                    contract=contract,
                    source=source,
                    project_root=project_root,
                )
            else:
                d2k2_step0_line, d2k2_failures = check_d2k2_remote_visual_surface(
                    contract=contract,
                    kernel_dir=kernel_dir,
                    source=source,
                    project_root=project_root,
                    route=remote_route,
                )
            input_source = remote_route["competition_source"] if remote_route else ""
            expected_root = (
                remote_route.get("mount_root")
                or remote_route.get("competition_mount_aliases", [""])[0]
                if remote_route
                else ""
            )
        failures.extend(d2k2_failures)
    else:
        d2k2_step0_line = None
        expected_sources = contract["dataset_sources"]
        if metadata.get("dataset_sources") != expected_sources:
            failures.append("dataset_sources drift")
        expected_kernel_sources = contract.get("kernel_sources")
        if (
            expected_kernel_sources is not None
            and metadata.get("kernel_sources", []) != expected_kernel_sources
        ):
            failures.append("kernel_sources drift")
        for key, expected in contract["kernel_properties"].items():
            if metadata.get(key) != expected:
                failures.append(f"kernel property drift: {key}")

        input_source = contract["exclusive_dataset_source"]
        owner = registry.get("dataset_ownership", {}).get(input_source)
        if owner != contract["lane_owner"]:
            failures.append("dataset source is not exclusively registered to this lane")

        expected_root = contract["expected_mount_root"]
        template = registry["policy"]["private_dataset_mount_template"]
        dataset_owner, dataset_slug = input_source.split("/", 1)
        rendered = template.format(owner=dataset_owner, slug=dataset_slug)
        if expected_root != rendered:
            failures.append("contract mount root does not match the registry template")
        if expected_root not in source:
            failures.append("code does not contain the exact owner-qualified mount root")

    allow_pristine_discovery = contract.get("allow_pristine_broad_discovery") is True
    if allow_pristine_discovery and contract.get("execution_profile") not in {
        "bisection_step0_only",
        "bisection_telemetry_no_input_hash",
    }:
        failures.append("only the registered Step-0 bisection stage may retain pristine broad discovery")
    if not allow_pristine_discovery:
        for snippet in FORBIDDEN_DISCOVERY_SNIPPETS:
            if snippet in source:
                failures.append(f"forbidden broad input discovery: {snippet}")

    step0_only = contract.get("execution_profile") == "bisection_step0_only"
    if not remote_visual_only:
        logging_contract = contract.get("run_logging")
        required_logging = {
            "schema": "celltrack-alpha.run-log.v1",
            "stdout_jsonl": True,
            "persisted_with_artifacts": True,
            "download_on_success": True,
            "inventory_before_resolution": True,
            "failure_traceback_with_state": True,
        }
        if step0_only:
            if logging_contract != {"mode": "intentionally_absent_for_first_bisection_stage"}:
                failures.append("Step-0-only bisection logging exception is malformed")
        elif not isinstance(logging_contract, dict) or any(
            logging_contract.get(key) != expected
            for key, expected in required_logging.items()
        ):
            failures.append("mandatory comprehensive run-logging contract is absent or malformed")
        required_logging_markers = (
            "RUN_LOG.jsonl",
            "inventory_input",
            "log_lookup",
            "stage_end",
            "preprocess_counts",
            "_run_failure",
        )
        missing_logging_markers = [marker for marker in required_logging_markers if marker not in source]
        if missing_logging_markers and not step0_only:
            failures.append(
                "code lacks mandatory comprehensive run-logging markers: "
                + ", ".join(missing_logging_markers)
            )

    sibling_modules = {
        path.stem for path in kernel_dir.glob("*.py") if path.resolve() != code_path.resolve()
    }
    imported = set()
    for node in ast.walk(tree):
        imported.update(imported_roots(node))
    overlap = sorted(imported & sibling_modules)
    if overlap:
        failures.append(f"code_file imports unpackaged sibling modules: {overlap}")

    if remote_visual_only:
        first_step0_line = d2k2_step0_line if d2k2_step0_line is not None else 10**9
    else:
        try:
            first_step0_line = step0_line(tree)
        except GuardError as exc:
            failures.append(str(exc))
            first_step0_line = 10**9
        for node in tree.body:
            if getattr(node, "lineno", 10**9) >= first_step0_line:
                break
            heavy = imported_roots(node) & HEAVY_IMPORTS
            if heavy:
                failures.append(f"heavy import before STEP0: {sorted(heavy)}")

    code_sha = sha256(code_path)
    if code_sha != contract["code_sha256"]:
        failures.append("code SHA drift")
    if not remote_visual_only:
        try:
            manifest_sha = _sha_field(
                contract.get("input_manifest_sha256"), "input manifest hash"
            )
        except GuardError as exc:
            failures.append(str(exc))
            manifest_sha = ""
        if manifest_sha not in source:
            failures.append("pinned input-manifest SHA absent from code")
        try:
            checkpoint_sha = _sha_field(contract.get("checkpoint_sha256"), "checkpoint hash")
        except GuardError as exc:
            failures.append(str(exc))
            checkpoint_sha = ""
        if checkpoint_sha not in source:
            failures.append("pinned checkpoint SHA absent from code")
        try:
            population_sha = _sha_field(
                contract.get("population_manifest_sha256"), "population manifest hash"
            )
        except GuardError as exc:
            failures.append(str(exc))
            population_sha = ""
        if population_sha not in source:
            failures.append("pinned population-manifest SHA absent from code")
        request_count = int(contract["request_count"])
        request_file_count = int(contract["request_file_count"])
        wheel_count = int(contract["wheel_count"])
        if request_count <= 0:
            failures.append("request_count must be positive")
        if request_file_count <= 0:
            failures.append("request_file_count must be positive")
        if wheel_count < 0:
            failures.append("wheel_count must be nonnegative")
        if wheel_count == 0 and contract.get("offline_dependency_mode") != "kaggle_base_image_only":
            failures.append(
                "zero-wheel packages must declare offline_dependency_mode=kaggle_base_image_only"
            )
        if str(request_count) not in source.replace("_", ""):
            failures.append("positive exact request count absent from code")
        if str(request_file_count) not in source.replace("_", ""):
            failures.append("positive exact request-file count absent from code")
        if wheel_count > 0 and str(wheel_count) not in source.replace("_", ""):
            failures.append("positive exact wheel count absent from code")

    if phase == "inline":
        if remote_visual_only:
            if contract.get("execution_profile") == CELLECT_TWO_CARRIER_EXECUTION_PROFILE:
                expected_mode = CELLECT_TWO_CARRIER_STEP0_MODE
            elif contract.get("execution_profile") == CELLECT_P100_EXECUTION_PROFILE:
                expected_mode = CELLECT_P100_STEP0_MODE
            elif contract.get("execution_profile") == LINAJEA_LEGACY_GPU_EXECUTION_PROFILE:
                expected_mode = LINAJEA_LEGACY_GPU_STEP0_MODE
            elif remote_route == REMOTE_VISUAL_ONLY_D2K5_COMPETITION_ROUTE:
                expected_mode = D2K5_INLINE_STEP0_MODE
            elif remote_route == REMOTE_VISUAL_ONLY_D2K3_COMPETITION_ROUTE:
                expected_mode = D2K3_INLINE_STEP0_MODE
            else:
                expected_mode = "d2k2_step0_before_native_release.v1"
            if contract.get("remote_preflight_mode") != expected_mode:
                failures.append("remote visual inline step-0 release mode is not declared")
        else:
            if contract.get("remote_preflight_mode") != "inline_step0_before_heavy_work":
                failures.append("inline launch mode is not declared")
            # This is the only coherent one-version mode: the actual remote input
            # assertion is the first executable kernel statement, before imports
            # or the banked runner.  A remote result cannot exist before this same
            # one-shot version is pushed.
            if "STEP0 = run_step0()" not in source:
                failures.append("inline step-0 execution boundary is absent")
            if contract.get("step0_competition_path_visits") == 0:
                pre_step0_source = "\n".join(source.splitlines()[:first_step0_line - 1])
                if "/kaggle/input/competitions/" in pre_step0_source or "/kaggle/input/biohub-cell-tracking" in pre_step0_source:
                    failures.append("competition path is referenced before inline step 0")
    if phase == "launch":
        if contract.get("remote_preflight_kernel_id") != expected_kernel_id:
            failures.append("remote preflight is not bound to the launch kernel ID")
        if not remote_visual_only:
            attestation = contract.get("remote_preflight_attestation")
            if not isinstance(attestation, dict) or attestation.get("status") != "PASS":
                failures.append("remote preflight PASS attestation missing")
            else:
                exact = {
                    "dataset_source": input_source,
                    "dataset_version": contract["dataset_version"],
                    "dataset_status": "ready",
                    "input_manifest_sha256": manifest_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "population_manifest_sha256": population_sha,
                    "request_count": request_count,
                    "request_file_count": request_file_count,
                    "population_file_count": int(contract["population_file_count"]),
                    "wheel_count": wheel_count,
                    "enable_gpu": contract["kernel_properties"].get("enable_gpu"),
                    "heavy_work_started": False,
                }
                for key, expected in exact.items():
                    if attestation.get(key) != expected:
                        failures.append(f"remote preflight attestation drift: {key}")
                if (
                    contract.get("step0_elapsed_is_hard_failure", True)
                    and float(attestation.get("step0_elapsed_seconds", 999.0)) >= float(
                        contract["step0_limit_seconds"]
                    )
                ):
                    failures.append("remote preflight exceeded step-0 limit")
                if contract.get("step0_competition_path_visits") == 0 and attestation.get("competition_path_visits") != 0:
                    failures.append("remote preflight competition-path visit drift")

    if expected_initial_kernel_version not in {None, 1}:
        failures.append("expected initial kernel version must be exactly 1")

    visual_preflight, visual_failures = check_local_visual_preflight(
        contract=contract,
        code_sha256=code_sha,
        code_path=code_path,
        kernel_dir=kernel_dir,
        source=source,
        project_root=project_root,
    )
    failures.extend(visual_failures)

    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None if first_step0_line == 10**9 else first_step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
        local_visual_preflight=visual_preflight,
        project_root=project_root,
        contract_path=contract_path,
        registry_path=registry_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--phase", choices=("static", "inline", "launch"), default="static")
    parser.add_argument("--expected-initial-kernel-version", type=int, choices=(1,))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = check_kernel(
        args.kernel_dir,
        load_json(args.contract),
        load_json(args.registry),
        args.phase,
        args.expected_initial_kernel_version,
        project_root=PROJECT_ROOT,
        contract_path=args.contract.resolve(),
        registry_path=args.registry.resolve(),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
