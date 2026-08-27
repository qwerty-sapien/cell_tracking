from pathlib import Path
import importlib.util, json
ROOT=Path(__file__).resolve().parents[5]
NODE=ROOT/"dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d28_fresh_reconciliation_time_central_release_v1"
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);assert s and s.loader;s.loader.exec_module(m);return m
def test_d28_route_and_cmux_exception():
 g=load(Path(__file__).with_name("kaggle_prelaunch_guard.py"),"d24g");c=load(ROOT/"scripts/cmux_ops.py","d24c")
 contract=json.loads((NODE/"central_integration_v1/CENTRAL_PRELAUNCH_CONTRACT.json").read_text());registry=json.loads(Path(__file__).with_name("KAGGLE_FAILURE_REGISTRY.json").read_text());package=NODE/"native_cert_package_v1"
 kw=dict(expected_initial_kernel_version=1,project_root=ROOT,contract_path=NODE/"central_integration_v1/CENTRAL_PRELAUNCH_CONTRACT.json",registry_path=Path(__file__).with_name("KAGGLE_FAILURE_REGISTRY.json"))
 assert g.check_kernel(package,contract,registry,"static",**kw)["status"]=="PASS"
 inline=g.check_kernel(package,contract,registry,"inline",**kw);assert inline["status"]=="PASS" and inline["eligible"] is True
 assert c.REMOTE_NATIVE_CERTIFICATION_EXCEPTIONS[contract["kernel_id"]]["exception_id"]=="d28-fresh-reconciliation-time-central-release-v1"
