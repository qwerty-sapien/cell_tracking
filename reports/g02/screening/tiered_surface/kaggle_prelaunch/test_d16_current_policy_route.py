from pathlib import Path
import importlib.util, json
ROOT=Path(__file__).resolve().parents[5]
NODE=ROOT/"dag_prompts/2026-08-23-model-clade-renewal-v1/portfolio_parallel_nonm_v1/d_v16_dual_dag_finalization_v1"
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_d16_route_and_cmux_exception():
 g=load(Path(__file__).with_name("kaggle_prelaunch_guard.py"),"d16g");c=load(ROOT/"scripts/cmux_ops.py","d16c")
 contract=json.loads((NODE/"CENTRAL_PRELAUNCH_CONTRACT.json").read_text());rp=Path(__file__).with_name("KAGGLE_FAILURE_REGISTRY.json");reg=json.loads(rp.read_text());pkg=NODE/"native_cert_package_v1"
 kw=dict(expected_initial_kernel_version=1,project_root=ROOT,contract_path=NODE/"CENTRAL_PRELAUNCH_CONTRACT.json",registry_path=rp)
 assert g.check_kernel(pkg,contract,reg,"static",**kw)["status"]=="PASS"
 inline=g.check_kernel(pkg,contract,reg,"inline",**kw);assert inline["status"]=="PASS" and inline["eligible"] is True
 assert c.REMOTE_NATIVE_CERTIFICATION_EXCEPTIONS[contract["kernel_id"]]["exception_id"]=="d16-current-policy-native-cert-v1"
