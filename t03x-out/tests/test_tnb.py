import ast, hashlib, importlib.util, json, math, subprocess, sys, tempfile, unittest
import numpy as np
from pathlib import Path
from types import ModuleType
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]; INPUT_ROOT=ROOT.parent; P=ROOT/'kaggle_kernel'/'train_nb.py'
def _artifact(name):
 p=ROOT/name; return p if p.exists() else INPUT_ROOT/name
class T(unittest.TestCase):
 def test_ast_and_embedded(self):
  ast.parse(P.read_text()); s=importlib.util.spec_from_file_location('tnb',P); m=importlib.util.module_from_spec(s); sys.modules['tnb']=m; s.loader.exec_module(m)
  for n,h in [('split','ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769'),('provenance','6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f')]:
   self.assertEqual(m.EMBEDDED[n][1],h); self.assertIsInstance(m.embedded_manifest(n),dict)
  projection=m.runtime_registry_projection(); self.assertEqual(projection['source_registry_sha256'],m.REGISTRY_SHA256); self.assertEqual(m.EMBEDDED['registry_projection'][1],m.REGISTRY_PROJECTION_SHA256)
 def test_kernel_main_argv(self):
  tree=ast.parse(P.read_text()); calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='main']
  empty=[n for n in calls if len(n.args)==1 and isinstance(n.args[0],ast.List) and not n.args[0].elts and not n.keywords]
  self.assertEqual(len(empty),1); self.assertFalse(any(not n.args and not n.keywords for n in calls))
  terminal=tree.body[-1]; self.assertIsInstance(terminal,ast.If); self.assertEqual(len(terminal.body),1); self.assertIsInstance(terminal.body[0],ast.Raise)
  outer=terminal.body[0].exc; self.assertIsInstance(outer,ast.Call); self.assertEqual(len(outer.args),1); self.assertIsInstance(outer.args[0],ast.Call)
  call=outer.args[0]; self.assertIsInstance(call.func,ast.Name); self.assertEqual(call.func.id,'main'); self.assertEqual(len(call.args),1); self.assertIsInstance(call.args[0],ast.List); self.assertFalse(call.args[0].elts)
 def test_projection_semantic_equivalence(self):
  s=importlib.util.spec_from_file_location('tnb_projection',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_projection']=m; s.loader.exec_module(m)
  raw=_artifact('reports/g02/dataset_registry_v1.json').read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(),m.REGISTRY_SHA256); full=json.loads(raw)
  split,prov=m.embedded_manifest('split'),m.embedded_manifest('provenance'); projection=m.runtime_registry_projection(); fields=('group_id','video_count','contains_nontrain_member','video_ids','splits')
  union=sorted(set().union(*(set(m._fold_data(f,split,prov)[0]) for f in (0,1,2)))); wanted=set(union); indexed={item['group_id']:item for item in full['groups'] if item['group_id'] in wanted}
  rebuilt={'artifact':'runtime-registry-projection-v1','source_registry_sha256':m.REGISTRY_SHA256,'groups':[{key:indexed[group][key] for key in fields} for group in union]}
  self.assertEqual(m._canonical_json(projection),m._canonical_json(rebuilt)); self.assertEqual(hashlib.sha256(m._canonical_json(projection)).hexdigest(),m.REGISTRY_PROJECTION_SHA256)
  for fold in (0,1,2):
   _,forbidden=m._fold_data(fold,split,prov)
   for fraction in m.FRACTIONS:
    selected=m.select_fit_groups(fold,split,prov,fraction,17); self.assertEqual(m.resolve_fit_records(selected,projection,forbidden),m.resolve_fit_records(selected,full,forbidden))
 def test_dry_folds_and_repeat(self):
  out=[]
  for fold in (0,0,1,2):
   with tempfile.TemporaryDirectory() as d:
    q=Path(d); r=subprocess.run([sys.executable,str(P),'--dry-run','--fold',str(fold),'--output-dir',str(q)],capture_output=True,text=True,check=True); out.append((r.stdout,hashlib.sha256((q/'prelaunch.json').read_bytes()).hexdigest()))
  self.assertEqual(out[0],out[1])
 def test_no_repo_import_or_forbidden_ops(self):
  t=P.read_text(); self.assertNotIn('celltrack_alpha',t); self.assertNotIn('tracking_cellmot',t)
  for x in ('dask','polars','scipy','subprocess','requests','pip install','git ','kaggle.api','submit') : self.assertNotIn(x,t)
  masked='\n'.join('EMBEDDED = {}' if line.startswith('EMBEDDED = ') else line for line in t.splitlines())
  for x in ('tracksdata','/kaggle/input/test','test_root','np.zeros((n,n','transition_matrix','pair_matrix','array[:]') : self.assertNotIn(x,masked.lower())
  self.assertIn("[start:start+2, ::1, ::4, ::4]",masked)
 def test_package_identity_and_checkpoint_search(self):
  s=importlib.util.spec_from_file_location('tnb5',P); m=importlib.util.module_from_spec(s); sys.modules['tnb5']=m; s.loader.exec_module(m)
  pack=json.loads((ROOT/'kaggle_kernel'/'tr-pack.json').read_text())
  for name,info in pack['package_files'].items():
   data=_artifact(name).read_bytes(); self.assertEqual((len(data),hashlib.sha256(data).hexdigest()),(info['bytes'],info['sha256']))
  self.assertTrue(pack['notebook']['self_contained']); self.assertFalse(pack['notebook']['sidecar_required'])
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); original=m.Path
   with mock.patch.object(m,'Path',lambda value: root if str(value)=='/kaggle/input' else original(value)):
    with self.assertRaises(m.LaunchError): m._find_input('edge_predictor_best.pth',m.WEIGHTS_BYTES,m.WEIGHTS_SHA256)
    for index in (0,1):
     path=root/str(index)/'edge_predictor_best.pth'; path.parent.mkdir(); path.touch()
     with path.open('r+b') as handle: handle.truncate(m.WEIGHTS_BYTES)
    with mock.patch.object(m,'_sha_file',return_value=m.WEIGHTS_SHA256):
     with self.assertRaises(m.LaunchError): m._find_input('edge_predictor_best.pth',m.WEIGHTS_BYTES,m.WEIGHTS_SHA256)
 def test_direct_v3_geff_nx2_window_and_structured(self):
  s=importlib.util.spec_from_file_location('tnb2',P); m=importlib.util.module_from_spec(s); sys.modules['tnb2']=m; s.loader.exec_module(m)
  tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
  b=Path(tmp.name); z=b/'short.zarr'; g=b/'short.geff'
  for q in (z/'0',g/'nodes/ids',g/'edges/ids',*(g/'nodes/props'/k/'values' for k in ('t','z','y','x'))): q.mkdir(parents=True,exist_ok=True)
  for q in (z/'zarr.json',z/'0/zarr.json',g/'zarr.json',g/'nodes/ids/zarr.json',g/'edges/ids/zarr.json',*(g/'nodes/props'/k/'values/zarr.json' for k in ('t','z','y','x'))): q.write_text('{"zarr_format":3}')
  class A:
   shape=(2,4,8,8)
   def __getitem__(self,i): self.i=i; return np.zeros((2,4,2,2),dtype='float32')
  a=A(); edge=np.array([[1,2]],dtype=np.uint64)
  class G(dict): pass
  ig=G({'0':a}); ig.attrs={'multiscales':[{'axes':[{'name':'t'},{'name':'z'},{'name':'y'},{'name':'x'}],'datasets':[{'coordinateTransformations':[{'type':'scale','scale':[1,1.625,.40625,.40625]}]}]}],'image_statistics':{'quantiles':{'0.001':0,'0.999':1}}}
  gg=G({'nodes/ids':np.array([1,2],dtype=np.uint64),'nodes/props/t/values':np.array([0,1]),'nodes/props/z/values':np.array([1,1]),'nodes/props/y/values':np.array([4,4]),'nodes/props/x/values':np.array([4,4]),'edges/ids':edge})
  gg.attrs={'geff':{'geff_version':'1.1','directed':True,'axes':[{'name':'t','scale':1},{'name':'z','scale':1.625},{'name':'y','scale':.40625},{'name':'x','scale':.40625}]}}
  zm=ModuleType('zarr'); zm.open_group=lambda p,mode='r': ig if str(p).endswith('.zarr') else gg
  r=m.SourceRecord('g02-01db8d94baa3ee0ee831','train:6bba_edf14583','short'); src=m.SparseWindowSource(b,(1,4,4))
  with mock.patch.dict(sys.modules,{'zarr':zm}):
   win=list(src._windows(r)); self.assertEqual(len(win),1); batch=src._read(win[0][1],win[0][0])
  self.assertEqual(tuple(batch['frames'].shape),(1,2,1,4,2,2)); self.assertEqual(a.i[0],slice(0,2)); self.assertEqual(batch['voxel_size_um'],(1.625,1.625,1.625))
  gg['edges/ids']=np.array([(1,2)],dtype=[('source_id','u8'),('target_id','u8')])
  with mock.patch.dict(sys.modules,{'zarr':zm}): self.assertEqual(src._metadata(r)[2],[{'source_id':1,'target_id':2}])
 def test_rejects_malformed_geff_edges_and_metadata(self):
  s=importlib.util.spec_from_file_location('tnb4',P); m=importlib.util.module_from_spec(s); sys.modules['tnb4']=m; s.loader.exec_module(m)
  tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
  b=Path(tmp.name); z=b/'short.zarr'; g=b/'short.geff'
  for q in (z/'0',g/'nodes/ids',g/'edges/ids',*(g/'nodes/props'/k/'values' for k in ('t','z','y','x'))): q.mkdir(parents=True,exist_ok=True)
  for q in (z/'zarr.json',z/'0/zarr.json',g/'zarr.json',g/'nodes/ids/zarr.json',g/'edges/ids/zarr.json',*(g/'nodes/props'/k/'values/zarr.json' for k in ('t','z','y','x'))): q.write_text('{"zarr_format":3}')
  class A:
   shape=(2,4,8,8)
   def __getitem__(self,i): return np.zeros((2,4,2,2),dtype='float32')
  class G(dict): pass
  ig=G({'0':A()}); ig.attrs={'multiscales':[{'axes':[{'name':'t'},{'name':'z'},{'name':'y'},{'name':'x'}],'datasets':[{'coordinateTransformations':[{'type':'scale','scale':[1,1.625,.40625,.40625]}]}]}],'image_statistics':{'quantiles':{'0.001':0,'0.999':1}}}
  gg=G({'nodes/ids':np.array([1,2,3],dtype=np.uint64),'nodes/props/t/values':np.array([0,0,1]),'nodes/props/z/values':np.array([1,1,1]),'nodes/props/y/values':np.array([4,4,4]),'nodes/props/x/values':np.array([4,4,4]),'edges/ids':np.array([[1,3]],dtype=np.uint64)})
  gg.attrs={'geff':{'geff_version':'1.1','directed':True,'axes':[{'name':'t','scale':1},{'name':'z','scale':1.625},{'name':'y','scale':.40625},{'name':'x','scale':.40625}]}}
  zm=ModuleType('zarr'); zm.open_group=lambda p,mode='r': ig if str(p).endswith('.zarr') else gg
  r=m.SourceRecord('g02-01db8d94baa3ee0ee831','train:6bba_edf14583','short'); src=m.SparseWindowSource(b,(1,4,4))
  cases=[np.zeros((1,3),dtype=np.uint64),np.zeros((1,2),dtype=np.float32),np.array([[1,3],[2,3]],dtype=np.uint64),np.array([[1,99]],dtype=np.uint64),np.array([[3,1]],dtype=np.uint64),np.array([[-1,3]],dtype=np.int64)]
  for edge in cases:
   gg['edges/ids']=edge
   with mock.patch.dict(sys.modules,{'zarr':zm}):
    with self.assertRaises(m.LaunchError): src._metadata(r)
  gg['edges/ids']=np.array([[1,3]],dtype=np.uint64); gg.attrs['geff']['directed']=False
  with mock.patch.dict(sys.modules,{'zarr':zm}):
   with self.assertRaises(m.LaunchError): src._metadata(r)
  gg.attrs['geff']['directed']=True; gg.attrs['geff']['axes'][1]['scale']=1.5
  with mock.patch.dict(sys.modules,{'zarr':zm}):
   with self.assertRaises(m.LaunchError): src._metadata(r)
 def test_synthetic_loss_and_caps(self):
  s=importlib.util.spec_from_file_location('tnb3',P); m=importlib.util.module_from_spec(s); sys.modules['tnb3']=m; s.loader.exec_module(m)
  class Tiny(__import__('torch').nn.Module):
   def __init__(self): super().__init__(); self.w=__import__('torch').nn.Parameter(__import__('torch').tensor(.5))
   def forward(self,x,scenario='universal'): return x.repeat(1,1,6,1,1,1)*self.w
  b={'frames':__import__('torch').ones(1,2,1,4,4,4),'nodes':[{'id':1,'batch':0,'time':0,'z':1,'y':1,'x':1},{'id':2,'batch':0,'time':1,'z':1,'y':2,'x':1}],'edges':[{'source_id':1,'target_id':2}],'voxel_size_um':(1.,1.,1.),'window_id':'tiny'}
  q=Tiny(); opt=__import__('torch').optim.AdamW(q.parameters(),lr=.1); r,_=m.train_steps(q,opt,[b],negative_weight=.001,device=__import__('torch').device('cpu'),deadline=m.time.monotonic()+5,max_steps=1); self.assertEqual(r['steps'],1); self.assertTrue(__import__('math').isfinite(r['loss_components']['total']))
  r,_=m.train_steps(q,opt,[b],negative_weight=.001,device=__import__('torch').device('cpu'),deadline=m.time.monotonic()-1,max_steps=1); self.assertEqual((r['status'],r['steps']),('time_cap',0))
 def test_notebook_identity_and_names(self):
  nb=json.loads((ROOT/'kaggle_kernel'/'train_nb.ipynb').read_text()); h=hashlib.sha256(P.read_bytes()).hexdigest(); self.assertEqual(nb['metadata']['t03_script_sha256'],h); self.assertEqual(''.join(nb['cells'][0]['source']).encode(),P.read_bytes()); self.assertIn('private',json.dumps(nb['metadata']).lower())
  for f in (P,ROOT/'kaggle_kernel'/'train_nb.ipynb',ROOT/'kaggle_kernel'/'tr-meta.json',ROOT/'kaggle_kernel'/'tr-pack.json',ROOT/'tests'/'test_tnb.py',ROOT/'reports'/'g02'/'trial-v1.json'):
   self.assertLess(len(f.name),15)
if __name__=='__main__': unittest.main()
