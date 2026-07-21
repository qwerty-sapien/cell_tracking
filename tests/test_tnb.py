import ast, hashlib, importlib.util, json, math, sys, tempfile, unittest, zipfile
import numpy as np
from pathlib import Path
from types import ModuleType
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'kaggle_kernel'/'t04.py'
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
  outer=terminal.body[0].exc; self.assertIsInstance(outer,ast.Call); self.assertEqual(len(outer.args),1); self.assertIs(outer.args[0],empty[0])
 def test_compact_args_and_armed_training(self):
  s=importlib.util.spec_from_file_location('tnb_armed',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_armed']=m; s.loader.exec_module(m)
  self.assertTrue(m._args([]).armed); self.assertFalse(m._args(['--dry-run']).armed)
  bad=m._args([]); delattr(bad,'armed')
  with self.assertRaisesRegex(m.LaunchError,'argument namespace is incomplete'): m._validate_arg_contract(bad)
  self.assertEqual(m._runtime_preflight(m._args(['--dry-run'])),{'mode':'dry','dependency_preflight':False})
  torch=__import__('torch')
  class Tiny(torch.nn.Module):
   def __init__(self): super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.5))
   def forward(self,x,scenario='universal'): return x.repeat(1,1,6,1,1,1)*self.w
  b={'frames':torch.ones(1,2,1,4,4,4),'nodes':[{'id':1,'batch':0,'time':0,'z':1,'y':1,'x':1},{'id':2,'batch':0,'time':1,'z':1,'y':2,'x':1}],'edges':[{'source_id':1,'target_id':2}],'voxel_size_um':(1.,1.,1.),'window_id':'armed'}
  record=m.SourceRecord('g02-local-armed','train:6bba_edf14583','6bba_edf14583')
  with tempfile.TemporaryDirectory() as d:
   q=Path(d); a=m._args([]); a.output_dir=q; a.data_root=None; a.max_steps=1; a.maximum_hours=.01
   result=m._run_bounded_training(a,[record],'cpu',model_factory=lambda _a,_d:Tiny(),batch_factory=lambda _s,_d:[b])
   self.assertEqual((result['status'],result['completed_steps']),('completed',1))
   self.assertTrue(all((q/name).is_file() for name in ('checkpoint.pt','metrics.json','resume.json')))
 def test_projection_semantic_equivalence(self):
  s=importlib.util.spec_from_file_location('tnb_projection',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_projection']=m; s.loader.exec_module(m)
  raw=(ROOT/'reports'/'g02'/'dataset_registry_v1.json').read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(),m.REGISTRY_SHA256); full=json.loads(raw)
  split,prov=m.embedded_manifest('split'),m.embedded_manifest('provenance'); projection=m.runtime_registry_projection(); fields=('group_id','video_count','contains_nontrain_member','video_ids','splits')
  union=sorted(set().union(*(set(m._fold_data(f,split,prov)[0]) for f in (0,1,2)))); wanted=set(union); indexed={item['group_id']:item for item in full['groups'] if item['group_id'] in wanted}
  rebuilt={'artifact':'runtime-registry-projection-v1','source_registry_sha256':m.REGISTRY_SHA256,'groups':[{key:indexed[group][key] for key in fields} for group in union]}
  self.assertEqual(m._canonical_json(projection),m._canonical_json(rebuilt)); self.assertEqual(hashlib.sha256(m._canonical_json(projection)).hexdigest(),m.REGISTRY_PROJECTION_SHA256)
  for fold in (0,1,2):
   _,forbidden=m._fold_data(fold,split,prov)
   for fraction in m.FRACTIONS:
    selected=m.select_fit_groups(fold,split,prov,fraction,17); self.assertEqual(m.resolve_fit_records(selected,projection,forbidden),m.resolve_fit_records(selected,full,forbidden))
 def test_dry_folds_and_repeat(self):
  s=importlib.util.spec_from_file_location('tnb_dry',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_dry']=m; s.loader.exec_module(m); out=[]
  for fold in (0,0,1,2):
   with tempfile.TemporaryDirectory() as d:
    q=Path(d)
    with mock.patch('builtins.print') as emit: self.assertEqual(m.main(['--dry-run','--fold',str(fold),'--output-dir',str(q)]),0)
    out.append((emit.call_args.args[0],hashlib.sha256((q/'prelaunch.json').read_bytes()).hexdigest()))
  self.assertEqual(out[0],out[1])
 def test_no_repo_import_or_forbidden_ops(self):
  t=P.read_text(); self.assertNotIn('celltrack_alpha',t); self.assertNotIn('tracking_cellmot',t)
  for x in ('dask','polars','scipy','subprocess','requests','pip install','git ','kaggle.api','submit') : self.assertNotIn(x,t)
  masked='\n'.join('EMBEDDED = {}' if line.startswith('EMBEDDED = ') else line for line in t.splitlines())
  for x in ('tracksdata','/kaggle/input/test','test_root','np.zeros((n,n','transition_matrix','pair_matrix','array[:]') : self.assertNotIn(x,masked.lower())
  self.assertIn("[start:start+2, ::1, ::4, ::4]",masked)
 def test_offline_zarr_bootstrap(self):
  s=importlib.util.spec_from_file_location('tnb_zarr',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_zarr']=m; s.loader.exec_module(m)
  def make(path,members):
   path.parent.mkdir(parents=True,exist_ok=True)
   with zipfile.ZipFile(path,'w') as archive:
    for name,data in members: archive.writestr(name,data)
   data=path.read_bytes(); return (path.name,len(data),hashlib.sha256(data).hexdigest())
  def missing(_): raise ModuleNotFoundError('zarr')
  with tempfile.TemporaryDirectory() as d:
   base=Path(d); support=base/'support'; dest=base/'deps'
   real=ROOT/'external'/'Tracking_Support'; self.assertEqual(m._wheel_root(roots=(real,)),(real/'wheels').resolve())
   m._extract_wheels(real/'wheels',base/'real-deps'); self.assertRegex((base/'real-deps'/'.ready').read_text(),r'^[0-9a-f]{64}\n$')
   spec=make(support/'wheels'/'zarr-test.whl',[('zarr/__init__.py','VALUE=1\n')])
   token=object(); loaded=[]
   def importer(name):
    loaded.append(name)
    if not (dest/'zarr/__init__.py').is_file(): raise ModuleNotFoundError('zarr')
    return token
   self.assertIs(m._ensure_zarr((spec,),(support,),dest,importer),token); self.assertEqual(loaded,list(m.RUNTIME_IMPORTS)); before=(dest/'.ready').read_bytes(); loaded.clear()
   self.assertIs(m._ensure_zarr((spec,),(support,),dest,importer),token); self.assertEqual((dest/'.ready').read_bytes(),before)
   self.assertEqual(loaded,list(m.RUNTIME_IMPORTS))
   with self.assertRaises(m.LaunchError): m._ensure_zarr((spec,),(base/'missing',),base/'x',missing)
   twin=base/'twin'; make(twin/'wheels'/spec[0],[('zarr/__init__.py','VALUE=1\n')])
   with self.assertRaises(m.LaunchError): m._ensure_zarr((spec,),(support,twin),base/'y',missing)
   bad=base/'bad'; badspec=make(bad/'wheels'/'bad.whl',[('../escape','x')])
   with self.assertRaises(m.LaunchError): m._ensure_zarr((badspec,),(bad,),base/'z',missing)
   self.assertFalse((base/'escape').exists())
 def test_package_identity_and_checkpoint_search(self):
  s=importlib.util.spec_from_file_location('tnb5',P); m=importlib.util.module_from_spec(s); sys.modules['tnb5']=m; s.loader.exec_module(m)
  pack=json.loads((ROOT/'kaggle_kernel'/'t04.pack.json').read_text())
  for name,info in pack['package_files'].items():
   data=(ROOT/name).read_bytes(); self.assertEqual((len(data),hashlib.sha256(data).hexdigest()),(info['bytes'],info['sha256']))
  self.assertFalse(pack['notebook']['runtime_self_contained']); self.assertTrue(pack['notebook']['support_input_required'])
  self.assertTrue(pack['runtime']['fp32_only']); self.assertTrue(pack['runtime']['support_input_required'])
  self.assertLess(pack['code_package_bytes'],pack['code_package_under_bytes'])
  self.assertNotIn('rglob(',P.read_text())
  self.assertNotIn(".validate(selected)",P.read_text())
  self.assertIn('cellmot-baseline-artifacts/weights/unet_transformer/split_0',P.read_text())
  self.assertNotIn('biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0',P.read_text())
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); primary=root/'primary'; secondary=root/'secondary'
   with mock.patch.object(m,'CHECKPOINT_DIRS',(primary,secondary)):
    diagnostic=root/'input'; diagnostic.mkdir(); (diagnostic/'mounted').mkdir()
    with self.assertRaises(m.LaunchError) as error: m._find_input('edge_predictor_best.pth',m.WEIGHTS_BYTES,m.WEIGHTS_SHA256,diagnostic)
    self.assertIn('"input_entries":["mounted"]',str(error.exception)); self.assertIn('"probes"',str(error.exception))
    for directory in (primary,secondary):
     path=directory/'edge_predictor_best.pth'; path.parent.mkdir(); path.touch()
     with path.open('r+b') as handle: handle.truncate(m.WEIGHTS_BYTES)
    with mock.patch.object(m,'_sha_file',return_value=m.WEIGHTS_SHA256):
     with self.assertRaises(m.LaunchError): m._find_input('edge_predictor_best.pth',m.WEIGHTS_BYTES,m.WEIGHTS_SHA256)
    secondary.joinpath('edge_predictor_best.pth').unlink()
    with mock.patch.object(m,'_sha_file',return_value=m.WEIGHTS_SHA256): self.assertEqual(m._find_input('edge_predictor_best.pth',m.WEIGHTS_BYTES,m.WEIGHTS_SHA256),primary/'edge_predictor_best.pth')
    weight=root/'locked'/'edge_predictor_best.pth'; weight.parent.mkdir(); weight.touch(); config=weight.with_name('config.json'); config.touch()
    with config.open('r+b') as handle: handle.truncate(m.CONFIG_BYTES)
    with mock.patch.object(m,'_sha_file',return_value=m.CONFIG_SHA256): self.assertEqual(m._sibling_config(weight),config)
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
  gg=G({'nodes/ids':np.array([1,2],dtype=np.uint64),'nodes/props/t/values':np.array([0,1]),'nodes/props/z/values':np.array([1,1]),'nodes/props/y/values':np.array([7,7]),'nodes/props/x/values':np.array([7,7]),'edges/ids':edge})
  gg.attrs={'geff':{'geff_version':'1.1','directed':True,'axes':[{'name':'t','scale':1},{'name':'z','scale':1.625},{'name':'y','scale':.40625},{'name':'x','scale':.40625}]}}
  zm=ModuleType('zarr'); zm.open_group=lambda p,mode='r': ig if str(p).endswith('.zarr') else gg
  r=m.SourceRecord('g02-01db8d94baa3ee0ee831','train:6bba_edf14583','short'); src=m.SparseWindowSource(b,(1,4,4))
  with mock.patch.object(m,'_ensure_zarr',return_value=zm):
   win=list(src._windows(r)); self.assertEqual(len(win),1); batch=src._read(win[0][1],win[0][0]); self.assertEqual(src.validate([r])['windows'],1)
  self.assertEqual(tuple(batch['frames'].shape),(1,2,1,4,2,2)); self.assertEqual(a.i[0],slice(0,2)); self.assertEqual(batch['voxel_size_um'],(1.625,1.625,1.625))
  self.assertTrue(all(node['y']==1 and node['x']==1 for node in batch['nodes'])); m.build_sparse_targets(__import__('torch').zeros((1,2,6,4,2,2)),batch['nodes'],batch['edges'],batch['voxel_size_um'])
  gg['edges/ids']=np.array([(1,2)],dtype=[('source_id','u8'),('target_id','u8')])
  with mock.patch.object(m,'_ensure_zarr',return_value=zm): self.assertEqual(src._metadata(r)[2],[{'source_id':1,'target_id':2}])
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
   with mock.patch.object(m,'_ensure_zarr',return_value=zm):
    with self.assertRaises(m.LaunchError): src._metadata(r)
  gg['edges/ids']=np.array([[1,3]],dtype=np.uint64); gg.attrs['geff']['directed']=False
  with mock.patch.object(m,'_ensure_zarr',return_value=zm):
   with self.assertRaises(m.LaunchError): src._metadata(r)
  gg.attrs['geff']['directed']=True; gg.attrs['geff']['axes'][1]['scale']=1.5
  with mock.patch.object(m,'_ensure_zarr',return_value=zm):
   with self.assertRaises(m.LaunchError): src._metadata(r)
 def test_synthetic_loss_and_caps(self):
  s=importlib.util.spec_from_file_location('tnb3',P); m=importlib.util.module_from_spec(s); sys.modules['tnb3']=m; s.loader.exec_module(m)
  self.assertFalse(m.USE_MIXED_PRECISION)
  class Tiny(__import__('torch').nn.Module):
   def __init__(self): super().__init__(); self.w=__import__('torch').nn.Parameter(__import__('torch').tensor(.5))
   def forward(self,x,scenario='universal'): return x.repeat(1,1,6,1,1,1)*self.w
  b={'frames':__import__('torch').ones(1,2,1,4,4,4),'nodes':[{'id':1,'batch':0,'time':0,'z':1,'y':1,'x':1},{'id':2,'batch':0,'time':1,'z':1,'y':2,'x':1}],'edges':[{'source_id':1,'target_id':2}],'voxel_size_um':(1.,1.,1.),'window_id':'tiny'}
  q=Tiny(); opt=__import__('torch').optim.AdamW(q.parameters(),lr=.1); r,_=m.train_steps(q,opt,[b],device=__import__('torch').device('cpu'),deadline=m.time.monotonic()+5,max_steps=1); self.assertEqual(r['steps'],1); self.assertTrue(__import__('math').isfinite(r['loss_components']['total']))
  out=__import__('torch').zeros((1,2,6,4,4,4),dtype=__import__('torch').float32,requires_grad=True)
  with __import__('torch').no_grad(): out[:,:,1:4].fill_(100); out[:,:,4:5].fill_(-10)
  target=m.build_sparse_targets(out,b['nodes'],b['edges'],b['voxel_size_um']); stress=m.multitask_loss(out,target); stress['total'].backward()
  self.assertTrue(all(__import__('torch').isfinite(value).all() for value in stress.values())); self.assertTrue(__import__('torch').isfinite(out.grad).all())
  self.assertEqual((m.CENTER_SUPERVISION,m.CENTER_LOSS),('annotated-only','report-only-excluded')); self.assertEqual(int(target.center_supervision_mask.sum()),2)
  r,_=m.train_steps(q,opt,[b],device=__import__('torch').device('cpu'),deadline=m.time.monotonic()-1,max_steps=1); self.assertEqual((r['status'],r['steps']),('time_cap',0))
 def test_periodic_checkpoint_callback(self):
  s=importlib.util.spec_from_file_location('tnb_progress',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_progress']=m; s.loader.exec_module(m)
  torch=__import__('torch')
  class Tiny(torch.nn.Module):
   def __init__(self): super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.5))
   def forward(self,x,scenario='universal'): return x.repeat(1,1,6,1,1,1)*self.w
  b={'frames':torch.ones(1,2,1,4,4,4),'nodes':[{'id':1,'batch':0,'time':0,'z':1,'y':1,'x':1},{'id':2,'batch':0,'time':1,'z':1,'y':2,'x':1}],'edges':[{'source_id':1,'target_id':2}],'voxel_size_um':(1.,1.,1.),'window_id':'progress'}
  q=Tiny(); opt=torch.optim.AdamW(q.parameters(),lr=.1); seen=[]
  with mock.patch.object(m,'CHECKPOINT_EVERY_STEPS',1):
   result,_=m.train_steps(q,opt,[b,b],device=torch.device('cpu'),deadline=m.time.monotonic()+5,max_steps=2,checkpoint_callback=lambda item,_s:seen.append(item))
  self.assertEqual((result['status'],result['steps']),('completed',2)); self.assertEqual([item['steps'] for item in seen],[1,2]); self.assertTrue(all(item['status']=='running' for item in seen))
 def test_end_to_end_cap_prevents_source_access(self):
  s=importlib.util.spec_from_file_location('tnb_cap',P); m=importlib.util.module_from_spec(s); sys.modules['tnb_cap']=m; s.loader.exec_module(m)
  torch=__import__('torch')
  class Tiny(torch.nn.Module):
   def __init__(self): super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.5))
   def forward(self,x,scenario='universal'): return x.repeat(1,1,6,1,1,1)*self.w
  record=m.SourceRecord('g02-cap','train:6bba_edf14583','short'); called=[]
  with tempfile.TemporaryDirectory() as d:
   args=m._args([]); args.output_dir=Path(d); args.data_root=None; args.started_monotonic=m.time.monotonic()-3600
   result=m._run_bounded_training(args,[record],'cpu',model_factory=lambda _a,_d:Tiny(),batch_factory=lambda _s,_d:called.append(True) or ())
  self.assertEqual((result['status'],result['completed_steps']),('time_cap',0)); self.assertEqual(called,[])
 def test_builder_emits_standard_nbformat_kernelspec(self):
  path=ROOT/'scripts'/'pack_t04.py'; s=importlib.util.spec_from_file_location('t04_pack',path); m=importlib.util.module_from_spec(s); sys.modules['t04_pack']=m; s.loader.exec_module(m)
  notebook=json.loads(m.notebook_bytes(b"print('kernelspec regression')\n")); metadata=notebook['metadata']
  self.assertEqual(notebook['nbformat'],4); self.assertEqual(notebook['nbformat_minor'],5)
  self.assertEqual(metadata['kernelspec'],{'display_name':'Python 3','language':'python','name':'python3'}); self.assertNotIn('kernel_spec',metadata)
 def test_notebook_identity_and_names(self):
  nb=json.loads((ROOT/'kaggle_kernel'/'t04.ipynb').read_text()); h=hashlib.sha256(P.read_bytes()).hexdigest(); self.assertEqual(nb['metadata']['t04_script_sha256'],h); self.assertEqual(nb['cells'][0]['id'],'t04-entry'); self.assertEqual(''.join(nb['cells'][0]['source']).encode(),P.read_bytes()); self.assertEqual(nb['metadata']['kernelspec'],{'display_name':'Python 3','language':'python','name':'python3'}); self.assertEqual(nb['metadata']['language_info'],{'name':'python'}); self.assertNotIn('kernel_spec',nb['metadata']); self.assertIn('private',json.dumps(nb['metadata']).lower()); self.assertFalse(nb['metadata']['runtime_self_contained']); self.assertTrue(nb['metadata']['support_input_required'])
  for f in (P,ROOT/'kaggle_kernel'/'t04.ipynb',ROOT/'kaggle_kernel'/'t04.meta.json',ROOT/'kaggle_kernel'/'t04.pack.json',ROOT/'scripts'/'pack_t04.py',ROOT/'tests'/'test_tnb.py',ROOT/'reports'/'g02'/'t04-trial.json'):
   self.assertLess(len(f.name),15)
 def test_t04_trial_and_historical_boundaries(self):
  s=importlib.util.spec_from_file_location('t04_trial',P); m=importlib.util.module_from_spec(s); sys.modules['t04_trial']=m; s.loader.exec_module(m)
  expected={'batch_size':1,'center_loss':'report-only-excluded','center_supervision':'annotated-only','division_enabled':False,'epochs':1,'fit_fraction':.1,'fold':0,'learning_rate':1e-5,'max_steps':1000,'released_detector_frozen':True,'scenario':'universal','seed':17,'unfreeze':0}
  self.assertEqual(m.trial_record()['candidate'],expected); m._exact_candidate(m._args([])); self.assertNotIn('sparse_negative_weight',P.read_text()); self.assertIn('warn_only=False',P.read_text()); self.assertIn("Path('/kaggle/working/t04')",P.read_text())
  support=ROOT/'external'/'Tracking_Support'; manifest=json.loads((support/'ARTIFACT_MANIFEST.json').read_text()); weight=support/'weights'/'unet_transformer'/'split_0'/'edge_predictor_best.pth'
  self.assertEqual((m.WEIGHTS_BYTES,m.WEIGHTS_SHA256),(8357783,'347915de9c33883cb2ee69832a8e4552c88b1ec692d0fbfe956422467d3d4235')); self.assertNotEqual(manifest['model']['weight_sha256'],m.WEIGHTS_SHA256); self.assertNotEqual(hashlib.sha256(weight.read_bytes()).hexdigest(),m.WEIGHTS_SHA256); self.assertEqual(m.trial_record()['immutable_inputs']['checkpoint_sha256'],m.WEIGHTS_SHA256); self.assertIn('cellmot-baseline-artifacts',m.trial_record()['immutable_inputs']['checkpoint_source'])
  legacy=json.loads((ROOT/'reports'/'g02'/'trial-v1.json').read_text())
  with mock.patch.object(m,'trial_record',return_value=legacy):
   with self.assertRaises(m.LaunchError): m._exact_candidate(m._args([]))
  pack=json.loads((ROOT/'kaggle_kernel'/'t04.pack.json').read_text()); historical={'reports/g02/trial-v1.json':'01083d1e5f1f2f5bc62e6de8336499f2b00295042987884d656f3c41e100b5c2','reports/g02/trial-v3.json':'9dbf157b6e59e7358baece018d3942126a45f67f7399095dd888401b5ea79f79','reports/g02/trial-v4.json':'dcd94b091ed965f924e3c4ab915fafabf2b8951a9de0b4e78e5d43f71fbc8691'}
  self.assertEqual(pack['historical_hashes'],historical)
  for name,digest in historical.items(): self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
  meta=json.loads((ROOT/'kaggle_kernel'/'t04.meta.json').read_text()); self.assertEqual(meta['trial']['path'],'reports/g02/t04-trial.json'); self.assertTrue(meta['support_input_required']); self.assertTrue(meta['checkpoint_input']['roots']); self.assertEqual(meta['checkpoint_input']['weight_sha256'],m.WEIGHTS_SHA256); self.assertEqual(meta['support_input']['wheel_count'],9)
if __name__=='__main__': unittest.main()
