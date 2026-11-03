#!/usr/bin/env python3
"""Compare sampled interactive energy pixels with sdir's actual volume shader.

Input RMLIT001 must be exported losslessly by inspect_energy from the same
energy .rmesh. The input capture must be linear RGB with its camera sidecar.
This checks real image rays without pretending vertex baking is exact.
"""
import argparse,json,struct,sys,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from sdir_checkpoint_reference import load_module
from compare_linear_renders import compare
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--sdir',type=Path,required=True);p.add_argument('--mesh-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--width',type=int,default=16);p.add_argument('--height',type=int,default=9);p.add_argument('--unshadowed',action='store_true');a=p.parse_args()
torch.set_num_threads(8);torch.set_grad_enabled(False);a.output.parent.mkdir(parents=True,exist_ok=True)
wp.config.kernel_cache_dir=str(a.output.parent/'warp-cache')
root=a.mesh_source.resolve();sys.path[:0]=[str(root),str(a.sdir)]
for name,relative in [('radiance_meshes',''),('radiance_meshes.delaunay_rasterization','delaunay_rasterization'),('radiance_meshes.delaunay_rasterization.internal','delaunay_rasterization/internal')]:
    mod=types.ModuleType(name);mod.__path__=[str(root/relative)];sys.modules[name]=mod
wt=load_module('radiance_meshes.delaunay_rasterization.internal.warp_tracer',root/'delaunay_rasterization/internal/warp_tracer.py')
vr=load_module('energy_sdir_volume_reference',a.sdir/'renderer/volumetric_relight.py')
with a.scene.open('rb') as f:magic=f.read(8);nv,nt,mode,eps=struct.unpack('<IIIf',f.read(16))
assert magic==b'RMLIT001' and mode==4
mapped=np.memmap(a.scene,mode='c',offset=24,dtype='<f4');off=0
def take(shape,dtype='<f4'):
    global off
    size=int(np.prod(shape));v=mapped[off:off+size].view(dtype).reshape(shape);off+=size;return torch.from_numpy(v)
verts=take((nv,3));normals=take((nv,3));tets=take((nt,4),'<i4');rho=take((nt,));mat=take((nt,9))
primary=torch.zeros(nt,15);primary[:,0]=rho;primary[:,7]=mat[:,0];primary[:,12:15]=mat[:,1:4];material=mat[:,4:].contiguous()
camera=json.loads(Path(str(a.capture)+'.json').read_text());capture=np.load(a.capture)
x=np.linspace(0,camera['width']-1,a.width).round().astype(int);y=np.linspace(0,camera['height']-1,a.height).round().astype(int)
yy,xx=np.meshgrid(y,x,indexing='ij');fx,fy,cx,cy=camera['intrinsics']
directions=torch.tensor(np.stack([(xx+.5-cx)/fx,(yy+.5-cy)/fy,np.ones_like(xx)],-1).reshape(-1,3),dtype=torch.float32)
c2w=torch.tensor(camera['c2w_columns'])[:,:3].T;directions=F.normalize(directions@c2w.T,dim=-1)
eye=torch.tensor(camera['camera_position'][:3],dtype=torch.float32);rays=torch.cat([eye.expand(len(directions),3),directions],-1).contiguous()
print(f'Building full scene BVH: {nv} vertices, {nt} cells',flush=True)
tracer=wt.WarpFaceTracer(verts,tets)
print('Preparing matched rays',flush=True);q=vr.prepare_volume(tracer,rays,primary,normals,min_t=camera['near'],samples=4)
print('Shading',len(q['ids']),'samples',flush=True);result=torch.zeros(len(rays),3)
shadow=primary
if a.unshadowed:
    shadow=primary.clone();shadow[:,0]=0
np.savez(str(a.output)+'.quadrature.npz',**{k:v.numpy() for k,v in q.items() if isinstance(v,torch.Tensor)})
for (pos,axis,rgb),intensity in zip(camera['light_position_direction_rgb'],camera['light_intensities']):
    lamp=types.SimpleNamespace(type='point',position=torch.tensor(pos,dtype=torch.float32),direction=torch.tensor(axis,dtype=torch.float32),color=torch.tensor(rgb,dtype=torch.float32)*intensity,hemi_soft=0.)
    result+=vr.shade_volume(tracer,primary,shadow,q,lamp,material=material,eye=eye,material_mode=4)[:,:3]
reference=result.reshape(a.height,a.width,3).numpy();candidate=capture[yy,xx]
np.save(a.output,reference);np.save(str(a.output)+'.candidate.npy',candidate);rays.numpy().tofile(str(a.output)+'.rays.bin')
report=compare(reference,candidate,reference_layout='HWC');report['scope']=('Every pixel' if (a.width,a.height)==(camera['width'],camera['height']) else 'Uniform grid of matched rays, not all pixels')+'; energy export fields, sdir CPU Warp/PyTorch'
Path(str(a.output)+'.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

raise SystemExit(0 if report["passed"] else 1)
