#!/usr/bin/env python3
"""Generate independent sdir fixtures for native per-sample lighting validation."""
import argparse
import importlib.util
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch
import warp as wp

from sdir_checkpoint_reference import load_module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdir',type=Path,required=True)
    p.add_argument('--mesh-source',type=Path,required=True)
    p.add_argument('--mode',type=int,choices=[3,4],default=3)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=str(args.output_dir/'warp-cache')
    root=args.mesh_source.resolve();sys.path.insert(0,str(root));sys.path.insert(0,str(args.sdir.resolve()))
    for name,relative in [('radiance_meshes',''),('radiance_meshes.delaunay_rasterization','delaunay_rasterization'),('radiance_meshes.delaunay_rasterization.internal','delaunay_rasterization/internal')]:
        mod=types.ModuleType(name);mod.__path__=[str(root/relative)];sys.modules[name]=mod
    wt=load_module('radiance_meshes.delaunay_rasterization.internal.warp_tracer',root/'delaunay_rasterization/internal/warp_tracer.py')
    vr=load_module('sdir_volumetric_reference',args.sdir/'renderer/volumetric_relight.py')
    for blocker in [False,True]:
        verts=[[-2.,-2.,1.],[2.,-2.,1.],[0.,2.,1.],[0.,0.,3.]]
        if blocker: verts += [[2.2,-2.,-1.],[2.2,2.,-1.],[2.2,0.,3.],[2.8,0.,1.]]
        verts=torch.tensor(verts); tets=torch.arange(len(verts),dtype=torch.int32).reshape(-1,4)
        normals=torch.tensor([[0.,0.,1.]]).repeat(len(verts),1)
        primary=torch.zeros(len(tets),15);primary[:,0]=1.3;primary[:,7]=.5;primary[:,12:15]=torch.tensor([.7,.8,.9])
        material=torch.tensor([[.3,.1,.2,.05,.1]]).repeat(len(tets),1)
        table=torch.cat([primary[:,7:8],primary[:,12:15],material],1)
        eps=max(float((verts.max(0).values-verts.min(0).values).norm())*1e-7,1e-7)
        name='disconnected' if blocker else 'single'
        with (args.output_dir/f'{name}.rmlit').open('wb') as f:
            f.write(b'RMLIT001');f.write(struct.pack('<IIIf',len(verts),len(tets),args.mode,eps))
            for a in [verts,normals,tets,primary[:,0].contiguous(),table]: a.numpy().tofile(f)
        rays=torch.tensor([[0.,0.,0.,x,y,1.] for y in [-.2,0.,.2] for x in [-.2,0.,.2]])
        rays.numpy().tofile(args.output_dir/f'{name}.rays.bin')
        tracer=wt.WarpFaceTracer(verts,tets)
        q=vr.prepare_volume(tracer,rays,primary,normals,samples=4,min_t=.1)
        lamp=types.SimpleNamespace(type='point',position=torch.tensor([3.,0.,0.]),direction=None,color=torch.tensor([2.,1.76,1.44]),hemi_soft=0.)
        out=vr.shade_volume(tracer,primary,primary,q,lamp,material=material,eye=torch.zeros(3),material_mode=args.mode)
        np.save(args.output_dir/f'{name}.npy',out[:,:3].reshape(3,3,3).numpy())
        print(name,out[:,0].tolist(),flush=True)


if __name__=='__main__':main()
