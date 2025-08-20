import torch
import torch.nn as nn
import trimesh
from pamo import PaMO, PaSP
import numpy as np
import time
import networkx as nx
import argparse


import os

def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', type=str, default='./mesh/BirdHouse_B019SXLRJ2_MetalLeafRoofGreenWalls_TU.obj')
    args.add_argument('-o', '--output', type=str, default='./BirdHouse_pamo.obj')
    args.add_argument('-r', '--ratio', type=float, default=0.1)
    args.add_argument('-mv', '--min-vertex', type=int, default=0)
    args.add_argument('--disable_stage1', action='store_true', help="Disable remeshing")
    args.add_argument('--disable_stage3', action='store_true', help="Disable safe projection")
    args = args.parse_args()


    input_mesh = trimesh.load(args.input, force='mesh', process=False)
    print("# of input verts : {}".format(len(input_mesh.vertices)))
    print("# of input faces : {}".format(len(input_mesh.faces)))
    
    pamo = PaMO(input_mesh,
                use_stage1 = not args.disable_stage1,
                use_stage3 = not args.disable_stage3)
    start = time.time()
    verts, faces = pamo.run(torch.from_numpy(input_mesh.vertices).float().cuda(), 
                            torch.from_numpy(input_mesh.faces).int().cuda(), 
                            min_verts=args.min_vertex,
                            ratio = args.ratio)
    end = time.time()
    print("Total time: ", end-start)
    
    output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    print("# of output verts : {}".format(len(output_mesh.vertices)))
    print("# of output faces : {}".format(len(output_mesh.faces)))
    output_mesh.export(args.output)

    # # test PaSP
    # pasp = PaSP()
    # start = time.time()
    # verts, faces = pasp.run(torch.from_numpy(input_mesh.vertices).float().cuda(), torch.from_numpy(input_mesh.faces).int().cuda(), iter=10000, threshold=0.001)
    # end = time.time()
    # print("PaSP time: ", end-start)
    
    # verts = verts.cpu().numpy()
    # faces = faces.cpu().numpy()
    # output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # output_mesh.export('./crab_pasp.obj')


main()
