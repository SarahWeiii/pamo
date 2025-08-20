import os
import copy
import torch
from torch import nn
from torch.autograd import Function
import trimesh
import numpy
from pdmc import DMC
import time
import torch.nn.functional as F
from . import _C
import numpy as np
import trimesh
import pamo_safe_project
import torchcumesh2sdf


class PaMO(nn.Module):
    def __init__(self, input_mesh, use_stage1 = True, use_stage3 = True):
        super().__init__()
        pamo = _C.CUDSP_Free()

        self.use_stage1 = use_stage1
        self.use_stage3 = use_stage3

        print("Stage1 : ", self.use_stage1)
        print("Stage3 : ", self.use_stage3)
        
        self.bbox = input_mesh.bounding_box.bounds
        diameter = np.abs(self.bbox[1] - self.bbox[0]).max()
        scale = 1.0 / diameter
        self.gt_mesh = copy.deepcopy(input_mesh)
        
        if self.use_stage3:
            self.config = pamo_safe_project.config.Stage3Config()  # default config
            self.system = pamo_safe_project.system.Stage3System(self.config)  # create a system (with all the cuda arrays)

        class DSPFunction(Function):
            @staticmethod
            def forward(ctx, points, triangles, vertices_undo, num_vertices_undo, scale, threshold, is_stuck, init):
                verts, faces, verts_occ, verts_map, verts_undo = pamo.forward(points, triangles, vertices_undo, num_vertices_undo, scale, threshold, is_stuck, init)
                ctx.points = points
                ctx.triangles = triangles
                return verts, faces, verts_occ, verts_map, verts_undo

        self.func = DSPFunction
        # vol2mesh params
        self.vol2mesh = DMC(dtype=torch.float32).cuda()
        # mesh2vol params
        self.R = 256
        self.band = 3 / self.R # 3
        self.margin = self.band * 2 + 1 #2
        self.target_faces = None

    def tri_area(self, v0, v1, v2):
        cross_prod = torch.cross(v1 - v0, v2 - v0)
        return 0.5 * torch.norm(cross_prod, dim=1)


    def preprocess_mesh(self, points, triangles, band, margin):
        tris = points[triangles]
        tris = tris.cpu().numpy()
        
        tris_mean = tris.mean(axis=1).mean(axis=0)
        tris = tris - tris_mean
        
        tris_min = tris.min(0).min(0)
        tris = tris - tris_min
        tris_max = tris.max()
        tris = (tris / tris_max + band) / margin
        
        return tris, tris_min, tris_max, tris_mean

    
    def remesh(self, tris, tris_min, tris_max, tris_mean):
        #print("preprocess start")
        # tris, tris_min, tris_max, tris_mean = self.preprocess_mesh(points, triangles, self.band, self.margin)
        
        # tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')
        # torch.cuda.synchronize()
        
        d = torchcumesh2sdf.get_sdf(tris, self.R, self.band)
        d = d - 0.9 / self.R
        
        v, f = self.vol2mesh(d, return_quads=True) #Dual MC

        v, f = v.cpu().numpy(), f.cpu().numpy()
        v = (((v * self.R +0.5)/(self.R+1)* self.margin - self.band) * tris_max + tris_min)
        
        v = torch.from_numpy(v).float().cuda()
        f = torch.from_numpy(f).int().cuda()
        # divide the quad into two triangles maximize the smallest angle within each triangle
        face_config1 = torch.tensor([[0, 1, 3], [1, 2, 3]])
        face_config2 = torch.tensor([[0, 1, 2], [0, 2, 3]])
        angles1, angles2 = [], []
        
        for i in range(len(face_config1)):
            v0, v1, v2 = torch.unbind(v[f[:, face_config1[i]]], dim=-2)
            cos1 = (F.normalize(v1-v0, dim=-1) * F.normalize(v2-v0, dim=-1)).sum(-1)
            cos2 = (F.normalize(v2-v1, dim=-1) * F.normalize(v0-v1, dim=-1)).sum(-1)
            cos3 = (F.normalize(v0-v2, dim=-1) * F.normalize(v1-v2, dim=-1)).sum(-1)
            angles1.append(torch.max(torch.stack([cos1, cos2, cos3], dim=-1), dim=-1)[0])
        for i in range(len(face_config2)):
            v0, v1, v2 = torch.unbind(v[f[:, face_config2[i]]], dim=-2)
            cos1 = (F.normalize(v1-v0, dim=-1) * F.normalize(v2-v0, dim=-1)).sum(-1)
            cos2 = (F.normalize(v2-v1, dim=-1) * F.normalize(v0-v1, dim=-1)).sum(-1)
            cos3 = (F.normalize(v0-v2, dim=-1) * F.normalize(v1-v2, dim=-1)).sum(-1)
            angles2.append(torch.max(torch.stack([cos1, cos2, cos3], dim=-1), dim=-1)[0])

        angles1 = torch.stack(angles1, dim=-1)
        angles2 = torch.stack(angles2, dim=-1)

        angles1 = torch.max(angles1, dim=1)[0]
        angles2 = torch.max(angles2, dim=1)[0]

        faces_1 = f[angles1 < angles2]
        faces_2 = f[angles1 >= angles2]
        f = torch.cat([faces_1[:, [0, 1, 3, 1, 2, 3]].view(-1, 3), faces_2[:, [0, 1, 2, 0, 2, 3]].view(-1, 3)], dim=0)
        
        return v, f

    def run(self, points, triangles, ratio, tolerance=4, threshold=1e-3, iter=1000000, min_verts=10000000000):
        
        self.target_faces = max(int(ratio * len(triangles)), min_verts)
        print("Target faces : {}".format(self.target_faces))

        # scale the input mesh
        tris, tris_min, tris_max, tris_mean = self.preprocess_mesh(points, triangles, self.band, self.margin)
        tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')

        # stage1 (Remeshing)
        if self.use_stage1:
            # Default 256
            if self.target_faces <= 1000:
                self.R = 128
            if self.target_faces <= 50:
                self.R = 64

            start_stage1 = time.time()
            verts, faces = self.remesh(tris, tris_min, tris_max, tris_mean)
            end_stage1 = time.time()
            print(f"Time for Remeshing: {end_stage1 - start_stage1} sec")
        else:
            verts = points - torch.from_numpy(tris_mean).to(points.device)
            faces = triangles

        # stage2 (Simplification)
        start_stage2 =time.time()
        verts_undo = torch.empty(0, dtype=torch.int32, device='cuda')
        n_verts_undo = 0
        count = 0
        is_stuck = 0
        scale = max(max(verts[:,0].max()-verts[:,0].min(), verts[:,1].max()-verts[:,1].min()), verts[:,2].max()-verts[:,2].min())
        init = True
        for it in range(iter):
            num_faces_prev = faces.shape[0]
            # Simplify
            verts, faces, verts_occ, verts_map, verts_undo = self.func.apply(verts, faces, verts_undo, n_verts_undo, scale, threshold, is_stuck, init)
            init = False
            n_verts_undo = verts_undo.shape[0]
            
            # set verts and faces after 1 step of simplification
            verts = verts[verts_occ.view(-1).bool()]
            faces = faces[faces[:, 0] >= 0]
            faces[:,0] = verts_map[faces[:,0].long()].view(-1)
            faces[:,1] = verts_map[faces[:,1].long()].view(-1)
            faces[:,2] = verts_map[faces[:,2].long()].view(-1)
            
            num_faces_current = faces.shape[0]
            
            if num_faces_current <= self.target_faces or num_faces_current <= 10:
                break # simplified to target ratio
            
            if num_faces_current == num_faces_prev:
                count += 1
            else:
                count = 0
                is_stuck = 0

            if count >= 2:
                is_stuck = 1
                
            if count == tolerance:
                print("Not enough edges available to be collapsed.")
                break
        
        end_stage2 = time.time()
        verts = verts.cpu().numpy()+ tris_mean
        faces = faces.cpu().numpy()
        print(f"Time for Simplification: {end_stage2 - start_stage2} sec")
        
        # stage3 (Safe projection)
        if self.use_stage3 == True:
            stage2_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            verts, faces = pamo_safe_project.process(
                self.gt_mesh.vertices,
                self.gt_mesh.faces,
                stage2_mesh.vertices,
                stage2_mesh.faces,
                5,
                system=self.system,  # if provided, reuse the same system to avoid memory allocation
                config=self.config,  # if system is not provided, use this config to create a new system
            )
            
        return verts, faces


class PaSP(nn.Module):
    def __init__(self):
        super().__init__()
        sp = _C.CUDSP()

        class PaSPFunction(Function):
            @staticmethod
            def forward(ctx, points, triangles, scale, threshold, init):
                verts, faces, verts_occ, verts_map = sp.forward(points, triangles, scale, threshold, init)
                ctx.points = points
                ctx.triangles = triangles
                return verts, faces, verts_occ, verts_map

        self.func = PaSPFunction

    def run(self, points, triangles, threshold=0.001, iter=1000):
        verts = points
        faces = triangles
        scale = max(max(verts[:,0].max()-verts[:,0].min(), verts[:,1].max()-verts[:,1].min()), verts[:,2].max()-verts[:,2].min())
        init = True
        for it in range(iter):
            # if it < 20:
            #     t = threshold / (21 - it) * 2
            # else:
            #     t = threshold
            # print(t)
            num_faces = faces.shape[0]
            verts, faces, verts_occ, verts_map = self.func.apply(verts, faces, scale, threshold, init)
            verts = verts[verts_occ.view(-1).bool()]
            faces = faces[faces[:, 0] >= 0]
            faces[:,0] = verts_map[faces[:,0].long()].view(-1)
            faces[:,1] = verts_map[faces[:,1].long()].view(-1)
            faces[:,2] = verts_map[faces[:,2].long()].view(-1)
            init = False
            # if faces.shape[0] < 4500:
            #     break
            if faces.shape[0] == num_faces:
                print("Converged at iteration {}".format(it))
                break

            # v = verts.cpu().numpy()
            # f = faces.cpu().numpy()
            # output_mesh = trimesh.Trimesh(vertices=v, faces=f)
            # output_mesh.export('/home/sarahwei/code/simp_parallel/{}.obj'.format(it))

        
        return verts, faces