import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from . import _C

class DMC(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        if dtype == torch.float32:
            dmc = _C.CUDMCFloat()
        elif dtype == torch.float64:
            dmc = _C.CUDMCDouble()

        class DMCFunction(Function):
            @staticmethod
            def forward(ctx, grid, isovalue):
                verts, quads, tris = dmc.forward(grid, isovalue)
                ctx.isovalue = isovalue
                return verts, quads, tris
            
        self.func = DMCFunction

    
    def forward(self, grid, isovalue=0.0, return_quads=False, normalize=True):
        if grid.min() >= isovalue or grid.max() <= isovalue:
            return torch.zeros((0, 3), dtype=self.dtype, device=grid.device), torch.zeros((0, 4), dtype=torch.int32, device=grid.device)
        dimX, dimY, dimZ = grid.shape
        grid = F.pad(grid, (1, 1, 1, 1, 1, 1), "constant", isovalue+1)
        verts, quads, tris = self.func.apply(grid, isovalue)
        verts = verts - 1
        if normalize:
            verts = verts / (
                torch.tensor([dimX, dimY, dimZ], dtype=verts.dtype, device=verts.device) - 1
            )
        if return_quads:
            return verts, quads.long()
        else:
            return verts, tris.long()