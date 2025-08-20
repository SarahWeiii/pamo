"""
Test the Hausdorff distance and Chamfer distance between
Stage 2 outputs and ground truth.  

Motivation: Get familiar with the evaluation metrics and
data formats. 

Author: Xiaodi

Date: 05/02/2024
"""

import argparse
import os

import numpy as np
import trimesh

from stage3.metrics import compute_min_angle


if __name__ == '__main__':
    V = np.array([[0, 0, 0], [1, 0, 0], [0, np.sqrt(3), 0]], dtype=np.float32)
    F = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=V, faces=F)
    min_angle = compute_min_angle(mesh)
    print(min_angle * 180 / np.pi)
    