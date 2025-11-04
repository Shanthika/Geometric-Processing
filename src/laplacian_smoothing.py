import torch
from torch import nn
import numpy as np
import potpourri3d as pp3d
from scipy import sparse, spatial
import trimesh
from scipy.sparse.linalg import lsqr, cg, eigsh, inv
from pdb import set_trace as breakpoint

def get_index(verts, color):  
    unique_col = np.array([[255,   0,   0, 255],[255,   255,   255, 255]])
    anchor = []
    for i in range(len(verts)):
        if(np.array_equal(color[i],unique_col[0])):
            anchor.append(i) 

    return anchor 


def get_lmatrix(L, anchor_indices, num_verts):
    L = L.toarray().copy()
    num_anchors = len(anchor_indices) 
    
    bmatrix = -L[:,anchor_indices]
    
    bmatrix[ anchor_indices, range(num_anchors) ] = 0.0
    
    # Set the off-diagonal boundary columns to 0
    Ldiag = np.diag(L).copy()

    L[:, anchor_indices] = 0
    np.fill_diagonal(L, Ldiag)
    
    L = sparse.coo_matrix(L, shape=(L.shape)).tocsr()
    bmatrix = sparse.coo_matrix(bmatrix, shape=(bmatrix.shape)).tocsr()
    breakpoint()
    return L, bmatrix


def solve_laplacian(L, delta, num_verts):
    updated_verts = np.zeros((num_verts,3))
    for i in range(3):
        updated_verts[:, i] = lsqr(L, delta[:, i])[0]
        
    return updated_verts
    

def main(mesh, anchor_indices):

    verts = mesh.vertices
    faces = mesh.faces
    num_verts = verts.shape[0]
    
    num_anchor_verts = len(anchor)  

    # Laplacian matrix L
    L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)

    # L' with new weights for anchors and handles
    L, L_dense = get_lmatrix( L, anchor_indices, num_verts)

    # del' = L'.V
    delta =  L_dense @ verts[anchor_indices]

    #solve for V' = L'^(-1) . del'
    updated_verts = solve_laplacian(L, delta, num_verts )
    updated_verts[anchor_indices] = verts[anchor_indices]  # keep anchor vertices fixed
    mesh.vertices = updated_verts
    mesh.export('results/smooth.ply')
    
    
    



if __name__ == '__main__':

    mesh = trimesh.load("data/meshes/pipe.obj",process=False, maintain_order=True)
    verts = mesh.vertices
    color = mesh.visual.vertex_colors
    ########### pick up anchor and handles from mesh vertex colors
    anchor  = get_index(verts,color)  
    # define new handle positions
    main(mesh,anchor)