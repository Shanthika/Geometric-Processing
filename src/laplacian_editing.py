import torch
from torch import nn
import numpy as np
import potpourri3d as pp3d
from scipy import sparse, spatial
import trimesh
from scipy.sparse.linalg import lsqr, cg, eigsh, inv

def get_index(verts, color):  
    unique_col = np.array([[ 0, 255,   0, 255],[255,   0,   0, 255],[0,   0,   0, 255]])
    anchor = []
    handle = [] 
    for i in range(len(verts)):
        if(np.array_equal(color[i],unique_col[0])):
            handle.append(i)
        elif(np.array_equal(color[i],unique_col[1])):
            anchor.append(i) 

    return anchor, handle 


def get_lmatrix(L, anchor_indices,handles, num_verts):
    L= L.toarray()
    num_anchors = len(anchor_indices) 
    num_handles = len(handles)
    
    amatrix = np.zeros((num_anchors,num_verts))
    hmatrix = np.zeros((num_handles,num_verts))
    
    for i in range(num_anchors):
        amatrix[i, anchor_indices[i]] = 1.0
    for i in range(num_handles):
        hmatrix[i, handles[i]] = 1.0
    
    L = np.vstack((L,amatrix,hmatrix))     
    L = np.vstack((L,amatrix,hmatrix))     
    L = sparse.coo_matrix(L, shape=(L.shape)).tocsr()
    return L


def solve_laplacian(L, delta, num_verts):
    
    updated_verts = np.zeros((num_verts,3))
    for i in range(3):
        updated_verts[:, i] = lsqr(L, delta[:, i])[0]
        
    return updated_verts
    

def main(mesh, anchor_indices, handles,new_pos):

    verts = mesh.vertices
    faces = mesh.faces
    num_verts = verts.shape[0]
    
    num_anchor_verts = len(anchor)  

    # Laplacian matrix L
    L_dense = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)

    # L' with new weights for anchors and handles
    L_dense = get_lmatrix( L_dense, anchor_indices,handles, num_verts)

    # del' = L'.V
    delta =  L_dense @ verts

    # update delta to define new handle vertices
    for i in range(len(handles)):
        delta[num_verts+num_anchor_verts+i, :] = new_pos[i] 
    
    #solve for V' = L'^(-1) . del'
    updated_verts = solve_laplacian(L_dense, delta, num_verts )

    mesh.vertices = updated_verts
    mesh.export('results/laplacian_edit.ply')
    
    
    



if __name__ == '__main__':

    mesh = trimesh.load("data/meshes/bunny_to_edit.obj",process=False, maintain_order=True)
    verts = mesh.vertices
    color = mesh.visual.vertex_colors
    ########### pick up anchor and handles from mesh vertex colors
    anchor,handles  = get_index(verts,color)  
    # define new handle positions
    new_vert_position = np.array([[0.02,0.1814,-0.0096]])
    main(mesh,anchor,handles,new_vert_position)