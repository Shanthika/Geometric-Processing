import numpy as np
import scipy.sparse as sp

def face_B_matrices(V, F):
    # V: (N_v, 3), F: (F, 3)
    Fcount = F.shape[0]
    B = np.zeros((Fcount, 3, 3))
    for t in range(Fcount):
        i0, i1, i2 = F[t]
        p0, p1, p2 = V[i0], V[i1], V[i2]
        e1 = p1 - p0
        e2 = p2 - p0
        M = np.column_stack([e1, e2])  # 3x2
        MTM = M.T @ M                  # 2x2
        invMTM = np.linalg.inv(MTM)    # 2x2
        P = invMTM @ M.T               # 2x3  -> rows are grad u and grad v (row vectors)
        grad_u = P[0, :]               # shape (3,)
        grad_v = P[1, :]
        grad_phi1 = grad_u
        grad_phi2 = grad_v
        grad_phi0 = -grad_u - grad_v
        B[t, :, 0] = grad_phi0   # column for i0
        B[t, :, 1] = grad_phi1   # column for i1
        B[t, :, 2] = grad_phi2   # column for i2
    return B  # shape (F, 3, 3)

def assemble_G(V, F):
    N_v = V.shape[0]
    Fcount = F.shape[0]
    B = face_B_matrices(V, F)
    rows, cols, data = [], [], []
    for t in range(Fcount):
        base_row = 3 * t
        for local_k in range(3):
            vi = F[t, local_k]
            g = B[t, :, local_k]   # 3-vector
            # place g into rows base_row..base_row+2 at column vi
            rows.extend([base_row + 0, base_row + 1, base_row + 2])
            cols.extend([vi, vi, vi])
            data.extend([g[0], g[1], g[2]])
    G = sp.coo_matrix((data, (rows, cols)), shape=(3*Fcount, N_v)).tocsr()
    return G


V = np.array([[3,0,1],[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], dtype=float)
F = np.array([[0,3,2],[0,4,3],[0,2,1],[0,1,4]], dtype=int)
G = assemble_G(V, F)
# test function f(x,y,z) = 2x - 3y + 5
f = 2*V[:,0] - 3*V[:,1] + 5
grad_faces = (G @ f).reshape((-1,3))
print("per-face gradients:\n", grad_faces)

print(G.todense())