import numpy as np
import matplotlib.pyplot as plt


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

def display(vertices):
    x,y,z = vertices
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.5, rstride=2, cstride=2, linewidth=0)

    # Labels and view
    ax.set_title("Tangent Vectors $x_u$ and $x_v$ on a Sphere")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()
