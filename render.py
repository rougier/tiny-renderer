# Tiny 3D Renderer
import numpy as np

def triangle(A, B, C, intensity):
    global coords, texture, image, zbuffer
    (v0, uv0), (v1, uv1), (v2, uv2) = A, B, C

    # Barycentric coordinates of points inside the triangle bounding box
    try:
        T = np.linalg.inv([[v0[0],v1[0],v2[0]], [v0[1],v1[1],v2[1]], [1,1,1]])
    except np.linalg.LinAlgError:
        return
    xmin = int(max(0,              min(v0[0], v1[0], v2[0])))
    xmax = int(min(image.shape[1], max(v0[0], v1[0], v2[0])+1))
    ymin = int(max(0,              min(v0[1], v1[1], v2[1])))
    ymax = int(min(image.shape[0], max(v0[1], v1[1], v2[1])+1))

    P = coords[:, xmin:xmax, ymin:ymax].reshape(2,-1)
    B = np.dot(T, np.vstack((P, np.ones((1, P.shape[1]), dtype=int))))

    # Cartesian coordinates of points inside the triangle
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y, Z = P[0,I], P[1,I], v0[2]*B[0,I] + v1[2]*B[1,I] + v2[2]*B[2,I]

    # Texture coordinates of points inside the triangle
    U = (    (uv0[0]*B[0,I] + uv1[0]*B[1,I] + uv2[0]*B[2,I]))*(texture.shape[0]-1)
    V = (1.0-(uv0[1]*B[0,I] + uv1[1]*B[1,I] + uv2[1]*B[2,I]))*(texture.shape[1]-1)
    C = texture[V.astype(int), U.astype(int)]

    # Z-Buffer test
    I = np.argwhere(zbuffer[Y,X] < Z)[:,0]
    X, Y, Z, C = X[I], Y[I], Z[I], C[I]
    zbuffer[Y, X] = Z
    image[Y, X] = C * (intensity, intensity, intensity, 1)


def obj_load(filename):
    V, T, Vi, Ti = [], [], [], []
    with open(filename) as f:
       for line in f.readlines():
           if line.startswith('#'): continue
           values = line.split()
           if not values: continue
           if values[0] == 'v':
               V.append([float(x) for x in values[1:4]])
           elif values[0] == 'vt':
               T.append([float(x) for x in values[1:3]])
           elif values[0] == 'f' :
               Vi.append([int(indices.split('/')[0]) for indices in values[1:]])
               Ti.append([int(indices.split('/')[1]) for indices in values[1:]])
    return np.array(V), np.array(T), np.array(Vi)-1, np.array(Ti)-1


def lookat(eye, center, up):
    normalize = lambda x: x/np.linalg.norm(x)
    M = np.eye(4)
    z = normalize(eye-center)
    x = normalize(np.cross(up,z))
    y = normalize(np.cross(z,x))
    M[0,:3], M[1,:3], M[2,:3], M[:3,3] = x, y, z, -center
    return M

def viewport(x, y, w, h, d):
    return np.array([[w/2, 0, 0, x+w/2],
                     [0, h/2, 0, y+h/2],
                     [0, 0, d/2,   d/2],
                     [0, 0, 0,       1]])

def cross(a, b): # numpy version is slow for 3d vectors
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    
if __name__ == '__main__':
    import time
    from imageio.core.functions import imwrite, imread

    width, height = 1000,1000
    light         = np.array([0,0,-1])
    eye           = np.array([1,1,3])
    center        = np.array([0,0,0])
    up            = np.array([0,1,0])

    image = np.zeros((height,width,4), dtype=np.uint8)
    zbuffer = -1000*np.ones((height,width))
    coords = np.mgrid[0:width, 0:height].astype(int)

    V, T, Vi, Ti = obj_load("head.obj")
    texture = imread("uv-grid.png")
    viewport = viewport(32, 32, width-64, height-64, 1000)
    modelview = lookat(eye, center, up)

    Vh = np.c_[V, np.ones(len(V))] # Homogenous coordinates
    V = Vh @ modelview.T           # World coordinates
    Vs = V @ viewport.T            # Screen coordinates
    V, Vs = V[:,:3],  Vs[:,:3]
    
    start = time.time()
    for (i0,i1,i2), (j0,j1,j2) in zip(Vi,Ti):
        v0,v1,v2 = V[i0], V[i1], V[i2]
        vs0,vs1,vs2 = Vs[i0], Vs[i1], Vs[i2]
        uv0, uv1, uv2 = T[j0], T[j1], T[j2]

        n = cross(v2-v0, v1-v0)
        n = n / np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
        intensity = np.dot(n, light)
        
        if intensity >= 0:
            triangle( (vs0,uv0), (vs1,uv1), (vs2,uv2), intensity)
    end = time.time()
    print("Rendering time: {}".format(end-start))
    imwrite("output.png", image[::-1,:,:])
