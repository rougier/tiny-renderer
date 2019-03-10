# Tiny 3D Renderer (with outlines)
import numpy as np

def triangle(t, v0, v1, v2, intensity):
    global coords, image, zbuffer

    # Barycentric coordinates of points inside the triangle bounding box
    xmin = int(max(0,              min(v0[0], v1[0], v2[0])))
    xmax = int(min(image.shape[1], max(v0[0], v1[0], v2[0])+1))
    ymin = int(max(0,              min(v0[1], v1[1], v2[1])))
    ymax = int(min(image.shape[0], max(v0[1], v1[1], v2[1])+1))
    P = coords[:, xmin:xmax, ymin:ymax].reshape(2,-1)
    B = np.dot(t, np.vstack((P, np.ones((1, P.shape[1]), dtype=int))))

    # Cartesian coordinates of points inside the triangle
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y, Z = P[0,I], P[1,I], v0[2]*B[0,I] + v1[2]*B[1,I] + v2[2]*B[2,I]

    # Z-Buffer test
    I = np.argwhere(zbuffer[Y,X] < Z)[:,0]
    X, Y, Z = X[I], Y[I], Z[I]
    zbuffer[Y, X] = Z
    image[Y, X] = intensity, intensity, intensity, 255

    # Outline (black color)
    P = []
    P.extend(line(v0,v1))
    P.extend(line(v1,v2))
    P.extend(line(v2,v0))
    P = np.array(P).T
    B = np.dot(t, np.vstack((P, np.ones((1, P.shape[1]), dtype=int))))
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y, Z = P[0,I], P[1,I], v0[2]*B[0,I] + v1[2]*B[1,I] + v2[2]*B[2,I]
    I = np.argwhere(zbuffer[Y,X] <= Z)[:,0]
    X, Y, Z = X[I], Y[I], Z[I]
    image[Y, X] = 0, 0, 0, 255


def line(A, B):
    (x0, y0, _), (x1, y1, _) = np.array(A).astype(int), np.array(B).astype(int)
    P = []
    steep = False
    if abs(x0-x1) < abs(y0-y1): 
        steep, x0, y0, x1, y1 = True, y0, x0, y1, x1
    if x0 > x1: x0, x1, y0, y1 = x1, x0, y1, y0
    dx, dy = x1-x0, y1-y0
    y, error2, derror2 = y0, 0, abs(dy)*2
    for x in range(x0,x1+1):
        if steep: P.append((y,x))
        else:     P.append((x,y))
        error2 += derror2; 
        if error2 > dx:
            y += 1 if y1 > y0 else -1 
            error2 -= dx*2
    return P

def obj_load(filename):
    V, Vi = [], []
    with open(filename) as f:
       for line in f.readlines():
           if line.startswith('#'): continue
           values = line.split()
           if not values: continue
           if values[0] == 'v':
               V.append([float(x) for x in values[1:4]])
           elif values[0] == 'f' :
               Vi.append([int(x) for x in values[1:4]])
    return np.array(V), np.array(Vi)-1


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

    
if __name__ == '__main__':
    import time
    import PIL.Image

    width, height = 1200,1200
    light         = np.array([0,0,-1])
    eye           = np.array([-1,1,3])
    center        = np.array([0,0,0])
    up            = np.array([0,1,0])

    image = np.zeros((height,width,4), dtype=np.uint8)
    zbuffer = -1000*np.ones((height,width))
    coords = np.mgrid[0:width, 0:height].astype(int)

    V, Vi = obj_load("bunny.obj")

    # Centering and scaling
    vmin, vmax = V.min(), V.max()
    V = (2*(V-vmin)/(vmax-vmin) - 1)*1.25
    xmin, xmax = V[:,0].min(), V[:,0].max()
    V[:,0] = V[:,0] - xmin - (xmax-xmin)/2
    ymin, ymax = V[:,1].min(), V[:,1].max()
    V[:,1] = V[:,1] - ymin - (ymax-ymin)/2

    viewport = viewport(32, 32, width-64, height-64, 1000)
    modelview = lookat(eye, center, up)

    Vh = np.c_[V, np.ones(len(V))] # Homogenous coordinates
    V = Vh @ modelview.T           # World coordinates
    Vs = V @ viewport.T            # Screen coordinates
    V, Vs = V[:,:3],  Vs[:,:3]     # Back to cartesian coordinates

    V, Vs = V[Vi], Vs[Vi]
    
    # Pre-compute tri-linear coordinates
    T = np.transpose(Vs, axes=[0,2,1]).copy()
    T[:,2,:] = 1
    T = np.linalg.inv(T)

    # Pre-compute normal vectors and intensity
    N = np.cross(V[:,2]-V[:,0], V[:,1]-V[:,0])
    N = N / np.linalg.norm(N,axis=1).reshape(len(N),1)
    I = np.dot(N, light)*255

    start = time.time()
    for i in np.argwhere(I>=0)[:,0]:
        (vs0, vs1, vs2) = Vs[i]
        triangle(T[i], vs0, vs1, vs2, I[i])
        #line(vs0, vs1, (0,0,0,255))
        #line(vs1, vs2, (0,0,0,255))
        #line(vs2, vs0, (0,0,0,255))
        
    end = time.time()
    
    print("Rendering time: {}".format(end-start))
    PIL.Image.fromarray(image[::-1,:,:]).save("bunny.png")
