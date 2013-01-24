import pickle
import scipy
from scipy import linalg
from scipy.spatial import cKDTree

import morphic
reload(morphic)

import morphic.utils
reload(morphic.utils)

# from morphic import viewer

# if "fig" not in locals():
#     fig = viewer.Figure()
def cH_node_values(values):
    return numpy.array([
        [values[0], 1, 0.0, 0, 0, 0, 0, 0],
        [values[1], 0, 1, 0, 0, 0, 0, 0],
        [values[2], 0, 0, 0, 1, 0, 0, 0]])

def print_header(fp):
 fp.write(' CMISS Version 2.1  ipnode File Version 2\n')
 fp.write(' Heading:\n\n')
 
 fp.write(' The number of nodes is [     1]: 8\n')
 fp.write(' Number of coordinates [3]: 3\n')
 fp.write(' Do you want prompting for different versions of nj=1 [N]? N\n')
 fp.write(' Do you want prompting for different versions of nj=2 [N]? N\n')
 fp.write(' Do you want prompting for different versions of nj=3 [N]? N\n')
 fp.write(' The number of derivatives for coordinate 1 is [0]: 7\n')
 fp.write(' The number of derivatives for coordinate 2 is [0]: 7\n')
 fp.write(' The number of derivatives for coordinate 3 is [0]: 7\n\n')



def print_node(fp, id, x):
     fp.write(' Node number [     1]: %d\n' % (id))
     fp.write(' The Xj(1) coordinate is [ 0.00000E+00]: %f\n' % (x[0]))
     fp.write(' The derivative wrt direction 1 is [ 0.00000E+00]: %f\n' % (x[1]))
     fp.write(' The derivative wrt direction 2 is [ 0.00000E+00]: %f\n' % (x[2]))
     fp.write(' The derivative wrt directions 1 & 2 is [ 0.00000E+00]: %f\n' % (x[3]))
     fp.write(' The derivative wrt direction 3 is [ 0.00000E+00]: %f\n' % (x[4]))
     fp.write(' The derivative wrt directions 1 & 3 is [ 0.00000E+00]: %f\n' % (x[5]))
     fp.write(' The derivative wrt directions 2 & 3 is [ 0.00000E+00]: %f\n' % (x[6]))
     fp.write(' The derivative wrt directions 1, 2 & 3 is [ 0.00000E+00]: %f\n' % (x[7]))
     fp.write(' The Xj(2) coordinate is [ 0.00000E+00]: %f\n' % (x[8]))
     fp.write(' The derivative wrt direction 1 is [ 0.00000E+00]: %f\n' % (x[9]))
     fp.write(' The derivative wrt direction 2 is [ 0.00000E+00]: %f\n' % (x[10]))
     fp.write(' The derivative wrt directions 1 & 2 is [ 0.00000E+00]: %f\n' % (x[11]))
     fp.write(' The derivative wrt direction 3 is [ 0.00000E+00]: %f\n' % (x[12]))
     fp.write(' The derivative wrt directions 1 & 3 is [ 0.00000E+00]: %f\n' % (x[13]))
     fp.write(' The derivative wrt directions 2 & 3 is [ 0.00000E+00]: %f\n' % (x[14]))
     fp.write(' The derivative wrt directions 1, 2 & 3 is [ 0.00000E+00]: %f\n' % (x[15]))
     fp.write(' The Xj(3) coordinate is [ 0.00000E+00]: %f\n' % (x[16]))
     fp.write(' The derivative wrt direction 1 is [ 0.00000E+00]: %f\n' % (x[17]))
     fp.write(' The derivative wrt direction 2 is [ 0.00000E+00]: %f\n' % (x[18]))
     fp.write(' The derivative wrt directions 1 & 2 is [ 0.00000E+00]: %f\n' % (x[19]))
     fp.write(' The derivative wrt direction 3 is [ 0.00000E+00]: %f\n' % (x[20]))
     fp.write(' The derivative wrt directions 1 & 3 is [ 0.00000E+00]: %f\n' % (x[21]))
     fp.write(' The derivative wrt directions 2 & 3 is [ 0.00000E+00]: %f\n' % (x[22]))
     fp.write(' The derivative wrt directions 1, 2 & 3 is [ 0.00000E+00]: %f\n\n' % (x[23]))
 

def calc_volume(mesh):
    Xi = morphic.utils.grid(50, 3)
    Xm = mesh.elements[1].evaluate(Xi)
    tree = cKDTree(Xm)
    Xn = mesh.get_nodes()
    Xmin, Xmax =  Xn.min(0), Xn.max(0)
    dX = Xmax - Xmin
    print dX
    N = 50
    Xi = morphic.utils.grid(N, 3)
    dv = (dX / N).prod()
    dlimit = (dX / N).max()
    print dlimit, dv
    X = Xmin + dX * Xi
    d, i = tree.query(X.tolist())
    ii = scipy.argwhere(d < dlimit)
    print ii.shape[0] * dv, ii.shape, X.shape

def calc_vol2(X, T):
    minz = X[:,2].min()
    X[:,2] -= minz
    N = X.shape[0] / 6
    sc = [-1, 1, -1, 1, -1, 1]
    V = 0
    for i in range(6):
        i1 = i * N
        i2 = (i+1) * N
        for t in T[i1:i2]:
            x = X[t,:]
            h = x[:,2].mean(0)
            v1 = morphic.utils.vector(x[0,:2], x[1,:2])
            v2 = morphic.utils.vector(x[0,:2], x[2,:2])
            a = 0.5 * morphic.utils.length(scipy.cross(v1, v2))
            #print a
            V += sc[i] * a * h
    return V
    
mesh = morphic.Mesh()
mesh.auto_add_faces = True


Xn = pickle.load(open('data/prone.pkl', 'r'))
Elements = pickle.load(open('data/elements.pkl', 'r'))    

fp = open('file.ipnode', 'w')
print_header(fp)

elem_index = 7
for i, xn in enumerate(Xn[Elements[elem_index], :]):
    print_node(fp, i+1, xn)
    x = xn.reshape((3, 8))
    mesh.add_stdnode(i + 1, x)
mesh.add_element(1, ['H3', 'H3', 'H3'], [1, 2, 3, 4, 5, 6, 7, 8])

fp.close()

mesh.generate()






V = mesh.elements[1].volume()
# print 'Volume %8.0f %8.0f %8.0f' % (V - Vcm, V, Vcm)
print 'Volume %8.0f' % (V)
#calc_volume(mesh)




Xs, Ts = mesh.get_faces(40)
print calc_vol2(Xs, Ts)

# Xn = mesh.get_nodes()

# fig.plot_surfaces('lsurface', Xs, Ts, color=(0, 0.8, 1))
# fig.plot_points('nodes', Xn, size=2)

mesh = morphic.Mesh()
mesh.auto_add_faces = True

Xn = scipy.array([
    cH_node_values([0,0,0]),
    cH_node_values([1,0,0]),
    cH_node_values([0,1,0]),
    cH_node_values([1,1,0]),
    cH_node_values([0,0,1]),
    cH_node_values([1,0,1]),
    cH_node_values([0,1,1]),
    cH_node_values([1,1,1])])

for i, xn in enumerate(Xn):
    mesh.add_stdnode(i + 1, xn)

mesh.add_element(1, ['H3', 'H3', 'H3'], [1, 2, 3, 4, 5, 6, 7, 8])

mesh.generate()

V = mesh.elements[1].volume()
# print 'Volume %8.0f %8.0f %8.0f' % (V - Vcm, V, Vcm)
print 'Volume %8.0f' % (V)
X, T = mesh.get_faces(32)
print calc_vol2(X, T)