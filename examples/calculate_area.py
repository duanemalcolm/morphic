"""
To calculate the volume of an element we can:
1. use Gaussian-quadrature to integrate the divergence of the 
deformation vectors at gauss points.
2. calculate the determinant of the Jacobian
3. numerically integrate the subdivided element - this can be crude
"""

import pickle
import scipy

import morphic
reload(morphic)

import morphic.utils
reload(morphic.utils)

from morphic import viewer

def cHnode(x, dx, y, dy):
    return [[x, dx, 0, 0], [y, 0, dy, 0, 0]]

def area(mesh, elem_id):
    X, T = mesh.get_surfaces(50)
    A = 0
    for t in T:
        v1 = morphic.utils.vector(X[t[1]], X[t[0]])
        v2 = morphic.utils.vector(X[t[2]], X[t[0]])
        A += 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])
    return A

def generate_lagrange_mesh(cHmesh, order):
    lmesh = morphic.Mesh()
    Xi1, Xi2 = numpy.mgrid[0:order+1, 0:order+1]
    Xi = numpy.array([Xi2.flatten(), Xi1.flatten()]).T * 1./order
    X = cHmesh.elements[1].evaluate(Xi)
    for i,x in enumerate(X):
        lmesh.add_stdnode(i+1, x)
    lmesh.add_element(1, ['L' + str(order), 'L' + str(order)],
        range(1, (order + 1)**2 + 1))
    lmesh.generate()
    return lmesh

if "fig" not in locals():
    fig = viewer.Figure()
 
mesh = morphic.Mesh()
mesh.add_stdnode(1, [[0.1, 1.2, 0, 0], [0, 0.1, 1.1, 0]])
mesh.add_stdnode(2, [[1.1, 1.0, 0, 0], [0, 0.0, 1.0, 0]])
mesh.add_stdnode(3, [[0.0, 1.1, 0, 0], [1.0, 0.1, 0.9, 0]])
mesh.add_stdnode(4, [[0.9, 0.9, 0, 0], [0.9, -0.1, 0.8, 0]])
mesh.add_element(1, ['H3', 'H3'], [1, 2, 3, 4])
mesh.generate()
print 'H3: %7.4f %7.4f' % (area(mesh, 1), mesh.elements[1].area())

lmesh = generate_lagrange_mesh(mesh, 1)
print 'L1: %7.4f %7.4f' % (area(lmesh, 1), lmesh.elements[1].area())

lmesh = generate_lagrange_mesh(mesh, 2)
print 'L2: %7.4f %7.4f' % (area(lmesh, 1), lmesh.elements[1].area())

lmesh = generate_lagrange_mesh(mesh, 3)
print 'L3: %7.4f %7.4f' % (area(lmesh, 1), lmesh.elements[1].area())

lmesh = generate_lagrange_mesh(mesh, 4)
print 'L4: %7.4f %7.4f' % (area(lmesh, 1), lmesh.elements[1].area())


