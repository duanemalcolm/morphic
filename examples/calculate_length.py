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

def integrate_length(mesh, elem_id):
    Xi = scipy.linspace(0, 1, 100)
    X = mesh.elements[elem_id].evaluate(Xi)
    dX = X[1:, :] - X[:-1, :]
    return scipy.sqrt((dX*dX).sum(1)).sum()


if "fig" not in locals():
    fig = viewer.Figure()
 
mesh = morphic.Mesh()
mesh.add_stdnode(1, [0, 0])
mesh.add_stdnode(2, [0.3, 0.5])
mesh.add_stdnode(3, [0.65, 0.4])
mesh.add_stdnode(4, [1, -0.2])
mesh.add_element(1, ['L1'], [1, 4])
mesh.add_element(2, ['L2'], [1, 3, 4])
mesh.add_element(3, ['L3'], [1, 2, 3, 4])
mesh.generate()

print 'L1: %7.4f %7.4f' % (integrate_length(mesh, 1), mesh.elements[1].length())
print 'L2: %7.4f %7.4f' % (integrate_length(mesh, 2), mesh.elements[2].length())
print 'L3: %7.4f %7.4f' % (integrate_length(mesh, 3), mesh.elements[3].length())

x1 = numpy.array([
    mesh.elements[3].evaluate([0]), mesh.elements[3].evaluate([0], deriv=[1])]).T
x2 = numpy.array([
    mesh.elements[3].evaluate([1]), mesh.elements[3].evaluate([1], deriv=[1])]).T

meshH3 = morphic.Mesh()
meshH3.add_stdnode(1, x1)
meshH3.add_stdnode(2, x2)
meshH3.add_element(1, ['H3'], [1, 2])
meshH3.generate()
print 'H3: %7.4f %7.4f' % (integrate_length(meshH3, 1), meshH3.elements[1].length())




