import numpy
import morphic
from scipy.spatial import cKDTree

def element_dimensions(basis):
    dimensions = 0
    for base in basis:
        if base[0] == 'T':
            dimensions += 2
        else:
            dimensions += 1
    return dimensions

def convert_hermite_lagrange(cHmesh, tol=1e-9):
    Xi3d = numpy.mgrid[0:4, 0:4, 0:4] * (1./3.)
    Xi3d = numpy.array([Xi3d[2, :, :].flatten(),
            Xi3d[1, :, :].flatten(), Xi3d[0, :, :].flatten()]).T
    Xi2d = numpy.mgrid[0:4, 0:4] * (1./3.)
    Xi2d = numpy.array([Xi2d[1, :].flatten(), Xi2d[0, :].flatten()]).T
    Xi1d = numpy.mgrid[0:4] * (1./3.)
    Xi1d = Xi1d.flatten().T
    
    X = []
    
    mesh = morphic.Mesh() # lagrange mesh
    mesh.auto_add_faces = cHmesh.auto_add_faces
    
    nid = 0
    eid = 0
    # add cubic-Hermite nodes first to preserve node ids
    for node in cHmesh.nodes:
        nid += 1
        xn = node.values[:, 0]
        mesh.add_stdnode(nid, xn)
        X.append(xn)
    
    for element in cHmesh.elements:
        element_nodes = []
        tree = cKDTree(X)
        if element.dimensions == 1:
            print "1D element conversion unchecked"
            Xg = element.evaluate(Xi1d)
            for xg in Xg:
                r, index = tree.query(xg.tolist())
                if r > tol:
                    nid += 1
                    mesh.add_stdnode(nid, xg)
                    X.append(xg)
                    element_nodes.append(nid)
                else:
                    element_nodes.append(index + 1)
            eid += 1
            mesh.add_element(eid, ['L3'], element_nodes)
        elif element.dimensions == 2:
            print "2D element conversion unchecked"
            Xg = element.evaluate(Xi2d)
            for xg in Xg:
                r, index = tree.query(xg.tolist())
                if r > tol:
                    nid += 1
                    mesh.add_stdnode(nid, xg)
                    X.append(xg)
                    element_nodes.append(nid)
                else:
                    element_nodes.append(index + 1)
            eid += 1
            mesh.add_element(eid, ['L3', 'L3'], element_nodes)
        elif element.dimensions == 3:
            Xg = element.evaluate(Xi3d)
            for xg in Xg:
                r, index = tree.query(xg.tolist())
                if r > tol:
                    nid += 1
                    mesh.add_stdnode(nid, xg)
                    X.append(xg)
                    element_nodes.append(nid)
                else:
                    element_nodes.append(index + 1)
            eid += 1
            mesh.add_element(eid, ['L3', 'L3', 'L3'], element_nodes)
        else:
            raise ValueError('Element conversion: element dimension not supported')
    
    mesh.generate()
    
    return mesh
    
    
    