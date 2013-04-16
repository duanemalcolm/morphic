import numpy
import morphic
from scipy.spatial import cKDTree

def grid(divs=10, dims=2):
    if isinstance(divs, int):
        divs = [divs for i in range(dims)]
    if dims == 1:
        return numpy.linspace(0, 1, divs[0]+1)
    elif dims == 2:
        Xi = numpy.mgrid[0:divs[1]+1, 0:divs[2]+1]
        return numpy.array([
            Xi[1, :].flatten() * (1./divs[0]),
            Xi[0, :].flatten() * (1./divs[1])]).T
    elif dims == 3:
        Xi = numpy.mgrid[0:divs[2]+1, 0:divs[1]+1, 0:divs[0]+1]
        return numpy.array([
            Xi[2, :].flatten() * (1./divs[0]),
            Xi[1, :].flatten() * (1./divs[1]),
            Xi[0, :].flatten() * (1./divs[2])]).T

def element_dimensions(basis):
    if basis == None:
        return None
    dimensions = 0
    for base in basis:
        if base[0] == 'T':
            dimensions += 2
        else:
            dimensions += 1
    return dimensions

def convert_hermite_lagrange(cHmesh, tol=1e-9):
    if isinstance(cHmesh, str):
        cHmesh = morphic.Mesh(cHmesh)
    
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
        dims = element_dimensions(element.interp)
        if dims == 1:
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
        elif dims == 2:
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
        elif dims == 3:
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
    
    
#################
### GEOMETRIC ###
#################

def length(v):
    """
    Calculates the length of a vector (v)
    """
    return numpy.sqrt((v * v).sum())

def vector(x1, x2, normalize=False):
    """
    Calculates a vector from two points and normalizes if normalize=True
    """
    v = numpy.array(x2) - numpy.array(x1)
    if normalize:
        return v / length(v)
    return v

def dot(v1, v2, normalize=False):
    """
    Calculates the dot product of two vectors and normalizes if normalize=True
    """
    pass

def normal(v1, v2, normalize=False):
    """
    Calculates the normal of two vectors and normalizes if normalize=True
    """
    pass
    