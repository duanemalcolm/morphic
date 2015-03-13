import numpy
import morphic

class PCAMesh(object):

    def __init__(self, groups=None):
        self.X = []
        self.pca = None
        self.num_modes = 5
        self.input_mesh = None
        self.groups = groups
        self.mesh = None

    def add_mesh(self, mesh):
        if self.input_mesh == None:
            self.input_mesh = mesh
        if isinstance(mesh, str):
            mesh = morphic.Mesh(mesh)
        x = []
        if self.groups is None:
            for node in mesh.nodes:
                if not isinstance(node, morphic.mesher.DepNode):
                    x.extend(node.values.flatten().tolist())
        else:
            for node in mesh.nodes:
                if node.in_group(self.groups):
                    x.extend(node.values.flatten().tolist())
        self.X.append(x)

    def generate(self, num_modes=5, package='sklearn'):
        self.X = numpy.array(self.X)
        self.num_modes = num_modes
        if package == 'sklearn':
            from sklearn import decomposition
            self.pca = decomposition.PCA(n_components=num_modes)
            self.pca.fit(self.X)
            self.mean = self.pca.mean_
            self.components = self.pca.components_.T
            self.variance = self.pca.explained_variance_
        else:
            import mdp
            self.pca = mdp.nodes.PCANode(output_dim=num_modes)
            self.pca.execute(self.X)
            self.mean = self.pca.avg[0]
            self.components = self.pca.v
            self.variance = self.pca.d
        self.generate_mesh()
        return self.mesh

    def generate_mesh(self):
        ### Generate mesh from PCA results
        self.mesh = morphic.Mesh()
        weights = numpy.zeros(self.num_modes + 1)
        weights[0] = 1.
        self.mesh.add_stdnode('weights', weights)
        variance = numpy.zeros(self.num_modes + 1)
        variance[0] = 1.0
        variance[1:] = numpy.sqrt(self.variance)
        self.mesh.add_stdnode('variance', variance)
        
        # Add node values
        idx = 0
        if self.groups is None:
            for node in self.input_mesh.nodes:
                x = self.get_pca_node_values(node, idx)
                self.mesh.add_pcanode(node.id, x, 'weights', 'variance', group='pca')
                idx += nsize
        else:
            for node in self.input_mesh.nodes:
                nsize = node.values.size
                if node.in_group(self.groups):
                    x = self.get_pca_node_values(node, idx)
                    self.mesh.add_pcanode(node.id, x, 'weights', 'variance', group='pca')
                    idx += nsize
                else:
                    if isinstance(node, morphic.mesher.StdNode):
                        self.mesh.add_stdnode(node.id, node.values)
                    elif isinstance(node, morphic.mesher.DepNode):
                        self.mesh.add_depnode(node.id, node.element, node.node, shape=node.shape, scale=node.scale)
                    if isinstance(node, morphic.mesher.PCANode):
                        raise Exception("Not implemented")

        for element in self.input_mesh.elements:
            self.mesh.add_element(element.id, element.basis, element.node_ids)
        
        self.mesh.generate()

    def get_pca_node_values(self, node, idx):
        nsize = node.values.size
        if len(node.shape) == 1:
            pca_node_shape = (node.shape[0], 1, self.num_modes)
            x = numpy.zeros((node.shape[0], 1, self.num_modes + 1)) # +1 to include mean
            x[:, 0, 0] = self.mean[idx:idx+nsize].reshape(node.shape) # mean values
            x[:, :, 1:] = self.components[idx:idx+nsize,:].reshape(pca_node_shape) # mode values
            return x
        elif len(node.shape) == 2:
            pca_node_shape = (node.shape[0], node.shape[1], self.num_modes)
            x = numpy.zeros((node.shape[0], node.shape[1], self.num_modes + 1)) # +1 to include mean
            x[:, :, 0] = self.mean[idx:idx+nsize].reshape(node.shape) # mean values
            x[:, :, 1:] = self.components[idx:idx+nsize,:].reshape(pca_node_shape) # mode values
            return x
        else:
            print 'Cannot reshape this node when genrating pca mesh'


def grid(divs=10, dims=2):
    if isinstance(divs, int):
        divs = [divs for i in range(dims)]
    if dims == 1:
        return numpy.linspace(0, 1, divs[0]+1)
    elif dims == 2:
        Xi = numpy.mgrid[0:divs[1]+1, 0:divs[0]+1]
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
        dims = element_dimensions(element.basis)
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
    