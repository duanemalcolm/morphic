import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import scipy.linalg
from scipy.spatial import cKDTree
import sys
import pickle

def normalise(v):
    return v/scipy.sqrt(scipy.sum(v*v))

class Fit():
    
    
    ####################################################################
    
    class ElementPoints():
        """ Holds a dictinary of element points
        """
        
        class ElementPoint():
            """ An element point has element number and xi location on the
                element
            """
            def __init__(self, id, element, xi):
                self.id = id
                self.element = element
                self.xi = xi
                self.dof_index = None
                self.dof_weights = None
            
            def update_from_mesh(self, mesh):
                self.dof_index = mesh.ENodes\
                    [mesh.EMap[self.element,1]:mesh.EMap[self.element,2]]
                self.dof_weights = mesh.BasisFunctions\
                    [mesh.EBFn[self.element]](scipy.array([self.xi]))[0,:]
            
            def matrix_rows(self):
                if self.dof_index==None or self.dof_weights==None:
                    raise NameError('DoF of point %(self.id)d has not been set.')
                
                cols = scipy.array([i*3 for i in self.dof_index])
                data = scipy.array([phi for phi in self.dof_weights])
                
                Rows = [[cols, data]]
                
                return Rows, [0,1,2], None
    
        
        def __init__(self):
            self.count = 0
            self.points = {}
        
        def add(self, el, xi):
            id = self.count
            self.points[self.count] = self.ElementPoint(self.count, el, xi)
            self.count += 1
            return id
            
        def __getitem__(self, key):
            return self.points[key]
        
        def __iter__(self):
            keys = self.points.keys()
            keys.sort()
            for key in keys:
                yield self.points[key]
        
        def __len__(self):
            return len(self.points.keys())
    
    
    
    ####################################################################
    ####################################################################
    ####################################################################
    class BindPoints():
        """ Holds a dictinary of element points
        """
        
        class BindPoint():
            """ An element point has element number and xi location on the
                element
            """
            def __init__(self, id, node, element, xi, weight=1, smooth=False):
                self.id = id
                self.node = node
                self.element = element
                self.xi = xi
                self.nodes = None
                self.Phi = None
                self.binding_weight = weight
                self.smooth = smooth
            
            def update_from_mesh(self, mesh):
                self.nodes = mesh.ENodes\
                    [mesh.EMap[self.element,1]:mesh.EMap[self.element,2]]
                self.Phi = mesh.BasisFunctions\
                    [mesh.EBFn[self.element]](scipy.array([self.xi]))[0,:]
            
            def matrix_rows(self):
                if self.node==None or self.Phi==None:
                    raise NameError('DoF of point %(self.id)d has not been set.')
                
                cols = scipy.array([i*3 for i in self.nodes])
                data = scipy.array([phi for phi in self.Phi])
                
                cols = scipy.append(cols, 3*self.node)
                data = self.binding_weight*scipy.append(data, -1)
                
                return [[cols, data]], [0,1,2], None
    
        
        def __init__(self):
            self.count = 0
            self.points = {}
        
        def add(self, node, element, xi, weight=1, smooth=False):
            id = self.count
            self.points[self.count] = self.BindPoint(self.count, node, element, xi, weight=weight, smooth=smooth)
            self.count += 1
            return id
            
        def __getitem__(self, key):
            return self.points[key]
        
        def __iter__(self):
            keys = self.points.keys()
            keys.sort()
            for key in keys:
                yield self.points[key]
        
        def __len__(self):
            return len(self.points.keys())
    
    
    
    ####################################################################
    ####################################################################
    ####################################################################
    class FixPoints():
        """
        Holds a dictinary of points to fix
        """
        
        class FixPoint():
            """
            A node point to fix
            """
            def __init__(self, id, point, values, weight=1):
                self.id = id
                self.point = point
                self.values = [v for v in values if v != None]
                self.dims = [i for i, v in enumerate(values) if v != None]
                self.NDims = len(self.dims)
                self.nodes = None
                self.Phi = None
                self.weight = weight
            
            def update_from_mesh(self, mesh):
                pass
            
            def matrix_rows(self):
                cols = scipy.array([self.point*3])
                data = scipy.array([self.weight])
                return [[cols, data]], self.dims, \
                    [v * self.weight for v in self.values]
    
        
        def __init__(self):
            self.count = 0
            self.points = {}
        
        def add(self, point, values, weight=1):
            id = self.count
            self.points[self.count] = self.FixPoint(self.count, point, values, weight=weight)
            self.count += 1
            return id
            
        def __getitem__(self, key):
            return self.points[key]
        
        def __iter__(self):
            keys = self.points.keys()
            keys.sort()
            for key in keys:
                yield self.points[key]
        
        def __len__(self):
            return len(self.points.keys())
    
    
    
    ####################################################################
    ####################################################################
    ####################################################################
    class SmoothPoints():
        """ Holds a dictinary of points to be smoothed
        """
        
        class SmoothPoint():
            """ A point to smooth
            """
            
            class SmoothVector():
                
                class Elem():
                    def __init__(self, el=None, xi=None, xi_weights=None):
                        self.num = el
                        self.xi = xi
                        self.xi_axis = None
                    
                    def compute_nodal_weights(self,mesh):
                        self.Nodes, self.Phi = \
                            mesh.getElementInwardVectorNodalWeights(
                            self.num, self.xi, self.xi_axis, normalise=True)
                            
                    
                def __init__(self, point=None, weight=1):
                    
                    self.weight = weight
                    self.EL =[self.Elem(), self.Elem()]
                    self.axis_aligned = True
                    
                    if point:
                        for i,p in enumerate(point):
                            self.EL[i].num = int(p[0])
                            self.EL[i].xi = p[1]
                            self.EL[i].xi_axis = p[2]                    
                    
                def update_from_mesh(self, mesh):
                    
                    if self.EL[0].xi_axis==None or self.EL[1].xi_axis==None:
                        nodes0 = mesh.ENodes[mesh.EMap[self.EL[0].num,1]:mesh.EMap[self.EL[0].num,2]]
                        nodes1 = mesh.ENodes[mesh.EMap[self.EL[1].num,1]:mesh.EMap[self.EL[1].num,2]]
                        
                        common_nodes = list(set(nodes0) & set(nodes1))
                        
                        for ne in [0,1]:
                            xi1 = mesh.getElementNodeXi(self.EL[ne].num, common_nodes[0])[0][0]
                            xi2 = mesh.getElementNodeXi(self.EL[ne].num, common_nodes[1])[0][0]
                            
                            if xi1[0]==xi2[0]:
                                self.EL[ne].xi_axis = 1
                            elif xi1[1]==xi2[1]:
                                self.EL[ne].xi_axis = 0
                            else:
                                self.EL[ne].xi_axis = -1
                    
                    for el in self.EL:
                        el.compute_nodal_weights(mesh)
                    
                    return 
                
                def matrix_rows(self):                    
                    Cols , Data = [], []
                    Cols.append(self.EL[0].Nodes*3)
                    Data.append(self.weight*self.EL[0].Phi)
                    Cols.append(self.EL[1].Nodes*3)
                    Data.append(self.weight*self.EL[1].Phi)
                    return Cols, Data
                    
                    
            def __init__(self, parent, id, point, N=5, xi_range=[0,1], weight=1E4):
                self.parent = parent
                self.id = id
                self.point = point
                self.line = None
                self.node = None
                self.bound_node = False
                self.N = N
                self.xi_range = xi_range
                self.vectors = []
                
                self.SWeight = weight
                self.smooth = True
                
                if isinstance(point, int):
                    self.node = point
                elif isinstance(point, str):
                    self.line = point
                elif isinstance(point, list):
                    self.vectors.append(self.SmoothVector(point, self.SWeight))
                else:
                    raise NameError('Unknown point description')
            
            
            def add_vector(self, point=None, weight=1E3):
                vector = self.SmoothVector(point=point, weight=weight)
                self.vectors.append(vector)
                return vector
            
            
            def update_from_mesh(self, mesh):
                if self.node:
                    
                    Elements = mesh.getNodeElements(self.node)
                    Elements.sort()
                    NElements = len(Elements)
                    
                    if self.bound_node:
                        vec = self.vectors[0]
                        vec.EL[1].num = Elements[0]
                        xi = mesh.getElementNodeXi(Elements[0], self.node)[0][0]
                        vec.EL[1].xi = xi.tolist()
                        
                    elif NElements==1:
                        self.smooth = False
                        
                    elif NElements==2:
                        vec = self.add_vector(weight=self.SWeight)
                        for ne, elem in enumerate(Elements):
                            vec.EL[ne].num = elem
                            vec.EL[ne].xi = mesh.getElementNodeXi(elem, self.node)[0][0]
                        
                    else:
                        Boundaries = []
                        for i in range(0,len(Elements)-1):
                            for j in range(i+1,len(Elements)):
                                el1 = Elements[i]
                                el2 = Elements[j]
                                nodes1 = mesh.ENodes[mesh.EMap[el1,1]:mesh.EMap[el1,2]]
                                nodes2 = mesh.ENodes[mesh.EMap[el2,1]:mesh.EMap[el2,2]]
                                common_nodes = list(set(nodes1) & set(nodes2))
                                if len(common_nodes)>1:
                                    Boundaries.append([el1,el2])
                        
                        for boundary in Boundaries:
                            vec = self.add_vector(weight=self.SWeight)
                            for ne, elem in enumerate(boundary):
                                vec.EL[ne].num = elem
                                vec.EL[ne].xi = mesh.getElementNodeXi(elem, self.node)[0][0]
                
                elif self.line:
                    line = mesh.lines[self.line]
                    xi_axis = []
                    constant_xi = []
                    for element in line.elements:
                        for node in line.nodes:
                            loc = mesh.getNodeElementLocation(node, element)
                            if loc in [2,4,6,8,10]:
                                break
                        if loc==2:
                            xi_axis.append(0)
                            constant_xi.append(0)
                        elif loc==4:
                            xi_axis.append(1)
                            constant_xi.append(0)
                        elif loc==6:
                            xi_axis.append(1)
                            constant_xi.append(1)
                        elif loc==8:
                            xi_axis.append(0)
                            constant_xi.append(1)
                        elif loc==10:
                            xi_axis.append(-1)
                            constant_xi.append(-1)
                    
                    El1_Xi, El2_Xi = mesh.getElementBoundaryXi( \
                        line.elements[0], line.elements[1],
                        self.xi_range[0], self.xi_range[1], self.N) 
                    
                    for i in range(El1_Xi.shape[0]):
                        vec = self.add_vector(weight=self.SWeight)
                        for ne in range(2):
                            vec.EL[ne].num = line.elements[ne]
                            self.xi_axis = xi_axis[ne]
                        vec.EL[0].xi = El1_Xi[i,:]
                        vec.EL[1].xi = El2_Xi[i,:]
                            
                        
                if self.smooth:
                    for vector in self.vectors:
                        vector.update_from_mesh(mesh)
                
                
            def matrix_rows(self):
                Rows = []
                for vector in self.vectors:
                    Cols, Data = vector.matrix_rows()
                    Rows.append([Cols, Data])
                return Rows
        
        
        def __init__(self):
            self.count = 0
            self.nodes = {}
        
        def add(self, parent, point, N=5, xi_range=[0,1], weight=1):
            id = self.count
            self.nodes[self.count] = self.SmoothPoint(parent, self.count, point, N=N, xi_range=xi_range, weight=weight)
            self.count += 1
            return id
            
        def __getitem__(self, key):
            return self.nodes[key]
            
        def __iter__(self):
            keys = self.nodes.keys()
            keys.sort()
            for key in keys:
                yield self.nodes[key]
        
        def __len__(self):
            return len(self.nodes.keys())
            
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    
    def __init__(self, filepath=None):
        self.description = 'New fit'
        self.A = None
        self.invA = None
        self.x = None
        self._NDoF = None
        self.NPoints = 0
        self.NBind = 0
        self.NFix = 0
        self.NSmooth = 0
        
        self.ElementXiIndices = {}
        self.Xi = []
        
        self.points = self.ElementPoints()
        self.bind_points = self.BindPoints()
        self.fix_points = self.FixPoints()
        self.smooth_points = self.SmoothPoints()
        
        self.List_of_Points = [self.points, self.bind_points, 
                               self.smooth_points, self.fix_points]
        
        self.Node_Weights = []
        self.NWMap = []
        self.Nodes = []
        self.Weights = []
        
        
        self.Xd = None
        self.data_filter = 'one-to-one'
        self.ndp = 1
        self.XdKDTree = None
        
        self.svd_UT = None
        self.svd_S = None
        self.svd_VT = None
        self.svd_invA = None
        
        if filepath!=None:
            self.load(filepath)
    
    
    def save(self, filepath):
        import tables
        fitNPBS = scipy.array([self.NPoints, self.NBind, self.NFix, self.NSmooth])
        atom1 = tables.Float64Atom()
        atom2 = tables.Int32Atom()
        filters = tables.Filters(complevel=5, complib='zlib')
        h5f = tables.openFile(filepath, 'w')
        h5NWMap = h5f.createCArray(h5f.root, 'NWMap', atom2, self.NWMap.shape, filters=filters)
        h5NWMap[:,:] = self.NWMap
        h5Nodes = h5f.createCArray(h5f.root, 'Nodes', atom2, self.Nodes.shape, filters=filters)
        h5Nodes[:] = self.Nodes
        h5Weights = h5f.createCArray(h5f.root, 'Weights', atom1, self.Weights.shape, filters=filters)
        h5Weights[:] = self.Weights
        invA = h5f.createCArray(h5f.root, 'invA', atom1, self.svd_invA.shape, filters=filters)
        invA[:,:] = self.svd_invA
        b = h5f.createCArray(h5f.root, 'b', atom1, self.b.shape, filters=filters)
        b[:] = self.b
        fixed = h5f.createCArray(h5f.root, 'fixed', atom1, self.fixed.shape, filters=filters)
        fixed[:] = self.fixed
        npbs = h5f.createCArray(h5f.root, 'npbs', atom2, fitNPBS.shape)
        npbs[:] = fitNPBS
        h5f.close()
            
        
    def load(self, filepath):
        import tables
        h5f = tables.openFile(filepath)
        self.NWMap = h5f.root.NWMap[:,:]
        self.Nodes = h5f.root.Nodes[:]
        self.Weights = h5f.root.Weights[:]
        self.svd_invA = h5f.root.invA[:,:]
        self.b = h5f.root.b[:]
        self.fixed = h5f.root.fixed[:]
        self.NPoints = h5f.root.npbs[0]
        self.NBind = h5f.root.npbs[1]
        self.NFix = h5f.root.npbs[2]
        self.NSmooth = h5f.root.npbs[3]
        h5f.close()
        self._NDoF = self.svd_invA.shape[0]
        
    
    def add_element_points(self, Elements, Xi):
        Ids = []
        if isinstance(Elements, list):
            for ne, elem in enumerate(Elements):
                for np, xi in enumerate(Xi):
                    Ids.append(self.points.add(Elements[ne], xi))
        else:
            for np, xi in enumerate(Xi):
                Ids.append(self.points.add(Elements, xi))
        return Ids
        
    def smooth_point(self, point, weight=1):
        return self.smooth_points.add(self, point, weight=weight)
    
    def smooth_line(self, point, N=5, xi_range=[0,1], weight=1):
        return self.smooth_points.add(self, point, N=N, xi_range=xi_range, weight=weight)
        
    def bind_point(self, node, element, xi, weight=1, smooth=False):
        id = self.bind_points.add(node, element, xi, weight=weight, smooth=smooth)
        if smooth:
            pid = self.smooth_point([[element, xi, None], [-1, None, None]])
            self.smooth_points[pid].node = node
            self.smooth_points[pid].bound_node = True
            
        return id
        
    def fix_point(self, point, values, weight=1):
        id = self.fix_points.add(point, values, weight=weight)
        return id
        
    def generate(self, mesh):
        for Points in self.List_of_Points:
            for point in Points:
                point.update_from_mesh(mesh)
        
        self.update_element_xi()
        
        self.set_NDoF(mesh)
        self.generate_matrix()
        self.assemble_node_weights()
        
    def assemble_node_weights(self):
        NNW = -1
        for point in self.points:
            if len(self.Node_Weights)==0:
                NNW += 1
                self.Node_Weights.append({'element':point.element, \
                    'nodes':point.dof_index, 'Phi':[]})
            else:
                if point.element!=self.Node_Weights[NNW]['element']:
                    self.Node_Weights[NNW]['Phi'] = scipy.array(self.Node_Weights[NNW]['Phi'])
                    NNW += 1
                    self.Node_Weights.append({'element':point.element, \
                        'nodes':point.dof_index, 'Phi':[]})
            self.Node_Weights[NNW]['Phi'].append(point.dof_weights)
        self.Node_Weights[NNW]['Phi'] = scipy.array(self.Node_Weights[NNW]['Phi'])
        
        self.NWMap = scipy.zeros((len(self.Node_Weights), 5), dtype=int)
        Nodes = []
        Weights = []
        NN, NW = 0, 0
        for i, EL in enumerate(self.Node_Weights):
            self.NWMap[i,0] = EL['element']
            self.NWMap[i,1] = NN
            self.NWMap[i,3] = NW
            Nodes.extend(EL['nodes'])
            Weights.extend(EL['Phi'].reshape((EL['Phi'].size)).tolist())
            NN = len(Nodes)
            NW = len(Weights)
            self.NWMap[i,2] = NN
            self.NWMap[i,4] = NW
        
        self.Nodes = scipy.array([n for n in Nodes])
        self.Weights = scipy.array([w for w in Weights])            
        
    def invert_matrix(self):
        from sparsesvd import sparsesvd
        self.svd_UT, self.svd_S, self.svd_VT = sparsesvd(self.A, self.A.shape[1])
        self.svd_invA = scipy.dot(\
            scipy.dot(self.svd_VT.T,scipy.linalg.inv(scipy.diag(self.svd_S))),self.svd_UT)

        
    def update_element_xi(self):
        self.Xi = []
        self.ElementXiIndices = {}
        for np, point in enumerate(self.points):
            self.Xi.append(point.xi)
            el = point.element
            if el not in self.ElementXiIndices.keys():
                self.ElementXiIndices[el] = []
            self.ElementXiIndices[el].append(np)
        
    def set_NDoF(self, mesh):
        self._NDoF = mesh.DoF.shape[0]
        
    def NDoF(self):
        if self._NDoF==None:
            raise NameError('Number of columns (N) not set.')
        return self._NDoF
    
    def NRows(self):
        self.NPoints = len(self.points)
        self.NSmooth = 0
        for point in self.smooth_points:
            if point.smooth:
                self.NSmooth += len(point.vectors)
        
        self.NBind = 0
        for point in self.bind_points:
            self.NBind += 1
                
        self.NFix = 0
        for point in self.fix_points:
            self.NFix += point.NDims
                
        M = 3*(len(self.points)+self.NSmooth+self.NBind)+self.NFix
        
        return M
        
        
    def generate_matrix(self):
        
        A = sparse.lil_matrix((self.NRows(), self.NDoF()))
        self.b = scipy.zeros((self.NRows()))
        self.fixed = scipy.zeros((self.NRows()))
        NR = -1
        
        for Points in self.List_of_Points:
            for point in Points:
                Rows, dims, b = point.matrix_rows()
                for row in Rows:
                    Cols = row[0]
                    Data = row[1]
                    for dim_ind, dim in enumerate(dims):
                        NR += 1
                        for i, cols in enumerate(Cols):
                            A[NR, cols+dim] += Data[i]
                        if b != None:
                            self.b[NR] = b[dim_ind]
                            self.fixed[NR] = 1
        
        self.A = A.tocsc()
        
    def set_data(self, Xd, mode='one-to-one', ndp=1):
        self.Xd = Xd
        self.data_filter = mode
        self.ndp = ndp
        if self.data_filter=='closest':
            self.XdKDTree = cKDTree(Xd)
            
    def compute_surface_points(self, Xn):
        Xe = scipy.empty((self.NPoints,3))
        np = 0
        for ne, nwmap in enumerate(self.NWMap):
            nodes = self.Nodes[nwmap[1]:nwmap[2]]
            NN = nodes.shape[0]
            Phi = self.Weights[nwmap[3]:nwmap[4]]
            Phi = Phi.reshape((Phi.size/NN,NN))
            NPhi = Phi.shape[0]
            Xe[np:np+NPhi,:] = scipy.dot(Phi, Xn[nodes,:])
            np += NPhi
        return Xe
    
    def find_closest_data_indices(self, Xn):
        if self.data_filter=='closest':
            self.Xe = self.compute_surface_points(Xn)
            return self.XdKDTree.query(list(self.Xe), k=self.Nd)[1]
        else:
            return None
            
    def get_data_for_solve(self, X=None):
        if self.data_filter=='one-to-one':
            if X:
                Xdr = X.reshape((X.size))
            else:
                Xdr = self.Xd.reshape((self.Xd.size))
        elif self.data_filter=='closest':
            if X==None:
                raise NameError('Need mesh node values for find closest data points')
            self.DI = self.find_closest_data_indices(X)
            if self.Nd>1:
                Xdr = scipy.array([self.Xd[di,:].mean(axis=0) for di in self.DI])
            else:
                Xdr = self.Xd[self.DI,:]
            Xdr = Xdr.reshape((3*len(self.DI)))
            self.Xd_closest = Xdr
        b = scipy.append(Xdr, scipy.zeros((3*(self.NSmooth+self.NBind)+self.NFix)))
        return b + self.b
        
        
    def solve_iteration(self, Xdr):
        if self.svd_invA==None:
            self.lsqr_result = linalg.lsqr(self.A, Xdr)
            self.x = self.lsqr_result[0].reshape((self.NDoF()/3,3))
        else:
            svd_x = scipy.dot(self.svd_invA, Xdr)
            self.x = svd_x.reshape((self.NDoF()/3,3))
        
        
    def solve(self, X=None, maxiter=1, drms=1e-9, Nd=1, output=False):
        self.Nd = Nd
        Xdr = self.get_data_for_solve(X=X)
        rms0 = self.rms_error(X)
        
        for i in xrange(1,maxiter+1):
            self.solve_iteration(Xdr)
            rms = self.rms_error()
            if output:
                sys.stdout.write("Iteration: %5d, RMS Error: %f\r" % (i, rms))
                sys.stdout.flush()
            if rms0-rms<=drms:
                return rms
            if i>=maxiter:
                return rms
            rms0 = rms
            Xdr = self.get_data_for_solve(X=self.x)
            
    
    def rms_error(self, Xn=None):
        if Xn==None:
            Xn = self.x
        Xe = self.compute_surface_points(Xn)
        dx = Xe - self.Xd_closest.reshape((self.Xd_closest.shape[0]/3,3))
        return scipy.sqrt(scipy.sum(dx*dx)/Xe.shape[0])
    
    
