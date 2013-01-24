'''
This module manages the low level parameters describing the mesh.
'''
import interpolator
import string
import random
import numpy

def dimensions(basis):
    dimensions = 0
    for base in basis:
        if base[0] == 'T':
            dimensions += 2
        else:
            dimensions += 1
    return dimensions

def element_face_nodes(basis, node_ids):
    dims = dimensions(basis)
    for base in basis:
        if base[0] == 'T':
            raise ValueError('Triangular basis are not supported')
    if dims is 1:
        return None
    elif dims is 2:
        return basis, node_ids
    elif dims is 3:
        shape = []
        for base in basis:
            if base[0] == 'L':
                shape.append(int(base[1:]) + 1)
            elif base[0] == 'H':
                if base[1:] is not '3':
                    ValueError('Only 3rd-order hermites are supported')
                shape.append(2)
            else:
                raise ValueError('Basis is not supported')
    else:
        raise ValueError('Dimensions >3 is not supported')
    
    face_basis = []
    face_basis.append([basis[0], basis[1]])
    face_basis.append([basis[0], basis[1]])
    face_basis.append([basis[0], basis[2]])
    face_basis.append([basis[0], basis[2]])
    face_basis.append([basis[1], basis[2]])
    face_basis.append([basis[1], basis[2]])
    
    shape.reverse()
    nids = numpy.array(node_ids).reshape(shape)
    
    faces = []
    faces.append(nids[0, :, :].flatten().tolist())
    faces.append(nids[shape[0] - 1, :, :].flatten().tolist())
    faces.append(nids[:, 0, :].flatten().tolist())
    faces.append(nids[:, shape[1] - 1, :].flatten().tolist())
    faces.append(nids[:, :, 0].flatten().tolist())
    faces.append(nids[:, :, shape[2] - 1].flatten().tolist())
    
    return faces
    
class ObjectList:
    '''
    This object is used by a few morphic modules to store collections
    of objects. For example, nodes, elements, fixed points.
    '''
    
    def __init__(self):
        self._objects = []
        self._object_ids = {}
        self._id_counter = 0
        self.groups = {}
        
    def size(self):
        '''
        Returns the number of objects in the list.
        '''
        return len(self._objects)
        
    def keys(self):
        return self._object_ids.keys()
    
    def add(self, obj, group=None):
        '''
        Adds an object to the collection. The object requires an id
        attribute (e.g., obj.id) to be able to be added and referenced.
        If a group is specified then the object will be added to the
        group.
        '''
        self._objects.append(obj)
        if hasattr(obj, 'id'):
            oid = obj.id
        else:
            oid = self.get_unique_id()
        self._object_ids[oid] = obj
        if group:
            self.add_to_group(oid, group)
        return oid
        
    def set_counter(self, value):
        '''
        Sets the counter value for finding unique ids.
        
        Typically, this function would be used if you wanted to start
        the counter at 1 instead of 0. In which case, the numbering of
        objects, like nodes or elements, will start at 1.
        
        This function can be used to reset the counter to zero, for
        example, so that unused object ids can be reused. Unused object
        ids might occur when objects are deleted from the list.
        
        :param value: value to reset the counter to
        :type X: int
        :return: None
        '''
        self._id_counter = value
        
    def get_unique_id(self, random_chars=0):
        existing_ids = self._object_ids.keys()
        if random_chars > 0:
            random_id = existing_ids[0]
            while random_id in existing_ids:
                random_id = ''.join(random.choice(
                        string.ascii_letters + string.digits)
                        for x in range(random_chars))
            return random_id
        else:
            while True:
                if self._id_counter in existing_ids:
                    self._id_counter += 1
                else:
                    return self._id_counter
    
    def add_to_group(self, uids, group):
        if not isinstance(uids, list):
            uids = [uids]
        for uid in uids:
            if group not in self.groups.keys():
                self.groups[group] = []
            obj = self._object_ids[uid]
            if obj not in self.groups[group]:
                self.groups[group].append(obj)
    
    def reset_object_list(self):
        self._objects = []
        self._object_ids = {}
        self._id_counter = 0
        self.groups = {}
    
    def _get_group(self, group):
        if group in self.groups.keys():
            return self.groups[group]
        else:
            return []
    
    def get_groups(self, groups):
        if not isinstance(groups, list):
            groups = [groups]
        objs = []
        for group in groups:
            objs.extend(self._get_group(group))
        return [obj for obj in set(objs)]
    
    def _save_dict(self):
        objlist_dict = {}
        objlist_dict['groups'] = {}
        for key in self.groups.keys():
            ids = []
            for obj in self.groups[key]:
                ids.append(obj.id)
            objlist_dict['groups'][key] = ids
        return objlist_dict
    
    def _load_dict(self, objlist_dict):
        self.groups = {}
        for group in objlist_dict['groups'].keys():
            self.add_to_group(objlist_dict['groups'][group], group)
    
    def __contains__(self, item):
        return item in self._object_ids.keys()
            
            
    def __getitem__(self, keys):
        if isinstance(keys, list):
            return [self._object_ids[key] for key in keys]
        else:
            return self._object_ids[keys]
    
    def __call__(self, group=None):
        return self._get_group(group)
        
    def __iter__(self):
        return self._objects.__iter__()
        
        
class Core():
    
    def __init__(self):
        self.P = numpy.array([])
        self.EFn = []
        self.EMap = []
        self.DNMap = []
        self.PCAMap = []
        self.fixed = numpy.array([])
        self.idx_unfixed = []
        self.variable_ids = []
        
        self.gauss_points = {}
        self.gauss_points[2] = [
                numpy.array([[0.21132486540518708], [0.78867513459481287]]),
                numpy.array([0.5, 0.5])]
        self.gauss_points[3] = [
                numpy.array([[0.1127016653792583], [0.5], [0.8872983346207417]]),
                numpy.array([5./18., 4./9., 5./18])]
        self.gauss_points[4] = [numpy.array([[0.33000947820757187, 0.6699905217924281, 0.06943184420297371, 0.9305681557970262]]).T,
                numpy.array([0.32607257743127305, 0.32607257743127305, 0.1739274225687269, 0.1739274225687269])]
        self.gauss_points[5] = [numpy.array([[0.5, 0.230765344947, 0.769234655053, 0.0469100770307, 0.953089922969]]).T,
                numpy.array([0.284444444444, 0.23931433525, 0.23931433525, 0.118463442528, 0.118463442528])]
        self.gauss_points[6] = [numpy.array([[0.8306046932331322, 0.1693953067668678, 0.3806904069584016, 0.6193095930415985, 0.0337652428984240, 0.9662347571015760]]).T,
                numpy.array([0.1803807865240693, 0.1803807865240693, 0.2339569672863455, 0.2339569672863455, 0.0856622461895852, 0.0856622461895852])]
    
        
    def add_params(self, params):
        i0 = self.P.size
        self.P = numpy.append(self.P, params)
        self.fixed = numpy.append(self.fixed, [False for p in params])
        return range(i0, self.P.size)
    
    def update_params(self, cids, params):
        self.P[cids] = params
        return True
        
    def fix_parameters(self, cids, fixed):
        self.fixed[cids] = fixed
    
    def generate_fixed_index(self):
        self.idx_unfixed = numpy.array([i 
                for i, f in  enumerate(self.fixed) if f == False])
    
    def add_variables(self, cids):
        if isinstance(cids, int):
            cids = [cids]
        for cid in cids:
            if cid not in self.variable_ids:
                self.variable_ids.append(cid)
    
    def remove_variables(self, cids):
        if not isinstance(cids, list):
            cids = [cids]
        for cid in cids:
            if cid in self.variable_ids:
                self.variable_ids.remove(cid)
    
    def get_variables(self):
        #~ return self.P[self.idx_unfixed]
        return self.P[self.variable_ids]
    
    def set_variables(self, variables):
        #~ self.P[self.idx_unfixed] = variables
        self.P[self.variable_ids] = variables
    
    def get_gauss_points(self, ng):
        if isinstance(ng, int):
            return self.gauss_points.get(ng)
        elif isinstance(ng, list):
            if len(ng) > 3:
                raise Exception('Gauss points for 4 dimensions' +
                        'and above not supported')
            if len(ng) == 2:
                Xi1, W1 = self.get_gauss_points(ng[0])
                Xi2, W2 = self.get_gauss_points(ng[1])
                Xi1g, Xi2g = numpy.meshgrid(Xi1.flatten(), Xi2.flatten())
                Xi1 = numpy.array([Xi1g.flatten(), Xi2g.flatten()]).T
                W1g, W2g = numpy.meshgrid(W1.flatten(), W2.flatten())
                W1 = W1g.flatten() * W2g.flatten()
                return Xi1, W1
            elif len(ng) == 3:
                Xi1, W1 = self.get_gauss_points(ng[0])
                Xi2, W2 = self.get_gauss_points(ng[1])
                Xi3, W3 = self.get_gauss_points(ng[2])
                gindex = numpy.mgrid[0:ng[0], 0:ng[1], 0:ng[2]]
                gindex = numpy.array([
                    gindex[2, :, :].flatten(),
                    gindex[1, :, :].flatten(),
                    gindex[0, :, :].flatten()]).T
                Xi = numpy.array([
                    Xi1[gindex[:,0]], Xi2[gindex[:,1]], Xi3[gindex[:,2]]])[:,:,0].T
                W = numpy.array([
                    W1[gindex[:,0]], W2[gindex[:,1]], W3[gindex[:,2,]]]).T.prod(1)
                return Xi, W
            
        raise Exception('Invalid number of gauss points')
        return None, None    
            
    
    def generate_element_map(self, mesh):
        self.EFn = []
        self.EMap = []
        cid = 0
        for elem in mesh.elements:
            self.EFn.append(elem.interp)
            self.EMap.append(elem._get_param_indicies())
            elem.set_core_id(cid)
            cid += 1
    
    def generate_dependent_node_map(self, mesh):
        self.DNMap = []
        for node in mesh.nodes:
            if node._type == 'dependent':
                elem = node.mesh.elements[node.element]
                pnode = node.mesh.nodes[node.node]
                self.DNMap.append([elem.cid, pnode.cids, node.cids])
    
    def update_dependent_nodes(self):
        # update dependent nodes
        for dn in self.DNMap:
            cid = dn[0]
            xi_cids = dn[1]
            dn_cids = dn[2]
            num_fields = len(self.EMap[cid])
            if num_fields == 1:
                xi = numpy.array([self.P[xi_cids]]).T
            else:
                xi = numpy.array([self.P[xi_cids]])
            Phi = interpolator.weights(self.EFn[cid], xi)
            for i in range(num_fields):
                self.P[dn_cids[i]] = numpy.dot(Phi, self.P[self.EMap[cid][i]])
    
    def add_pca_node(self, pca_node):
        self.PCAMap.append([
            pca_node.cids,
            pca_node.node.shape, pca_node.node.cids,
            pca_node.weights.cids, pca_node.variance.cids])
        return len(self.PCAMap) - 1
    
    def update_pca_nodes(self):
        for pcamap in self.PCAMap:
            self.P[pcamap[0]] = numpy.dot(
                self.P[pcamap[2]].reshape(pcamap[1]),
                self.P[pcamap[3]] * self.P[pcamap[4]])
    
    def weights(self, cid, xi, deriv=None):
        return interpolator.weights(self.EFn[cid], xi, deriv=deriv)
    
    def evaluate(self, cid, xi, deriv=None):
        num_fields = len(self.EMap[cid])
        #~ print self.EMap[cid].shape
        X = numpy.zeros((xi.shape[0], num_fields))
        Phi = interpolator.weights(self.EFn[cid], xi, deriv=deriv)
        for i in range(num_fields):
            X[:, i] = numpy.dot(Phi, self.P[self.EMap[cid][i]])
        return X
       
    def evaluates(self, cids, xi, deriv=None, X=None):
        num_fields = len(self.EMap[cids[0]])
        if X==None:
            X = numpy.zeros((len(cids) * xi.shape[0], num_fields))
        Phi = interpolator.weights(self.EFn[cids[0]], xi, deriv=deriv)
        Nxi = xi.shape[0]
        ind = 0
        for cid in cids:
            for i in range(num_fields):
                X[ind:ind+Nxi, i] = numpy.dot(Phi, self.P[self.EMap[cid][i]])
            ind += Nxi
        return X
    
    def evaluates_weights(self, cids, Phi, X=None):
        num_fields = len(self.EMap[cids[0]])
        if X==None:
            X = numpy.zeros((len(cids) * Phi.shape[0], num_fields))
        Nxi = Phi.shape[0]
        ind = 0
        for cid in cids:
            for i in range(num_fields):
                X[ind:ind+Nxi, i] = numpy.dot(Phi, self.P[self.EMap[cid][i]])
            ind += Nxi
        return X
    
    def evaluate_fields(self, cid, xi, fields):
        num_fields = len(fields)
        X = numpy.zeros((xi.shape[0], num_fields))
        for i, field in enumerate(fields):
            Phi = interpolator.weights(self.EFn[cid], xi, deriv=field[1:])
            X[:, i] = numpy.dot(Phi, self.P[self.EMap[cid][field[0]]])
        return X
     
