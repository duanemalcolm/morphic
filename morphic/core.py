'''
This module manages the low level parameters describing the mesh.
'''
import interpolator
import string
import random
import numpy

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
    
    def get_variables(self):
        return self.P[self.idx_unfixed]
    
    def set_variables(self, variables):
        self.P[self.idx_unfixed] = variables
    
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
