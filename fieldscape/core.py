'''
This module manages the low level parameters describing the mesh.
'''
import interpolants
import scipy

class Core():
    
    def __init__(self):
        self.P = scipy.array([])
        self.EFn = []
        self.EMap = []
        self.DNMap = []
    
    def add_params(self, params):
        i0 = self.P.size
        self.P = scipy.append(self.P, params)
        return range(i0, self.P.size)
    
    def update_params(self, pind, params):
        self.P[pind] = params
        return True
        
    def generate_parameter_list(self, mesh):
        self.P = scipy.array(self.P)
    
    def generate_element_map(self, mesh):
        self.EFn = []
        self.EMap = []
        cid = 0
        for elem in mesh.elements:
            self.EFn.append(elem.interp)
            self.EMap.append(elem.get_param_indicies())
            elem.set_core_id(cid)
            cid += 1
        print self.EFn
        print self.EMap
    
    def generate_dependent_node_map(self, mesh):
        self.DNMap = []
        for node in mesh.nodes:
            if node._type == 'dependent':
                elem = node.mesh.elements[node.element]
                pnode = node.mesh.nodes[node.node]
                self.DNMap.append([elem.cid, pnode.pind, node.pind])
    
    def update_dependent_nodes(self):
        # update dependent nodes
        for dn in self.DNMap:
            cid = dn[0]
            xi_pind = dn[1]
            dn_pind = dn[2]
            num_fields = len(self.EMap[cid])
            if num_fields == 1:
                xi = scipy.array([self.P[xi_pind]]).T
            else:
                xi = scipy.array([self.P[xi_pind]])
            Phi = interpolants.weights(self.EFn[cid], xi)
            for i in range(num_fields):
                self.P[dn_pind[i]] = scipy.dot(Phi, self.P[self.EMap[cid][i]])
    
    def interpolate(self, cid, xi):
        num_fields = len(self.EMap[cid])
        X = scipy.zeros((xi.shape[0], num_fields))
        Phi = interpolants.weights(self.EFn[cid], xi)
        for i in range(num_fields):
            X[:, i] = scipy.dot(Phi, self.P[self.EMap[cid][i]])
        return X
