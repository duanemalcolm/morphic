'''
This module contains the higher level functions for creating and
manipulating meshes.
'''

import numpy
import scipy

import core
reload(core)
        
class Node:

    def __init__(self, mesh, uid, values):
        self._type = 'normal'
        self.mesh = mesh 
        self.id = uid
        self._added = False
        self.set_values(values)
    
    def __getattr__(self, name):
        if name == 'values':
            return self.mesh._core.P[self.pind]
        else:
            raise AttributeError
    
    def set_values(self, values):
        values = numpy.array(values, dtype='float')
        self.num_values = values.size
        self.num_fields = values.shape[0]
        self.num_components = 1
        
        if len(values.shape) == 2:
            self.num_components = values.shape[1]
        
        params = values.reshape(self.num_values)    
        if self._added:
            self.mesh._core.update_params(self.pind, params)
        else:
            self.pind = self.mesh._core.add_params(params)
        
        self._added = True
        self.mesh._regenerate = True
        self.mesh._reupdate = True
        
    def get_param_indicies(self):
        pi = []
        ind = 0
        for f in range(self.num_fields):
            pi.append(self.pind[ind:ind+self.num_components])
            ind += self.num_components
        return pi
        
    
class DepNode:

    def __init__(self, mesh, uid, element, node):
        self._type = 'dependent'
        self.mesh = mesh 
        self.id = uid
        self._added = False
        
        self.element = element
        self.node = node
        
        self.num_values = 0
        self.num_fields = 0
        self.num_components = 0
        
        self.mesh._regenerate = True
        self.mesh._reupdate = True
    
    def __getattr__(self, name):
        if name == 'values':
            return self.mesh._core.P[self.pind]
        else:
            raise AttributeError
    
    def set_values(self, values):
        values = numpy.array(values, dtype='float')
        self.num_values = values.size
        self.num_fields = values.shape[0]
        self.num_components = 1
        
        if len(values.shape) == 2:
            self.num_components = values.shape[1]
        
        params = values.reshape(self.num_values)    
        if self._added:
            self.mesh._core.update_params(self.pind, params)
        else:
            self.pind = self.mesh._core.add_params(params)
        
        self._added = True
        self.mesh._regenerate = True
        self.mesh._reupdate = True
        
    def get_param_indicies(self):
        return [[pi] for pi in self.pind]
    
    
class Element:

    def __init__(self, mesh, uid, interp, nodes):
        self.mesh = mesh 
        self.interp = interp 
        self.id = uid
        self.nodes = nodes
        
        self.shape = 'quad'
        if self.interp[0][0] == 'T':
            self.shape = 'tri'
        
        self.cid = None
        
        self.mesh._regenerate = True
        self.mesh._reupdate = True
        
    
    def get_param_indicies(self):
        PI = None
        for node in self:
            if PI == None:
                self.num_fields = node.num_fields
                PI = []
                for i in range(node.num_fields):
                    PI.append([])
            npi = node.get_param_indicies()
            for i, pi in enumerate(npi):
                PI[i].extend(pi)
        
        return PI
        
    def set_core_id(self, cid):
        self.cid = cid
        
    def __iter__(self):
        return [self.mesh.nodes[i] for i in self.nodes].__iter__()
        

class MeshObjectList:
    
    def __init__(self):
        self._objects = []
        self._object_ids = {}
        self._id_counter = 0
        self.groups = {}
        
    def size(self):
        return len(self._objects)
    
    def add(self, obj, group=None):
        self._objects.append(obj)
        self._object_ids[obj.id] = obj
        if group:
            self.add_to_group(obj.id, group)
        
    def set_counter(self, value):
        self._id_counter = value
        
    def get_unique_id(self):
        existing_ids = self._object_ids.keys()
        while True:
            if self._id_counter in existing_ids:
                self._id_counter += 1
            else:
                return self._id_counter
    
    def add_to_group(self, uid, group):
        if group not in self.groups.keys():
            self.groups[group] = []
        obj = self._object_ids[uid]
        if obj not in self.groups[group]:
            self.groups[group].append(obj)
    
    def get_group(self, group):
        if group in self.groups.keys():
            return self.groups[group]
        else:
            return []
    
    def __getitem__(self, keys):
        if isinstance(keys, list):
            return [self._object_ids[key] for key in keys]
        else:
            return self._object_ids[keys]
    
    def __iter__(self):
        return self._objects.__iter__()
    
    
class Mesh():
    
    def __init__(self, filename=None, label='/', units='m'):
        
        self.label = label
        self.nodes = MeshObjectList()
        self.elements = MeshObjectList()
        
        self._core = core.Core()
        self._regenerate = True
        self._reupdate = True
        
        #~ if filename:
            #~ self.load(filename)
        
    def add_node(self, uid, *args, **kwargs):
        '''
        Adds a node to a mesh.
        
        If None is given for the uid, the mesh will automatically
        increment a counter (starting at 1) and assign a unique id. If
        you would like to start the counter at 1, create your mesh as
        follows:
        >>> mesh = Mesh()
        >>> mesh.nodes.set_counter(1)
        >>> id = mesh.nodes.get_unique_id()
        >>> print id
        1
        
        There are two forms for entering node values for the variable x:
        - Simple form for when there is one type of value for each
        dimension e.g., [x, y, z] or [xi1, xi2] values
        - Component form for when each dimension has different types of
        values, e.g., x = [[x, dx1, dx2, dx12], [y, dy1, dy2, dy12]]
        
        >>> mesh = Mesh()
        >>> node1 = mesh.add_node(1, [0.2, 0.1, 3.])
        >>> node2 = mesh.add_node(None, [0.1])
        >>> node3 = mesh.add_node(None, [0.2])
        >>> print node1.id, node1.values
        1 [ 0.2  0.1  3. ]
        >>> print node2.id, node2.values
        0 [ 0.1]
        >>> print node3.id, node3.values
        2 [ 0.2]
        '''
        if uid==None:
            uid = self.nodes.get_unique_id()
        
        group = kwargs.get('group', '_default')
        nodetype = kwargs.get('nodetype', None)
        
        num_args = len(args)
        
        if nodetype == None:
            if num_args == 1 and isinstance(args[0], list):
                nodetype = 'normal'
            elif num_args == 2:
                nodetype = 'dependent'
                
        if nodetype == 'normal':
            x = args[0]
            node = Node(self, uid, x)
        elif nodetype == 'dependent':
            element = args[0]
            node_id = args[1]
            node = DepNode(self, uid, element, node_id)
        self.nodes.add(node, group=group)
        return node
    
    def add_element(self, uid, interp, node_ids):
        '''
        Adds a element to a mesh.
        
        >>> mesh = Mesh()
        >>> n1 = mesh.add_node(1, [0.1])
        >>> n2 = mesh.add_node(2, [0.2])
        >>> elem = mesh.add_element(1, ['L1'], [1, 2])
        >>> print elem.id, elem.interp, elem.nodes
        1 ['L1'] [1, 2]
        '''
        if uid==None:
            uid = self.elements.get_unique_id()
        elem = Element(self, uid, interp, node_ids)
        self.elements.add(elem)
        return elem
    
    def save(self, filename):
        pass
    
    def load(self, filename):
        pass
    
    def generate(self):
        '''
        Generates a flat representation of the mesh for faster
        computation.
        '''
        if self._regenerate == True:
            self._update_dependent_nodes()
            self._core.generate_parameter_list(self)
            self._core.generate_element_map(self)
            self._core.generate_dependent_node_map(self)
            self._regenerate = False
            self._reupdate = True
        
        if self._reupdate == True:
            self._core.update_dependent_nodes()
            self._reupdate = False
        
    def update(self):
        '''
        Updates the dependent node. This update may be required if some
        mesh parameters have been changed.
        '''
        if self._reupdate == True:
            self._core.update_dependent_nodes()
            self._reupdate = False
    
    def _update_dependent_nodes(self):
        for node in self.nodes:
            if node._type == 'dependent' and node._added == False:
                elem = self.elements[node.element]
                for enode in elem:
                    if enode.num_values > 0:
                        node.set_values(scipy.zeros(enode.num_values))
                        break
        
    
    def get_num_parameters(self):
        num_values = 0
        for node in self.nodes:
            num_values += node.num_values
        return num_values
    
    def get_nodes(self, group='_default'):
        self.generate()
        nodes = self.nodes.get_group(group)
        return scipy.array([n.values for n in nodes])
    
    def get_lines(self, res=8):
        self.generate()
        Xl = []
        xi = scipy.array([scipy.linspace(0, 1, res)]).T
        for i, elem in enumerate(self.elements):
            Xl.append(self._core.interpolate(elem.cid, xi))
        return Xl
        
    def get_surfaces(self, res=8):
        self.generate()
        
        Elements = self.elements
        
        nx = res+1
        NPT = int(0.5*nx*(nx-1)+nx)
        NPQ = int(nx*nx)
        NTT = int(res*res)
        NTQ = int(2*(res*res))
        
        NP = 0
        NT = 0
        for elem in Elements:
            if elem.shape == 'tri':
                NP += NPT
                NT += NTT
            elif elem.shape == 'quad':
                NP += NPQ
                NT += NTQ
        
        X = scipy.zeros((NP, 3))
        T = scipy.zeros((NT, 3), dtype='uint32')
        
        xi = scipy.linspace(0,1,res+1)
        
        XiT = scipy.zeros([NPT,2])
        TT = scipy.zeros((NTT, 3), dtype='uint32')
        NodesPerLine = range(res, 0, -1)
        np = 0
        for row in range(nx):
            for col in range(nx-row):
                XiT[np,0] = xi[col]
                XiT[np,1] = xi[row]
                np += 1
        
        np = 0
        ns = 0
        for row in range(res):
            for col in range(res-row):
                TT[np,:] = [ns,ns+1,ns+nx-row]
                np += 1
                if col!=res-row-1:
                    TT[np,:] = [ns+1,ns+nx-row+1, ns+nx-row]
                    np += 1
                ns += 1
            ns += 1
        
        xi1,xi2 = scipy.meshgrid(xi, xi)
        xi1 = xi1.reshape([xi1.size])
        xi2 = xi2.reshape([xi2.size])
        XiQ = scipy.array([xi1, xi2]).T
        TQ = scipy.zeros((NTQ, 3), dtype='uint32')
        np = 0
        for row in range(res):
            for col in range(res):
                NPPR = row*nx
                TQ[np,:] = [NPPR+col,NPPR+col+1,NPPR+col+nx]
                np += 1
                TQ[np,:] = [NPPR+col+1,NPPR+col+nx+1,NPPR+col+nx]
                np += 1
        
        np = 0
        nt = 0
        for elem in Elements:
            if elem.shape == 'tri':
                X[np:np+NPT,:] = self._core.interpolate(elem.cid, XiT)
                T[nt:nt+NTT,:] = TT+np
                np += NPT
                nt += NTT
            elif elem.shape == 'quad':
                X[np:np+NPQ,:] = self._core.interpolate(elem.cid, XiQ)
                T[nt:nt+NTQ,:] = TQ+np
                np += NPQ
                nt += NTQ
                
        return X, T
