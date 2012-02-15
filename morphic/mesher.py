'''
This module contains the higher level functions for creating and
manipulating meshes.

Todo:

Fix values for fitting purposes.

MapNode - ability to map values from other nodes. Essentially, use their
parameters.

DepNode with calculation of derivatives to allow calculating derivatives
on lagrange elements to be used by a hanging node for a cubic-Hermite
element. Might also allow the flipping of derivatives in case the
hanging element has an opposite orientation. This is included here
because it already does calculations.

Field class to describe the field structure. Maybe we can access values
using node.x or node.x.0 for values and node.x.1 for other components.

Different interpolation types for each field. For example, one might
have x, y, z and Temperature field. Could describe x, y, z using a
bilinear interpolation but temperature as a bicubic-Hermite
interpolation.


'''

import numpy
import scipy

import core
import discretizer

class Node:
    '''
    Node is the super-class for StdNode, DepNode, PCANode, and MapNode.
    
    Node.values
        Returns only the first component the fields, i.e., the field
        values only. For example, if the field values are set as
        [[1, 2], [4, 6]], Node.values will return [1, 4]
    
    Node.all_values
        Returns all fields including components, e.g., [[1, 2], [4, 6]]
    
    Node.all_values_flat
        Returns all fields including components but in a flat structure,
        e.g., [1, 2, 4, 6]
        
    >>> mesh = Mesh()
    >>> node = Node(mesh, 1)
    >>> node.set_values([[1, 2], [4, 6]])
    >>> print node.values
    [ 1.  4.]
    >>> print node.all_values
    [[ 1.  2.]
     [ 4.  6.]]
    >>> print node.all_values_flat
    [ 1.  2.  4.  6.]
    '''
    def __init__(self, mesh, uid):
        self._type = 'standard'
        self.mesh = mesh 
        self.id = uid
        self.fixed = None
        self.cids = None
        self.num_values = 0
        self.num_fields = 0
        self.num_components = 0
        self._added = False
        self.mesh._regenerate = True
        self.mesh._reupdate = True
        
    def __getattr__(self, name):
        if name == 'values':
            return self.mesh._core.P[self.cids][0:self.num_values:
                                                self.num_components]
        elif name == 'all_values':
            return self.mesh._core.P[self.cids].reshape(
                    (self.num_fields, self.num_components))
        elif name == 'all_values_flat':
            return self.mesh._core.P[self.cids]
        elif name == 'field_cids':
            return self._get_param_indicies()
        else:
            raise AttributeError
    
    def set_value(self, field, component, value):
        cid = self.cids[field * self.num_components + component]
        self.mesh._core.update_params([cid], value)
    
    def set_values(self, values):
        '''
        Sets the values for the node by adding them to core.
        If the nodes values have already been set, it will update the
        values. Note, the number of fields
        and components cannot change when updating the values.
        
        ``values`` can be of the form:
            - [0.2, 0.5] or [[0.2], [0.5]] which sets two field values
            - [[0.2 1, 0], [0.5, .3, 1]] which sets two fields with two components each.
        '''
        # Sets the number of values, fields and components.
        values = numpy.array(values, dtype='float')
        self.num_values = values.size
        self.num_fields = values.shape[0]
        self.num_components = 1
        
        if len(values.shape) == 2:
            self.num_components = values.shape[1]
        
        # Updates the values in core if they exist otherwise adds them.
        params = values.reshape(self.num_values) 
        if self._added:
            self.mesh._core.update_params(self.cids, params)
        else:
            self.cids = self.mesh._core.add_params(params)
        
        self._added = True
        self.mesh._regenerate = True
        self.mesh._reupdate = True
    
    def fix(self, fix):
        '''
        This function will the values of the node, which is used for
        fitting. The values can be fixed in three forms:
          - node.fix(True) will fix all the values
          - node.fix([False, True, False]) will fix all the values in 
            field 2 and unfix all others.
          - node.fix([[True, False], [False, True]] will fix the first 
            component in field 1 and the second component in field 2.
        
        '''
        if isinstance(fix, bool):
            self.fixed = fix * scipy.ones(self.num_values, dtype=bool)
        elif isinstance(fix, list):
            fixed = []
            for f1 in fix:
                if isinstance(f1, bool):
                    fixed.extend([f1 for ncomp in range(self.num_components)])
                else:
                    fixed.extend([f2 for f2 in f1])
            self.fixed = fixed
        self.mesh._core.fix_parameters(self.cids, self.fixed)
        
    def _get_param_indicies(self):
        pi = []
        ind = 0
        for f in range(self.num_fields):
            pi.append(self.cids[ind:ind+self.num_components])
            ind += self.num_components
        return pi
        
    
class StdNode(Node):
    
    def __init__(self, mesh, uid, values):
        Node.__init__(self, mesh, uid)
        self._type = 'standard'
        self.set_values(values)
        
    
class DepNode(Node):

    def __init__(self, mesh, uid, element, node):
        Node.__init__(self, mesh, uid)
        self._type = 'dependent'
        self.element = element
        self.node = node
        
        
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
        
    def _get_param_indicies(self):
        PI = None
        for node in self:
            if PI == None:
                self.num_fields = node.num_fields
                PI = []
                for i in range(node.num_fields):
                    PI.append([])
            npi = node._get_param_indicies()
            for i, pi in enumerate(npi):
                PI[i].extend(pi)
        
        return PI
        
    def set_core_id(self, cid):
        self.cid = cid
        
    def grid(self, res=[8, 8]):
        return discretizer.xi_grid(
                shape=self.shape, res=res, units='div')[0]
                
    def weights(self, xi, deriv=None):
        return self.mesh._core.weights(self.cid, xi, deriv=deriv)
        
    def interpolate(self, xi, deriv=None):
        return self.mesh._core.interpolate(self.cid, xi, deriv=deriv)
    
    def normal(self, Xi):
        '''
        Calculates the surface normals at xi locations on an element.
        
        Xi is an ``mxn`` ndarray where ``m`` is the number of element
        locations and ``n`` is the number of element dimensions.
        
        For example,
        Xi = numpy.array([[0.1, 0.1], [0.3, 0.2], [0.7, 0.2]])
        '''
        dx1 = self.mesh._core.interpolate(self.cid, Xi, deriv=[1, 0])
        dx2 = self.mesh._core.interpolate(self.cid, Xi, deriv=[0, 1])
        return scipy.cross(dx1, dx2)
    
    def _project_objfn(self, xi, x, args):
        xe = self.interpolate(scipy.array([xi]))
        dx = xe - x
        return scipy.sum(dx * dx)
    
    def project(self, x, xi=None):
        from scipy.optimize import fmin
        if xi == None:
            xi = 0.5
        xi_opt = fmin(self._project_objfn, xi, (x), disp=False)
        return xi_opt
        
    def __iter__(self):
        return [self.mesh.nodes[i] for i in self.nodes].__iter__()
        
    
class Mesh():
    '''
    This is the top level object for a mesh which allows the:
    
    * generation of a mesh, i.e., add `Generating a Mesh`_
    * extraction details about the mesh for analysis or rendering
      (`Analysing a Mesh`_)
    * saving and loading meshes (`Saving and Loading`_).
    
    To create the simplist of elements, a linear 1D element, and
    interpolate values along the element:
    
    >>> mesh = Mesh()
    >>> n = mesh.add_stdnode(1, [0, 0])
    >>> n = mesh.add_stdnode(2, [1, 0.5])
    >>> e = mesh.add_element(1, ['L1'], [1, 2])
    >>> xi = [0, 0.25, 0.5, 0.75, 1.0]
    >>> mesh.interpolate(1, xi)
    array([[ 0.   ,  0.   ],
           [ 0.25 ,  0.125],
           [ 0.5  ,  0.25 ],
           [ 0.75 ,  0.375],
           [ 1.   ,  0.5  ]])
    
    These interpolated values can be used for analysis or for plotting.
    '''
    
    def __init__(self, filename=None, label='/', units='m'):
        self.label = label
        self.nodes = core.ObjectList()
        self.elements = core.ObjectList()
        
        self._core = core.Core()
        self._regenerate = True
        self._reupdate = True
        
        #~ if filename:
            #~ self.load(filename)
        
    def add_stdnode(self, uid, values, group='_default'):
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
        >>> node1 = mesh.add_stdnode(1, [0.2, 0.1, 3.], group='xyz')
        >>> node2 = mesh.add_stdnode(None, [0.1], group='xi')
        >>> node3 = mesh.add_stdnode(None, [0.2], group='xi')
        >>> print node1.id, node1.values
        1 [ 0.2  0.1  3. ]
        >>> print node2.id, node2.values
        0 [ 0.1]
        >>> print node3.id, node3.values
        2 [ 0.2]
        
        '''
        if uid==None:
            uid = self.nodes.get_unique_id()
        node = StdNode(self, uid, values)
        self.nodes.add(node, group=group)
        return node
    
    def add_depnode(self, uid, element, node_id, group='_default'):
        '''
        Adds a dependent node to a mesh. A dependent node is typically
        an interpolated location on an element. The location on the 
        element is defined by a node.
        
        >>> mesh = Mesh()
        >>> node1 = mesh.add_stdnode(1, [0, 0])
        >>> node2 = mesh.add_stdnode(2, [2, 1])
        >>> elem1 = mesh.add_element('elem1', ['L1'], [1, 2])
        >>> hang1 = mesh.add_stdnode('xi', [0.6]) # element location
        >>> node3 = mesh.add_depnode(3, 'elem1', 'xi') # hanging node
        >>> print mesh.get_nodes([3])
        [[ 1.2  0.6]]
        '''
        if uid==None:
            uid = self.nodes.get_unique_id()
        node = DepNode(self, uid, element, node_id)
        self.nodes.add(node, group=group)
        return node
    
    def fix_values(self, uids, fix):
        if not isinstance(uid, list):
            uids = [uids]
        for uid in uids:
            self.nodes[uid].fix(fix)
    
    
    def add_element(self, uid, interp, node_ids, group='_default'):
        '''
        Adds a element to a mesh.
        
        >>> mesh = Mesh()
        >>> n1 = mesh.add_stdnode(1, [0.1])
        >>> n2 = mesh.add_stdnode(2, [0.2])
        >>> elem = mesh.add_element(1, ['L1'], [1, 2])
        >>> print elem.id, elem.interp, elem.nodes
        1 ['L1'] [1, 2]
        '''
        if uid==None:
            uid = self.elements.get_unique_id()
        elem = Element(self, uid, interp, node_ids)
        self.elements.add(elem, group=group)
        return elem
    
    def save(self, filename):
        '''
        TODO
        '''
        pass
    
    def load(self, filename):
        '''
        TODO
        '''
        pass
    
    def generate(self, force=False):
        '''
        Generates a flat representation of the mesh for faster
        computation.
        '''
        if self._regenerate == True or force:
            self._update_dependent_nodes()
            self._core.generate_element_map(self)
            self._core.generate_dependent_node_map(self)
            self._regenerate = False
            self._reupdate = True
        
        if self._reupdate == True:
            self._core.update_dependent_nodes()
            self._reupdate = False
        
    def update(self, force=False):
        '''
        Updates the dependent node. This update may be required if some
        mesh nodes or parameters have been changed. This function can be
        called with `force=True` to force an update, e.g.,
        mesh.update(force=True)
        '''
        if self._reupdate == True or force:
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
    
    def get_variables(self):
        return self._core.get_variables()
    
    def set_variables(self, variables):
        self._core.set_variables(variables)
    
    def update_parameters(self, param_ids, values):
        self._core.update_params(param_ids, values)
    
    def interpolate(self, element_ids, xi, deriv=None):
        self.generate()
        if isinstance(xi, list):
            xi = scipy.array(xi)
            if len(xi.shape) == 1:
                xi = scipy.array([xi]).T
        else:
            if len(xi.shape) == 1:
                xi = scipy.array([xi]).T
                
        if not isinstance(element_ids, list):
            element_ids = [element_ids]
        
        node0 = self.nodes[self.elements[element_ids[0]].nodes[0]]
        num_fields = node0.num_fields
        num_elements = len(element_ids)
        num_xi = xi.shape[0]
        X = scipy.zeros((num_xi * num_elements, num_fields))
        
        ind = 0
        for element in self.elements[element_ids]:
            X[ind:ind+num_xi, :] = element.interpolate(xi, deriv=deriv)
            ind += num_xi
            
        return X
    
    def get_nodes(self, nodes=None, group='_default'):
        self.generate()
        if nodes:
            if not isinstance(nodes, list):
                nodes = [nodes]
            nodes = self.nodes[nodes]
        else:
            nodes = self.nodes(group)
        return scipy.array([n.values for n in nodes])
    
    def get_lines(self, res=8, group='_default'):
        self.generate()
        Xl = []
        xi = scipy.array([scipy.linspace(0, 1, res)]).T
        for i, elem in enumerate(self.elements(group)):
            Xl.append(self._core.interpolate(elem.cid, xi))
        return Xl
        
    def get_surfaces(self, res=8):
        self.generate()
        
        Elements = self.elements
        
        XiT, TT = discretizer.xi_grid(shape='tri', res=res)
        XiQ, TQ = discretizer.xi_grid(shape='quad', res=res)
        NPT, NTT = XiT.shape[0], TT.shape[0]
        NPQ, NTQ = XiQ.shape[0], TQ.shape[0]
        
        NP, NT = 0, 0
        for elem in Elements:
            if elem.shape == 'tri':
                NP += NPT
                NT += NTT
            elif elem.shape == 'quad':
                NP += NPQ
                NT += NTQ
                
        X = scipy.zeros((NP, 3))
        T = scipy.zeros((NT, 3), dtype='uint32')
        np, nt = 0, 0
        for elem in Elements:
            if elem.shape == 'tri':
                X[np:np+NPT,:] = self._core.interpolate(elem.cid, XiT)
                T[nt:nt+NTT,:] = TT + np
                np += NPT
                nt += NTT
            elif elem.shape == 'quad':
                X[np:np+NPQ,:] = self._core.interpolate(elem.cid, XiQ)
                T[nt:nt+NTQ,:] = TQ + np
                np += NPQ
                nt += NTQ
                
        return X, T
