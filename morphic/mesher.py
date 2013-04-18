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
import os
import sys
import numpy

from scipy import linalg

import core
import discretizer
import utils

class Values(numpy.ndarray):
    '''
    This is a temporary object passed to the user when setting node
    values.
    '''
    def __new__(cls, input_array, node, cids):
        obj = numpy.asarray(input_array).view(cls)
        obj.node = node
        obj.cids = cids
        return obj
    
    def __setslice__(self, i, j, sequence):
        # Required for 1D arrays
        if j > self.size:
            j = None
        self.__setitem__(slice(i, j, None), sequence)
        
    def __setitem__(self, name, values):
        flat_cids = numpy.array(self.cids).reshape(self.shape)[name].flatten()
        self.node._set_values(flat_cids, values)


class NodeValues(object):
    
    def __get__(self, instance, owner):
        x = instance.mesh._core.P[instance.cids].reshape(instance.shape)
        return Values(x, instance, instance.cids)
    
    def __set__(self, instance, values):
        if values.shape != instance.shape:
            raise IndexError('Cannot set values with a different shaped'
                    + ' array. User node.set_values(values) instead') 
        instance.mesh._core.P[instance.cids] = values.flatten()
        


class Node(object):
    '''
    This is the super-class for StdNode and DepNode.
    '''
    
    values = NodeValues()
    
    def __init__(self, mesh, uid):
        self._type = 'standard'
        self.mesh = mesh 
        self.id = uid
        self.fixed = None
        self.cids = None
        self.num_values = 0
        self.num_fields = 0
        self.num_components = 0
        self.num_modes = 0
        self.shape = (0, 0, 0)
        self._added = False
        self._uptodate = False
        self.mesh._regenerate = True
        self.mesh._reupdate = True
        
    @property
    def field_cids(self):
        return self._get_param_indicies()
    
    def _save_dict(self):
        node_dict = {}
        node_dict['id'] = self.id
        node_dict['fixed'] = self.fixed
        node_dict['cids'] = self.cids
        node_dict['num_values'] = self.num_values
        node_dict['num_fields'] = self.num_fields
        node_dict['num_components'] = self.num_components
        node_dict['num_modes'] = self.num_modes
        node_dict['shape'] = self.shape
        return node_dict
        
    def _load_dict(self, node_dict):
        if node_dict['id'] != self.id:
            raise 'Ids do not match'
        self.fixed = node_dict['fixed']
        self.cids = node_dict['cids'] 
        self.num_values = node_dict['num_values']
        self.num_fields = node_dict['num_fields']
        self.num_components = node_dict['num_components']
        if 'num_modes' in node_dict.keys():
            self.num_modes = node_dict['num_modes']
        else:
            self.num_modes = 0
        if 'shape' in node_dict.keys():
            self.shape = node_dict['shape']
        else:
            self.shape = (self.num_fields, self.num_components)
        self._added = True
        
    def _set_values(self, pids, values):
        self.mesh._core.P[pids] = values
    
    def add_to_group(self, groups):
        if not isinstance(groups, list):
            groups = [groups]
        for group in groups:
            self.mesh.nodes.add_to_group(self.id, group)
    
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
        self.shape = values.shape
        
        if len(values.shape) >= 2:
            self.num_components = values.shape[1]
        
        if len(values.shape) >= 3:
            self.num_modes = values.shape[2]
        
        # Updates the values in core if they exist otherwise adds them.
        params = values.reshape(self.num_values)
        if self._added:
            self.mesh._core.update_params(self.cids, params)
        else:
            self.cids = self.mesh._core.add_params(params)
        
        self._added = True
        self.mesh._regenerate = True
        self.mesh._reupdate = True
        
    def get_values(self, index=None):
        if index == None:
            return self.mesh._core.P[self.cids].reshape(
                    (self.num_fields, self.num_components))
    
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
            self.fixed = fix * numpy.ones(self.num_values, dtype=bool)
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
        #~ if self.num_modes == 0:
            #~ pi = []
            #~ ind = 0
            #~ for f in range(self.num_fields):
                #~ pi.append(self.cids[ind:ind+self.num_components])
                #~ ind += self.num_components
        #~ else:
            #~ pi = []
            #~ ind = 0
            #~ for f in range(self.num_fields):
                #~ pf = []
                #~ for 
                #~ pi.append(self.cids[ind:ind+self.num_components])
                #~ ind += self.num_components
        if len(self.shape) == 1:
            shape = (self.shape[0], 1)
        else:
            shape = self.shape
        pi1 = numpy.array(self.cids).reshape(shape).tolist()
        #~ print self.shape, pi, '=', pi1, '\n'
        
        return pi1
    
    def variables(self, state=True, fields=None):
        if isinstance(fields, int):
            fields = [fields]
        if fields == None:
            cids = self.cids
        else:
            cids = numpy.array(self.cids).reshape(self.shape)[fields]
        if state == True:
            self.mesh._core.add_variables(cids)
        else:
            self.mesh._core.remove_variables(cids)
                
        
    
class StdNode(Node):
    '''
    .. autoclass:: morphic.mesher.Node
    '''
    
    def __init__(self, mesh, uid, values):
        Node.__init__(self, mesh, uid)
        self._type = 'standard'
        if values != None:
            self.set_values(values)
        
    def _save_dict(self):
        node_dict = Node._save_dict(self)
        node_dict['type'] = self._type
        return node_dict
        
    def _load_dict(self, node_dict):
        Node._load_dict(self, node_dict)
        
        
class DepNode(Node):

    def __init__(self, mesh, uid, element, node):
        Node.__init__(self, mesh, uid)
        self._type = 'dependent'
        self.element = element
        self.node = node
     
    def _save_dict(self):
        node_dict = Node._save_dict(self)
        node_dict['type'] = self._type   
        node_dict['element'] = self.element   
        node_dict['node'] = self.node
        return node_dict
    
    def _load_dict(self, node_dict):
        Node._load_dict(self, node_dict)
        self.element = node_dict['element']
        self.node = node_dict['node'] 
    
class PCANode(Node):

    def __init__(self, mesh, uid, node_id, weights_id, variance_id):
        Node.__init__(self, mesh, uid)
        self._type = 'pca'
        self.node_id = node_id
        self.weights_id = weights_id
        self.variance_id = variance_id
        
        self._initialise()
        
    def _initialise(self, loading=False):
        OK = True
        if OK and self.node_id in self.mesh.nodes:
            self.node = self.mesh.nodes[self.node_id]
        else:
            #~ print 'Warning: PCA node not initialised'
            OK = False
        
        if OK and self.weights_id in self.mesh.nodes:
            self.weights = self.mesh.nodes[self.weights_id]
        else:
            #~ print 'Warning: PCA node not initialised'
            OK = False
        
        if OK and self.variance_id in self.mesh.nodes:
            self.variance = self.mesh.nodes[self.variance_id]
        else:
            #~ print 'Warning: PCA node not initialised'
            OK = False
        
        if OK and not loading:
            self.set_values(self.node.values[:,:,0])
            #~ self.set_values(numpy.zeros(self.shape))
        
        if OK:
            self._pca_id = self.mesh._core.add_pca_node(self)
            self._added = self._pca_id > -1
    
    def _save_dict(self):
        node_dict = Node._save_dict(self)
        node_dict['type'] = self._type   
        node_dict['node_id'] = self.node_id   
        node_dict['weights_id'] = self.weights_id
        node_dict['variance_id'] = self.variance_id
        return node_dict
    
    def _load_dict(self, node_dict):
        Node._load_dict(self, node_dict)
        self.node_id = node_dict['node_id']
        self.weights_id = node_dict['weights_id'] 
        self.variance_id = node_dict['variance_id'] 
        self._initialise(True)
        
        
class Element(object):

    def __init__(self, mesh, uid, interp, node_ids):
        self._type = 'element'
        self.mesh = mesh 
        self.core = mesh._core 
        self.interp = interp 
        self.basis = interp
        self.dimensions = utils.element_dimensions(interp)
        self.id = uid
        self.node_ids = node_ids
        self.cid = None
        
        self._set_shape()
        
        self.mesh._regenerate = True
        self.mesh._reupdate = True
    
    @property
    def nodes(self):
        return self.mesh.nodes[self.node_ids]
        
    @nodes.setter
    def nodes(self, nodes):
        self.node_ids = [node.id for node in nodes]
    
    def _set_shape(self):
        if self.interp:
            if self.dimensions == None:
                pass
            elif self.dimensions == 1:
                self.shape = 'line'
            elif self.dimensions == 2:
                self.shape = 'quad'
                if self.interp[0][0] == 'T':
                    self.shape = 'tri'
            elif self.dimensions == 3:
                self.shape = 'hexagonal'
            else:
                raise NotImplementedError('Cannot set shape for dimensions'
                    + ' higher than 3')
    
    def _save_dict(self):
        elem_dict = {}
        elem_dict['id'] = self.id
        elem_dict['interp'] = self.interp
        elem_dict['nodes'] = self.node_ids
        elem_dict['shape'] = self.shape
        return elem_dict
     
    def _load_dict(self, elem_dict):
        if elem_dict['id'] != self.id:
            raise 'Ids do not match'
        self.interp = elem_dict['interp']
        self.node_ids = elem_dict['nodes'] 
        self.shape = elem_dict['shape']
        self._set_shape()
        if self.mesh.auto_add_faces:
            self.add_faces()
        
        
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
        
        PI = self._filter_face_param_indices(PI)
        
        return PI
    
    def _filter_face_param_indices(self, PI):
        return PI
    
    def add_to_group(self, groups):
        if not isinstance(groups, list):
            groups = [groups]
        for group in groups:
            self.mesh.elements.add_to_group(self.id, group)
        
    def set_core_id(self, cid):
        self.cid = cid
    
    def add_faces(self):
        if utils.element_dimensions(self.interp) == 2:
            self.mesh.add_face(self.id, 0, self.node_ids)
        elif utils.element_dimensions(self.interp) == 3:
            face_nodes = core.element_face_nodes(self.interp, self.node_ids)
            for face_index in range(6):
                self.mesh.add_face(self.id, face_index, face_nodes[face_index])
    
    def grid(self, res=[8, 8]):
        return discretizer.xi_grid(
                shape=self.shape, res=res, units='div')[0]
    
    def get_field_cids(self, field_index):
        return self.core.EMap[self.cid][field_index]

    def weights(self, xi, deriv=None):
        return self.mesh._core.weights(self.cid, xi, deriv=deriv)
    
    def interpolate(self, xi, deriv=None):
        print 'Interpolate deprecated. Use evaluate instead.'
        return self.evaluate(xi, deriv=deriv)
        
    def evaluate(self, xi, deriv=None):
        if isinstance(xi, int) or isinstance(xi, float):
            if self.shape is not 'line':
                raise Exception('Invalid xi to an element greater than 1d')
            xi = numpy.array([[xi]])
            return self.mesh._core.evaluate(self.cid, xi, deriv=deriv)[0]
        
        else:
            if isinstance(xi, list):
                xi = numpy.asarray(xi)
                
            if self.shape is 'line':
                if len(xi.shape) == 1:
                    xi = numpy.array([xi]).T
                    if xi.shape[0] == 1:
                        return self.mesh._core.evaluate(
                                self.cid, xi, deriv=deriv)[0]
                    else:
                        return self.mesh._core.evaluate(
                                self.cid, xi, deriv=deriv)
                else:
                    return self.mesh._core.evaluate(self.cid, xi, deriv=deriv)
            else:
                if len(xi.shape) == 1:
                    xi = numpy.array([xi])
                    return self.mesh._core.evaluate(
                            self.cid, xi, deriv=deriv)[0]
                else:
                    return self.mesh._core.evaluate(
                            self.cid, xi, deriv=deriv)
        
    
    def integrate(self, fields, func=None, ng=4):
        '''
        Integration using gaussian quadrature.
        
        Only supports up to two dimensions. The generation of the gauss
        points for 3 dimensions and up is required to extend this.
        
        Input:
          - fields is a list of fields to integrate. The format is
            [[field no, derivative wrt x1, derivative wrt x2,... ], ...]
            For example, [[0, 0, 0], [2, 1, 0], [1, 0, 2]] will return
            the integral of Field0, dField2/dx1, and d^2Field1/dx2^2.
          - func is a function that will take the field values process them
            and return the processed values for the fields
          - ng is the number of gauss point. Default is 4. Max is 6.
            For 2D, the number of gauss points in each direction is
            given using [ng1, ng2, ...].
        
        Returns:
          - integral of the fields or processed fields by 'func'
        '''
        if self.shape is 'line':
            Xi, W = self.core.get_gauss_points(ng)
            X = self.mesh._core.evaluate_fields(self.cid, Xi, fields)
            if func is not None:
                X = func(X)
            integral = numpy.dot(W, X)
        elif self.shape in ['quad', 'tri']:
            Xi, W = self.core.get_gauss_points([ng, ng])
            X = self.mesh._core.evaluate_fields(self.cid, Xi, fields)
            if func is not None:
                X = func(X)
            integral = numpy.dot(W, X)
        elif self.shape is 'hexagonal':
            Xi, W = self.core.get_gauss_points([ng, ng, ng])
            X = self.mesh._core.evaluate_fields(self.cid, Xi, fields)
            if func is not None:
                X = func(X)
            integral = numpy.dot(W, X)
        else:
            raise ValueError('Unknown element shape for integral.')
            
        return integral
    
    def length(self, ng=3):
        
        def _length_integral(X):
            return numpy.sqrt((X**2).sum(1))
            
        if self.shape is 'line':
            fields = []
            for i in range(self.num_fields):
                fields.append([i, 1])
            return self.integrate(fields, func=_length_integral, ng=ng)
        else:
            raise TypeError('You can only calculate the length '
                + 'of a 1D element.')
            
    def area(self, ng=3):
        
        def _area_integral(X):
            A = []
            for x in X:
                x = x.reshape((2, x.size/2))
                c = numpy.cross(x[0], x[1])
                A.append(numpy.sqrt((c * c).sum()))
            return A
            
        if self.shape is 'quad':
            fields = []
            for i in range(self.num_fields):
                fields.append([i, 1, 0])
                fields.append([i, 0, 1])
            return self.integrate(fields, func=_area_integral, ng=ng)
        else:
            raise TypeError('You can only calculate the area '
                + 'of a 2D quad element. Triangles not implemented.')
    
    def volume(self, ng=3):
        
        def _volume_integral0(X):
            V = []
            for x in X:
                V.append(linalg.det(x.reshape((3, x.size/3))))
            return V
            
        def _volume_integral(X):
            V = X[:,0] * (X[:,4] * X[:,8] - X[:,5] * X[:,7]) - X[:,1] * (X[:,3] * X[:,8] - X[:,5] * X[:,6]) + X[:,2] * (X[:,3] * X[:,7] - X[:,4] * X[:,6])
            return V
            
        if self.shape is 'hexagonal':
            fields = []
            for i in range(self.num_fields):
                fields.append([i, 1, 0, 0])
                fields.append([i, 0, 1, 0])
                fields.append([i, 0, 0, 1])
            return abs(self.integrate(fields, func=_volume_integral, ng=ng))
        else:
            raise TypeError('You can only calculate the area '
                + 'of a 3D hexagonal element. Triangles not implemented.')
    
    
    def normal(self, Xi):
        '''
        Calculates the surface normals at xi locations on an element.
        
        Xi is an ``mxn`` ndarray where ``m`` is the number of element
        locations and ``n`` is the number of element dimensions.
        
        For example,
        
        .. code-block:: python
        
            Xi = numpy.array([[0.1, 0.1], [0.3, 0.2], [0.7, 0.2]])
        
        '''
        dx1 = self.mesh._core.evaluate(self.cid, Xi, deriv=[1, 0])
        dx2 = self.mesh._core.evaluate(self.cid, Xi, deriv=[0, 1])
        return numpy.cross(dx1, dx2)
    
    def _project_objfn(self, xi, *args):
        x = args[0]
        xe = self.evaluate(xi)
        dx = xe - x
        return numpy.sum(dx * dx)
    
    def project(self, x, xi=None, xtol=1e-4, ftol=1e-4):
        from scipy.optimize import fmin
        if xi == None:
            if self.shape == 'line':
                xi = 0.5
            else:
                xi = [0.5, 0.5]
        xi_opt = fmin(self._project_objfn, xi, args=(x),
                xtol=xtol, ftol=ftol, disp=False)
        return xi_opt
        
    def __iter__(self):
        return self.nodes.__iter__()

class Face(object):
    
    def __init__(self, mesh, uid):
        self.id = uid
        self.mesh = mesh
        self.element_faces = []
        self.shape = 'quad'
        
    def add_element(self, element_id, face_index=0):
        element_face = [element_id, face_index]
        if element_face not in self.element_faces:
            self.element_faces.append(element_face)
    
    @property
    def nodes(self):
        return self.mesh.nodes[self.node_ids]
        
    @nodes.setter
    def nodes(self, nodes):
        self.node_ids = [node.id for node in nodes]
    
    
class Mesh(object):
    '''
    This is the top level object for a mesh which allows:
    
        * building a mesh
        * analysis or rendering of a mesh
        * saving and loading meshes.
    
    A mesh consists of nodes and elements. There are two types of nodes
    that can be added to a mesh:
    
        * standard nodes, which is stores fields values, for example, 
          x, y, and z coordinates.
        
        * dependent nodes, which are embedded in an element. A hanging 
          node can be added to a mesh using a dependent node.
        
    There are many interpolation schemes that can be used to
    create an element:
        * 1D and 2D lagrange basis up to 4th order
        * 1D and 2D Hermite basis (cubic only)
        * 2D triangular elements up to 4th order.
    
    .. note::
    
        Higher order interpolants (3D) will be added in the future.
    
    
    **Building a 1D Mesh**
    
    Here is an example of a simple 1D mesh which consists of two 1D
    linear element. First, we initialise a mesh,
    
    >>> mesh = Mesh()
    
    Add a few nodes,
    
    >>> n = mesh.add_stdnode(1, [0, 0])
    >>> n = mesh.add_stdnode(2, [1, 0.5])
    >>> n = mesh.add_stdnode(3, [2, 0.3])
    
    Add two linear elements,
    
    >>> e = mesh.add_element(1, ['L1'], [1, 2])
    >>> e = mesh.add_element(2, ['L1'], [2, 3])
    
    >>> xi = [0, 0.25, 0.5, 0.75, 1.0]
    >>> X = mesh.evaluate(1, xi)
    >>> X
    array([[ 0.   ,  0.   ],
           [ 0.25 ,  0.125],
           [ 0.5  ,  0.25 ],
           [ 0.75 ,  0.375],
           [ 1.   ,  0.5  ]])
    
    
    **Building a 2D Mesh**
    
    Here is an example of a 2D mesh which consists of two quadratic-linear
    elements. First, we initialise a mesh,
    
    >>> mesh = Mesh()
    
    Add ten nodes,
    
    >>> n = mesh.add_stdnode(1, [0, 0, 0])
    >>> n = mesh.add_stdnode(2, [1, 0, 0.5])
    >>> n = mesh.add_stdnode(3, [2, 0, 0.3])
    >>> n = mesh.add_stdnode(4, [3, 0, 0.2])
    >>> n = mesh.add_stdnode(5, [4, 0, -0.1])
    >>> n = mesh.add_stdnode(6, [0, 1.2, 0])
    >>> n = mesh.add_stdnode(7, [1, 1.2, 0.5])
    >>> n = mesh.add_stdnode(8, [2, 1.2, 0.3])
    >>> n = mesh.add_stdnode(9, [3, 1.2, 0.2])
    >>> n = mesh.add_stdnode(10, [4, 1.2, -0.1])
    
    
    Add two quadratic-linear lagrange elements,
    
    >>> e = mesh.add_element(1, ['L2', 'L1'], [1, 2, 3, 6, 7, 8])
    >>> e = mesh.add_element(2, ['L2', 'L1'], [3, 4, 5, 8, 9, 10])
    
    >>> # Analyse (evaluate, calculate derivatives and normals)
    >>> xi = [[0.3, 0.5]]
    
    Interpolate coordinates at xi=[0.3, 0.5] on element 1,
    
    >>> X = mesh.evaluate(1, xi)
    
    Calculate derivatives at xi = [0.3, 0.5] on element 2,
    
    >>> dx1 = mesh.evaluate(2, xi, deriv=[1, 0]) # dx/dxi1
    >>> dx2 = mesh.evaluate(2, xi, deriv=[0, 1]) # dx/dxi2
    >>> dx12 = mesh.evaluate(2, xi, deriv=[1, 1]) # d2x/dxi1.dxi2
    
    Calculate the element normal vector at xi=[0.3, 0.5] on element 1,
    
    >>> dn = mesh.normal(1, xi) # normal vector
    
    
    '''
    
    def __init__(self, filepath=None, label='/', units='m'):
        self._version = 1
        self.label = label
        self.units = units
        self.nodes = core.ObjectList()
        self.elements = core.ObjectList()
        self.faces = core.ObjectList()
        
        self._core = core.Core()
        self._regenerate = True
        self._reupdate = True
        
        self.auto_add_faces = True;
        
        self.filepath = filepath
        if filepath != None:
            self.load(filepath)
    
    def add_node(self, uid, values, group='_default'):
        return self.add_stdnode(uid, values, group=group)
        
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
        
    def add_pcanode(self, uid, values, weights_nid, variance_nid,
                    group='_default'):
        '''
        Adds a pca node to a mesh.
        '''
        if uid==None:
            uid = self.nodes.get_unique_id()
        if isinstance(values, list) or isinstance(values, numpy.ndarray):
            values_nid = '__'+self.nodes.get_unique_id(random_chars=12)
            std_node = self.add_stdnode(values_nid, values,
                    group='__sys_'+group)
        else:
            values_nid = values
            
        node = PCANode(self, uid, values_nid, weights_nid, variance_nid)
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
        >>> print elem.id, elem.interp, elem.node_ids
        1 ['L1'] [1, 2]
        '''
        if uid==None:
            uid = self.elements.get_unique_id()
        if isinstance(node_ids, numpy.ndarray):
            node_ids = node_ids.tolist()
        elem = Element(self, uid, interp, node_ids)
        self.elements.add(elem, group=group)
        if self.auto_add_faces:
            elem.add_faces()
            
        return elem
        
    
    def add_face(self, element, face_index=0, nodes=None):
        '''
        Adds a face to the mesh, typically used to store element faces
        for graphics
        - basis: the intepolation functions in each dimension
            e.g., ['L2', 'H3']
        - nodes: the node indices for the face
        - face index: the face position on a cube, range [0:7]
        '''
        if nodes == None:
            nodes = core.element_face_nodes(elem.interp, elem.node_ids)[face_index]
        sorted_nodes = [n for n in nodes]
        sorted_nodes.sort()
        face_id = '_' + '_'.join([str(i) for i in sorted_nodes])
        if face_id not in self.faces:
            face = Face(self, face_id)
            self.faces.add(face)
        else:
            face = self.faces[face_id]
        face.add_element(element, face_index)
        return face
    
    def groups(self, group_type=None):
        if group_type == None:
            return {
                'nodes':[k for k in self.nodes.groups.keys()],
                'elements':[k for k in self.nodes.groups.keys()]}
        elif group_type == 'nodes':
            return {'nodes':[k for k in self.nodes.groups.keys()]}
        elif group_type == 'elements':
            return {'elements':[k for k in self.elements.groups.keys()]}

        return None
        
    def _save_dict(self):
        import datetime
        mesh_dict = {}
        mesh_dict['_version'] = self._version
        mesh_dict['_datetime'] = datetime.datetime.now()
        mesh_dict['label'] = self.label
        mesh_dict['units'] = self.units
        mesh_dict['nodes'] = []
        mesh_dict['elements'] = []
        for node in self.nodes:
            mesh_dict['nodes'].append(node._save_dict())
        for elem in self.elements:
            mesh_dict['elements'].append(elem._save_dict())
        
        mesh_dict['node_objlist'] = self.nodes._save_dict()
        mesh_dict['element_objlist'] = self.elements._save_dict()
        
        mesh_dict['values'] = self._core.P
        return mesh_dict
        
    def save(self, filepath):
        '''
        Saves a mesh.
        
        >>> mesh = Mesh()
        >>> mesh.save('data/cube.mesh')
        
        '''
        import pickle
        mesh_dict = self._save_dict()
        pickle.dump(mesh_dict, open(filepath, "w"))
        
    def _load_dict(self, mesh_dict):
        self.label = mesh_dict['label']
        self.units = mesh_dict['units']
        for node_dict in mesh_dict['nodes']:
            if node_dict['type'] == 'standard':
                node = self.add_stdnode(node_dict['id'], None)
            elif node_dict['type'] == 'dependent':
                node = self.add_depnode(node_dict['id'], None, None)
            elif node_dict['type'] == 'pca':
                node = self.add_pcanode(node_dict['id'], None, None, None)
            #~ else:
                #~ raise InputError()
            node._load_dict(node_dict)
        for elem_dict in mesh_dict['elements']:
            elem = self.add_element(elem_dict['id'], None, None)
            elem._load_dict(elem_dict)
        self._core.P = mesh_dict['values']
        
        if 'node_objlist' in mesh_dict.keys():
            self.nodes._load_dict(mesh_dict['node_objlist'])
        if 'element_objlist' in mesh_dict.keys():
            self.elements._load_dict(mesh_dict['element_objlist'])
        
    
    def load(self, filepath):
        '''
        Loads a mesh.
        
        >>> mesh = Mesh('data/cube.mesh')
        
        or
        
        >>> mesh = Mesh()
        >>> mesh.load('data/cube.mesh')
        
        '''
        import pickle
        self._load_dict(pickle.load(open(filepath, "r")))
        self.generate(True)
        
    
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
            self._core.update_pca_nodes()
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
            self._core.update_pca_nodes()
            self._core.update_dependent_nodes()
            self._reupdate = False
    
    def _update_dependent_nodes(self):
        for node in self.nodes:
            if node._type == 'dependent' and node._added == False:
                elem = self.elements[node.element]
                for enode in elem:
                    if enode.num_values > 0:
                        node.set_values(numpy.zeros(enode.num_values))
                        break
                        
    def update_pca_nodes(self):
        self._core.update_pca_nodes()
    
    def get_variables(self):
        return self._core.get_variables()
    
    def set_variables(self, variables):
        self._core.set_variables(variables)
    
    def get_element_cids(self, elements=None):
        if elements == None:
            elements = self.elements
        cids = []
        for elem in elements:
            cids.append(elem.cid)
        return cids
    
    def update_parameters(self, param_ids, values):
        self._core.update_params(param_ids, values)
    
    def interpolate(self, element_ids, xi, deriv=None):
        print 'Interpolate deprecated. Use evaluate instead.'
        return self.evaluate(element_ids, xi, deriv=deriv)
        
    def evaluate(self, element_ids, xi, deriv=None):
        self.generate()
        if isinstance(xi, list):
            xi = numpy.array(xi)
            if len(xi.shape) == 1:
                xi = numpy.array([xi]).T
        else:
            if len(xi.shape) == 1:
                xi = numpy.array([xi]).T
                
        if not isinstance(element_ids, list):
            element_ids = [element_ids]
        
        node0 = self.elements[element_ids[0]].nodes[0]
        num_fields = node0.num_fields
        num_elements = len(element_ids)
        num_xi = xi.shape[0]
        X = numpy.zeros((num_xi * num_elements, num_fields))
        
        ind = 0
        for element in self.elements[element_ids]:
            X[ind:ind+num_xi, :] = element.evaluate(xi, deriv=deriv)
            ind += num_xi
            
        return X
    
    def normal(self, element_ids, xi, normalise=False):
        self.generate()
        if isinstance(xi, list):
            xi = numpy.array(xi)
            if len(xi.shape) == 1:
                xi = numpy.array([xi]).T
        else:
            if len(xi.shape) == 1:
                xi = numpy.array([xi]).T
                
        if not isinstance(element_ids, list):
            element_ids = [element_ids]
        
        node0 = self.elements[element_ids[0]].nodes[0]
        num_fields = node0.num_fields
        num_elements = len(element_ids)
        num_xi = xi.shape[0]
        X = numpy.zeros((num_xi * num_elements, num_fields))
        
        ind = 0
        for element in self.elements[element_ids]:
            X[ind:ind+num_xi, :] = element.normal(xi)
            ind += num_xi
        
        if normalise:
            R = numpy.sqrt(numpy.sum(X * X, axis=1))
            for axis in range(X.shape[1]):
                X[:,axis] /= R
            
        return X
    
    def deformation_gradient_tensor(self, deformed_mesh, xi):
        dx1 = self.elements[1].evaluate(xi, deriv=[1,0])
        dx2 = self.elements[1].evaluate(xi, deriv=[0,1])
        dx = numpy.array([dx1, dx2])
        invdx = linalg.inv(dx)
        
        dX1 = deformed_mesh.elements[1].evaluate(xi, deriv=[1,0])
        dX2 = deformed_mesh.elements[1].evaluate(xi, deriv=[0,1])
        dX = numpy.array([dX1, dX2])
        invdX = linalg.inv(dX)
        F = numpy.dot(invdx, dX.T)
        invF = linalg.inv(F)
        #~ print dx
        #~ print dX
        #~ print F
        return F, invF
    
    def grid(self, res=[8, 8], shape='quad'):
        return discretizer.xi_grid(
                shape=shape, res=res, units='div')[0]
                
    def get_nodes(self, nodes=None, group='_default'):
        self.generate()
        if nodes != None:
            if not isinstance(nodes, list):
                nodes = [nodes]
            nodes = self.nodes[nodes]
        else:
            nodes = self.nodes(group)
        Xn = []
        for node in nodes:
            if len(node.shape) == 1:
                Xn.append(node.values)
            else:
                Xn.append(node.values[:, 0])
        return numpy.array([xn for xn in Xn])
        
    def get_node_ids(self, nodes=None, group='_default'):
        self.generate()
        if nodes != None:
            if not isinstance(nodes, list):
                nodes = [nodes]
            nodes = self.nodes[nodes]
        else:
            nodes = self.nodes(group)
        Xn = []
        labels = []
        for node in nodes:
            labels.append(node.id)
            if len(node.shape) == 1:
                Xn.append(node.values)
            else:
                Xn.append(node.values[:, 0])
        return numpy.array([xn for xn in Xn]), labels
    
    def get_lines(self, res=8, group='_default'):
        self.generate()
        Xl = []
        xi = numpy.array([numpy.linspace(0, 1, res)]).T
        for i, elem in enumerate(self.elements(group)):
            Xl.append(self._core.evaluate(elem.cid, xi))
        return Xl
        
    def get_surfaces(self, res=8, elements=None, groups=None, include_xi=False):
        self.generate()
        
        if elements == None:
            if groups == None:
                Elements = self.elements
            else:
                Elements = self.elements.get_groups(groups)
        else:
            Elements = self.elements[elements]
        
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
                
        X = numpy.zeros((NP, elem.nodes[0].num_fields))
        T = numpy.zeros((NT, 3), dtype='uint32')
        if include_xi:
            Xi = numpy.zeros((NP, 2))
        np, nt = 0, 0
        for elem in Elements:
            if elem.shape == 'tri':
                X[np:np+NPT,:] = self._core.evaluate(elem.cid, XiT)
                if include_xi:
                    Xi[np:np+NPT,:] = XiT
                T[nt:nt+NTT,:] = TT + np
                np += NPT
                nt += NTT
            elif elem.shape == 'quad':
                X[np:np+NPQ,:] = self._core.evaluate(elem.cid, XiQ)
                T[nt:nt+NTQ,:] = TQ + np
                if include_xi:
                    Xi[np:np+NPQ,:] = XiQ
                np += NPQ
                nt += NTQ
        if include_xi:
            return X, T, Xi
        return X, T
        
    def get_faces(self, res=8, exterior_only=True, include_xi=False):
        self.generate()
        
        if exterior_only:
            Faces = [face for face in self.faces if len(face.element_faces) == 1]
        else:
            Faces = self.faces
        
        XiT, TT = discretizer.xi_grid(shape='tri', res=res)
        XiQ, TQ = discretizer.xi_grid(shape='quad', res=res)
        NPT, NTT = XiT.shape[0], TT.shape[0]
        NPQ, NTQ = XiQ.shape[0], TQ.shape[0]
        
        XiQ0 = numpy.zeros(NPQ)
        XiQ1 = numpy.ones(NPQ)
        
        NP, NT = 0, 0
        for face in Faces:
            if face.shape == 'tri':
                NP += NPT
                NT += NTT
            elif face.shape == 'quad':
                NP += NPQ
                NT += NTQ
                
        X = numpy.zeros((NP, 3))#######TODO#####face.nodes[0].num_fields))
        T = numpy.zeros((NT, 3), dtype='uint32')
        if include_xi:
            Xi = numpy.zeros((NP, 2))
        np, nt = 0, 0
        for face in Faces:
            if face.shape == 'tri':
                X[np:np+NPT,:] = self._core.evaluate(face.cid, XiT)
                if include_xi:
                    Xi[np:np+NPT,:] = XiT
                T[nt:nt+NTT,:] = TT + np
                np += NPT
                nt += NTT
            elif face.shape == 'quad':
                elem = self.elements[face.element_faces[0][0]]
                face_index = face.element_faces[0][1]
                if face_index == 0:
                    X[np:np+NPQ,:] = self._core.evaluate(elem.cid,
                        numpy.array([XiQ[:,0], XiQ[:,1], XiQ0]).T)
                elif face_index == 1:
                    X[np:np+NPQ,:] = self._core.evaluate(elem.cid,
                        numpy.array([XiQ[:,0], XiQ[:,1], XiQ1]).T)
                elif face_index == 2:
                    X[np:np+NPQ,:] = self._core.evaluate(elem.cid,
                        numpy.array([XiQ[:,0], XiQ0, XiQ[:,1]]).T)
                elif face_index == 3:
                    X[np:np+NPQ,:] = self._core.evaluate(elem.cid,
                        numpy.array([XiQ[:,0], XiQ1, XiQ[:,1]]).T)
                elif face_index == 4:
                    X[np:np+NPQ,:] = self._core.evaluate(elem.cid,
                        numpy.array([XiQ0, XiQ[:,0], XiQ[:,1]]).T)
                elif face_index == 5:
                    X[np:np+NPQ,:] = self._core.evaluate(elem.cid,
                        numpy.array([XiQ1, XiQ[:,0], XiQ[:,1]]).T)
                    
                T[nt:nt+NTQ,:] = TQ + np
                if include_xi:
                    Xi[np:np+NPQ,:] = XiQ
                np += NPQ
                nt += NTQ
        if include_xi:
            return X, T, Xi
        return X, T
        
        
        
        
