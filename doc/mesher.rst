*******
Meshing
*******

.. toctree::
   :maxdepth: 2

===============
The Mesh Object
===============

A mesh uses nodes and elements in order to describe a field, such as
geometric fields that represent a surface.

--------
New Mesh
--------

.. code-block:: python
    
    mesh = Mesh()


---------
Load Mesh
---------

.. automethod:: morphic.mesher.Mesh.load


---------
Save Mesh
---------

.. automethod:: morphic.mesher.Mesh.save


------
Export
------

.. note::
    
    TODO: Export to cm files, cmgui files, fieldml


------------------------
Retrieving Mesh Entities
------------------------


=====
Nodes
=====

There are two types of nodes that can be added to a mesh:

  **Standard Nodes**
    Stores field values. The fields can include components, for example,
    in the case where field derivatives or PCA modes are included.
    
  **Dependent Nodes**
    Describes a node that depends on other parts of a mesh, typically,
    a node embedded in an element.

A standard node can be added to the mesh by,

.. code-block:: python
    
    node = mesh.add_stdnode(id, values)
    
where ``id`` is the unique identified for nodes, and ``values`` are the
field values for the node. This command will return a node object.

The ``id`` variable can be defined by user as integer, string or
``None``.  If set to ``None`` a unique integer id will be assigned.

The ``value`` variable can be a one or two dimensional list or numpy
array of field values. In the case of a one-dimensional array, e.g.,
``values = [0.2, 1.5, -0.4]``, each value is assumed to be a field
value. In the case of a two-dimensional array, e.g.,
``values = [[0.2, 1, 0, 0], [1.5, 0, 1, 0], [-0.4, 0, 0, 0]]``, the rows
represent the fields and the columns represents the field components.
Examples of field components are field derivative or mode vectors for a
PCA model.

---------------
Accessing Nodes
---------------

Nodes are stored in a mesh as a list of node objects which can be
accessed through a list or by direct reference by node id.

.. code-block:: python
    
    list_of_nodes = mesh.nodes
    node = mesh.nodes[5]
    node = mesh.nodes['dd']
    
.. note::
    
    TODO

-----------
Node Values
-----------

A standard node can be added to the mesh by,

.. code-block:: python
    
    field_values = node.get_values()
    
    node.set_values(values)
    
.. note::
    
    TODO




========
Elements
========

An element can be added to a mesh by,

.. code-block:: python
    
    elem = mesh.add_element(id, interp, nodes)

where ``id`` is the unique identified for elements, ``interp`` is the 
interpolation functions in each dimension, and ``nodes`` are the node
ids for the element. This command will return a element object.

The ``id`` variable can be defined by user as integer, string or
``None``.  If set to ``None`` a unique integer id will be assigned.

The ``interp`` variable is a list of strings each representing the 
interpolation scheme in each dimension, for example,
``interp = ['L1', 'H3'] for a linear-cubic-Hermite two-dimensional
element.

Interpolation schemes include:
    - L1 - linear lagrange
    - L2 - quadratic lagrange
    - L3 - cubic lagrange
    - L4 - quartic lagrange
    - H3 - cubic-Hermite
    - T11 - linear 2d-simplex
    - T22 - quadratic 2d-simplex
    - T33 - cubic 2d-simplex
    - T44 - quartic 2d-simplex
        
Some examples of interpolation schemes:
    - ['L1', 'L1'] = bilinear (2d)
    - ['L3', 'L2'] = cubic-quadratic (2d)
    - ['H3', 'L1', 'L1'] = cubic-Hermite-bilinear (3d) - note warning below.
    - ['T22'] = biquadratic simplex (2d triangle)
    - ['T11', 'L1'] = a linear prism (3d) - note warning below.

.. warning::
    
    Morphic only supports one and two dimensional elements. Morphic can
    can support higher order elements but this is not fully implemented
    or throughly tested.

------------------
Accessing Elements
------------------
.. note::
    
    TODO

------------------
Element Properties 
------------------
.. note::
    
    TODO: interpolating values, derivatives, normals, neighbours, etc...

-----------------
Dividing Elements
-----------------

-------------------
Converting Elements
-------------------




