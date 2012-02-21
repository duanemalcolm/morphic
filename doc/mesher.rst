*******
Meshing
*******

.. toctree::
   :maxdepth: 2



A mesh describes fields, such as a surface, using nodes and elements.

The typical process for building a mesh is to define a bunch of nodes
and then elements that use the nodes. First we will show you how to
create, view and analyse a simple 2D mesh and then go into the details
of nodes, elements and the analysis of a mesh.

There are two types of nodes that can be added to a mesh:

  **Standard Nodes**
    Stores field values. The fields can include components, for example,
    in the case where field derivatives or PCA modes are included.
    
  **Dependent Nodes**
    Describes a node that depends on other parts of a mesh, typically,
    a node embedded in an element.


====
Mesh
====

.. autoclass:: morphic.mesher.Mesh


-------------------------
Saving and Loading a Mesh
-------------------------

.. automethod:: morphic.mesher.Mesh.save

.. automethod:: morphic.mesher.Mesh.load



==============
Standard Nodes
==============

.. autoclass:: morphic.mesher.StdNode


----------
Attributes
----------


-------
Methods
-------

.. automethod:: morphic.mesher.StdNode.set_values

.. automethod:: morphic.mesher.StdNode.set_value

.. automethod:: morphic.mesher.StdNode.fix


========
Elements
========

----------
Attributes
----------


-------
Methods
-------

.. automethod:: morphic.mesher.Element.normal

============
DESIGN IDEAS
============

The following is the ideal meshing design:

Creating a mesh:

.. code-block:: python
    
    mesh = Mesh()

Adding standard nodes and dependent nodes:

.. code-block:: python
    
    mesh.add_stdnode(id, values)
    mesh.add_depnode(id, element_id, xi_location)

Adding elements:

.. code-block:: python
    
    mesh.add_element(id, basis, nodes)

========
Elements
========

Adding elements:

.. code-block:: python
    
    mesh.add_element(id, basis, nodes)

where ``id`` is the unique identifier for elements, 


