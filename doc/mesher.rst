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


================
A Simple 2D Mesh
================


=====
Nodes
=====
There are two types of nodes that can be added to a mesh:

  **Standard Nodes** ``mesh.add_stdnode(id, values)``
    Stores field values. The fields can include components, for example,
    in the case where field derivatives or PCA modes are included.
    
  **Dependent Nodes** ``mesh.add_stdnode(id, element_id, node_id)``
    Describes a node that depends on other parts of a mesh, typically,
    a node embedded in an element.



========
Elements
========




.. automethod:: morphic.mesher.Element.normal


================
Analysing a Mesh
================


==================
Saving and Loading
==================

.. attention::
    
    To be implemented
