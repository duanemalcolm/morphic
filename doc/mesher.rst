Meshing
=======

.. toctree::
   :maxdepth: 2
  
This module calculates the basis function weights for values and
derivatives for various types of interpolants.

----
Mesh
----

.. autoclass:: morphic.mesher.Mesh


-----------------
Generating a Mesh
-----------------

^^^^^
Nodes
^^^^^

There are four types of nodes that can be added to a mesh:

  **Standard Nodes** ``mesh.add_stdnode(id, values)``
    Stores field values. The fields can include components, for example,
    in the case where field derivatives or PCA modes are included.
    
  **Dependent Nodes**
    Describes a node that depends on other parts of a mesh, typically,
    a node embedded in an element.
    
  **PCA Nodes** (TODO)
    This is a convinient description of a node derived from  principal
    component analysis. This would store the mean node values and it's
    modes, and the associated weights and variance, which are stored as
    standard nodes.
    
  **Mapping Nodes** (TODO)
    Descibes a node which uses values from other
    nodes. This is typically used when creating a Hermite node based of
    derivative values from a lagrange element or when a there are
    different versions of node values.


.. automethod:: morphic.mesher.Mesh.add_stdnode

.. automethod:: morphic.mesher.Mesh.add_depnode

^^^^^^^^
Elements
^^^^^^^^

.. automethod:: morphic.mesher.Mesh.add_element


----------------
Analysing a Mesh
----------------

------------------
Saving and Loading
------------------

