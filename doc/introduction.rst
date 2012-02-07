Introduction
============
   
.. toctree::
   :maxdepth: 2
  
Fieldscape is a convinient module for mesh creation, analysis,
rendering and fitting to data.

Currently it can handle 1D and 2D meshes with:
    - **Lagrange elements** (up to 4th-order)
    - **Triangle elements** (up to 4th order)
    - **Hermite elements** (cubic-Hermites only)

Fieldscape also offers four types of nodes allowing creation of complex
meshes:
    - **Standard nodes** which store field values, e.g., x, y, z and
      temperature, with components such as derivatives.
    - **Dependent node** which can be calculated from a location on another
      element, for example, hanging nodes.
    - **PCA nodes** to handle PCA models (todo)
    - **Mapping nodes** which can be based on values from other nodes.
      This allows versions of nodes or the creation of a node for a
      cubic-Hermite element where the derivatives might be computed from a
      lagrange element.

.. note::
    3D and higher-dimensional meshes are possible but it
    has not been fully implemented.

The following tutorials will go through the process of building meshes
and fitting meshes. 
