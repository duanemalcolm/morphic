Examples
========

.. toctree::
   :maxdepth: 2
  
Description

-----------
1D Elements
-----------

In this example we create a mesh containing four lines representing a
sine wave.

First we some data point along the sine wave as nodes in the mesh,

.. literalinclude:: ../examples/lines.py
    :start-after: # Start Generate Data
    :end-before: # End Generate Data

Then we create the mesh. First we define all the node and add the nodes
we will

.. literalinclude:: ../examples/lines.py
    :start-after: # Start Generate Mesh
    :end-before: # End Generate Mesh
    
Now we can plot the mesh.

.. literalinclude:: ../examples/lines.py
    :start-after: # Sphinx Start Tag: Plotting
    :end-before: # Sphinx End Tag: Plotting

.. _lines_plot:
.. figure::  ./images/lines.png
   :align:   center

   From bottom to top, 4 linear, 2 quadratics, 2 cubic and 2 quartic
   lagrange elements.
   
We can also analyse the mesh in a number of ways:

-----------
2D Elements
-----------
