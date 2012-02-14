================================
Tutorial: Fitting Meshes to Data
================================

.. toctree::
   :maxdepth: 2
  

In this tutorial we build a 2D mesh using cubic-Hermite elements and fit
the mesh to data generated using a cosine function.

For this tutorial we'll need to import ``scipy`` and ``morphic``.

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # sphinx tag start import
    :end-before: # sphinx tag end import


---------------
Building a Mesh
---------------
First, we create some node point which will be arranged in a regular 3x3
grid ranging from ``x = [-pi, pi]`` and ``y = [-pi, pi]``. The z values
are set to zero.


.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # sphinx tag start generate node values
    :end-before: # Default derivatives values for cubic-Hermite nodes


Because we are using cubic-Hermite element, the nodal values require
derivative values which we default to,

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # Default derivatives values for cubic-Hermite nodes
    :end-before: # sphinx tag end generate node values

Now, we create the mesh,

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # Create mesh
    :end-before: # Add nodes

We add the nodes where for each node value we append the derivative
values,

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # Add nodes
    :end-before: # Add elements

Now we add four bicubic-Hermite elements, which are ``['H3', 'H3']``
basis,

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # Add elements
    :end-before: # Generate the mesh
    
And finally generate the mesh structures,

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # Generate the mesh
    :end-before: # sphinx tag end generate mesh


-------------
Generate Data
-------------
The data cloud is based on a cosine function, ``z = cos(x + 1) cos(y + 1)``.
A 200 x 200 grid of points are created in the x and y dimensions on which
the z values are generated.

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # Generate a data cloud for fitting
    :end-before: # sphinx tag end generate data

--------
Plotting
--------

Here we want to view our initial mesh and data. First we need to extract
the coordinates of the nodes (without derivatives) and a surface of the
mesh, which is done using the following commands,

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # sphinx start get node values and surface
    :end-before: # sphinx start plotting

Next we want to plot the mesh nodes and surface and the data which is
done using the ``morphic.viewer`` module,

.. literalinclude:: ../examples/example_2d_fit_lse.py
    :start-after: # sphinx start plotting
    :end-before: # sphinx tag end plotting

The resultant plot is shown where the nodes are plotted in green, the
mesh surface in blue and the data in red,
 
.. figure::  ./images/example_2d_fit_lse_mesh.png
    :align:   center
    :width: 800px


-------
Fitting
-------


