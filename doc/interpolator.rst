Interpolation
=============
   
.. toctree::
   :maxdepth: 2
  
This module calculates the basis function weights for values and
derivatives for various types of interpolants.

The interpolants include:

- `Lagrange (1D)`_ interpolants:
    linear ('L1'), quadratic ('L2'), cubic ('L3'), quartic ('L4')
- `Hermite (1D)`_ interpolator:
    cubic-Hermite ('H3')
- `Triangular (2D)`_ interpolator:
    bilinear ('T11'), biquadratic ('T22'), bicubic ('T33), biquartic ('T44')

:func:`morphic.interpolator.weights` is the main function used for
calculating weights for the values or derivatives for an interpolant.
Multiple interpolants can be combined to create higher
order interpolants.

**Examples**

The basic usage of :func:`morphic.interpolator.weights` requires a
list of interpolants (``basis``) and a list of point (``X``) for which
the weights are calculated.

To calculate the weights for a one-dimensional linear lagrange interpolant::

    X = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    W = weights(['L1'], X) # values
    dWdx = weights(['L1'], X, deriv=[1]) # dW/dx
    dWdxdx = weights(['L1'], X, deriv=[2]) # dW/dxdx

.. warning::
    
    The second derivative for some interpolants may not be implemented.

Similarly, for a cubic-Hermite interpolant::

    W = weights(['H3'], X) # values
    W = weights(['H3'], X, deriv=[1]) # dW/dx

For two-dimensional interpolants we create a 2D list of points. The
easiest way to create a 2D regular grid of points is to use
:func:`numpy.meshgrid`

    import numpy
    x = numpy.arange(0, 1.001, 0.2)
    y = numpy.arange(0, 1.001, 0.2)
    X, Y = numpy.meshgrid(x, y)
    XY = numpy.array([X.reshape(X.size), Y.reshape(Y.size)]).T

Then to calculate the weights for a cubic-quadratic element::

    W = weights(['L3', 'L2'], XY) # values
    dWdx = weights(['L3', 'L2'], XY, deriv=[1, 0]) # dW/dx
    dWdy = weights(['L3', 'L2'], XY, deriv=[0, 1])) # dW/dy
    dWdxdy = weights(['L3', 'L2'], XY, deriv=[1, 1])) # dW/dxdy

For the two-dimensional triangular interpolants, the range of values in
the points list (X) must satisfy the follow to stay in the bounds of the
element, ``0 <= x <= 1``, ``0<=y<=1``, and ``x + y <= 1``. We can filter
the list of points (X) generated above to satify these constrains by:

    XY_tri = numpy.array([xy for xy in XY if xy[0] + xy[1] <= 1.])

Then we can calculate the values and derivatives for a triangular
interpolant, in this case, the bi-quartic triangle::

    W = weights(['T44'], XY_tri) # values
    dWdx = weights(['T44'], XY_tri, deriv=[[1, 0]]) # dW/dx
    dWdy = weights(['T44'], XY_tri, deriv=[[0, 1]])) # dW/dy

Interpolants can be combined to create higher-order elements. For example,::

    W = weights(['L1', 'L1', 'L3'], X) # bilinear-cubic
    W = weights(['H3', 'H3', 'L2'], X) # bicubic-Hermite-quadratic
    W = weights(['T11', 'L1'], X) # a prism **Needs testing**
    W = weights(['T11', 'L1', 'H3', 'L4', 'T44'], X) # something crazy

.. warning::
    
    Higher order (>2) elements need to be implemented.

.. note::
    
    To develop and add more intepolants see the developers section (ToDo).


Calculating Weights
-------------------

.. automodule:: morphic.interpolator
    :members: weights
    

Lagrange (1D)
-------------

Linear Lagrange
+++++++++++++++
.. automodule:: morphic.interpolator
    :members: L1, L1d1, L1d1d1
        
Quadratic Lagrange
++++++++++++++++++
.. automodule:: morphic.interpolator
    :members: L2, L2d1

Cubic Lagrange
++++++++++++++
.. automodule:: morphic.interpolator
    :members: L3, L3d1

Quartic Lagrange
++++++++++++++++
.. automodule:: morphic.interpolator
    :members: L4, L4d1


Hermite (1D)
------------

Cubic-Hermite
+++++++++++++
.. automodule:: morphic.interpolator
    :members: H3, H3d1


Triangular (2D)
---------------

Linear Triangle
+++++++++++++++
.. automodule:: morphic.interpolator
    :members: T11

Quadratic Triangle
++++++++++++++++++
.. automodule:: morphic.interpolator
    :members: T22

Cubic Triangle
++++++++++++++
.. automodule:: morphic.interpolator
    :members: T33

Quartic Triangle
++++++++++++++++
.. automodule:: morphic.interpolator
    :members: T44, T44d1, T44d2

