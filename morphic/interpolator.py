import numpy
import numpy.linalg

def weights(basis, X, deriv=None):
    """
    Calculates the interpolant value or derivative weights for points X.
    
    :param basis: interpolation function in each direction, eg,
        ``['L1', 'L1']`` for bilinear.
    :type basis: list of strings
    :param X: locations to calculate interpolant weights
    :type X: list or numpy array (npoints, ndims)
    :param deriv: derivative in each dimension, e.g., ``deriv=[1, 1]``
    :type deriv: list of integers
    :return: basis weights (ndims)
    :rtype: numpy array, size: (npoints, nweights)
    
    >>> import numpy
    >>> x = numpy.array([[0.13, 0.23], [0.77, 0.06]])
    >>> weights(['L1', 'L2'], x, deriv=[0, 1])
    array([[-1.8096, -0.2704,  1.8792,  0.2808, -0.0696, -0.0104],
           [-0.6348, -2.1252,  0.8096,  2.7104, -0.1748, -0.5852]])
    
    """
    
    basis_functions, dimensions = _get_basis_functions(basis, deriv)
    X = _process_x(X, dimensions)
    
    W = []
    for bf in basis_functions:
        if bf[0].__name__[0] == 'T':
            W.append(bf[0](X[:, bf[1]]))
        else:
            W.append(bf[0](X[:, bf[1]])[0])
    
    BPInd = _get_basis_product_indices(basis, dimensions, W)             
    
    if BPInd is None:
        return W[0]
        
    WW = numpy.zeros((X.shape[0], len(BPInd)))
    if dimensions == 3:
        for ind, ii in enumerate(BPInd):
            WW[:, ind] = W[0][:, ii[0]] * W[1][:, ii[1]] * W[2][:, ii[2]]
    else:
        for ind, ii in enumerate(BPInd):
            WW[:, ind] = W[0][:, ii[0]] * W[1][:, ii[1]]
    
    return WW


def _get_basis_product_indices(basis, dimensions, W):
    """
    Returns the indicies for the product between the weights for each
    interpolant for basis functions.
    """
    BPInd = None
    if dimensions == 1:
        return None
    elif dimensions == 2:
        if basis[0][0] == 'T':
            return None
        elif len(basis) == 2:
            if (basis[0][0] == 'L' and basis[1][0] == 'L') or \
               (basis[0][0] == 'L' and basis[1][0] == 'H') or \
               (basis[0][0] == 'H' and basis[1][0] == 'L'):
                BPInd = []
                for ind1 in range(W[1].shape[1]):
                    for ind0 in range(W[0].shape[1]):
                        BPInd.append([ind0, ind1])
            elif basis == ['H3', 'H3']:
                BPInd = [[0, 0], [1, 0], [0, 1], [1, 1],
                      [2, 0], [3, 0], [2, 1], [3, 1],
                      [0, 2], [1, 2], [0, 3], [1, 3],
                      [2, 2], [3, 2], [2, 3], [3, 3]]
            else:
                raise ValueError('Basis combination not supported')
    elif dimensions == 3:
        if len(basis) == 3:
            if (basis[0][0] == 'L' and basis[1][0] == 'L' and basis[2][0] == 'L'):
                BPInd = []
                for ind2 in range(W[2].shape[1]):
                    for ind1 in range(W[1].shape[1]):
                        for ind0 in range(W[0].shape[1]):
                            BPInd.append([ind0, ind1, ind2])
            elif basis == ['H3', 'H3', 'H3']:
                BPInd = [
                    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
                    [2, 0, 0], [3, 0, 0], [2, 1, 0], [3, 1, 0],
                    [2, 0, 1], [3, 0, 1], [2, 1, 1], [3, 1, 1],
                    [0, 2, 0], [1, 2, 0], [0, 3, 0], [1, 3, 0],
                    [0, 2, 1], [1, 2, 1], [0, 3, 1], [1, 3, 1],
                    [2, 2, 0], [3, 2, 0], [2, 3, 0], [3, 3, 0],
                    [2, 2, 1], [3, 2, 1], [2, 3, 1], [3, 3, 1],
                    
                    [0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2],
                    [0, 0, 3], [1, 0, 3], [0, 1, 3], [1, 1, 3],
                    [2, 0, 2], [3, 0, 2], [2, 1, 2], [3, 1, 2],
                    [2, 0, 3], [3, 0, 3], [2, 1, 3], [3, 1, 3],
                    [0, 2, 2], [1, 2, 2], [0, 3, 2], [1, 3, 2],
                    [0, 2, 3], [1, 2, 3], [0, 3, 3], [1, 3, 3],
                    [2, 2, 2], [3, 2, 2], [2, 3, 2], [3, 3, 2],
                    [2, 2, 3], [3, 2, 3], [2, 3, 3], [3, 3, 3]]
            else:
                raise ValueError('Basis combination not supported')
        else:
            raise ValueError('Basis combination not supported')
    else:
        raise ValueError('%d dimensions not supported' % (len(basis)))
        
    return BPInd
    
    
def _get_basis_functions(basis, deriv):
    """
    Returns a list of interpolation function for the interpolation
    definition and derivatives specified by the user. Also returns the
    number of dimensions as defined in the basis parameter.
    """
    # List of basis functions
    bsfn_list = {
        'L1': [L1, L1d1, L1d1d1],
        'L2': [L2, L2d1],
        'L3': [L3, L3d1],
        'L4': [L4, L4d1],
        'H3': [H3, H3d1, H3d1d1],
        'T11': [T11],
        'T22': [T22],
        'T33': [T33, T33d1, T33d2],
        'T44': [T44, T44d1, T44d2]}
    
    # Set the index of the basis function in BFn from the deriv input
    di = []
    if deriv == None:
        for bs in basis:
            di.append(0)
    else:
        ind = 0
        for bs in basis:
            if bs[0] == 'T':
                if deriv[ind:ind+2] == [0, 0]:
                    di.append(0)
                elif deriv[ind:ind+2] == [1, 0]:
                    di.append(1)
                elif deriv[ind:ind+2] == [0, 1]:
                    di.append(2)
                else:
                    raise ValueError(
                        'Derivative (%d) for %s basis not implemented' %
                        (ind, bs))
                ind += 2
            else:
                di.append(deriv[ind])
                ind += 1
    
    # Set the basis functions pointers and index in X for each basis in
    # the basis input 
    dimensions = 0
    basis_functions = []
    for ind, bs in enumerate(basis):
        if bs[0] == 'T':
            if bs in bsfn_list.keys():
                basis_functions.append([bsfn_list[bs][di[ind]],
                    [dimensions, dimensions + 1]])
            dimensions += 2
        else:
            if bs in bsfn_list.keys():
                basis_functions.append([bsfn_list[bs][di[ind]],
                    [dimensions]])
            dimensions += 1
    
    return basis_functions, dimensions


def _process_x(X, dimensions):
    """
    Converts the X parameter to the correct numpy array for the
    interpolation functions. The return numpy array should be size
    (npoints, ndims).
    """
    # Converting X to a numpy array if the input is a list
    if isinstance(X, list):
        if isinstance(X[0], list):
            X = numpy.array([x for x in X])
        else:
            if dimensions == 1:
                X = numpy.array([[x for x in X]]).T
            else:
                X = numpy.array([x for x in X])
    
    if X.shape[1] != dimensions:
        raise ValueError(
            'X dimensions does not match the number of basis')
    
    return X

# Lagrange basis functions
def L1(x):
    """
    Linear lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array (npoints, 2)
    """
    return numpy.array([1. - x, x]).T

def L1d1(x):
    """
    First derivative for the linear lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array (npoints, 2)
    """
    W = numpy.ones((x.shape[0], 2))
    W[:, 0] -= 2
    return numpy.array([W])

def L1d1d1(x):
    """
    Second derivative for the linear lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array (npoints, 2)
    """
    return numpy.zeros((x.shape[0], 2))

def L2(x):
    """
    Quadratic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 3)
    """
    L1, L2 = 1-x, x
    Phi = numpy.array([
        L1 * (2.0 * L1 - 1),
        4.0 * L1 * L2,
        L2 * (2.0 * L2 - 1)])
    return Phi.T

def L2d1(x):
    """
    First derivative of the quadratic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 3)
    """
    L1 = 1-x
    return numpy.array([
        1.0 - 4.0 * L1,
        4.0 * L1 - 4.0 * x,
        4.0 * x - 1.]).T

# .. todo: L2dxdx

def L3(x):
    """
    Cubic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 4)
    """
    L1, L2 = 1-x, x
    sc = 9./2.
    return numpy.array([
        0.5*L1*(3*L1-1)*(3*L1-2),
        sc*L1*L2*(3*L1-1),
        sc*L1*L2*(3*L2-1),
        0.5*L2*(3*L2-1)*(3*L2-2)]).T

def L3d1(x):
    """
    First derivative of the cubic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 4)
    """
    L1 = x*x
    return numpy.array([
        -(27.*L1-36.*x+11.)/2.,
        (81.*L1-90.*x+18.)/2.,
        -(81.*L1-72.*x+9.)/2.,
        (27.*L1-18.*x+2.)/2.]).T

# .. todo: L3dxdx

def L4(x):
    """
    Quartic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 5)
    """
    sc = 1/3.
    x2 = x*x
    x3 = x2*x
    x4 = x3*x
    return numpy.array([
        sc*(32*x4-80*x3+70*x2-25*x+3),
        sc*(-128*x4+288*x3-208*x2+48*x),
        sc*(192*x4-384*x3+228*x2-36*x),
        sc*(-128*x4+224*x3-112*x2+16*x),
        sc*(32*x4-48*x3+22*x2-3*x)]).T

def L4d1(x):
    """
    First derivative of the quartic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 5)
    """
    sc = 1/3.
    x2 = x*x
    x3 = x2*x
    return numpy.array([ \
        sc*(128*x3-240*x2+140*x-25), \
        sc*(-512*x3+864*x2-416*x+48), \
        sc*(768*x3-1152*x2+456*x-36), \
        sc*(-512*x3+672*x2-224*x+16), \
        sc*(128*x3-144*x2+44*x-3)]).T

# .. todo: L4d2

# Hemite basis functions
def H3(x):
    """
    Cubic-Hermite basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 4)
    """
    x2 = x*x
    Phi = numpy.array([ \
        1-3*x2+2*x*x2,
        x*(x-1)*(x-1),
        x2*(3-2*x),
        x2*(x-1)])
    return Phi.T
    
def H3d1(x):
    """
    First derivative of the cubic-Hermite basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 4)
    """
    x2 = x*x
    Phi = numpy.array([ \
        6*x*(x-1),
        3*x2-4*x+1,
        6*x*(1-x),
        x*(3*x-2)])
    return Phi.T
    
def H3d1d1(x):
    """
    First derivative of the cubic-Hermite basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 4)
    """
    Phi = numpy.array([ \
        12*x-6,
        6*x-4,
        6-12*x,
        6*x-2])
    return Phi.T
    
    
# Triangle Elements
def T11(x): # Linear-Linear
    """
    Linear lagrange triangle element.
    
    :param x: points to interpolate 0<=x<=1, x1+x2<=1
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 3)
    """
    L1, L2, L3 = 1-x[:, 0]-x[:, 1], x[:, 0], x[:, 1]
    return numpy.array([L1, L2, L3]).T


def T22(x): # Quadratic-Quadratic
    """
    Quadratic lagrange triangle element.
    
    :param x: points to interpolate 0<=x<=1, x1+x2<=1
    :type x: numpy array (npoints, 2)
    :return: basis weights
    :rtype: numpy array(npoints, 6)
    """
    L1, L2, L3 = 1-x[:, 0]-x[:, 1], x[:, 0], x[:, 1]
    Phi = numpy.array([ \
        L1*(2.0*L1-1.0), 4.0*L1*L2, L2*(2.0*L2-1.0), \
        4.0*L1*L3, 4.0*L2*L3, L3*(2.0*L3-1.0)])
    return Phi.T


def T33(x): # Cubic-Cubic
    """
    Cubic lagrange triangle element.
    
    :param x: points to interpolate 0<=x<=1, x1+x2<=1
    :type x: numpy array (npoints, 2)
    :return: basis weights
    :rtype: numpy array(npoints, 10)
    """
    L1, L2, L3 = 1-x[:, 0]-x[:, 1], x[:, 0], x[:, 1]
    sc = 9./2.
    Phi = numpy.array([ \
        0.5*L1*(3*L1-1)*(3*L1-2), sc*L1*L2*(3*L1-1), sc*L1*L2*(3*L2-1), \
        0.5*L2*(3*L2-1)*(3*L2-2), sc*L1*L3*(3*L1-1), 27*L1*L2*L3, \
        sc*L2*L3*(3*L2-1), sc*L1*L3*(3*L3-1), sc*L2*L3*(3*L3-1), \
        0.5*L3*(3*L3-1)*(3*L3-2)])
    return Phi.T

def T33d1(x): # Cubic-Cubic - from Ju Zhang
    """
    First derivative in dimension 1 for the cubic lagrange triangle
    element.
    
    :param x: points to interpolate 0<=x<=1, x1+x2<=1
    :type x: numpy array(npoints, 2)
    :return: basis weights
    :rtype: numpy array(npoints, 10)
    """
    L1, L2, L3 = 1-x[:, 0]-x[:, 1], x[:, 0], x[:, 1]
    v0 = [ 0.0, 0.0]
    v1 = [ 1.0, 0.0]
    v2 = [ 0.0, 1.0]
    b = numpy.array( [ v1[1] - v2[1],\
                 v2[1] - v0[1],\
                 v0[1] - v1[1] ] )
    D = 0.5 * numpy.linalg.det( [ [1.0, v0[0], v0[1]],\
                [1.0, v1[0], v1[1]],\
                [1.0, v2[0], v2[1]] ] )
    D2 = 2.0 * D             
    D16 = 16.0*D**3.0
    D916 = -9.0 / D16
    D27 = 27.0 / (8.0*D**3.0)
    D4 = D * 4.0
    D38 = -3.0 / (8.0*D**3.0)
    D98 = -9.0 / (8.0*D**3.0)
    
    Phi = numpy.array([  (b[0]*(D2 - 3.0*L1)*(D4 - 3.0*L1) - 3.0*b[0]*L1*(D4 - 3.0*L1) - 3.0*b[0]*L1*(D2 - 3.0*L1) )/ D16,\
                D916*b[0]*L2*(D2 - 3.0*L1) + D916*b[1]*L1*(D2 - 3.0*L1) + 0.5*D27*b[0]*L1*L2,\
                D916*b[0]*L2*(D2 - 3.0*L2) + D916*b[1]*L1*(D2 - 3.0*L2) + 0.5*D27*b[1]*L1*L2,\
               (b[1]*(D2 - 3.0*L2)*(D4 - 3.0*L2) - 3.0*b[1]*L2*(D4 - 3.0*L2) - 3.0*b[1]*L2*(D2 - 3.0*L2) )/ D16,\
                D916*b[0]*L3*(D2 - 3.0*L1) + D916*b[2]*L1*(D2 - 3.0*L1) + 0.5*D27*b[0]*L1*L3,\
                D27*( b[0]*L2*L3 + b[1]*L1*L3 + b[2]*L1*L2 ),\
                D916*b[1]*L3*(D2 - 3.0*L2) + D916*b[2]*L2*(D2 - 3.0*L2) + 0.5*D27*b[1]*L2*L3,\
                D916*b[0]*L3*(D2 - 3.0*L3) + D916*b[2]*L1*(D2 - 3.0*L3) + 0.5*D27*b[2]*L1*L3,\
                D916*b[1]*L3*(D2 - 3.0*L3) + D916*b[2]*L2*(D2 - 3.0*L3) + 0.5*D27*b[2]*L2*L3,\
               (b[2]*(D2 - 3.0*L3)*(D4 - 3.0*L3) - 3.0*b[2]*L3*(D4 - 3.0*L3) - 3.0*b[2]*L3*(D2 - 3.0*L3) )/ D16 ] )
    
    return Phi.T

def T33d2(x): # Cubic-Cubic - from Ju Zhang
    """
    First derivative in dimension 2 for the cubic lagrange triangle
    element.
    
    :param x: points to interpolate 0<=x<=1, x1+x2<=1
    :type x: numpy array(npoints, 2)
    :return: basis weights
    :rtype: numpy array(npoints, 10)
    """
    L1, L2, L3 = 1-x[:, 0]-x[:, 1], x[:, 0], x[:, 1]
    v0 = [ 0.0, 0.0]
    v1 = [ 1.0, 0.0]
    v2 = [ 0.0, 1.0]
    c = numpy.array( [ v2[0] - v1[0],\
                 v0[0] - v2[0],\
                 v1[0] - v0[0] ] )
    D = 0.5 * numpy.linalg.det( [ [1.0, v0[0], v0[1]],\
            [1.0, v1[0], v1[1]],\
            [1.0, v2[0], v2[1]] ] )
    D2 = 2.0 * D             
    D16 = 16.0*D**3.0
    D916 = -9.0 / D16
    D27 = 27.0 / (8.0*D**3.0)
    D4 = D * 4.0
    D38 = -3.0 / (8.0*D**3.0)
    D98 = -9.0 / (8.0*D**3.0)
    Phi = numpy.array([  (c[0]*(D2 - 3.0*L1)*(D4 - 3.0*L1) - 3.0*c[0]*L1*(D4 - 3.0*L1) - 3.0*c[0]*L1*(D2 - 3.0*L1) )/ D16,\
					    D916*c[0]*L2*(D2 - 3.0*L1) + D916*c[1]*L1*(D2 - 3.0*L1) + 0.5*D27*c[0]*L1*L2,\
					    D916*c[0]*L2*(D2 - 3.0*L2) + D916*c[1]*L1*(D2 - 3.0*L2) + 0.5*D27*c[1]*L1*L2,\
					   (c[1]*(D2 - 3.0*L2)*(D4 - 3.0*L2) - 3.0*c[1]*L2*(D4 - 3.0*L2) - 3.0*c[1]*L2*(D2 - 3.0*L2) )/ D16,\
						D916*c[0]*L3*(D2 - 3.0*L1) + D916*c[2]*L1*(D2 - 3.0*L1) + 0.5*D27*c[0]*L1*L3,\
						D27*( c[0]*L2*L3 + c[1]*L1*L3 + c[2]*L1*L2 ),\
						D916*c[1]*L3*(D2 - 3.0*L2) + D916*c[2]*L2*(D2 - 3.0*L2) + 0.5*D27*c[1]*L2*L3,\
						D916*c[0]*L3*(D2 - 3.0*L3) + D916*c[2]*L1*(D2 - 3.0*L3) + 0.5*D27*c[2]*L1*L3,\
						D916*c[1]*L3*(D2 - 3.0*L3) + D916*c[2]*L2*(D2 - 3.0*L3) + 0.5*D27*c[2]*L2*L3,\
					   (c[2]*(D2 - 3.0*L3)*(D4 - 3.0*L3) - 3.0*c[2]*L3*(D4 - 3.0*L3) - 3.0*c[2]*L3*(D2 - 3.0*L3) )/ D16 ] )

    return Phi.T
                 
                 

def T44(x): # Quartic-Quartic
    """
    Quartic lagrange triangle element.
    
    :param x: points to interpolate 0<=x<=1, x1+x2<=1
    :type x: numpy array(npoints, 2)
    :return: basis weights
    :rtype: numpy array(npoints, 15)
    """
    L1, L2, L3 = 1-x[:, 0]-x[:, 1], x[:, 0], x[:, 1]
    Phi = numpy.array([ \
        (32/3.)*(L1-0.75)*(L1-0.5)*(L1-0.25)*L1, \
        (128/3.)*(L1-0.5)*(L1-0.25)*L1*L2, \
        64*(L1-0.25)*L1*(L2-0.25)*L2, \
        (128/3.)*L1*(L2-0.5)*(L2-0.25)*L2, \
        (32/3.)*(L2-0.75)*(L2-0.5)*(L2-0.25)*L2, \
        (128/3.)*(L1-0.5)*(L1-0.25)*L1*L3, \
        128*(L1-0.25)*L1*L2*L3, \
        128*L1*(L2-0.25)*L2*L3, \
        (128/3.)*(L2-0.5)*(L2-0.25)*L2*L3, \
        64*(L1-0.25)*L1*(L3-0.25)*L3, \
        128*L1*L2*(L3-0.25)*L3, \
        64*(L2-0.25)*L2*(L3-0.25)*L3, \
        (128/3.)*L1*(L3-0.5)*(L3-0.25)*L3, \
        (128/3.)*L2*(L3-0.5)*(L3-0.25)*L3, \
        (32/3.)*(L3-0.75)*(L3-0.5)*(L3-0.25)*L3 \
        ])
    return Phi.T
    
def T44d1(x): # Quartic-Quartic
    """
    First derivative in dimension 1 for the quartic lagrange triangle
    element.
    
    :param x: points to interpolate 0<=x<=1, x1+x2<=1
    :type x: numpy array(npoints, 2)
    :return: basis weights
    :rtype: numpy array(npoints, 15)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    x1s = x1*x1
    x1c = x1s*x1
    x2s = x2*x2
    x2c = x2s*x2
    dPhi = numpy.array([
        ((8*x2+8*x1-5)*(16*x2s+(32*x1-20)*x2+16*x1s-20*x1+5))/3, \
        -(16*(8*x2c+(48*x1-18)*x2s+(72*x1s-72*x1+13)*x2+32*x1c-54*x1s+26*x1-3))/3, \
        4*(x2+2*x1-1)*((32*x1-4)*x2+32*x1s-32*x1+3), \
        -(16*((24*x1s-12*x1+1)*x2+32*x1c-42*x1s+14*x1-1))/3, \
        ((8*x1-3)*(16*x1s-12*x1+1))/3, \
        -(16*x2*(24*x2s+(48*x1-36)*x2+24*x1s-36*x1+13))/3, \
        32*x2*(4*x2s+(16*x1-7)*x2+12*x1s-14*x1+3), \
        -32*x2*((8*x1-1)*x2+12*x1s-10*x1+1), \
        (16*(24*x1s-12*x1+1)*x2)/3, \
        4*x2*(4*x2-1)*(8*x2+8*x1-7), \
        -32*x2*(x2+2*x1-1)*(4*x2-1), \
        4*(8*x1-1)*x2*(4*x2-1), \
        -(16*x2*(2*x2-1)*(4*x2-1))/3, \
        (16*x2*(2*x2-1)*(4*x2-1))/3, \
        0*x1])
    return dPhi.T
    
def T44d2(x): # Quartic-Quartic
    """
    First derivative in dimension 2 for the quartic lagrange triangle
    element.
    
    :param x: points to interpolate 0<=x1<=1, 0<=x2<=1, x1+x2<=1
    :type x: numpy array(npoints, 2)
    :return: basis weights
    :rtype: numpy array(npoints, 15)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    x1s = x1*x1
    x1c = x1s*x1
    x2s = x2*x2
    x2c = x2s*x2
    dPhi = numpy.array([
        ((8*x2+8*x1-5)*(16*x2s+(32*x1-20)*x2+16*x1s-20*x1+5))/3, \
        -(16*x1*(24*x2s+(48*x1-36)*x2+24*x1s-36*x1+13))/3, \
        4*x1*(4*x1-1)*(8*x2+8*x1-7), \
        -(16*x1*(2*x1-1)*(4*x1-1))/3, \
        0*x1, \
        -(16*(32*x2c+(72*x1-54)*x2s+(48*x1s-72*x1+26)*x2+8*x1c-18*x1s+13*x1-3))/3, \
        32*x1*(12*x2s+(16*x1-14)*x2+4*x1s-7*x1+3), \
        -32*x1*(4*x1-1)*(2*x2+x1-1), \
        (16*x1*(2*x1-1)*(4*x1-1))/3, \
        4*(2*x2+x1-1)*(32*x2s+(32*x1-32)*x2-4*x1+3), \
        -32*x1*(12*x2s+(8*x1-10)*x2-x1+1), \
        4*x1*(4*x1-1)*(8*x2-1), \
        -(16*(32*x2c+(24*x1-42)*x2s+(14-12*x1)*x2+x1-1))/3, \
        (16*x1*(24*x2s-12*x2+1))/3, \
        ((8*x2-3)*(16*x2s-12*x2+1))/3])
    return dPhi.T
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
