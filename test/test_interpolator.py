import unittest
import doctest

import numpy
from numpy import array
import numpy.testing

from morphic import interpolator

class Test(unittest.TestCase):
    """Unit tests for morphic interpolator."""

    def test_doctests(self):
        """Run interpolator doctests"""
        doctest.testmod(interpolator)
        
    def test_weights_get_functions(self):
        """Tests the preprocessing for weights function"""
        import numpy
        basisfn, dim = interpolator._get_basis_functions(['L1'], None)
        self.assertEqual(1, len(basisfn))
        self.assertEqual('L1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        
        basisfn, dim = interpolator._get_basis_functions(['L1'], None)
        self.assertEqual(1, len(basisfn))
        self.assertEqual('L1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        
        basisfn, dim = interpolator._get_basis_functions(['L1'], None)
        self.assertEqual(1, len(basisfn))
        self.assertEqual('L1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['L1', 'L3'], None)
        self.assertEqual(2, len(basisfn))
        self.assertEqual('L1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        self.assertEqual('L3', basisfn[1][0].__name__)
        self.assertEqual([1], basisfn[1][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['L1', 'T22'], None)
        self.assertEqual(2, len(basisfn))
        self.assertEqual('L1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        self.assertEqual('T22', basisfn[1][0].__name__)
        self.assertEqual([1, 2], basisfn[1][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['T44', 'L2'], None)
        self.assertEqual(2, len(basisfn))
        self.assertEqual('T44', basisfn[0][0].__name__)
        self.assertEqual([0, 1], basisfn[0][1])
        self.assertEqual('L2', basisfn[1][0].__name__)
        self.assertEqual([2], basisfn[1][1])
    
    def test_weights_get_functions_deriv(self):
        """Tests the preprocessing for weights function"""
        import numpy
        basisfn, dim = interpolator._get_basis_functions(['L1'],
            deriv=[0])
        self.assertEqual(1, len(basisfn))
        self.assertEqual('L1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        
        basisfn, dim = interpolator._get_basis_functions(['L1'],
            deriv=[1])
        self.assertEqual(1, len(basisfn))
        self.assertEqual('L1d1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['L1', 'L2', 'L3', 'L4'], deriv=[0, 0, 0, 0])
        self.assertEqual(4, len(basisfn))
        self.assertEqual('L1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        self.assertEqual('L2', basisfn[1][0].__name__)
        self.assertEqual([1], basisfn[1][1])
        self.assertEqual('L3', basisfn[2][0].__name__)
        self.assertEqual([2], basisfn[2][1])
        self.assertEqual('L4', basisfn[3][0].__name__)
        self.assertEqual([3], basisfn[3][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['L1', 'L2', 'L3', 'L4'], deriv=[1, 1, 1, 1])
        self.assertEqual(4, len(basisfn))
        self.assertEqual('L1d1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        self.assertEqual('L2d1', basisfn[1][0].__name__)
        self.assertEqual([1], basisfn[1][1])
        self.assertEqual('L3d1', basisfn[2][0].__name__)
        self.assertEqual([2], basisfn[2][1])
        self.assertEqual('L4d1', basisfn[3][0].__name__)
        self.assertEqual([3], basisfn[3][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['L1'], deriv=[2])
        self.assertEqual(1, len(basisfn))
        self.assertEqual('L1d1d1', basisfn[0][0].__name__)
        
        basisfn, dim = interpolator._get_basis_functions(
            ['H3'], deriv=[0])
        self.assertEqual(1, len(basisfn))
        self.assertEqual('H3', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['H3'], deriv=[1])
        self.assertEqual(1, len(basisfn))
        self.assertEqual('H3d1', basisfn[0][0].__name__)
        self.assertEqual([0], basisfn[0][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['T11', 'T22', 'T33', 'T44'],
            deriv=[0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(4, len(basisfn))
        self.assertEqual('T11', basisfn[0][0].__name__)
        self.assertEqual([0, 1], basisfn[0][1])
        self.assertEqual('T22', basisfn[1][0].__name__)
        self.assertEqual([2, 3], basisfn[1][1])
        self.assertEqual('T33', basisfn[2][0].__name__)
        self.assertEqual([4, 5], basisfn[2][1])
        self.assertEqual('T44', basisfn[3][0].__name__)
        self.assertEqual([6, 7], basisfn[3][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['T44', 'T44'],
            deriv=[1, 0, 0, 1])
        self.assertEqual(2, len(basisfn))
        self.assertEqual('T44d1', basisfn[0][0].__name__)
        self.assertEqual([0, 1], basisfn[0][1])
        self.assertEqual('T44d2', basisfn[1][0].__name__)
        self.assertEqual([2, 3], basisfn[1][1])
        
        basisfn, dim = interpolator._get_basis_functions(
            ['T44', 'L2'],
            deriv=[1, 0, 1])
        self.assertEqual(2, len(basisfn))
        self.assertEqual('T44d1', basisfn[0][0].__name__)
        self.assertEqual([0, 1], basisfn[0][1])
        self.assertEqual('L2d1', basisfn[1][0].__name__)
        self.assertEqual([2], basisfn[1][1])
        
    def test_weights_process_x(self):
        """Tests the preprocessing for weights function"""
        import numpy
        X = interpolator._process_x([0.1], 1)
        Xc = numpy.array([[0.1]])
        numpy.testing.assert_almost_equal(Xc, X)
        
        X = interpolator._process_x([0.1, 0.2, 0.3], 1)
        Xc = numpy.array([[0.1], [0.2], [0.3]])
        numpy.testing.assert_almost_equal(Xc, X)
        
        X = interpolator._process_x([[0.1], [0.2], [0.3]], 1)
        Xc = numpy.array([[0.1], [0.2], [0.3]])
        numpy.testing.assert_almost_equal(Xc, X)
        
        X = interpolator._process_x([[0.1, 0.2]], 2)
        Xc = numpy.array([[0.1, 0.2]])
        numpy.testing.assert_almost_equal(Xc, X)
        
        X = interpolator._process_x([[0.1, 0.2], [0.4, 0.5]], 2)
        Xc = numpy.array([[0.1, 0.2], [0.4, 0.5]])
        numpy.testing.assert_almost_equal(Xc, X)
        
        Xc = numpy.array([[0.1, 0.2], [0.4, 0.5]])
        X = interpolator._process_x(Xc, 2)
        numpy.testing.assert_almost_equal(Xc, X)
        
    def test_weights_values(self):
        """Tests the weights function"""
        import numpy
        Xi1 = numpy.array([[0.1], [0.3], [0.2]])
        Xi2 = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.2]])
        
        W = interpolator.weights(['L2'], Xi1)
        numpy.testing.assert_almost_equal(W, numpy.array([
                [0.72, 0.36, -0.08],
                [0.28, 0.84, -0.12],
                [0.48, 0.64, -0.12]]))
        
        W = interpolator.weights(['L3'], Xi1)
        numpy.testing.assert_almost_equal(W, numpy.array([
                [0.5355,  0.6885, -0.2835,  0.0595],
                [0.0385,  1.0395, -0.0945,  0.0165],
                [0.224,   1.008,  -0.288,   0.056 ]]))
                
        W = interpolator.weights(['H3'], Xi1)
        numpy.testing.assert_almost_equal(W, numpy.array([
                [ 0.972,  0.081,  0.028, -0.009],
                [ 0.784,  0.147,  0.216, -0.063],
                [ 0.896,  0.128,  0.104, -0.032]]))
        
        W = interpolator.weights(['L1', 'L1'], Xi2)
        numpy.testing.assert_almost_equal(W, numpy.array([
                [0.72, 0.08, 0.18, 0.02],
                [0.56, 0.14, 0.24, 0.06],
                [0.56, 0.24, 0.14, 0.06]]))
        
        W = interpolator.weights(['L1', 'L2'], Xi2)
        numpy.testing.assert_almost_equal(W, numpy.array([
                [0.432,  0.048,  0.576,  0.064, -0.108, -0.012],
                [0.224,  0.056,  0.672,  0.168, -0.096, -0.024],
                [0.336,  0.144,  0.448,  0.192, -0.084, -0.036]]))
        
        W = interpolator.weights(['L3', 'L2'], Xi2)
        numpy.testing.assert_almost_equal(W, numpy.array([
             [ 0.25704, 0.33048, -0.13608, 0.02856, 0.34272, 0.44064, -0.18144, 0.03808,
              -0.06426, -0.08262, 0.03402, -0.00714],
             [ 0.06272, 0.28224, -0.08064, 0.01568, 0.18816, 0.84672, -0.24192, 0.04704,
              -0.02688, -0.12096, 0.03456, -0.00672],
             [ 0.01848, 0.49896, -0.04536, 0.00792, 0.02464, 0.66528, -0.06048, 0.01056,
              -0.00462, -0.12474, 0.01134, -0.00198]]))
        
        W = interpolator.weights(['L3', 'L3'], Xi2)
        numpy.testing.assert_almost_equal(W, numpy.array([
             [  1.19952000e-01,  1.54224000e-01, -6.35040000e-02,  1.33280000e-02,
                5.39784000e-01,  6.94008000e-01, -2.85768000e-01,  5.99760000e-02,
               -1.54224000e-01, -1.98288000e-01,  8.16480000e-02, -1.71360000e-02,
                2.99880000e-02,  3.85560000e-02, -1.58760000e-02,  3.33200000e-03],
             [  8.62400000e-03,  3.88080000e-02, -1.10880000e-02,  2.15600000e-03,
                2.32848000e-01,  1.04781600e+00, -2.99376000e-01,  5.82120000e-02,
               -2.11680000e-02, -9.52560000e-02,  2.72160000e-02, -5.29200000e-03,
                3.69600000e-03,  1.66320000e-02, -4.75200000e-03,  9.24000000e-04],
             [  8.62400000e-03,  2.32848000e-01, -2.11680000e-02,  3.69600000e-03,
                3.88080000e-02,  1.04781600e+00, -9.52560000e-02,  1.66320000e-02,
               -1.10880000e-02, -2.99376000e-01,  2.72160000e-02, -4.75200000e-03,
                2.15600000e-03,  5.82120000e-02, -5.29200000e-03,  9.24000000e-04]]))
        
        W = interpolator.weights(['L4', 'L4'], Xi2)
        numpy.testing.assert_almost_equal(W, numpy.array([
             [  2.63577600e-02,  7.02873600e-02, -3.95366400e-02,  1.62201600e-02,
               -2.92864000e-03,  4.21724160e-01,  1.12459776e+00, -6.32586240e-01,
                2.59522560e-01, -4.68582400e-02, -1.05431040e-01, -2.81149440e-01,
                1.58146560e-01, -6.48806400e-02,  1.17145600e-02,  3.83385600e-02,
                1.02236160e-01, -5.75078400e-02,  2.35929600e-02, -4.25984000e-03,
               -6.58944000e-03, -1.75718400e-02,  9.88416000e-03, -4.05504000e-03,
                7.32160000e-04],
             [ -2.36544000e-03, -3.78470400e-02,  9.46176000e-03, -3.44064000e-03,
                5.91360000e-04,  5.67705600e-02,  9.08328960e-01, -2.27082240e-01,
                8.25753600e-02, -1.41926400e-02,  2.12889600e-02,  3.40623360e-01,
               -8.51558400e-02,  3.09657600e-02, -5.32224000e-03, -6.30784000e-03,
               -1.00925440e-01,  2.52313600e-02, -9.17504000e-03,  1.57696000e-03,
                1.01376000e-03,  1.62201600e-02, -4.05504000e-03,  1.47456000e-03,
               -2.53440000e-04],
             [ -2.36544000e-03,  5.67705600e-02,  2.12889600e-02, -6.30784000e-03,
                1.01376000e-03, -3.78470400e-02,  9.08328960e-01,  3.40623360e-01,
               -1.00925440e-01,  1.62201600e-02,  9.46176000e-03, -2.27082240e-01,
               -8.51558400e-02,  2.52313600e-02, -4.05504000e-03, -3.44064000e-03,
                8.25753600e-02,  3.09657600e-02, -9.17504000e-03,  1.47456000e-03,
                5.91360000e-04, -1.41926400e-02, -5.32224000e-03,  1.57696000e-03,
               -2.53440000e-04]]))
        
        W = interpolator.weights(['H3', 'H3'], Xi2)
        numpy.testing.assert_almost_equal(W, numpy.array([
             [  8.70912000e-01,   7.25760000e-02,   1.24416000e-01,   1.03680000e-02,
                2.50880000e-02,  -8.06400000e-03,   3.58400000e-03,  -1.15200000e-03,
                1.01088000e-01,   8.42400000e-03,  -3.11040000e-02,  -2.59200000e-03,
                2.91200000e-03,  -9.36000000e-04,  -8.96000000e-04,   2.88000000e-04],
             [  7.02464000e-01,   1.00352000e-01,   1.31712000e-01,   1.88160000e-02,
                8.15360000e-02,  -2.50880000e-02,   1.52880000e-02,  -4.70400000e-03,
                1.93536000e-01,   2.76480000e-02,  -5.64480000e-02,  -8.06400000e-03,
                2.24640000e-02,  -6.91200000e-03,  -6.55200000e-03,   2.01600000e-03],
             [  7.02464000e-01,   1.31712000e-01,   1.00352000e-01,   1.88160000e-02,
                1.93536000e-01,  -5.64480000e-02,   2.76480000e-02,  -8.06400000e-03,
                8.15360000e-02,   1.52880000e-02,  -2.50880000e-02,  -4.70400000e-03,
                2.24640000e-02,  -6.55200000e-03,  -6.91200000e-03,   2.01600000e-03]]))
        
        W = interpolator.weights(['H3', 'L1'], Xi2)
        numpy.testing.assert_almost_equal(W, numpy.array([
                 [ 0.7776,  0.0648,  0.0224, -0.0072,  0.1944,  0.0162,  0.0056, -0.0018],
                 [ 0.6272,  0.0896,  0.0728, -0.0224,  0.2688,  0.0384,  0.0312, -0.0096],
                 [ 0.6272,  0.1176,  0.1728, -0.0504,  0.1568,  0.0294,  0.0432, -0.0126]]))
        
        W = interpolator.weights(['T22'], Xi2)
        numpy.testing.assert_almost_equal(W, numpy.array([
                 [  2.80000000e-01,   2.80000000e-01,  -8.00000000e-02,
                    5.60000000e-01,   8.00000000e-02,  -1.20000000e-01],
                 [  0.00000000e+00,   4.00000000e-01,  -1.20000000e-01,
                    6.00000000e-01,   2.40000000e-01,  -1.20000000e-01],
                 [ -5.55111512e-17,   6.00000000e-01,  -1.20000000e-01,
                    4.00000000e-01,   2.40000000e-01,  -1.20000000e-01]]))
        
        ### TODO: test ['L1', 'H3'] and higher order lagrange
        ### interpolations, e.g., L2, L3, L4...
        
        # Derivatives
        W = interpolator.weights(['L4'], Xi1, deriv=[1])
        numpy.testing.assert_almost_equal(W, numpy.array([
                [-4.424     ,  4.84266667, -0.384     , -0.064     ,  0.02933333],
                [-0.38133333, -4.288     ,  5.952     , -1.51466667,  0.232     ],
                [-1.85866667, -1.57866667,  5.088     , -2.00533333,  0.35466667]]))
        
        W = interpolator.weights(['L3', 'L3'], Xi2, deriv=[1, 0])
        numpy.testing.assert_almost_equal(W, numpy.array([
               [ -8.59040000e-01,   1.09872000e+00,  -2.92320000e-01,
                  5.26400000e-02,  -3.86568000e+00,   4.94424000e+00,
                 -1.31544000e+00,   2.36880000e-01,   1.10448000e+00,
                 -1.41264000e+00,   3.75840000e-01,  -6.76800000e-02,
                 -2.14760000e-01,   2.74680000e-01,  -7.30800000e-02,
                  1.31600000e-02],
               [ -9.39400000e-02,   6.23700000e-02,   4.15800000e-02,
                 -1.00100000e-02,  -2.53638000e+00,   1.68399000e+00,
                  1.12266000e+00,  -2.70270000e-01,   2.30580000e-01,
                 -1.53090000e-01,  -1.02060000e-01,   2.45700000e-02,
                 -4.02600000e-02,   2.67300000e-02,   1.78200000e-02,
                 -4.29000000e-03],
               [ -2.94560000e-01,  -1.91520000e-01,   5.94720000e-01,
                 -1.08640000e-01,  -1.32552000e+00,  -8.61840000e-01,
                  2.67624000e+00,  -4.88880000e-01,   3.78720000e-01,
                  2.46240000e-01,  -7.64640000e-01,   1.39680000e-01,
                 -7.36400000e-02,  -4.78800000e-02,   1.48680000e-01,
                 -2.71600000e-02]]))
        
        W = interpolator.weights(['L3', 'L3'], Xi2, deriv=[1, 1])
        numpy.testing.assert_almost_equal(W, numpy.array([
               [  9.3574, -11.9682,   3.1842,  -0.5734,  -6.2127,   7.9461,
                 -2.1141,   0.3807,  -4.1418,   5.2974,  -1.4094,   0.2538,
                  0.9971,  -1.2753,   0.3393,  -0.0611],
               [  3.2086,  -2.1303,  -1.4202,   0.3419,   2.0862,  -1.3851,
                 -0.9234,   0.2223,  -6.4782,   4.3011,   2.8674,  -0.6903,
                  1.1834,  -0.7857,  -0.5238,   0.1261],
               [  3.2086,   2.0862,  -6.4782,   1.1834,  -2.1303,  -1.3851,
                  4.3011,  -0.7857,  -1.4202,  -0.9234,   2.8674,  -0.5238,
                  0.3419,   0.2223,  -0.6903,   0.1261]]))
        
        W = interpolator.weights(['H3', 'H3'], Xi2, deriv=[0, 1])
        numpy.testing.assert_almost_equal(W, numpy.array([
               [-0.93312, -0.07776,  0.31104,  0.02592, -0.02688,  0.00864,
                 0.00896, -0.00288,  0.93312,  0.07776, -0.27216, -0.02268,
                 0.02688, -0.00864, -0.00784,  0.00252],
               [-1.12896, -0.16128,  0.06272,  0.00896, -0.13104,  0.04032,
                 0.00728, -0.00224,  1.12896,  0.16128, -0.29568, -0.04224,
                 0.13104, -0.04032, -0.03432,  0.01056],
               [-0.75264, -0.14112,  0.25088,  0.04704, -0.20736,  0.06048,
                 0.06912, -0.02016,  0.75264,  0.14112, -0.21952, -0.04116,
                 0.20736, -0.06048, -0.06048,  0.01764]]))
                 
    
class TestBasisFunction(unittest.TestCase):
    """Unit tests for morphic interpolator."""
    
    def test_L1(self):
        x = numpy.array([0.13, 0.37, 0.669, 0.87])
        numpy.testing.assert_almost_equal(interpolator.L1(x),
            array([[ 0.87 ,  0.13 ],
                   [ 0.63 ,  0.37 ],
                   [ 0.331,  0.669],
                   [ 0.13 ,  0.87 ]]))
    
    def test_L1d1(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.L1d1(x),
            array([[-1.,  1.],
                   [-1.,  1.]]))
    
    def test_L1d1d1(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.L1d1d1(x),
            array([[ 0.,  0.],
                   [ 0.,  0.]]))
    
    def test_L2(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.L2(x),
            array([[ 0.6438,  0.4524, -0.0962],
                   [-0.1242,  0.7084,  0.4158]]))
    
    def test_L2d1(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.L2d1(x),
            array([[-2.48,  2.96, -0.48],
                   [ 0.08, -2.16,  2.08]]))
    
    def test_L3(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.L3(x),
            array([[ 0.4272135,  0.8194095, -0.3104595,  0.0638365],
                   [ 0.0467015, -0.2470545,  1.0440045,  0.1563485]]))
    
    def test_L3d1(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.L3d1(x),
            array([[-3.38815,  3.83445, -0.50445,  0.05815],
                   [ 0.35585, -1.63755, -0.79245,  2.07415]]))
    
    def test_L4(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.L4(x),
            array([[ 0.25545984,  1.10699264, -0.53853696,  0.21425664, -0.03817216],
                   [-0.00688896,  0.04080384, -0.11787776,  1.06089984,  0.02306304]]))
    
    def test_L4d1(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.L4d1(x),
            array([[-3.524928  ,  2.46557867,  1.832832  , -0.962688  ,  0.18920533],
                   [-0.35325867,  2.06690133, -5.761152  ,  2.73463467,  1.31287467]]))
    
    def test_H3(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.H3(x),
            array([[ 0.953694,  0.098397,  0.046306, -0.014703],
                   [ 0.134366,  0.040733,  0.865634, -0.136367]]))
        
    def test_H3d1(self):
        x = numpy.array([0.13, 0.77])
        numpy.testing.assert_almost_equal(interpolator.H3d1(x),
            array([[-0.6786,  0.5307,  0.6786, -0.2093],
                   [-1.0626, -0.3013,  1.0626,  0.2387]]))
        
    def test_T11(self):
        x = numpy.array([[0.13, 0.23], [0.77, 0.06]])
        numpy.testing.assert_almost_equal(interpolator.T11(x),
            array([[ 0.64,  0.13,  0.23],
                   [ 0.17,  0.77,  0.06]]))
    
    
    def test_T22(self):
        x = numpy.array([[0.13, 0.23], [0.77, 0.06]])
        numpy.testing.assert_almost_equal(interpolator.T22(x),
            array([[ 0.1792,  0.3328, -0.0962,  0.5888,  0.1196, -0.1242],
                   [-0.1122,  0.5236,  0.4158,  0.0408,  0.1848, -0.0528]]))
    
    def test_T33(self):
        x = numpy.array([[0.13, 0.23], [0.77, 0.06]])
        numpy.testing.assert_almost_equal(interpolator.T33(x),
            array([[-0.023552 ,  0.344448 , -0.228384 ,  0.0638365,  0.609408 ,
                     0.516672 , -0.0820755, -0.205344 , -0.0417105,  0.0467015],
                   [ 0.0620585, -0.2886345,  0.7716555,  0.1563485, -0.022491 ,
                     0.212058 ,  0.272349 , -0.037638 , -0.170478 ,  0.044772 ]]))
    
    def test_T44(self):
        x = numpy.array([[0.13, 0.23], [0.77, 0.06]])
        numpy.testing.assert_almost_equal(interpolator.T44(x),
            array([[-0.04100096,  0.19382272, -0.24920064,  0.15761408, -0.03817216,
                     0.34291712,  0.95526912, -0.29392896,  0.05664256, -0.07348224,
                    -0.04898816,  0.00459264,  0.03391488,  0.00688896, -0.00688896],
                   [-0.02776576,  0.14744576, -0.34850816,  0.78414336,  0.02306304,
                     0.01148928, -0.08042496,  0.52276224,  0.27675648,  0.00992256,
                    -0.19100928, -0.29213184,  0.03638272,  0.16479232, -0.03691776]]))
        
    def test_T44d1(self):
        x = numpy.array([[0.13, 0.23], [0.77, 0.06]])
        numpy.testing.assert_almost_equal(interpolator.T44d1(x),
            array([[  8.93226667e-02,  -6.93333333e-01,   1.18809600e+00,
                     -7.73290667e-01,   1.89205333e-01,  -3.86449067e+00,
                      3.40620800e+00,   6.47680000e-01,  -1.89397333e-01,
                      3.03232000e-01,  -3.00288000e-01,  -2.94400000e-03,
                     -5.29920000e-02,   5.29920000e-02,   0.00000000e+00],
                   [ -3.15754667e-01,   1.61403733e+00,  -3.42912000e+00,
                      8.17962667e-01,   1.31287467e+00,   1.10848000e-01,
                     -6.36672000e-01,  -1.39084800e+00,   1.91667200e+00,
                      6.56640000e-02,   8.75520000e-01,  -9.41184000e-01,
                     -2.14016000e-01,   2.14016000e-01,   0.00000000e+00]]))
        
    def test_T44d2(self):
        x = numpy.array([[0.13, 0.23], [0.77, 0.06]])
        numpy.testing.assert_almost_equal(interpolator.T44d2(x),
            array([[ 0.08932267, -2.18427733,  1.028352  , -0.246272  ,  0.        ,
                    -2.37354667,  0.211328  , -0.818688  ,  0.246272  ,  3.657856  ,
                     2.31296   , -0.209664  , -1.72689067, -0.34001067,  0.35325867],
                   [-0.31575467,  1.42254933, -2.306304  , -4.612608  ,  0.        ,
                     0.302336  , -1.87264   ,  5.637632  ,  4.612608  ,  0.178816  ,
                    -1.054592  , -3.331328  ,  0.11818667,  1.50468267, -0.283584  ]]))

        
if __name__ == "__main__":
    unittest.main()
