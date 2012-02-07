import doctest
import pickle
import sys
import unittest
import numpy.testing as npt

sys.path.append('../examples/')
        
class TestExamples(unittest.TestCase):
    """Unit tests for morphic Node superclass."""

    def test_example_1d_linear(self):
        example = 'example_1d_linear'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xl = locals()['mesh'].get_lines()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_equal(Xn, data['Xn'])
        npt.assert_equal(Xl, data['Xl'])
        
    def test_example_1d_quadratic(self):
        example = 'example_1d_quadratic'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xl = locals()['mesh'].get_lines()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_equal(Xn, data['Xn'])
        npt.assert_equal(Xl, data['Xl'])
        
    def test_example_1d_cubic(self):
        example = 'example_1d_cubic'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xl = locals()['mesh'].get_lines()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_equal(Xn, data['Xn'])
        npt.assert_equal(Xl, data['Xl'])
        
    def test_example_1d_quartic(self):
        example = 'example_1d_quartic'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xl = locals()['mesh'].get_lines()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_equal(Xn, data['Xn'])
        npt.assert_equal(Xl, data['Xl'])
        


if __name__ == "__main__":
    unittest.main()
