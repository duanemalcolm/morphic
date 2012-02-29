import doctest
import os
import pickle
import sys
import unittest
import numpy.testing as npt

sys.path.append('..')
import morphic
        
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
    
    def test_create_mesh(self):
        example = 'create_mesh'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xs, Ts = locals()['mesh'].get_surfaces()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_equal(Xn, data['Xn'])
        npt.assert_equal(Xs, data['Xs'])
        npt.assert_equal(Ts, data['Ts'])
    
    def test_load_mesh1(self):
        example = 'create_mesh'
        mesh = morphic.Mesh('data/'+example+'.mesh')
        Xn = mesh.get_nodes()
        Xs, Ts = mesh.get_surfaces()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_equal(Xn, data['Xn'])
        npt.assert_equal(Xs, data['Xs'])
        npt.assert_equal(Ts, data['Ts'])
        
    def test_load_mesh2(self):
        example = 'create_mesh'
        mesh = morphic.Mesh()
        mesh.load('data/'+example+'.mesh')
        Xn = mesh.get_nodes()
        Xs, Ts = mesh.get_surfaces()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_equal(Xn, data['Xn'])
        npt.assert_equal(Xs, data['Xs'])
        npt.assert_equal(Ts, data['Ts'])
        
    def test_example_2d_fit_lse(self):
        example = 'example_2d_fit_lse'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xs, Ts = locals()['mesh'].get_surfaces()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_equal(Xn, data['Xn'])
        npt.assert_equal(Xs, data['Xs'])
        npt.assert_equal(Ts, data['Ts'])
        


if __name__ == "__main__":
    unittest.main()
