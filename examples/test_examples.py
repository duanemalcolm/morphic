import doctest
import os
import pickle
import sys
import unittest
import numpy.testing as npt

sys.path.append('..')
import morphic
import random

        
class TestExamples(unittest.TestCase):
    """Unit tests for morphic Node superclass."""

    def test_example_1d_linear(self):
        example = 'example_1d_linear'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xl = locals()['mesh'].get_lines()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_almost_equal(Xn, data['Xn'])
        npt.assert_almost_equal(Xl, data['Xl'])
        
    def test_example_1d_quadratic(self):
        example = 'example_1d_quadratic'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xl = locals()['mesh'].get_lines()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_almost_equal(Xn, data['Xn'])
        npt.assert_almost_equal(Xl, data['Xl'])
        
    def test_example_1d_cubic(self):
        example = 'example_1d_cubic'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xl = locals()['mesh'].get_lines()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_almost_equal(Xn, data['Xn'])
        npt.assert_almost_equal(Xl, data['Xl'])
        
    def test_example_1d_quartic(self):
        example = 'example_1d_quartic'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xl = locals()['mesh'].get_lines()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_almost_equal(Xn, data['Xn'])
        npt.assert_almost_equal(Xl, data['Xl'])
    
    def test_create_mesh(self):
        example = 'create_mesh'
        execfile('../examples/'+example+'.py', globals(), locals())
        Xn = locals()['mesh'].get_nodes()
        Xs, Ts = locals()['mesh'].get_surfaces()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_almost_equal(Xn, data['Xn'])
        npt.assert_almost_equal(Xs, data['Xs'])
        npt.assert_almost_equal(Ts, data['Ts'])
    
    def test_load_mesh1(self):
        example = 'create_mesh'
        mesh = morphic.Mesh('data/'+example+'.mesh')
        Xn = mesh.get_nodes()
        Xs, Ts = mesh.get_surfaces()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_almost_equal(Xn, data['Xn'])
        npt.assert_almost_equal(Xs, data['Xs'])
        npt.assert_almost_equal(Ts, data['Ts'])
        
    def test_load_mesh2(self):
        example = 'create_mesh'
        mesh = morphic.Mesh()
        mesh.load('data/'+example+'.mesh')
        Xn = mesh.get_nodes()
        Xs, Ts = mesh.get_surfaces()
        
        data = pickle.load(open('data/'+example+'.pkl', 'r'))
        npt.assert_almost_equal(Xn, data['Xn'])
        npt.assert_almost_equal(Xs, data['Xs'])
        npt.assert_almost_equal(Ts, data['Ts'])
        
    def test_pca_mesh(self):
        sys.argv = ['', 'test']
        execfile('../examples/tutorial_1d_pca_mesh.py', globals(), locals())
        
        testmesh = locals()['pcamesh']
        
        testdata = pickle.load(open('data/pca_node_values.pkl', 'r'))
        testmesh.nodes['weights'].values[1:] *= 0
        testmesh.update_pca_nodes()
        npt.assert_almost_equal(testmesh.get_nodes(), testdata['Xn0'])
        testmesh.nodes['weights'].values[1] = -1.3
        testmesh.update_pca_nodes()
        npt.assert_almost_equal(testmesh.get_nodes(), testdata['Xn1'])
        testmesh.nodes['weights'].values[3] = 0.7
        testmesh.update_pca_nodes()
        npt.assert_almost_equal(testmesh.get_nodes(), testdata['Xn2'])
        
    def test_example_2d_fit_lse(self):
        if not fast:
            example = 'example_2d_fit_lse'
            execfile('../examples/'+example+'.py', globals(), locals())
            Xn = locals()['mesh'].get_nodes()
            Xs, Ts = locals()['mesh'].get_surfaces()
            
            data = pickle.load(open('data/'+example+'.pkl', 'r'))
            npt.assert_almost_equal(Xn, data['Xn'])
            npt.assert_almost_equal(Xs, data['Xs'])
            npt.assert_almost_equal(Ts, data['Ts'])
        



if __name__ == "__main__":
    fast = False
    if 'fast' in sys.argv:
        fast = True
    del sys.argv[1:]
    unittest.main()
