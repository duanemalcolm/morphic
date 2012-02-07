import unittest
import doctest

import numpy
import numpy.testing as npt

from morphic import fitter
from morphic import mesher

class TestBoundElementPoint(unittest.TestCase):
    """Unit tests for morphic."""

    def test_init(self):
        pt = fitter.BoundElementPoint(2, [0.1, 0.3], 'datacloud',
                data_index=0, weight=2)
        self.assertEqual(pt.eid, 2)
        self.assertEqual(pt.xi, [0.1, 0.3])
        self.assertEqual(pt.data, 'datacloud')
        self.assertEqual(pt.data_index, 0)
        self.assertEqual(pt.bind_weight, 2)
        self.assertEqual(pt.param_ids, None)
        self.assertEqual(pt.param_weights, None)
        self.assertEqual(pt.num_fields, 0)
        
    def test_update_from_mesh(self):
        mesh = mesher.Mesh()
        mesh.add_stdnode(1, [0, 0, 1])
        mesh.add_stdnode(2, [1, 0, 2])
        mesh.add_element(9, ['L1'], [1, 2])
        mesh.generate()
        pt = fitter.BoundElementPoint(9, [0.3], 'datacloud',
                data_index=0, weight=2)
        pt.update_from_mesh(mesh)
        self.assertEqual(pt.param_ids, [[0, 3], [1, 4], [2, 5]])
        npt.assert_almost_equal(pt.param_weights, [0.7, 0.3])
        self.assertEqual(pt.num_fields, 3)
        
        
class TestBoundNodePoint(unittest.TestCase):
    """Unit tests for morphic."""

    def test_init(self):
        pt = fitter.BoundNodePoint(2, 'datacloud', data_index=0,
                weight=2)
        self.assertEqual(pt.nid, 2)
        self.assertEqual(pt.data, 'datacloud')
        self.assertEqual(pt.data_index, 0)
        self.assertEqual(pt.bind_weight, 2)
        self.assertEqual(pt.param_ids, None)
        self.assertEqual(pt.param_weights, None)
        self.assertEqual(pt.num_fields, 0)
        
    def test_update_from_mesh(self):
        mesh = mesher.Mesh()
        mesh.add_stdnode(1, [0, 0, 1])
        mesh.add_stdnode(2, [1, 0, 2])
        mesh.add_element(9, ['L1'], [1, 2])
        mesh.generate()
        pt = fitter.BoundNodePoint(2, 'datacloud', data_index=0,
                weight=2)
        pt.update_from_mesh(mesh)
        self.assertEqual(pt.param_ids, [[3], [4], [5]])
        npt.assert_almost_equal(pt.param_weights, [1, 1, 1])
        self.assertEqual(pt.num_fields, 3)
        
        
class TestData(unittest.TestCase):
    """Unit tests for morphic."""

    def test_init(self):
        Xd = numpy.array([
                numpy.linspace(0, 10, 4),
                numpy.linspace(4, 6, 4)])
        
        fit = fitter.Fit()
        fit.set_data('mydata', Xd)
        d = fit.data['mydata']
        self.assertEqual(d.id, 'mydata')
        npt.assert_almost_equal(d.values, Xd)
        
        
class TestFitter(unittest.TestCase):
    """Unit tests for morphic."""

    def setUp(self):
        self.mesh = mesher.Mesh()
        self.mesh.add_stdnode(1, [0, 0])
        self.mesh.add_stdnode(2, [1, 0])
        self.mesh.add_element(9, ['L1'], [1, 2])
        self.mesh.generate()

    def test_init(self):
        fit = fitter.Fit()
        self.assertEqual(fit.use_sparse, True)
        
        
    def test_bind_element_point(self):
        fit = fitter.Fit()
        fit.bind_element_point(2, [0.1, 0.3], 'datacloud', weight=2)
        pt = fit.points[0]
        self.assertEqual(pt.eid, 2)
        self.assertEqual(pt.xi, [0.1, 0.3])
        self.assertEqual(pt.data, 'datacloud')
        self.assertEqual(pt.bind_weight, 2)
        self.assertEqual(pt.param_ids, None)
        self.assertEqual(pt.param_weights, None)
        
    def test_update_from_mesh(self):
        fit = fitter.Fit()
        fit.use_sparse = False
        fit.bind_element_point(9, [0.3], 'datacloud', 0, weight=2)
        fit.bind_element_point(9, [0.7], 'datacloud', 1, weight=1)
        fit.update_from_mesh(self.mesh)
        self.assertEqual(fit.num_dof, 4)
        self.assertEqual(fit.num_rows, 4)
        pt = fit.points[0]
        self.assertEqual(pt.param_ids, [[0, 2], [1, 3]])
        npt.assert_almost_equal(pt.param_weights, [0.7, 0.3])
        npt.assert_almost_equal(fit.A, [
                [1.4, 0, 0.6, 0],
                [0, 1.4, 0, 0.6],
                [0.3, 0, 0.7, 0],
                [0, 0.3, 0, 0.7]])
        self.assertEqual(fit.data_map, [[0, 0], [0, 1], [1, 0], [1, 1]])
        Xd = numpy.array([[0.1, 0.2], [0.3, 0.4]])
        fit.set_data('datacloud', Xd)
        npt.assert_almost_equal(fit.data['datacloud'].values, Xd)
        Xr = fit.get_data(self.mesh)
        npt.assert_almost_equal(Xr, [0.1, 0.2, 0.3, 0.4])
        
    def test_set_data(self):
        fit = fitter.Fit()
        Xd = numpy.array([[0.1, 0.2], [0.3, 0.4]])
        fit.set_data('datacloud', Xd)
        npt.assert_almost_equal(fit.data['datacloud'].values, Xd)

    def test_get_data(self):
        fit = fitter.Fit()
        fit.use_sparse = False
        fit.bind_element_point(9, [0.3], 'datacloud', 1, weight=2)
        fit.bind_element_point(9, [0.8], 'datacloud', 0, weight=2)
        fit.update_from_mesh(self.mesh)
        Xd = numpy.array([[0.1, 0.2], [0.3, 0.4]])
        fit.set_data('datacloud', Xd)
        Xr = fit.get_data(self.mesh)
        npt.assert_almost_equal(Xr, [0.3, 0.4, 0.1, 0.2])
        
    def test_solve(self):
        fit = fitter.Fit()
        fit.bind_element_point(9, [0.3], 'datacloud', 0, weight=2)
        fit.bind_element_point(9, [0.8], 'datacloud', 1, weight=2)
        fit.update_from_mesh(self.mesh)
        Xd = numpy.array([[0.3, 0.15], [0.8, 0.4]])
        fit.set_data('datacloud', Xd)
        mesh = fit.solve(self.mesh)
        npt.assert_almost_equal(mesh.get_nodes(), [[ 0, 0], [1.0, 0.5]])
        
        
    
        
        
        
        
        
if __name__ == "__main__":
    unittest.main()
