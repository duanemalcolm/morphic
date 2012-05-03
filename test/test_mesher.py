import sys
import unittest
import doctest

import numpy
from numpy import array
import numpy.testing as npt
#~ sys.path.insert(0, os.path.abspath('..'))

sys.path.append('..')
from morphic import core
from morphic import mesher

 
class TestNode(unittest.TestCase):
    """Unit tests for morphic Node superclass."""

    def test_node_init(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        self.assertEqual(node._type, 'standard')
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, '4')
        self.assertEqual(node.cids, None)
        self.assertEqual(node.num_values, 0)
        self.assertEqual(node.num_fields, 0)
        self.assertEqual(node.num_components, 0)
        self.assertEqual(node.num_modes, 0)
        self.assertEqual(node.shape, (0, 0, 0))
        self.assertEqual(node._added, False)
        self.assertEqual(node.mesh._regenerate, True)
        self.assertEqual(node.mesh._reupdate, True)
        
    def test_save(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        ns = node._save_dict()
        self.assertEqual(ns['id'], '4')
        self.assertEqual(ns['fixed'], None)
        self.assertEqual(ns['cids'], None)
        self.assertEqual(ns['num_values'], 0)
        self.assertEqual(ns['num_fields'], 0)
        self.assertEqual(ns['num_components'], 0)
        self.assertEqual(ns['shape'], (0, 0, 0))
        self.assertEqual(len(ns.keys()), 7) 
        
    def test_load(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.cids = [3, 4]
        node.fixed = [True, False]
        node.num_values = 4
        node.num_fields = 2
        node.num_components = 3
        node_dict = node._save_dict()
        
        mesh2 = mesher.Mesh()
        node2 = mesher.Node(mesh2, '4')
        node2._load_dict(node_dict)
        self.assertEqual(node2.id, '4')
        self.assertEqual(node2.cids, [3, 4])
        self.assertEqual(node2.fixed, [True, False])
        self.assertEqual(node2.num_values, 4)
        self.assertEqual(node2.num_fields, 2)
        self.assertEqual(node2.num_components, 3)
    
    
    def test_get_values(self):
        mesh = mesher.Mesh()
        Xn = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        node = mesh.add_stdnode(1, Xn)
        npt.assert_almost_equal(node.get_values(), Xn)
        npt.assert_almost_equal(node.get_values()[:,:], Xn)
        npt.assert_almost_equal(node.get_values()[:,0], Xn[:,0])
        
        
    def test_one_field_no_components(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([3])
        self.assertEqual(node.num_values, 1)
        self.assertEqual(node.num_fields, 1)
        self.assertEqual(node.num_components, 1)
        self.assertEqual(node.cids, [0])
        self.assertEqual(node._added, True)
        self.assertEqual(node.mesh._regenerate, True)
        self.assertEqual(node.mesh._reupdate, True)
        
        npt.assert_almost_equal(node.values, [3])
        npt.assert_almost_equal(node.values.flatten(), [3])
        npt.assert_almost_equal(node.cids, [0])
        npt.assert_almost_equal(node.field_cids, [[0]])
        
    def test_three_fields_no_components(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([3, 4, 6])
        self.assertEqual(node.num_values, 3)
        self.assertEqual(node.num_fields, 3)
        self.assertEqual(node.num_components, 1)
        self.assertEqual(node.cids, [0, 1, 2])
        self.assertEqual(node._added, True)
        self.assertEqual(node.mesh._regenerate, True)
        self.assertEqual(node.mesh._reupdate, True)
        
        npt.assert_almost_equal(node.values, [3, 4, 6])
        npt.assert_almost_equal(node.values.flatten(), [3, 4, 6])
        npt.assert_almost_equal(node.cids, [0, 1, 2])
        npt.assert_almost_equal(node.field_cids, [[0], [1], [2]])
        
    def test_three_fields_alt_no_components(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([[3], [4], [6]])
        self.assertEqual(node.num_values, 3)
        self.assertEqual(node.num_fields, 3)
        self.assertEqual(node.num_components, 1)
        self.assertEqual(node.cids, [0, 1, 2])
        self.assertEqual(node._added, True)
        self.assertEqual(node.mesh._regenerate, True)
        self.assertEqual(node.mesh._reupdate, True)
        
        npt.assert_almost_equal(node.values[:, 0], [3, 4, 6])
        npt.assert_almost_equal(node.values, [[3], [4], [6]])
        npt.assert_almost_equal(node.values.flatten(), [3, 4, 6])
        npt.assert_almost_equal(node.cids, [0, 1, 2])
        npt.assert_almost_equal(node.field_cids, [[0], [1], [2]])
        
    def test_one_field_with_components(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([[3, 1, 0, 0]])
        self.assertEqual(node.num_values, 4)
        self.assertEqual(node.num_fields, 1)
        self.assertEqual(node.num_components, 4)
        self.assertEqual(node.cids, [0, 1, 2, 3])
        self.assertEqual(node._added, True)
        self.assertEqual(node.mesh._regenerate, True)
        self.assertEqual(node.mesh._reupdate, True)
        
        npt.assert_almost_equal(node.values[0, 0], [3])
        npt.assert_almost_equal(node.values, [[3, 1, 0, 0]])
        npt.assert_almost_equal(node.values.flatten(), [3, 1, 0, 0])
        npt.assert_almost_equal(node.cids, [0, 1, 2, 3])
        npt.assert_almost_equal(node.field_cids, [[0, 1, 2, 3]])
        
    def test_three_field_with_components(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([[3, 1, 0, 0], [4, 2, 0, 0], [5, 3, 0, 0]])
        self.assertEqual(node.num_values, 12)
        self.assertEqual(node.num_fields, 3)
        self.assertEqual(node.num_components, 4)
        self.assertEqual(node.cids, range(12))
        self.assertEqual(node._added, True)
        self.assertEqual(node.mesh._regenerate, True)
        self.assertEqual(node.mesh._reupdate, True)
        
        npt.assert_almost_equal(node.values[:, 0], [3, 4, 5])
        npt.assert_almost_equal(node.values,
            [[3, 1, 0, 0], [4, 2, 0, 0], [5, 3, 0, 0]])
        npt.assert_almost_equal(node.values.flatten(),
            [3, 1, 0, 0, 4, 2, 0, 0, 5, 3, 0, 0])
        npt.assert_almost_equal(node.cids, range(12))
        npt.assert_almost_equal(node.field_cids,
            [range(4), range(4, 8), range(8, 12)])
            
    def test_2fields_2comps_3modes(self):
        mesh = mesher.Mesh()
        xn = numpy.array([
                [[0, 0.2, 0.1], [0.5, 0.11, 0.07]],
                [[1, 0.22, 0.11], [0.5, 0.11, 0.07]]])
        
        node = mesh.add_stdnode(1, xn)
        npt.assert_almost_equal(node.values, xn)
        npt.assert_almost_equal(node.cids, range(12))
        npt.assert_almost_equal(node._get_param_indicies(),
            [[range(3), range(3, 6)], [range(6, 9), range(9, 12)]])
            
    def test_set_value(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([[3, 1, 0, 0], [4, 2, 0, 0], [5, 3, 0, 0]])
        node.values[1, 3] = 7
        npt.assert_almost_equal(node.values.flatten(),
            [3, 1, 0, 0, 4, 2, 0, 7, 5, 3, 0, 0])
        node.values[0, 2] = 9
        npt.assert_almost_equal(node.values.flatten(),
            [3, 1, 9, 0, 4, 2, 0, 7, 5, 3, 0, 0])
            
    def test_fix_all(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([[3, 1], [4, 2], [5, 3]])
        node.fix(True)
        npt.assert_equal(node.fixed, [True, True, True, True, True, True])
        node.fix(False)
        npt.assert_equal(node.fixed, [False, False, False, False, False, False])
        
    def test_fix_fields(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([[3, 1], [4, 2], [5, 3]])
        node.fix([False, True, False])
        npt.assert_equal(node.fixed, [False, False, True, True, False, False])
        node.fix([True, False, True])
        npt.assert_equal(node.fixed, [True, True, False, False, True, True])
        
    def test_fix_components(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4')
        node.set_values([[3, 1], [4, 2], [5, 3]])
        fixed = [[False, True], [True, True], [False, False]]
        node.fix(fixed)
        npt.assert_equal(node.fixed, [False, True, True, True, False, False])
        
        
        
        
class TestStdNode(unittest.TestCase):
    """Unit tests for morphic interpolants."""

    def test_node_init(self):
        mesh = mesher.Mesh()
        node = mesher.StdNode(mesh, '4', [0])
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, '4')
        npt.assert_equal(node.values, [0.])
        
        node = mesher.StdNode(mesh, '4', [0.1, 0.2])
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, '4')
        npt.assert_equal(node.values, [0.1, 0.2])
        
    def test_save(self):
        mesh = mesher.Mesh()
        node = mesher.StdNode(mesh, '4', [[0.1, 0.2], [0.3, 0.4]])
        ns = node._save_dict()
        self.assertEqual(ns['type'], 'standard')
        self.assertEqual(ns['id'], '4')
        self.assertEqual(ns['fixed'], None)
        self.assertEqual(ns['cids'], [0, 1, 2, 3])
        self.assertEqual(ns['num_values'], 4)
        self.assertEqual(ns['num_fields'], 2)
        self.assertEqual(ns['num_components'], 2)
        self.assertEqual(ns['shape'], (2, 2))
        self.assertEqual(len(ns.keys()), 8) 
        
    def test_load(self):
        mesh = mesher.Mesh()
        node = mesher.StdNode(mesh, '4', None)
        node_dict = node._save_dict()
        
        mesh2 = mesher.Mesh()
        node2 = mesher.StdNode(mesh2, '4', None)
        node2._load_dict(node_dict)
        self.assertEqual(node2.id, '4')
        self.assertEqual(node2._type,'standard')
        
class TestDepNode(unittest.TestCase):
    """Unit tests for morphic interpolants."""

    def test_node_init(self):
        mesh = mesher.Mesh()
        node = mesher.DepNode(mesh, 7, 2, '4')
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, 7)
        npt.assert_equal(node.element, 2)
        npt.assert_equal(node.node, '4')
        
    def test_save(self):
        mesh = mesher.Mesh()
        node = mesher.DepNode(mesh, '4', '7', '2')
        ns = node._save_dict()
        self.assertEqual(ns['type'], 'dependent')
        self.assertEqual(ns['id'], '4')
        self.assertEqual(ns['fixed'], None)
        self.assertEqual(ns['cids'], None)
        self.assertEqual(ns['num_values'], 0)
        self.assertEqual(ns['num_fields'], 0)
        self.assertEqual(ns['num_components'], 0)
        self.assertEqual(ns['shape'], (0, 0, 0))
        self.assertEqual(ns['element'], '7')
        self.assertEqual(ns['node'], '2')
        self.assertEqual(len(ns.keys()), 10) 
     
    def test_load(self):
        mesh = mesher.Mesh()
        node = mesher.DepNode(mesh, '4', 3, 77)
        node_dict = node._save_dict()
        
        mesh2 = mesher.Mesh()
        node2 = mesher.DepNode(mesh2, '4', None, None)
        node2._load_dict(node_dict)
        self.assertEqual(node2.id, '4')
        self.assertEqual(node2._type, 'dependent')   
        self.assertEqual(node2.element, 3)   
        self.assertEqual(node2.node, 77)

class TestPCANode(unittest.TestCase):
    """Unit tests for morphic interpolants."""

    def test_node_init(self):
        mesh = mesher.Mesh()
        Xpca = numpy.array([
                [[1, 0.2, 0.1], [2, 0.55, 0.11]],
                [[2.1, 0.02, 0.01], [2.3, 0.15, 0.06]]])
        Weights = numpy.array([1, 0.0, 0.0])
        Variance = numpy.array([1, 1., 1.])
        mesh.add_stdnode(1, Xpca)
        mesh.add_stdnode('weights', Weights)
        mesh.add_stdnode('variance', Variance)
        node = mesher.PCANode(mesh, 2, 1, 'weights', 'variance')
        
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, 2)
        self.assertEqual(node.node_id, 1)
        self.assertEqual(node.weights_id, 'weights')
        self.assertEqual(node.variance_id, 'variance')
        
        Xn = numpy.array([[ 1., 2.], [2.1, 2.3]])
        mesh.update_pca_nodes()
        npt.assert_equal(node.values, Xn)
        
        mesh.nodes['weights'].values = numpy.array([1, 1.0, 0.0])
        Xn = numpy.array([[ 1.2, 2.55], [2.12, 2.45]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
        mesh.nodes['weights'].values = numpy.array([1, 0.0, 1.0])
        Xn = numpy.array([[ 1.1, 2.11], [2.11, 2.36]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
        mesh.nodes['weights'].values = numpy.array([1, 1.0, 1.0])
        Xn = numpy.array([[ 1.3, 2.66], [2.13, 2.51]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
        mesh.nodes['weights'].values = numpy.array([1, 2.0, -1.5])
        Xn = numpy.array([[ 1.25, 2.935], [2.125, 2.51]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
        mesh.nodes['weights'].values = numpy.array([1, 1.0, 1.0])
        
        mesh.nodes['variance'].values = numpy.array([1, 2., 1.])
        Xn = numpy.array([[ 1.5, 3.21], [2.15, 2.66]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
        mesh.nodes['variance'].values = numpy.array([1, 1., 2.])
        Xn = numpy.array([[ 1.4, 2.77], [2.14, 2.57]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
        mesh.nodes['variance'].values = numpy.array([1, 2., 2.])
        Xn = numpy.array([[ 1.6, 3.32], [2.16, 2.72]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
    def test_node_init_list(self):
        mesh = mesher.Mesh()
        Xpca = [[[1, 0.2, 0.1], [2, 0.55, 0.11]],
                [[2.1, 0.02, 0.01], [2.3, 0.15, 0.06]]]
        mesh.add_stdnode('weights', [1, 0.0, 0.0])
        mesh.add_stdnode('variance', [1, 1., 1.])
        node = mesh.add_pcanode(2, Xpca, 'weights', 'variance')
        
        Xn = numpy.array([[ 1., 2.], [2.1, 2.3]])
        mesh.update_pca_nodes()
        npt.assert_equal(node.values, Xn)
        
        mesh.nodes['weights'].values = numpy.array([1, 1.0, 0.0])
        Xn = numpy.array([[ 1.2, 2.55], [2.12, 2.45]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
    def test_node_init_ndarray(self):
        mesh = mesher.Mesh()
        Xpca = numpy.array([[[1, 0.2, 0.1], [2, 0.55, 0.11]],
                [[2.1, 0.02, 0.01], [2.3, 0.15, 0.06]]])
        Weights = numpy.array([1, 0.0, 0.0])
        Variance = numpy.array([1, 1., 1.])
        mesh.add_stdnode('weights', Weights)
        mesh.add_stdnode('variance', Variance)
        node = mesh.add_pcanode(2, Xpca, 'weights', 'variance')
        
        Xn = numpy.array([[ 1., 2.], [2.1, 2.3]])
        mesh.update_pca_nodes()
        npt.assert_equal(node.values, Xn)
        
        mesh.nodes['weights'].values = numpy.array([1, 1.0, 0.0])
        Xn = numpy.array([[ 1.2, 2.55], [2.12, 2.45]])
        mesh.update_pca_nodes()
        npt.assert_almost_equal(node.values, Xn)
        
    def test_save(self):
        mesh = mesher.Mesh()
        Xpca = numpy.array([[[1, 0.2, 0.1], [2, 0.55, 0.11]],
                [[2.1, 0.02, 0.01], [2.3, 0.15, 0.06]]])
        wn = mesh.add_stdnode('weights', [1, 0.0, 0.0])
        vn = mesh.add_stdnode('variance', [1, 1., 1.])
        node = mesh.add_pcanode(2, Xpca, 'weights', 'variance')
        ns = node._save_dict()
        self.assertEqual(ns['type'], 'pca')
        self.assertEqual(ns['id'], 2)
        self.assertEqual(ns['node_id'], node.node_id)
        self.assertEqual(ns['weights_id'], wn.id)
        self.assertEqual(ns['variance_id'], vn.id)
     
    def test_load(self):
        mesh = mesher.Mesh()
        Xpca = numpy.array([[[1, 0.2, 0.1], [2, 0.55, 0.11]],
                [[2.1, 0.02, 0.01], [2.3, 0.15, 0.06]]])
        wn = mesh.add_stdnode('weights', [1, 0.0, 0.0])
        vn = mesh.add_stdnode('variance', [1, 1., 1.])
        node = mesh.add_pcanode(2, Xpca, 'weights', 'variance')
        node_dict = node._save_dict()
        
        mesh2 = mesher.Mesh()
        node2 = mesher.PCANode(mesh2, 2, None, None, None)
        node2._load_dict(node_dict)
        self.assertEqual(node2.id, 2)
        self.assertEqual(node2._type, 'pca')   
        self.assertEqual(node2.node_id, node.node_id)   
        self.assertEqual(node2.weights_id, wn.id)
        self.assertEqual(node2.variance_id, vn.id)
        
        
class TestElement(unittest.TestCase):
    """Unit tests for morphic interpolants."""

    def test_elem_init(self):
        mesh = mesher.Mesh()
        elem = mesher.Element(mesh, 1, ['L1'], [1, 2])
        self.assertEqual(elem.mesh, mesh)
        self.assertEqual(elem.id, 1)
        self.assertEqual(elem.interp, ['L1'])
        self.assertEqual(elem.node_ids, [1, 2])
    
    def test_elem_nodes_access(self):
        mesh = mesher.Mesh()
        n1 = mesh.add_stdnode(2, [1, 2])
        n2 = mesh.add_stdnode(6, [2, 4])
        elem = mesh.add_element(1, ['L1'], [2, 6])
        self.assertEqual(elem.node_ids, [2, 6])
        self.assertEqual(elem.nodes, [n1, n2])
        
    def test_save_1d(self):
        mesh = mesher.Mesh()
        elem = mesher.Element(mesh, 1, ['L1'], [1, 2])
        es = elem._save_dict()
        self.assertEqual(es['id'], 1)
        self.assertEqual(es['interp'], ['L1'])
        self.assertEqual(es['nodes'], [1, 2])
        self.assertEqual(es['shape'], 'line')
        self.assertEqual(len(es.keys()), 4)
        
    def test_save_2d(self):
        mesh = mesher.Mesh()
        elem = mesher.Element(mesh, 1, ['L1', 'L1'], [1, 2, 3, 4])
        es = elem._save_dict()
        self.assertEqual(es['id'], 1)
        self.assertEqual(es['interp'], ['L1', 'L1'])
        self.assertEqual(es['nodes'], [1, 2, 3, 4])
        self.assertEqual(es['shape'], 'quad')
        self.assertEqual(len(es.keys()), 4)
        
    def test_load(self):
        mesh = mesher.Mesh()
        elem = mesher.Element(mesh, 1, ['L1', 'L1'], [1, 2, 3, 4])
        elem_dict = elem._save_dict()
        
        mesh2 = mesher.Mesh()
        elem2 = mesher.Element(mesh, 1, None, None)
        elem2._load_dict(elem_dict)
        self.assertEqual(elem2.id, 1)
        self.assertEqual(elem2.interp, ['L1', 'L1'])
        self.assertEqual(elem2.node_ids, [1, 2, 3, 4])
        self.assertEqual(elem2.shape, 'quad')
    
    def test_set_shape_line(self):
        mesh = mesher.Mesh()
        elem = mesher.Element(mesh, 1, ['L2'], [1, 2, 3])
        self.assertEqual(elem.shape, 'line')
        
    def test_set_shape_quad(self):
        mesh = mesher.Mesh()
        elem = mesher.Element(mesh, 1, ['L1', 'L1'], [1, 2, 3, 4])
        self.assertEqual(elem.shape, 'quad')
        
    def test_set_shape_tri(self):
        mesh = mesher.Mesh()
        elem = mesher.Element(mesh, 1, ['T11'], [1, 2, 3])
        self.assertEqual(elem.shape, 'tri')
        
    def test_elem_iter(self):
        mesh = mesher.Mesh()
        n1 = mesh.add_stdnode(1, [0.1])
        n2 = mesh.add_stdnode(2, [0.2])
        elem = mesher.Element(mesh, 1, ['L1'], [1, 2])
        Nodes = [n1, n2]
        for i, node in enumerate(elem):
            self.assertEqual(node, Nodes[i])
    
    def test_element_evaluate_1d_list_1pt(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0.5, 1])
        m.add_stdnode(2, [1, 0.5, 0.5])
        m.add_element(1, ['L1'], [1, 2])
        m.generate()
        x = m.elements[1].evaluate([0.2])
        npt.assert_almost_equal(x, [0.2, 0.5, 0.9])
    
    def test_element_evaluate_1d_list_2pts(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0.5, 1])
        m.add_stdnode(2, [1, 0.5, 0.5])
        m.add_element(1, ['L1'], [1, 2])
        m.generate()
        x = m.elements[1].evaluate([[0.2], [0.6]])
        npt.assert_almost_equal(x, 
            [[0.2, 0.5, 0.9], [0.6, 0.5, 0.7]])
        
    def test_element_evaluate_1d_array_1pt(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0.5, 1])
        m.add_stdnode(2, [1, 0.5, 0.5])
        m.add_element(1, ['L1'], [1, 2])
        m.generate()
        x = m.elements[1].evaluate(numpy.array([0.2]))
        npt.assert_almost_equal(x, [0.2, 0.5, 0.9])
    
    def test_element_evaluate_1d_array_2pts(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0.5, 1])
        m.add_stdnode(2, [1, 0.5, 0.5])
        m.add_element(1, ['L1'], [1, 2])
        m.generate()
        x = m.elements[1].evaluate(numpy.array([[0.2], [0.6]]))
        npt.assert_almost_equal(x, 
            [[0.2, 0.5, 0.9], [0.6, 0.5, 0.7]])
        
    def test_element_evaluate_2d_list_1pt(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0, 0])
        m.add_stdnode(2, [1, 0, 1])
        m.add_stdnode(3, [0, 1, 0])
        m.add_stdnode(4, [1, 1, 1])
        m.add_element(1, ['L1', 'L1'], [1, 2, 3, 4])
        m.generate()  
        x = m.elements[1].evaluate([0.2, 0.3])
        npt.assert_almost_equal(x, [0.2, 0.3, 0.2])
    
    def test_element_evaluate_2d_list_2pts(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0, 0])
        m.add_stdnode(2, [1, 0, 1])
        m.add_stdnode(3, [0, 1, 0])
        m.add_stdnode(4, [1, 1, 1])
        m.add_element(1, ['L1', 'L1'], [1, 2, 3, 4])
        m.generate()  
        x = m.elements[1].evaluate([[0.2, 0.3], [0.5, 0.6]])
        npt.assert_almost_equal(x, [[0.2, 0.3, 0.2], [0.5, 0.6, 0.5]])
    
    def test_element_evaluate_2d_array_1pt(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0, 0])
        m.add_stdnode(2, [1, 0, 1])
        m.add_stdnode(3, [0, 1, 0])
        m.add_stdnode(4, [1, 1, 1])
        m.add_element(1, ['L1', 'L1'], [1, 2, 3, 4])
        m.generate()  
        x = m.elements[1].evaluate(numpy.array([0.2, 0.3]))
        npt.assert_almost_equal(x, [0.2, 0.3, 0.2])
    
    def test_element_evaluate_2d_array_2pts(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0, 0])
        m.add_stdnode(2, [1, 0, 1])
        m.add_stdnode(3, [0, 1, 0])
        m.add_stdnode(4, [1, 1, 1])
        m.add_element(1, ['L1', 'L1'], [1, 2, 3, 4])
        m.generate()  
        x = m.elements[1].evaluate(
                numpy.array([[0.2, 0.3], [0.5, 0.6]]))
        npt.assert_almost_equal(x, [[0.2, 0.3, 0.2], [0.5, 0.6, 0.5]])
    
    def test_element_normal(self):
        m = mesher.Mesh()
        m.add_stdnode(1, [0, 0, 0])
        m.add_stdnode(2, [1, 0, 1])
        m.add_stdnode(3, [0, 1, 0])
        m.add_stdnode(4, [1, 1, 1])
        m.add_element(1, ['L1', 'L1'], [1, 2, 3, 4])
        m.generate()
        Xi = numpy.array([[0.1, 0.1], [0.3, 0.1], [0.7, 0.3]])
        dn = m.elements[1].normal(Xi)
        npt.assert_almost_equal(dn, [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        
    
        
class TestMesh(unittest.TestCase):
    """Unit tests for morphic interpolants."""

    def test_doctests(self):
        """Run interpolants doctests"""
        doctest.testmod(mesher)
        
    def test_mesh_init(self):
        mesh = mesher.Mesh()
        self.assertEqual(mesh.label, '/')
        self.assertEqual(isinstance(mesh.nodes, core.ObjectList),
                True)
    
    def test_save(self):
        mesh = mesher.Mesh(label='cube', units='mm')
        mesh.add_stdnode(0, [0.5])
        mesh.add_stdnode(1, [0.0, 0.3])
        mesh.add_stdnode(2, [1.0, 0.5])
        mesh.add_depnode(3, 1, 0)
        mesh.add_element(1, ['L1'], [1, 2])
        mesh.generate()
        md = mesh._save_dict()
        self.assertEqual(len(md), 7)
        self.assertEqual(md['_version'], mesh._version)
        self.assertTrue(md.has_key('_datetime'))
        self.assertEqual(md['label'], 'cube')
        self.assertEqual(md['units'], 'mm')
        self.assertEqual(len(md['nodes']), 4)
        self.assertEqual(len(md['elements']), 1)
        self.assertEqual(md['values'].shape[0], 7)
        npt.assert_almost_equal(md['values'], 
                [0.5, 0.0, 0.3, 1.0, 0.5, 0.5, 0.4])
        mesh.save('data/test_save.mesh')
    
    def test_load(self):
        mesh = mesher.Mesh('data/test_save.mesh')
        self.assertEqual(mesh.label, 'cube')
        self.assertEqual(mesh.units, 'mm')
        self.assertEqual(mesh.nodes.size(), 4)
        self.assertEqual(mesh.nodes[0].cids, [0])
        self.assertEqual(mesh.nodes[1].cids, [1, 2])
        self.assertEqual(mesh.nodes[2].cids, [3, 4])
        self.assertEqual(mesh.nodes[3].cids, [5, 6])
        npt.assert_almost_equal(mesh._core.P, 
                [0.5, 0.0, 0.3, 1.0, 0.5, 0.5, 0.4])
        
    def test_add_node(self):
        mesh = mesher.Mesh()
        node1 = mesh.add_stdnode(1, [0.0])
        nodeN1 = mesh.add_stdnode(None, [0.1])
        nodeN2 = mesh.add_stdnode(None, [0.2])
        noded = mesh.add_stdnode('d', [0.3])
        
        self.assertEqual(node1.id, 1)
        self.assertEqual(node1.values, [0.0])
        self.assertEqual(mesh.nodes[1].id, 1)
        self.assertEqual(mesh.nodes[1].values, [0.0])
        
        self.assertEqual(nodeN1.id, 0)
        self.assertEqual(nodeN1.values, [0.1])
        self.assertEqual(mesh.nodes[0].id, 0)
        self.assertEqual(mesh.nodes[0].values, [0.1])
        
        self.assertEqual(nodeN2.id, 2)
        self.assertEqual(nodeN2.values, [0.2])
        self.assertEqual(mesh.nodes[2].id, 2)
        self.assertEqual(mesh.nodes[2].values, [0.2])
        
        self.assertEqual(noded.id, 'd')
        self.assertEqual(noded.values, [0.3])
        self.assertEqual(mesh.nodes['d'].id, 'd')
        self.assertEqual(mesh.nodes['d'].values, [0.3])
        
        nids = [1, 0, 2, 'd']
        for i, node in enumerate(mesh.nodes):
            self.assertEqual(node.id, nids[i])
    
    def test_add_element(self):
        mesh = mesher.Mesh()
        node1 = mesh.add_stdnode(1, [0.1])
        node2 = mesh.add_stdnode(2, [0.2])
        elem1 = mesh.add_element(1, ['L1'], [1, 2])
        elem0 = mesh.add_element(None, ['L1'], [2, 1])
        
        self.assertEqual(elem1.id, 1)
        self.assertEqual(elem1.interp, ['L1'])
        self.assertEqual(elem1.node_ids, [1, 2])
        self.assertEqual(mesh.elements[1].id, 1)
        self.assertEqual(mesh.elements[1].interp, ['L1'])
        self.assertEqual(mesh.elements[1].node_ids, [1, 2])
        
        self.assertEqual(elem0.id, 0)
        self.assertEqual(elem0.interp, ['L1'])
        self.assertEqual(elem0.node_ids, [2, 1])
        self.assertEqual(mesh.elements[0].id, 0)
        self.assertEqual(mesh.elements[0].interp, ['L1'])
        self.assertEqual(mesh.elements[0].node_ids, [2, 1])
        
    def test_node_groups(self):
        mesh = mesher.Mesh()
        n1 = mesh.add_stdnode(1, [0.1], group='g1')
        n2 = mesh.add_stdnode(2, [0.2], group='g1')
        n3 = mesh.add_stdnode(3, [0.3], group='g2')
        n4 = mesh.add_stdnode(4, [0.4], group='g1')
        n5 = mesh.add_stdnode(5, [0.5], group='g3')
        
        mesh.nodes.add_to_group(1, 'g3')
        
        self.assertEqual(mesh.nodes('g1'), [n1, n2, n4])
        self.assertEqual(mesh.nodes('g2'), [n3])
        self.assertEqual(mesh.nodes('g3'), [n5, n1])
        
    def test_element_groups(self):
        mesh = mesher.Mesh()
        e1 = mesh.add_element(1, ['L1'], [1, 2], group='g1')
        e2 = mesh.add_element(2, ['L1'], [1, 2], group='g1')
        e3 = mesh.add_element(3, ['L1'], [1, 2], group='g2')
        e4 = mesh.add_element(4, ['L1'], [1, 2], group='g1')
        e5 = mesh.add_element(5, ['L1'], [1, 2], group='g3')
        
        mesh.elements.add_to_group(1, 'g3')
        
        self.assertEqual(mesh.elements('g1'), [e1, e2, e4])
        self.assertEqual(mesh.elements('g2'), [e3])
        self.assertEqual(mesh.elements('g3'), [e5, e1])
        
        
        
if __name__ == "__main__":
    unittest.main()
