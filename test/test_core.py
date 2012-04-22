import sys
import unittest
import doctest

import numpy
import numpy.testing as npt

sys.path.append('..')
from morphic import core
from morphic import mesher

class TestObjectList(unittest.TestCase):
    """Unit tests for morphic."""

    def test_init(self):
        mol = core.ObjectList()
        self.assertEqual(mol._objects, [])
        self.assertEqual(mol._object_ids, {})
        self.assertEqual(mol._id_counter, 0)
    
    def test_set_counter(self):
        mol = core.ObjectList()
        self.assertEqual(mol._id_counter, 0)
        mol.set_counter(3)
        self.assertEqual(mol._id_counter, 3)
    
    def test_get_unique_id(self):
        mol = core.ObjectList()
        self.assertEqual(mol.get_unique_id(), 0)
        mol._object_ids[0] = ''
        self.assertEqual(mol.get_unique_id(), 1)
        for i in range(10):
            mol._object_ids[i] = ''
        self.assertEqual(mol.get_unique_id(), 10)
        mol._object_ids.pop(5)
        self.assertEqual(mol.get_unique_id(), 10)
        mol._id_counter = 0
        self.assertEqual(mol.get_unique_id(), 5)
    
    def test_add(self):
        mesh = mesher.Mesh()
        node1 = mesher.StdNode(mesh, 1, [0.1])
        node2 = mesher.DepNode(mesh, 2, 1, 1)
        node3 = mesher.StdNode(mesh, 3, [0.3])
        nodea = mesher.StdNode(mesh, 'a', [0.3])
        mol = core.ObjectList()
        mol.add(node1)
        mol.add(node2)
        mol.add(node3)
        mol.add(nodea)
        self.assertEqual(mol._objects[0], node1)
        self.assertEqual(mol._objects[1], node2)
        self.assertEqual(mol._objects[2], node3)
        self.assertEqual(mol._objects[3], nodea)
        Nodes = [node1, node2, node3, nodea]
        for i, node in enumerate(mol):
            self.assertEqual(node, Nodes[i])
        Nodes = [node3, node1]
        for i, node in enumerate(mol[[3,1]]):
            self.assertEqual(node, Nodes[i])
        self.assertEqual(mol[1], node1)
        self.assertEqual(mol[2], node2)
        self.assertEqual(mol[3], node3)
        self.assertEqual(mol['a'], nodea)
        
    def test_add_no_uid(self):
        mol = core.ObjectList()
        uid = mol.add('item_with_no_uid')
        self.assertEqual(uid, 0)
        self.assertEqual(mol[0], 'item_with_no_uid')
    
    def test_group(self):
        mesh = mesher.Mesh()
        node1 = mesher.StdNode(mesh, 1, [0.1])
        node2 = mesher.DepNode(mesh, 2, 1, 1)
        node3 = mesher.StdNode(mesh, 3, [0.3])
        node4 = mesher.StdNode(mesh, 4, [0.3])
        mol = core.ObjectList()
        mol.add(node1, group='g1')
        mol.add(node2, group='g2')
        mol.add(node3, group='g3')
        mol.add(node4, group='g2')
        
        mol.add_to_group(2, 'dependent_nodes')
        mol.add_to_group([1, 3, 4], 'standard_nodes')
        
        self.assertEqual(mol._get_group('g1'), [node1])
        self.assertEqual(mol._get_group('g2'), [node2, node4])
        self.assertEqual(mol._get_group('g3'), [node3])
        
        self.assertEqual(mol._get_group('dependent_nodes'), [node2])
        self.assertEqual(mol._get_group('standard_nodes'),
            [node1, node3, node4])
    
    def test_contains(self):
        mesh = mesher.Mesh()
        node1 = mesher.StdNode(mesh, 1, [0.1])
        node2 = mesher.DepNode(mesh, 2, 1, 1)
        node3 = mesher.StdNode(mesh, 3, [0.3])
        nodea = mesher.StdNode(mesh, 'a', [0.3])
        mol = core.ObjectList()
        mol.add(node1)
        mol.add(node2)
        mol.add(node3)
        mol.add(nodea)
        self.assertTrue(1 in mol)
        self.assertTrue(2 in mol)
        self.assertTrue(3 in mol)
        self.assertTrue('a' in mol)
        self.assertFalse(5 in mol)
        self.assertFalse('b' in mol)
         

class TestCore(unittest.TestCase):
    """Unit tests for morphic."""

    def test_init(self):
        c = core.Core()
    
    def test_add_params(self):
        c = core.Core()
        cids = c.add_params(numpy.array([3, 6, 9]))
        self.assertEqual(cids, [0, 1, 2])
        npt.assert_equal(c.P, [3, 6, 9])
        cids = c.add_params(numpy.array([5, 2]))
        self.assertEqual(cids, [3, 4])
        npt.assert_equal(c.P, [3, 6, 9, 5, 2])
        
    def test_update_params(self):
        c = core.Core()
        cids = c.add_params(numpy.array([3, 6, 9, 5, 2]))
        c.update_params([3, 1], numpy.array([7, 8]))
        npt.assert_equal(c.P, [3, 8, 9, 7, 2])
        
    def test_fix_parameters(self):
        c = core.Core()
        cids = c.add_params(numpy.array([3, 6, 9, 5, 2]))
        npt.assert_equal(c.fixed, [False, False, False, False, False])
        c.fix_parameters(cids, [False, True, False, True, True])
        npt.assert_equal(c.fixed, [False, True, False, True, True])
        c.fix_parameters([2, 1], [True, False])
        npt.assert_equal(c.fixed, [False, False, True, True, True])
        
    def test_generate_fixed_index(self):
        c = core.Core()
        cids = c.add_params(numpy.array([3, 6, 9, 5, 2]))
        c.fix_parameters(cids, [False, True, False, True, True])
        c.generate_fixed_index()
        npt.assert_equal(c.idx_unfixed, [0, 2])
        
    def test_get_variables(self):
        c = core.Core()
        cids = c.add_params(numpy.array([3, 6, 9, 5, 2]))
        c.fix_parameters(cids, [False, True, False, True, True])
        c.generate_fixed_index()
        npt.assert_equal(c.get_variables(), [3, 9])
        
    def test_set_variables(self):
        c = core.Core()
        cids = c.add_params(numpy.array([3, 6, 9, 5, 2]))
        c.fix_parameters(cids, [False, True, False, True, True])
        c.generate_fixed_index()
        c.set_variables([7, 8])
        npt.assert_equal(c.P, [7, 6, 8, 5, 2])
        
        
if __name__ == "__main__":
    unittest.main()
