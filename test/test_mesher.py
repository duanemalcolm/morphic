import unittest
import doctest

import numpy
from numpy import array
import numpy.testing as npt

from fieldscape import mesher

class TestMeshObjectList(unittest.TestCase):
    """Unit tests for fieldscape."""

    def test_init(self):
        mol = mesher.MeshObjectList()
        self.assertEqual(mol._objects, [])
        self.assertEqual(mol._object_ids, {})
        self.assertEqual(mol._id_counter, 0)
    
    def test_set_counter(self):
        mol = mesher.MeshObjectList()
        self.assertEqual(mol._id_counter, 0)
        mol.set_counter(3)
        self.assertEqual(mol._id_counter, 3)
    
    def test_get_unique_id(self):
        mol = mesher.MeshObjectList()
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
        node1 = mesher.Node(mesh, 1, [0.1])
        node2 = mesher.Node(mesh, 2, [0.2])
        node3 = mesher.Node(mesh, 3, [0.3])
        nodea = mesher.Node(mesh, 'a', [0.3])
        mol = mesher.MeshObjectList()
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
         
        
class TestNode(unittest.TestCase):
    """Unit tests for fieldscape interpolants."""

    def test_node_init(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4', [0])
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, '4')
        #~ npt.assert_equal(node.values, [0.])
        
        node = mesher.Node(mesh, '4', [0.1, 0.2])
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, '4')
        #~ npt.assert_equal(node.values, [0.1, 0.2])
        

class TestDepNode(unittest.TestCase):
    """Unit tests for fieldscape interpolants."""

    def test_node_init(self):
        mesh = mesher.Mesh()
        node = mesher.Node(mesh, '4', [0])
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, '4')
        npt.assert_equal(node.values, [0.])
        node = mesher.Node(mesh, '4', [0.1, 0.2])
        self.assertEqual(node.mesh, mesh)
        self.assertEqual(node.id, '4')
        npt.assert_equal(node.values, [0.1, 0.2])
        

class TestElement(unittest.TestCase):
    """Unit tests for fieldscape interpolants."""

    def test_elem_init(self):
        mesh = mesher.Mesh()
        elem = mesher.Element(mesh, 1, ['L1'], [1, 2])
        self.assertEqual(elem.mesh, mesh)
        self.assertEqual(elem.id, 1)
        self.assertEqual(elem.interp, ['L1'])
        self.assertEqual(elem.nodes, [1, 2])
        
    def test_elem_iter(self):
        mesh = mesher.Mesh()
        n1 = mesh.add_node(1, [0.1])
        n2 = mesh.add_node(2, [0.2])
        elem = mesher.Element(mesh, 1, ['L1'], [1, 2])
        Nodes = [n1, n2]
        for i, node in enumerate(elem):
            self.assertEqual(node, Nodes[i])
        
        
class TestMesh(unittest.TestCase):
    """Unit tests for fieldscape interpolants."""

    def test_doctests(self):
        """Run interpolants doctests"""
        doctest.testmod(mesher)
        
    def test_mesh_init(self):
        mesh = mesher.Mesh()
        self.assertEqual(mesh.label, '/')
        self.assertEqual(isinstance(mesh.nodes, mesher.MeshObjectList),
                True)
        
    def test_add_node(self):
        mesh = mesher.Mesh()
        node1 = mesh.add_node(1, [0.0])
        nodeN1 = mesh.add_node(None, [0.1])
        nodeN2 = mesh.add_node(None, [0.2])
        noded = mesh.add_node('d', [0.3])
        
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
        node1 = mesh.add_node(1, [0.1])
        node2 = mesh.add_node(2, [0.2])
        elem1 = mesh.add_element(1, ['L1'], [1, 2])
        elem0 = mesh.add_element(None, ['L1'], [2, 1])
        
        self.assertEqual(elem1.id, 1)
        self.assertEqual(elem1.interp, ['L1'])
        self.assertEqual(elem1.nodes, [1, 2])
        self.assertEqual(mesh.elements[1].id, 1)
        self.assertEqual(mesh.elements[1].interp, ['L1'])
        self.assertEqual(mesh.elements[1].nodes, [1, 2])
        
        self.assertEqual(elem0.id, 0)
        self.assertEqual(elem0.interp, ['L1'])
        self.assertEqual(elem0.nodes, [2, 1])
        self.assertEqual(mesh.elements[0].id, 0)
        self.assertEqual(mesh.elements[0].interp, ['L1'])
        self.assertEqual(mesh.elements[0].nodes, [2, 1])
        
        
        
if __name__ == "__main__":
    unittest.main()
