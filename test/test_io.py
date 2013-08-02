import sys
import unittest
import doctest

import ddt

import numpy
from numpy import array
import numpy.testing as npt
#~ sys.path.insert(0, os.path.abspath('..'))

sys.path.append('..')
from morphic import mesher
        
@ddt.ddt
class TestPyTablesMesh(unittest.TestCase):
    """Unit tests for morphic interpolants."""
    
    @ddt.file_data('io_formats.json')
    def test_save_empty_mesh(self, value):
        filepath = 'data/%s.mesh' % (value)
        mesh0 = mesher.Mesh(label='cube', units='mm')
        mesh0.generate()
        # mesh0.save(filepath, format=value)

        # mesh1 = mesher.Mesh(filepath)
        # self.assertEqual(mesh0.label, mesh1.label)
        # self.assertEqual(mesh0.units, mesh1.units)
    
    @ddt.file_data('io_formats.json')
    def test_metadata(self, value):
        filepath = 'data/%s.mesh' % (value)
        mesh0 = mesher.Mesh(label='cube', units='mm')
        mesh0.add_stdnode(1, [0.5])
        mesh0.generate()
        mesh0.save(filepath, format=value)

        mesh1 = mesher.Mesh(filepath)
        self.assertEqual(mesh0.version, mesh1.version)
        self.assertEqual(mesh1.version, 1)
        self.assertEqual(mesh0.created_at, mesh1.created_at)
        self.assertEqual(mesh1.created_at[:2], "20")
        self.assertEqual(mesh1.created_at[-3:], "UTC")
        self.assertEqual(mesh1.saved_at[:2], "20")
        self.assertEqual(mesh1.saved_at[-3:], "UTC")
        self.assertEqual(mesh0.label, mesh1.label)
        self.assertEqual(mesh0.units, mesh1.units)
    
    @ddt.file_data('io_formats.json')
    def test_metadata(self, value):
        filepath = 'data/%s.mesh' % (value)
        mesh0 = mesher.Mesh(label='cube', units='mm')
        mesh0.metadata.name = 'Joe Bloggs'
        mesh0.metadata.age = 23
        mesh0.metadata.height = 1.68
        mesh0.metadata.list1 = ['c', 'a', 'b']
        mesh0.metadata.list2 = [2, 1, 3]
        mesh0.metadata.list3 = [2.1, 1.1, 3.1]
        mesh0.metadata.list4 = [1, 'a', 3.3]
        
        mesh0.add_stdnode(1, [0.5])
        mesh0.generate()
        mesh0.save(filepath, format=value)

        mesh1 = mesher.Mesh(filepath)
        self.assertEqual(mesh1.metadata.name, 'Joe Bloggs')
        self.assertEqual(mesh1.metadata.age, 23)
        self.assertEqual(mesh1.metadata.height, 1.68)
        self.assertEqual(mesh1.metadata.list1, ['c', 'a', 'b'])
        self.assertEqual(mesh1.metadata.list2, [2, 1, 3])
        self.assertEqual(mesh1.metadata.list3, [2.1, 1.1, 3.1])
        self.assertEqual(mesh1.metadata.list4, [1, 'a', 3.3])
    
    @ddt.file_data('io_formats.json')
    def test_stdnodes(self, value):
        filepath = 'data/%s.mesh' % (value)
        mesh0 = mesher.Mesh()
        mesh0.add_stdnode(1, [0.5, 0.5])
        mesh0.add_stdnode(2, [0.0, 0.3])
        mesh0.add_stdnode('2', [[1.0, 0.5],[0.9, 1.3]])
        mesh0.add_element(1, ['L1'], [1, 2])
        mesh0.generate()
        mesh0.save(filepath, format=value)

        mesh1 = mesher.Mesh(filepath)
        for node in mesh0.nodes:
            nid = node.id
            npt.assert_equal(node.id, mesh1.nodes[nid].id)
            npt.assert_equal(mesh0.nodes[nid].values, mesh1.nodes[nid].values)
        npt.assert_equal(mesh0.core.P, mesh1.core.P)

    @ddt.file_data('io_formats.json')
    def test_depnodes(self, value):
        filepath = 'data/%s.mesh' % (value)
        mesh0 = mesher.Mesh()
        mesh0.add_stdnode('xi', [0.5])
        mesh0.add_stdnode(1, [0.0, 0.0])
        mesh0.add_stdnode(2, [1, 0.5])
        mesh0.add_depnode(3, 1, 'xi')
        mesh0.add_element(1, ['L1'], [1, 2])
        mesh0.generate()
        mesh0.save(filepath, format=value)

        mesh1 = mesher.Mesh(filepath)
        npt.assert_equal(mesh0.nodes[3].values, mesh1.nodes[3].values)
        npt.assert_equal(mesh0.core.P, mesh1.core.P)
        
    @ddt.file_data('io_formats.json')
    def test_pcanodes(self, value):
        filepath = 'data/%s.mesh' % (value)
        mesh0 = mesher.Mesh()
        mesh0.add_stdnode('weights', [1, 1, -0.1])
        mesh0.add_stdnode('vars', [1, 0.1, 0.04])
        mesh0.add_pcanode(1, [[[0.5, 0.1, 0.01]], [[0.7, 0.2, 0.02]]], 'weights', 'vars')
        mesh0.add_pcanode(2, [[[0.0, -0.1, 0.01]], [[0.0, 0.02, -0.01]]], 'weights', 'vars')
        mesh0.add_element(1, ['L1'], [2, 1])
        mesh0.generate()
        mesh0.save(filepath, format=value)

        mesh1 = mesher.Mesh(filepath)
        npt.assert_equal(mesh0.nodes[1].values, mesh1.nodes[1].values)
        npt.assert_equal(mesh0.nodes[2].values, mesh1.nodes[2].values)
        npt.assert_equal(mesh0.elements[1].evaluate([0.3]),
                mesh1.elements[1].evaluate([0.3]))

        weights = numpy.array([1, -0.1, 0.2])
        mesh0.nodes['weights'].values = weights
        mesh0.update_pca_nodes()
        mesh1.nodes['weights'].values = weights
        mesh1.update_pca_nodes()

        npt.assert_equal(mesh0.nodes[1].values, mesh1.nodes[1].values)
        npt.assert_equal(mesh0.nodes[2].values, mesh1.nodes[2].values)
        npt.assert_equal(mesh0.elements[1].evaluate([0.3]),
                mesh1.elements[1].evaluate([0.3]))

        npt.assert_equal(mesh0.core.P, mesh1.core.P)


    @ddt.file_data('io_formats.json')
    def test_elements(self, value):
        filepath = 'data/%s.mesh' % (value)
        mesh0 = mesher.Mesh()
        mesh0.add_stdnode(1, [0.5, 0.7, 1.2])
        mesh0.add_stdnode(2, [0.0, 0.3, 0.4])
        mesh0.add_element(1, ['L1'], [1, 2])
        mesh0.add_element('-1', ['L1'], [2, 1])
        mesh0.generate()
        mesh0.save(filepath, format=value)

        mesh1 = mesher.Mesh(filepath)
        for ne, el0 in enumerate(mesh0.elements):
            el1 = mesh1.elements[el0.id]
            self.assertEqual(el0.id, el1.id)
            self.assertEqual(el0.basis, el1.basis)
            self.assertEqual(el0.node_ids, el1.node_ids)

        npt.assert_equal(mesh0.elements[1].evaluate([0.3]),
            mesh1.elements[1].evaluate([0.3]))
        npt.assert_equal(mesh0.elements[1].length(),
            mesh1.elements[1].length())

    @ddt.file_data('io_formats.json')
    def test_groups(self, value):
        def compare_node_groups(mesh0, mesh1, group):
            nids0 = [n.id for n in mesh0.nodes.groups[group]]
            nids1 = [n.id for n in mesh1.nodes.groups[group]]
            self.assertEqual(nids0, nids1)
        
        def compare_elem_groups(mesh0, mesh1, group):
            nids0 = [n.id for n in mesh0.elements.groups[group]]
            nids1 = [n.id for n in mesh1.elements.groups[group]]
            self.assertEqual(nids0, nids1)
        
        filepath = 'data/%s.mesh' % (value)
        mesh0 = mesher.Mesh()
        mesh0.add_stdnode(1, [0., 0.])
        mesh0.add_stdnode('2', [3., 0.])
        mesh0.add_stdnode(3, [0., 4.])
        mesh0.add_element(1, ['L1'], [1, '2'])
        mesh0.add_element(2, ['L1'], [1, 3])
        mesh0.add_element('hypotenuse', ['L1'], ['2', 3])
        mesh0.nodes.add_to_group(1, 'origin')
        mesh0.nodes.add_to_group(['2', 3], 5)
        mesh0.nodes.add_to_group(['2', 3, 1], 'all')
        mesh0.elements.add_to_group([1, 2], 12)
        mesh0.elements.add_to_group('hypotenuse', 'hypotenuse')
        mesh0.elements.add_to_group([1, 2, 'hypotenuse'], 'loop')
        mesh0.generate()
        mesh0.save(filepath, format=value)

        mesh1 = mesher.Mesh(filepath)
        compare_node_groups(mesh0, mesh1, 'origin')
        compare_node_groups(mesh0, mesh1, 5)
        compare_node_groups(mesh0, mesh1, 'all')
        compare_elem_groups(mesh0, mesh1, 12)
        compare_elem_groups(mesh0, mesh1, 'hypotenuse')
        compare_elem_groups(mesh0, mesh1, 'loop')

if __name__ == "__main__":
    unittest.main()
