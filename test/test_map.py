import unittest
import morphic
import numpy.testing as npt


class TestMap(unittest.TestCase):
    """Unit tests for morphic mapping paramters superclass."""

    def test_node_map(self):
        mesh = morphic.Mesh()
        mesh.add_stdnode(1, [[1, 2, 3], [3, 4, 5]])
        mesh.add_stdnode(2, [9, 9])
        mesh.add_map((1, 1, 1), (2, 1))
        self.assertTrue(mesh.core.has_maps)

        mesh.update_maps()
        npt.assert_almost_equal(mesh.nodes[2].values, [9., 4.])

    def test_node_maps(self):
        mesh = morphic.Mesh()
        mesh.add_stdnode(1, [[1, 2, 3], [3, 4, 5]])
        mesh.add_stdnode(2, [9, 9])
        mesh.add_map((1, 0, 0), (2, 0))
        mesh.add_map((1, 1, 0), (2, 1))

        mesh.update_maps()
        npt.assert_almost_equal(mesh.nodes[2].values, [1., 3.])

    def test_node_maps_with_scaling(self):
        mesh = morphic.Mesh()
        mesh.add_stdnode(1, [[1, 2, 3], [3, 4, 5]])
        mesh.add_stdnode(2, [9, 9])
        mesh.add_map((1, 0, 0), (2, 0), 2.)
        mesh.add_map((1, 1, 0), (2, 1), -1)

        mesh.update_maps()
        npt.assert_almost_equal(mesh.nodes[2].values, [2., -3.])


if __name__ == "__main__":
    unittest.main()
