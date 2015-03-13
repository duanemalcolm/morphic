import sys
import unittest
import doctest

import numpy as np
import numpy.testing as npt

sys.path.append('..')
from morphic import mesher
from morphic import fasteval


class TestFastEval(unittest.TestCase):
    """Unit tests for morphic."""

    def setUp(self):
        self.mesh = mesher.Mesh()
        self.mesh.add_stdnode(1, [0., 0., 0.])
        self.mesh.add_stdnode(2, [1., 0., 1.])
        self.mesh.add_stdnode(3, [0., 1., 2.])
        self.mesh.add_stdnode(4, [1., 1., 1.])
        self.mesh.add_element(1, ['L1', 'L1'], [1, 2, 3, 4])

        self.mesh.nodes[1].variables(True)
        self.mesh.nodes[2].variables(True)
        self.mesh.nodes[3].variables(True)
        self.mesh.nodes[4].variables(True)

        self.mesh.generate()

    def test_init_matrix(self):
        fe = fasteval.FEMatrix((5, 3))
        npt.assert_equal(fe.A.todense(), np.zeros((5, 3)))
        npt.assert_equal(fe.rhs, np.zeros((3)))

    def test_add_mesh(self):
        fe = fasteval.FEMatrix((5, 3))
        fe.add_mesh(self.mesh)
        self.assertEqual(fe.mesh, self.mesh)

    def test_auto_increment_row_id(self):
        elem_id = 1
        xi = np.array([[0.1, 0.7]])
        fe = fasteval.FEMatrix((3, 12))
        fe.add_mesh(self.mesh)
        self.assertEqual(fe.row_id, 0)
        fe.add_element_point(elem_id, xi, 0)
        self.assertEqual(fe.row_id, 0)
        fe.auto_increment_rows()
        fe.add_element_point(elem_id, xi, 0)
        self.assertEqual(fe.row_id, 1)
        fe.auto_increment_rows(False)
        fe.add_element_point(elem_id, xi, 0)
        self.assertEqual(fe.row_id, 1)
        fe.auto_increment_rows(True)
        fe.add_element_point(elem_id, xi, 0)
        self.assertEqual(fe.row_id, 2)
        fe.add_element_point(elem_id, xi, 0)
        self.assertEqual(fe.row_id, 3)

    def test_add_element_point(self):
        elem_id = 1
        xi = np.array([[0.1, 0.7]])
        A = np.array([[0.27, 0, 0, 0.03, 0, 0, 0.63, 0, 0, 0.07, 0, 0]])

        # FastEval contrust
        fe = fasteval.FEMatrix((1, 12))
        fe.add_mesh(self.mesh)
        fe.add_element_point(elem_id, xi, 0)
        npt.assert_almost_equal(fe.A.toarray(), A)

        # Check FastEval against Mesher evaluate
        x_matrix = np.dot(fe.A.toarray(), self.mesh.get_variables())
        npt.assert_almost_equal(x_matrix, self.mesh.evaluate(elem_id, xi)[0][0])

    def test_add_element_point_with_scalar(self):
        elem_id = 1
        xi = np.array([[0.1, 0.7]])
        scalar = -2.1;
        A = scalar * np.array([[0.27, 0, 0, 0.03, 0, 0, 0.63, 0, 0, 0.07, 0, 0]])

        # FastEval contrust
        fe = fasteval.FEMatrix((1, 12))
        fe.add_mesh(self.mesh)
        fe.add_element_point(elem_id, xi, 0, scalar=scalar)
        npt.assert_almost_equal(fe.A.toarray(), A)

    def test_add_element_points(self):
        elem_id = 1
        xi = np.array([[0.1, 0.7]])
        A = np.array([
            [0.27, 0, 0, 0.03, 0, 0, 0.63, 0, 0, 0.07, 0, 0],
            [0, 0.27, 0, 0, 0.03, 0, 0, 0.63, 0, 0, 0.07, 0],
            [0, 0, 0.27, 0, 0, 0.03, 0, 0, 0.63, 0, 0, 0.07]
        ])

        # FastEval construct
        fe = fasteval.FEMatrix((3, 12))
        fe.auto_increment_rows()
        fe.add_mesh(self.mesh)
        fe.add_element_point(elem_id, xi, 0)
        fe.add_element_point(elem_id, xi, 1)
        fe.add_element_point(elem_id, xi, 2)
        npt.assert_almost_equal(fe.A.toarray(), A)

        # Check FastEval against Mesher evaluate
        x_matrix = np.dot(fe.A.toarray(), self.mesh.get_variables())
        npt.assert_almost_equal(x_matrix, self.mesh.evaluate(elem_id, xi)[0])

    def test_add_element_points_deriv(self):
        elem_id = 1
        xi = np.array([[0.1, 0.7]])
        deriv = [0, 1]
        A = np.array([
            [-0.9, 0., 0., -0.1, 0., 0., 0.9, 0., 0., 0.1, 0., 0.],
            [0., -0.9, 0., 0., -0.1, 0., 0., 0.9, 0., 0., 0.1, 0.],
            [0., 0., -0.9, 0., 0., -0.1, 0., 0., 0.9, 0., 0., 0.1]
        ])

        # FastEval construct
        fe = fasteval.FEMatrix((3, 12))
        fe.auto_increment_rows()
        fe.add_mesh(self.mesh)
        fe.add_element_point(elem_id, xi, 0, deriv=deriv)
        fe.add_element_point(elem_id, xi, 1, deriv=deriv)
        fe.add_element_point(elem_id, xi, 2, deriv=deriv)
        npt.assert_almost_equal(fe.A.toarray(), A)

        # Check FastEval against Mesher evaluate
        x_matrix = np.dot(fe.A.toarray(), self.mesh.get_variables())
        npt.assert_almost_equal(x_matrix,
                                self.mesh.evaluate(1, xi, deriv=deriv)[0])

    def test_duplicate_row_entries(self):
        # FastEval for the vector between two points
        elem_id = 1
        xi1 = np.array([[0.1, 0.7]])
        xi2 = np.array([[0.6, 0.5]])
        A = np.array([
            [-0.07, 0., 0., 0.27, 0., 0., -0.43, 0., 0., 0.23, 0., 0.],
            [0., -0.07, 0., 0., 0.27, 0., 0., -0.43, 0., 0., 0.23, 0.],
            [0., 0., -0.07, 0., 0., 0.27, 0., 0., -0.43, 0., 0., 0.23]
        ])

        # FastEval construct
        fe = fasteval.FEMatrix((3, 12))
        fe.add_mesh(self.mesh)
        fe.add_element_point(elem_id, xi1, 0, scalar=-1)
        fe.add_element_point(elem_id, xi2, 0)
        fe.next_row()
        fe.add_element_point(elem_id, xi1, 1, scalar=-1)
        fe.add_element_point(elem_id, xi2, 1)
        fe.next_row()
        fe.add_element_point(elem_id, xi1, 2, scalar=-1)
        fe.add_element_point(elem_id, xi2, 2)
        fe.next_row()
        npt.assert_almost_equal(fe.A.toarray(), A)

        # Check FastEval against Mesher evaluate
        dx_matrix = np.dot(fe.A.toarray(), self.mesh.get_variables())
        x1 = self.mesh.evaluate(elem_id, xi1)[0]
        x2 = self.mesh.evaluate(elem_id, xi2)[0]
        npt.assert_almost_equal(dx_matrix, x2 - x1)


if __name__ == "__main__":
    unittest.main()
