import sys
import unittest
import doctest

import numpy
import numpy.testing as npt

sys.path.append('..')
import morphic
from morphic import core

class TestFace(unittest.TestCase):
    """Unit tests for morphic core."""
    

    def test_dimesions(self):
        self.assertEqual(core.dimensions(['L1']), 1)
        self.assertEqual(core.dimensions(['L2']), 1)
        self.assertEqual(core.dimensions(['L3']), 1)
        self.assertEqual(core.dimensions(['L4']), 1)
        self.assertEqual(core.dimensions(['H3']), 1)
        self.assertEqual(core.dimensions(['T33']), 2)
        self.assertEqual(core.dimensions(['L1', 'L1']), 2)
        self.assertEqual(core.dimensions(['L1', 'L2']), 2)
        self.assertEqual(core.dimensions(['L4', 'L1']), 2)
        self.assertEqual(core.dimensions(['H3', 'L3']), 2)
        self.assertEqual(core.dimensions(['L4', 'H3']), 2)
        self.assertEqual(core.dimensions(['L1', 'L1', 'L2']), 3)
        self.assertEqual(core.dimensions(['L4', 'L3', 'L2']), 3)
        self.assertEqual(core.dimensions(['L4', 'L4', 'L2']), 3)
        self.assertEqual(core.dimensions(['L3', 'H3', 'L2']), 3)
        self.assertEqual(core.dimensions(['H3', 'L1', 'H3']), 3)
        self.assertEqual(core.dimensions(['H3', 'H3', 'H3']), 3)
    
    def test_face_1d_elements(self):
        self.assertEqual(core.element_face_nodes(['L1'], [1, 2]), None)
        self.assertEqual(core.element_face_nodes(['L2'], [1, 2, 3]), None)
        self.assertEqual(core.element_face_nodes(['L3'], [1, 2, 3, 4]), None)
        self.assertEqual(core.element_face_nodes(['L4'], [1, 2, 3, 4, 5]), None)
        self.assertEqual(core.element_face_nodes(['H3'], [1, 2]), None)
        
    def test_face_2d_elements(self):
        self.assertEqual(core.element_face_nodes(['L1', 'L1'],
            [1, 2, 3, 4]),
            [1, 2, 3, 4])
        self.assertEqual(core.element_face_nodes(['L2', 'L1'],
            [1, 2, 3, 4, 5, 6]),
            [1, 2, 3, 4, 5, 6])
        self.assertEqual(core.element_face_nodes(['L2', 'H3'],
            [1, 2, 3, 4, 5, 6]),
            [1, 2, 3, 4, 5, 6])
    
    def test_face_3d_elements(self):
        self.assertEqual(core.element_face_nodes(['L1', 'L1', 'L1'],
            [1, 2, 3, 4, 5, 6, 7, 8]),
            [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [1, 3, 5, 7],
            [2, 4, 6, 8]])
        self.assertEqual(core.element_face_nodes(['H3', 'L1', 'L1'],
            [1, 2, 3, 4, 5, 6, 7, 8]),
            [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [1, 3, 5, 7],
            [2, 4, 6, 8]])
        self.assertEqual(core.element_face_nodes(['L1', 'H3', 'L1'],
            [1, 2, 3, 4, 5, 6, 7, 8]),
            [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [1, 3, 5, 7],
            [2, 4, 6, 8]])
        self.assertEqual(core.element_face_nodes(['L1', 'L1', 'H3'],
            [1, 2, 3, 4, 5, 6, 7, 8]),
            [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [1, 3, 5, 7],
            [2, 4, 6, 8]])
        self.assertEqual(core.element_face_nodes(['H3', 'H3', 'H3'],
            [1, 2, 3, 4, 5, 6, 7, 8]),
            [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [1, 3, 5, 7],
            [2, 4, 6, 8]])
        self.assertEqual(core.element_face_nodes(['L2', 'L1', 'L1'],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            [[1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [1, 2, 3, 7, 8, 9],
            [4, 5, 6, 10, 11, 12],
            [1, 4, 7, 10],
            [3, 6, 9, 12]])
        self.assertEqual(core.element_face_nodes(['L1', 'L2', 'L1'],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            [[1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [1, 2, 7, 8],
            [5, 6, 11, 12],
            [1, 3, 5, 7, 9, 11],
            [2, 4, 6, 8, 10, 12]])
        self.assertEqual(core.element_face_nodes(['L1', 'L1', 'L2'],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            [[1, 2, 3, 4],
            [9, 10, 11, 12],
            [1, 2, 5, 6, 9, 10],
            [3, 4, 7, 8, 11, 12],
            [1, 3, 5, 7, 9, 11],
            [2, 4, 6, 8, 10, 12]])
        self.assertEqual(core.element_face_nodes(['L2', 'L2', 'L2'],
            range(1, 28)),
            [[1, 2, 3, 4, 5, 6, 7, 8, 9],
            [19, 20, 21, 22, 23, 24, 25, 26, 27],
            [1, 2, 3, 10, 11, 12, 19, 20, 21],
            [7, 8, 9, 16, 17, 18, 25, 26, 27],
            [1, 4, 7, 10, 13, 16, 19, 22, 25],
            [3, 6, 9, 12, 15, 18, 21, 24, 27]])
    
    def test_add_face_to_mesh(self):
        mesh = morphic.Mesh()
        face = mesh.add_face(['L1', 'L1'], [7, 3, 22, 2], element=3)
        face = mesh.add_face(['L1', 'L2'], [5, 15, 22, 2, 7, 4], element=6)
        self.assertEqual(len(mesh.faces.keys()), 2)
        self.assertTrue('_2_3_7_22' in mesh.faces.keys())
        self.assertTrue('_2_4_5_7_15_22' in mesh.faces.keys())
        face = mesh.faces['_2_3_7_22']
        # self.assertEqual(face.interp, ['L1', 'L1'])
        # self.assertEqual(face.node_ids, [7, 3, 22, 2])
        # self.assertEqual(face.element_ids, [3])
        face = mesh.faces['_2_4_5_7_15_22']
        # self.assertEqual(face.interp, ['L1', 'L2'])
        # self.assertEqual(face.node_ids, [5, 15, 22, 2, 7, 4])
        # self.assertEqual(face.element_ids, [6])
        
    def test_add_duplicate_face_to_mesh(self):
        mesh = morphic.Mesh()
        face = mesh.add_face(1, 0)
        face = mesh.add_face(1, 0)
        self.assertEqual(len(mesh.faces.keys()), 1)
        self.assertTrue('_2_3_7_22' in mesh.core.keys())
        face = mesh.faces['_2_3_7_22']
        # self.assertEqual(face.interp, ['L1', 'L1'])
        # self.assertEqual(face.node_ids, [7, 3, 22, 2])
        # self.assertEqual(face.element_ids, [3, 6])
    
if __name__ == "__main__":
    unittest.main()
