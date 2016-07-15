import unittest
import morphic
import numpy.testing as npt


class TestData(unittest.TestCase):
    """Unit tests for morphic Node superclass."""

    def test_init_attributes(self):
        data = morphic.Data()
        self.assertTrue(data.values is None)
        self.assertTrue(data.kdtree is None)

    def test_load_vtk(self):
        data = morphic.Data('data/chest.vtk')
        self.assertTrue(data.values.shape[0] == 4)
        self.assertTrue(data.values.shape[1] == 3)
        npt.assert_almost_equal(data.values[0, :], [110.025, 22.275, -83.025])
        npt.assert_almost_equal(data.values[1, :], [111.375, 22.95, -83.025])
        npt.assert_almost_equal(data.values[2, :], [110.7, 22.275, -83.025])
        npt.assert_almost_equal(data.values[3, :], [112.7, 22.256, -83.251, ])

    def test_save_load_hdf5(self):
        data = morphic.Data('data/chest.vtk')
        data.save('data/chest.data')
        data = morphic.Data('data/chest.data')
        self.assertTrue(data.values.shape[0] == 4)
        self.assertTrue(data.values.shape[1] == 3)
        npt.assert_almost_equal(data.values[0, :], [110.025, 22.275, -83.025])
        npt.assert_almost_equal(data.values[1, :], [111.375, 22.95, -83.025])
        npt.assert_almost_equal(data.values[2, :], [110.7, 22.275, -83.025])
        npt.assert_almost_equal(data.values[3, :], [112.7, 22.256, -83.251, ])

    def test_metadata_hdf5(self):
        data = morphic.Data('data/chest.vtk')
        data.metadata.name = 'Joe Bloggs'
        data.metadata.age = 42
        data.metadata.height = 1.87
        data.metadata.address = {'street': '123 Abc St', 'city': 'Big City'}
        data.metadata.scans = ['foot', 'brain']
        data.save('data/chest.data')

        data = morphic.Data('data/chest.data')
        m = data.metadata
        self.assertEqual(m.name, 'Joe Bloggs')
        self.assertEqual(m.age, 42)
        self.assertEqual(m.height, 1.87)
        self.assertEqual(m.address, {'street': '123 Abc St', 'city': 'Big City'})
        self.assertEqual(m.scans, ['foot', 'brain'])


if __name__ == "__main__":
    unittest.main()
