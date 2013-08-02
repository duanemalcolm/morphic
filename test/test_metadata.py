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

    def test_init_attributes(self):
        mesh = mesher.Mesh()
        self.assertTrue(isinstance(mesh.userdata, mesher.Metadata))
    
    def test_set_attribute(self):
        mesh = mesher.Mesh()
        mesh.userdata.age = 23
        self.assertEqual(mesh.userdata.age, 23)
    
    def test_set_item(self):
        mesh = mesher.Mesh()
        mesh.userdata['age'] = 23
        self.assertEqual(mesh.userdata.age, 23)
    
    def test_set_method(self):
        mesh = mesher.Mesh()
        mesh.userdata.set('age', 23)
        self.assertEqual(mesh.userdata.age, 23)
    
    def test_get_attribute(self):
        mesh = mesher.Mesh()
        mesh.userdata.age = 23
        self.assertEqual(mesh.userdata.age, 23)
    
    def test_get_item(self):
        mesh = mesher.Mesh()
        mesh.userdata.age = 23
        self.assertEqual(mesh.userdata['age'], 23)
    
    def test_get_method(self):
        mesh = mesher.Mesh()
        mesh.userdata.age = 23
        self.assertEqual(mesh.userdata.get('age'), 23)
    
    def test_delete_method(self):
        mesh = mesher.Mesh()
        self.assertFalse('age' in mesh.userdata.keys())
        mesh.userdata.age = 23
        self.assertTrue('age' in mesh.userdata.keys())
        mesh.userdata.delete('age')
        self.assertFalse('age' in mesh.userdata.keys())
    
    def test_delete_method_nonexistant(self):
        mesh = mesher.Mesh()
        mesh.userdata.delete('age')
    
    def test_get_method_nonexistant(self):
        mesh = mesher.Mesh()
        age = mesh.userdata.get('age')
        self.assertEqual(age, None)
        
    def test_get_method_nonexistant_default(self):
        mesh = mesher.Mesh()
        age = mesh.userdata.get('age', 25)
        self.assertEqual(age, 25)
    
    def test_get_method_existant_default(self):
        mesh = mesher.Mesh()
        mesh.userdata.age = 23
        age = mesh.userdata.get('age', 25)
        self.assertEqual(age, 23)
    
    def test_set_method_overwrite(self):
        mesh = mesher.Mesh()
        mesh.userdata.age = 23
        self.assertEqual(mesh.userdata.age, 23)
        result = mesh.userdata.set('age', 25)
        self.assertTrue(result)
        self.assertEqual(mesh.userdata.age, 25)
        
    def test_set_method_no_overwrite(self):
        mesh = mesher.Mesh()
        mesh.userdata.age = 23
        self.assertEqual(mesh.userdata.age, 23)
        result = mesh.userdata.set('age', 25, False)
        self.assertFalse(result)
        self.assertEqual(mesh.userdata.age, 23)
        
    def test_contains(self):
        mesh = mesher.Mesh()
        mesh.userdata.age = 23
        self.assertTrue('age' in mesh.userdata)
        self.assertFalse('height' in mesh.userdata)
    
    def test_length(self):
        mesh = mesher.Mesh()
        mesh.userdata.name = 'Joe Bloggs'
        mesh.userdata.age = 23
        self.assertEqual(len(mesh.userdata), 2)
        
    def test_data_types(self):
        mesh = mesher.Mesh()
        mesh.userdata.name = 'Joe Bloggs'
        self.assertEqual(mesh.userdata.name, 'Joe Bloggs')
        mesh.userdata.age = 23
        self.assertEqual(mesh.userdata.age, 23)
        mesh.userdata.height = 1.68
        self.assertEqual(mesh.userdata.height, 1.68)
        
    def test_get_dict(self):
        mesh = mesher.Mesh()
        mesh.userdata.name = 'Joe Bloggs'
        mesh.userdata.age = 23
        mesh.userdata.height = 1.68
        d = mesh.userdata.get_dict()
        print d
        self.assertEqual(d,
                {'name':'Joe Bloggs', 'age':23, 'height':1.68})
    
    def test_set_dict(self):
        d = {'name':'Joe Bloggs', 'age':23, 'height':1.68}
        mesh = mesher.Mesh()
        mesh.userdata.set_dict(d)
        self.assertEqual(mesh.userdata.name, 'Joe Bloggs')
        self.assertEqual(mesh.userdata.age, 23)
        self.assertEqual(mesh.userdata.height, 1.68)
       
        
if __name__ == "__main__":
    unittest.main()
