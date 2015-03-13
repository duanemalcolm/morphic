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
from morphic import metadata

class TestNode(unittest.TestCase):
    """Unit tests for morphic Node superclass."""

    def test_init_attributes(self):
        mesh = mesher.Mesh()
        self.assertTrue(isinstance(mesh.metadata, metadata.Metadata))
    
    def test_set_attribute(self):
        mesh = mesher.Mesh()
        mesh.metadata.age = 23
        self.assertEqual(mesh.metadata.age, 23)
    
    def test_set_item(self):
        mesh = mesher.Mesh()
        mesh.metadata['age'] = 23
        self.assertEqual(mesh.metadata.age, 23)
    
    def test_set_method(self):
        mesh = mesher.Mesh()
        mesh.metadata.set('age', 23)
        self.assertEqual(mesh.metadata.age, 23)
    
    def test_get_attribute(self):
        mesh = mesher.Mesh()
        mesh.metadata.age = 23
        self.assertEqual(mesh.metadata.age, 23)
    
    def test_get_item(self):
        mesh = mesher.Mesh()
        mesh.metadata.age = 23
        self.assertEqual(mesh.metadata['age'], 23)
    
    def test_get_method(self):
        mesh = mesher.Mesh()
        mesh.metadata.age = 23
        self.assertEqual(mesh.metadata.get('age'), 23)
    
    def test_delete_method(self):
        mesh = mesher.Mesh()
        self.assertFalse('age' in mesh.metadata.keys())
        mesh.metadata.age = 23
        self.assertTrue('age' in mesh.metadata.keys())
        mesh.metadata.delete('age')
        self.assertFalse('age' in mesh.metadata.keys())
    
    def test_delete_method_nonexistant(self):
        mesh = mesher.Mesh()
        mesh.metadata.delete('age')
    
    def test_get_method_nonexistant(self):
        mesh = mesher.Mesh()
        age = mesh.metadata.get('age')
        self.assertEqual(age, None)
        
    def test_get_method_nonexistant_default(self):
        mesh = mesher.Mesh()
        age = mesh.metadata.get('age', default=25)
        self.assertEqual(age, 25)
    
    def test_get_method_existant_default(self):
        mesh = mesher.Mesh()
        mesh.metadata.age = 23
        age = mesh.metadata.get('age', default=25)
        self.assertEqual(age, 23)
    
    def test_set_method_overwrite(self):
        mesh = mesher.Mesh()
        mesh.metadata.age = 23
        self.assertEqual(mesh.metadata.age, 23)
        result = mesh.metadata.set('age', 25)
        self.assertTrue(result)
        self.assertEqual(mesh.metadata.age, 25)
        
    def test_set_method_no_overwrite(self):
        mesh = mesher.Mesh()
        mesh.metadata.age = 23
        self.assertEqual(mesh.metadata.age, 23)
        result = mesh.metadata.set('age', 25, False)
        self.assertFalse(result)
        self.assertEqual(mesh.metadata.age, 23)
        
    def test_contains(self):
        mesh = mesher.Mesh()
        mesh.metadata.age = 23
        self.assertTrue('age' in mesh.metadata)
        self.assertFalse('height' in mesh.metadata)
    
    def test_length(self):
        mesh = mesher.Mesh()
        mesh.metadata.name = 'Joe Bloggs'
        mesh.metadata.age = 23
        self.assertEqual(len(mesh.metadata), 2)
        
    def test_data_types(self):
        mesh = mesher.Mesh()
        mesh.metadata.name = 'Joe Bloggs'
        self.assertEqual(mesh.metadata.name, 'Joe Bloggs')
        mesh.metadata.age = 23
        self.assertEqual(mesh.metadata.age, 23)
        mesh.metadata.height = 1.68
        self.assertEqual(mesh.metadata.height, 1.68)
        
    def test_get_dict(self):
        mesh = mesher.Mesh()
        mesh.metadata.name = 'Joe Bloggs'
        mesh.metadata.age = 23
        mesh.metadata.height = 1.68
        d = mesh.metadata.get_dict()
        
        self.assertEqual(d,
                {'name':'Joe Bloggs', 'age':23, 'height':1.68})
    
    def test_set_dict(self):
        d = {'name':'Joe Bloggs', 'age':23, 'height':1.68}
        mesh = mesher.Mesh()
        mesh.metadata.set_dict(d)
        self.assertEqual(mesh.metadata.name, 'Joe Bloggs')
        self.assertEqual(mesh.metadata.age, 23)
        self.assertEqual(mesh.metadata.height, 1.68)
       
        
if __name__ == "__main__":
    unittest.main()
