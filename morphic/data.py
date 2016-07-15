import os
import numpy
import metadata


class Data:
    
    def __init__(self, filepath=None):
        self._version = 1
        self.label = None
        self.created_at = None
        self.saved_at = None
        self.units = None
        self.metadata = metadata.Metadata()
        self.values = None
        self.kdtree = None
        if filepath is not None:
            self.load(filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            if os.path.splitext(filepath)[1] == '.vtk':
                self.load_vtk(filepath)
            else:
                self.load_hdf5(filepath)
        else:
            print 'Data file does not exist'

    def save(self, filepath, type='hdf5'):
        self.save_hdf5(filepath)

    def load_vtk(self, filepath):
        fp = open(filepath, 'r')
        for i in range(5):
            fp.readline()  # skipping header lines
        self.values = numpy.array([[float(x) for x in line.split()] for line in fp])
        fp.close()

    def load_hdf5(self, filepath):
        import tables
        
        def get_attribute(h5node, key, default=None):
            if key in h5node._v_attrs:
                return h5node._v_attrs[key]
            return default
        
        h5f = tables.open_file(filepath, 'r')
        
        self._version = get_attribute(h5f.root, 'version')
        self.created_at = get_attribute(h5f.root, 'created_at')
        self._saved_at = get_attribute(h5f.root, 'saved_at')
        self.label = get_attribute(h5f.root, 'label')
        self.units = get_attribute(h5f.root, 'units')
        
        if 'metadata' in h5f.root:
            self.metadata.load_pytables(h5f.root.metadata)
        
        self.values = h5f.root.values.read()
        
        h5f.close()

    def save_hdf5(self, filepath):
        import tables
        import datetime
            
        h5f = tables.open_file(filepath, 'w')
        filters = tables.Filters(complevel=5, complib='zlib', shuffle=True)
        h5f.set_node_attr(h5f.root, 'version', self._version)
        h5f.set_node_attr(h5f.root, 'created_at', self.created_at)
        h5f.set_node_attr(h5f.root, 'saved_at', datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
        h5f.set_node_attr(h5f.root, 'label', self.label)
        h5f.set_node_attr(h5f.root, 'units', self.units)
        
        metadata_node = h5f.create_group(h5f.root, 'metadata')
        self.metadata.save_pytables(metadata_node)
        
        params = h5f.create_carray(h5f.root, 'values', tables.Float64Atom(), self.values.shape, filters=filters)
        params[:] = self.values
        
        h5f.close()   
