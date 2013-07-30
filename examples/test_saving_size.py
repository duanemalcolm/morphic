import tables
import morphic
import numpy
import os
reload(morphic)

mesh = morphic.Mesh('data/prone.mesh')

mesh.created_at
mesh.save("data/prone_h5py.mesh", 'h5py')
mesh.save("data/prone_pytables.mesh", 'pytables')

mesh = morphic.Mesh('data/prone_pca.mesh')
mesh.save("data/prone_pca_h5py.mesh", 'h5py')
mesh.save("data/prone_pca_pytables.mesh", 'pytables')

print '%9s %8s %8s' % ('', 'Standard', 'PCA')
print '%8s: %7dk %7dk' % ('Pickle',
                          os.path.getsize('data/prone.mesh') / 1000, 
                          os.path.getsize('data/prone_pca.mesh') / 1000)
print '%8s: %7dk %7dk' % ('PyTables',
                          os.path.getsize('data/prone_pytables.mesh') / 1000, 
                          os.path.getsize('data/prone_pca_pytables.mesh') / 1000)
print '%8s: %7dk %7dk' % ('H5Py',
                          os.path.getsize('data/prone_h5py.mesh') / 1000, 
                          os.path.getsize('data/prone_pca_h5py.mesh') / 1000)