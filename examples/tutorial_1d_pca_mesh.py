# import modules
import os
import sys
import scipy
import random
import pylab
import mdp
import morphic

testdatadir = 'data'
docimagedir = os.path.join('..', 'doc', 'images')
testdata = {}

plot = False
update = False
test = False

if 'plot' in sys.argv:
    plot = True
if 'update' in sys.argv:
    update = True
    plot = True
if 'test' in sys.argv:
    test = True

# Mesh randomiser
def randomise_mesh(mesh):
    import random
    import scipy
    
    xa = random.gauss(-2.5, 1)
    xb = random.gauss(2.5, 1)
    ya = random.gauss(-0.1, 0.5)
    yb = random.gauss(0.3, 0.5)
    dya = random.gauss(0, 0.1)
    dyb = random.gauss(0, 0.2)

    mesh.nodes[1].values = scipy.array([[xa, 5], [ya, dya + dyb]])
    mesh.nodes[2].values = scipy.array([[xb, 5], [yb, -dya]])
    
    return mesh
 
# GENERATE POPULATION OF MESHES
mesh = morphic.Mesh()
mesh.add_stdnode(1, scipy.zeros((2, 2)))
mesh.add_stdnode(2, scipy.zeros((2, 2)))
mesh.add_element(1, ['H3'], [1, 2])
mesh.generate()

# Now we can randomise the mesh
mesh = randomise_mesh(mesh)

if plot:
    pylab.figure(1)
    pylab.clf()
    pylab.hold('on')

# Create a population of meshes
Npop = 100 # number in the population
Xp = scipy.zeros((8, Npop))

for i in range(Npop):
    mesh = randomise_mesh(mesh)
    Xp[:,i] = mesh.elements[1].evaluate(
            [[0], [1/3.], [2/3.], [1]]).flatten()

    # and plot if need be
    if plot:
        Xl = mesh.get_lines(32)[0]
        pylab.plot(Xl[:,0], Xl[:,1])

if plot:
    pylab.title('Population of %d Lines' % (Npop))
    pylab.hold('off')
    filepath = os.path.join(docimagedir, 'population_of_meshes.png')
    pylab.savefig(filepath)

if update:
    import pickle
    filepath = os.path.join(testdatadir, 'pca_random_data.pkl')
    pickle.dump(Xp, open(filepath, 'w'))

if test:
    import pickle
    filepath = os.path.join(testdatadir, 'pca_random_data.pkl')
    Xp = pickle.load(open(filepath, 'r'))

# PERFORM PCA
pca = mdp.nodes.PCANode(svd=True)
pca.execute(Xp.T)
#~ print pca.avg    # Average mesh
#~ print pca.v    # The modes
#~ print pca.d   # Variance shows there are 6 significant modes.
# tag end pca

# GENERATE PCA MESH
pcamesh = morphic.Mesh()

Nnodes = 4 # number of nodes
Ndims = 2 # number of dimensions
Ncomps = 1 # number of components
Nmodes = 6 # number of modes

# Initialise the weights
weights = scipy.zeros((7)) # Average mesh plus 6 modes
weights[0] = 1.0 # the weight on the average mesh
pcamesh.add_stdnode('weights', weights, group='pca_init')

# Initialise the variance. Note, the variance is sqrt'ed so the weights
# can be defined as standard deviations
variance = scipy.zeros((7)) # Average mesh plus 7 modes
variance[0] = 1.0 # Set to 1 for the average mesh
variance[1:] = scipy.sqrt(pca.d[:6]) # variance for the 7 modes
pcamesh.add_stdnode('variance', variance, group='pca_init')

# Add the four cubic-Lagrange nodes
xn = scipy.zeros((Ndims, Ncomps, Nmodes+1)) # node values array
for i in range(Nnodes):
    idx = i * Ndims
    # Add a PCA node using the node values, weights and variance
    xn[:, 0, 0] = pca.avg[0, idx:idx+Ndims] # add mean values
    xn[:, 0, 1:] = pca.v[idx:idx+Ndims,:Nmodes] # add 7 mode values
    pcamesh.add_pcanode(i+1, xn, 'weights', 'variance', group='pca')

# Add element
pcamesh.add_element(1, ['L3'], [1, 2, 3, 4])

pcamesh.generate()

# PCA modes element
#~ #pcamesh.add_element('pca1', ['L3'], ['pca1', 'pca2', 'pca3', 'pca4'])
#~ #pcamesh.generate()

#~ #print 'Eval Node:', pcamesh.elements[1].evaluate([[0.333]])
#~ #print 'PCA Node:', pcamesh.elements['pca1'].evaluate([[0.333]])

if update:
    filepath = os.path.join(testdatadir, 'pca_mesh_orig.mesh')
    pcamesh.save(filepath)

filepath = os.path.join(testdatadir, 'pca_mesh_test.mesh')
pcamesh.save(filepath)

# Plot each mode into a subplot
def plot_modes(pcamesh, mode, sp, title=None):
    colors = {-2:'r', -1:'y', 0:'g', 1:'c', 2:'b'}
    
    pcamesh.nodes['weights'].values[1:] = 0 # Reset weights to zero
    pcamesh.update_pca_nodes()
    
    pylab.subplot(sp)
    pylab.hold('on')
    # Plotting modes over a range of weights values
    for w in scipy.linspace(-2, 2, 5):
        pcamesh.nodes['weights'].values[mode] = w
        pcamesh.update_pca_nodes()
        Xl = pcamesh.get_lines(32)[0]
        pylab.plot(Xl[:,0], Xl[:,1], colors[w])
        
    pylab.hold('off')
    pylab.xticks([])
    pylab.yticks([])
    if title != None:
        pylab.title(title)

# end plot loop
        
if plot:
    pylab.figure(2)
    pylab.clf()
    plot_modes(pcamesh, 1, 221, title='Mode 1')
    plot_modes(pcamesh, 2, 222, title='Mode 2')
    plot_modes(pcamesh, 3, 223, title='Mode 3')
    plot_modes(pcamesh, 4, 224, title='Mode 4') 
    pylab.suptitle('Top 4 Modes from PCA', fontsize=14)
    pylab.show()
    filepath = os.path.join(docimagedir, 'top_four_modes.png')
    pylab.savefig(filepath)

if update:
    pcamesh.nodes['weights'].values[1:] = 0
    pcamesh.update_pca_nodes()
    testdata['Xn0'] = pcamesh.get_nodes()
    pcamesh.nodes['weights'].values[1] = -1.3
    pcamesh.update_pca_nodes()
    testdata['Xn1'] = pcamesh.get_nodes()
    pcamesh.nodes['weights'].values[3] = 0.7
    pcamesh.update_pca_nodes()
    testdata['Xn2'] = pcamesh.get_nodes()
    
    import pickle
    filepath = os.path.join(testdatadir, 'pca_node_values.pkl')
    pickle.dump(testdata, open(filepath, 'w'))


