import os
import sys

action = None
if len(sys.argv) > 1:
    action = sys.argv[1]


example = 'create_mesh'
title = 'Create a one element biquadratic mesh'
testdatadir = 'data'
docimagedir = os.path.join('..', 'doc', 'images')


def plot_mesh_data(label, mesh, filepath):
    # Get node values and surface
    Xn = mesh.get_nodes()
    Xs,Ts = mesh.get_surfaces(res=32)
    
    view = (-52.321030743641359,
         61.965935835282515,
         6.0000000000000009,
         array([ 2.1530268 ,  1.04356322, -0.11091185]))
        
    # Plotting
    import morphic.viewer
    
    try: S.clear(label)
    except: S = morphic.viewer.Scenes(label, bgcolor=(1,1,1))
    
    S.plot_points('nodes', Xn, color=(1,0,1), size=0.1)
    S.plot_surfaces('surface', Xs, Ts, scalars=Xs[:,2])
    
    S.set_view(view)
    
    import mayavi
    mayavi.mlab.savefig(filepath, size=(600,400))
    
    return S


# sphinx tag start import
import scipy
import morphic

# sphinx tag start generate node values
# Generate a regular grid of X, Y and Z values
pi = scipy.pi

# Create mesh
mesh = morphic.Mesh()

# Add nodes
mesh.add_stdnode(1, [0, 0, 0])
mesh.add_stdnode(2, [1, 0, 0])
mesh.add_stdnode(3, [2, 0, 0])
mesh.add_stdnode(4, [3, 0, 0])
mesh.add_stdnode(5, [4, 0, 0])
mesh.add_stdnode(6, [0, 1, 0])
mesh.add_stdnode(7, [1, 1, 1])
mesh.add_stdnode(8, [2, 1, 0])
mesh.add_stdnode(9, [3, 1, -1])
mesh.add_stdnode(10, [4, 1, 0])
mesh.add_stdnode(11, [0, 2, 0])
mesh.add_stdnode(12, [1, 2, 0])
mesh.add_stdnode(13, [2, 2, 0])
mesh.add_stdnode(14, [3, 2, 0])
mesh.add_stdnode(15, [4, 2, 0])

# Add elements
mesh.add_element(1, ['L2', 'L2'], [1, 2, 3, 6, 7, 8, 11, 12, 13])
mesh.add_element(2, ['L2', 'L2'], [3, 4, 5, 8, 9, 10, 13, 14, 15])

# Generate the mesh
mesh.generate()

# Plotting
if action in ['update', 'plot']:
    filepath = os.path.join(docimagedir, example+'_mesh.png')
    plot1 = plot_mesh_data('mesh', mesh, filepath)

# sphinx start get node values and surface
Xn = mesh.get_nodes()
Xs,Ts = mesh.get_surfaces(res=24)
# sphinx end get node values and surface

if action == 'update':
    import pickle
    
    Xn = mesh.get_nodes()
    Xs, Ts = mesh.get_surfaces()
    
    data = {'Xn': Xn, 'Xs': Xs, 'Ts': Ts}
    filepath = os.path.join(testdatadir, example+'.pkl')
    pickle.dump(data, open(filepath, 'w'))

filepath = os.path.join(testdatadir, example+'.mesh')

# save mesh
mesh.save(filepath)

# load mesh
mesh = morphic.Mesh(filepath)

# OR
mesh = morphic.Mesh()
mesh.load(filepath)

# end load mesh
