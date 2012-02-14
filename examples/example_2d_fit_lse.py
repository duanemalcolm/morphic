import os
import sys

example = 'example_2d_fit_lse'
title = 'Fit 2D cubic-Hermite elements to sin^2'
testdatadir = os.path.join('..', 'test', 'data')
docimagedir = os.path.join('..', 'doc', 'images')


def plot_mesh_data(label, mesh, Xd, filepath):
    # Get node values and surface
    Xn = mesh.get_nodes()
    Xs,Ts = mesh.get_surfaces(res=24)
    
    view = (45.000000000000085,
             54.735610317245204,
             14.341069369356839,
             array([ 0.58479455,  0.56426392,  0.8258849 ]))
    
    # Plotting
    from morphic import viewer
    reload(morphic)
    
    try: S.clear(label)
    except: S = viewer.Scenes(label, bgcolor=(1,1,1))
    
    S.plotPoints('nodes', Xn, color=(0,1,0), size=0.1)
    S.plotPoints('data', Xd[::7,:], color=(1,0,0), mode='point')
    S.plotSurfaces('surface', Xs, Ts, color=(0.2,0.5,1))
    
    S.set_view(view)
    
    import mayavi
    mayavi.mlab.savefig(filepath, size=(800,600))
    
    return S


# sphinx tag start import
import scipy
import morphic
# sphinx tag end import

# sphinx tag start generate node values
# Generate a regular grid of X, Y and Z values
pi = scipy.pi
Xn = scipy.array([
        [-pi, -pi, 0],
        [  0, -pi, 0],
        [ pi, -pi, 0],
        [-pi,   0, 0],
        [  0,   0, 0],
        [ pi,   0, 0],
        [-pi,  pi, 0],
        [  0,  pi, 0],
        [ pi,  pi, 0]])

# Default derivatives values for cubic-Hermite nodes
deriv = scipy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# sphinx tag end generate node values

# sphinx tag start generate mesh
# Create mesh
mesh = morphic.Mesh()

# Add nodes
for i, xn in enumerate(Xn):
    xn_ch = scipy.append(scipy.array([xn]).T, deriv, axis=1)
    mesh.add_stdnode(i+1, xn_ch)

# Add elements
mesh.add_element(1, ['H3', 'H3'], [1, 2, 4, 5])
mesh.add_element(2, ['H3', 'H3'], [2, 3, 5, 6])
mesh.add_element(3, ['H3', 'H3'], [4, 5, 7, 8])
mesh.add_element(4, ['H3', 'H3'], [5, 6, 8, 9])

# Generate the mesh
mesh.generate()
# sphinx tag end generate mesh

# Generate a data cloud for fitting
xd = scipy.linspace(-scipy.pi, scipy.pi, 200)
X, Y = scipy.meshgrid(xd, xd.T)
Xd = scipy.array([
        X.reshape((X.size)),
        Y.reshape((Y.size)),
        scipy.zeros((X.size, 1))]).T
Xd[:, 2] = (scipy.cos(Xd[:, 0])+1) * (scipy.cos(Xd[:, 1])+1)
# sphinx tag end generate data

# Plottin'
filepath = os.path.join(docimagedir, example+'_mesh.png')
plot1 = plot_mesh_data('Mesh_and_Data', mesh, Xd, filepath)



#### FIT PART 1: NO FIXED NODE VALUES ###
# Generate a fit
fit = morphic.Fit()

# Add a grid of 10x10 points on each element to fit to the data cloud
Xi_fit = mesh.elements[1].grid(res=10)
for elem in mesh.elements:
    for xi in Xi_fit:
        fit.bind_element_point(elem.id, [xi], 'datacloud')

# Update and generate the fit data structures using the mesh.
fit.update_from_mesh(mesh)

# Add data to fit
fit.set_data('datacloud', Xd)
fit.generate_fast_data()

# Fit
fit.invert_matrix()
mesh = fit.solve(mesh, niter=100, output=True)

### PLOTTIN' ###
filepath = os.path.join(docimagedir, example+'_fit1.png')
plot2 = plot_mesh_data('Initial_Fit', mesh, Xd, filepath)



#### FIT PART 2: FIXED NODE VALUES ###
# Fix node values [x, y, z, dz/dxi1, dz/dxi2]
fix_nodes = [
        ['-pi', '-pi', 'zero'],
        ['zero', '-pi', 'zero'],
        ['pi', '-pi', 'zero'],
        ['-pi', 'zero', 'zero'],
        ['zero', 'zero', 'four'],
        ['pi', 'zero', 'zero'],
        ['-pi', 'pi', 'zero'],
        ['zero', 'pi', 'zero'],
        ['pi', 'pi', 'zero']]

weight1 = 100
for i, fix in enumerate(fix_nodes):
    node_id = i + 1
    # Fix node values
    fit.bind_node_value(node_id, 0, 0, fix[0], weight=weight1)
    fit.bind_node_value(node_id, 1, 0, fix[1], weight=weight1)
    fit.bind_node_value(node_id, 2, 0, fix[2], weight=weight1)
    
    # Fix node derivatives
    fit.bind_node_value(node_id, 0, 2, 'zero', weight=weight1) # dx/dxi2=0
    fit.bind_node_value(node_id, 1, 1, 'zero', weight=weight1) # dy/dxi1=0
    fit.bind_node_value(node_id, 2, 1, 'zero', weight=weight1) # dz/dxi1=0
    fit.bind_node_value(node_id, 2, 2, 'zero', weight=weight1) # dz/dxi2=0
    
# Flattens the corners. Stops z from dipping below zero, d2z/(dxi1.dxi2)=0
weight2 = 100
fit.bind_node_value(1, 2, 3, 'zero', weight=weight2)
fit.bind_node_value(3, 2, 3, 'zero', weight=weight2)
fit.bind_node_value(7, 2, 3, 'zero', weight=weight2)
fit.bind_node_value(9, 2, 3, 'zero', weight=weight2)

fit.update_from_mesh(mesh)

# Add data to fit
fit.set_data('pi', scipy.pi)
fit.set_data('-pi', -scipy.pi)
fit.set_data('zero', 0)
fit.set_data('four', 4.)
fit.generate_fast_data()

# Fit
fit.invert_matrix()
mesh = fit.solve(mesh, niter=100, output=True)

# Plottin'
filepath = os.path.join(docimagedir, example+'_fit2.png')
plot3 = plot_mesh_data('Final_Fit', mesh, Xd, filepath)


# sphinx start get node values and surface
Xn = mesh.get_nodes()
Xs,Ts = mesh.get_surfaces(res=24)

# sphinx start plotting
from morphic import viewer

S = viewer.Scenes('MyPlot', bgcolor=(1,1,1))

S.plotPoints('nodes', Xn, color=(0,1,0), size=0.1)
S.plotSurfaces('surface', Xs, Ts, color=(0.2,0.5,1))
S.plotPoints('data', Xd[::7,:], color=(1,0,0), mode='point')
# sphinx tag end plotting
