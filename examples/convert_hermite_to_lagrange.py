import pickle
import scipy

import morphic
reload(morphic)

import morphic.utils
reload(morphic.utils)

from morphic import viewer

if "fig" not in locals():
    fig = viewer.Figure()

Xn = pickle.load(open('data/prone.pkl', 'r'))
Elements = pickle.load(open('data/elements.pkl', 'r'))
    
mesh = morphic.Mesh()
mesh.auto_add_faces = True

for i, xn in enumerate(Xn):
    mesh.add_stdnode(i + 1, xn.reshape((3, 8)))

for i, nodes in enumerate(Elements):
    mesh.add_element(i + 1, ['H3', 'H3', 'H3'], nodes)

mesh.generate()

lmesh = morphic.utils.convert_hermite_lagrange(mesh, tol=1.)



Xs, Ts = mesh.get_faces(16, exterior_only=False)
Xls, Tls = lmesh.get_faces(16, exterior_only=False)
Xn = lmesh.get_nodes()

# fig.plot_surfaces('surface', Xs, Ts, color=(1, 0.8, 0.8))
fig.plot_surfaces('lsurface', Xls, Tls, color=(0, 0.8, 1))
fig.plot_points('nodes', Xn, size=1)
