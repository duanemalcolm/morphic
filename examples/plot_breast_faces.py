import pickle
import scipy

import morphic
reload(morphic)

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

X, T = mesh.get_faces(16)

fig.plot_surfaces('Faces', X, T, color=(1,0,0), opacity=0.5)

# Xn = pickle.load(open('data/supine.pkl', 'r'))
# Elements = pickle.load(open('data/elements.pkl', 'r'))
    
# mesh = morphic.Mesh()
# mesh.auto_add_faces = True

# for i, xn in enumerate(Xn):
#     mesh.add_stdnode(i + 1, xn.reshape((3, 8)))

# for i, nodes in enumerate(Elements):
#     mesh.add_element(i + 1, ['H3', 'H3', 'H3'], nodes)

# mesh.generate()

# X, T = mesh.get_faces(32)

# fig.plot_surfaces('Faces2', X, T, color=(0,0,1), opacity=0.5)
