import scipy

import morphic
reload(morphic)

from morphic import viewer

if "fig" not in locals():
    fig = viewer.Figure()
    
mesh = morphic.Mesh()
mesh.auto_add_faces = True

Xn = scipy.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    [0.5, -0.3, 0], [0.5, 1.2, 0], [0.5, -0.5, 1], [0.5, 1.2, 1]])

for i, xn in enumerate(Xn):
    mesh.add_stdnode(i + 1, xn)

mesh.add_element(1, ['L1', 'L1', 'L1'], [1, 2, 3, 4, 5, 6, 7, 8])
mesh.add_element(2, ['L2', 'L1', 'L1'], [1, 9, 2, 3, 10, 4, 5, 11, 6, 7, 12, 8])

mesh.generate()

X, T = mesh.get_faces()

fig.plot_surfaces('Faces', X, T, color=(0,0,1), opacity=0.2)

# res = 30
# g = numpy.mgrid[0:res, 0:res, 0:res]
# Xig = (1./(res - 1)) * numpy.array([g[2].flatten(), g[1].flatten(), g[0].flatten()]).T
# Xg = mesh.evaluate(2, Xig)

# fig.plot_points("Xg", Xg, size=0)
