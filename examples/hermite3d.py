import scipy

import morphic
reload(morphic)

from morphic import viewer

def cH_node_values(values):
    return numpy.array([
        [values[0], 1, 0.5, 0, 0, 0, 0, 0],
        [values[1], 0, 1, 0, 0, 0, 0, 0],
        [values[2], 0, 0, 0, 1, 1, 0, 0]])

if "fig" not in locals():
    fig = viewer.Figure()
    
mesh = morphic.Mesh()
mesh.auto_add_faces = True

Xn = scipy.array([
    cH_node_values([0,0,0]),
    cH_node_values([1,0,0]),
    cH_node_values([0,1,0]),
    cH_node_values([1,1,0]),
    cH_node_values([0,0,1]),
    cH_node_values([1,0,1]),
    cH_node_values([0,1,1]),
    cH_node_values([1,1,1])])

for i, xn in enumerate(Xn):
    mesh.add_stdnode(i + 1, xn)

mesh.add_element(1, ['H3', 'H3', 'H3'], [1, 2, 3, 4, 5, 6, 7, 8])

mesh.generate()

X, T = mesh.get_faces(32)

fig.plot_surfaces('Faces', X, T)


# res = 30
# g = numpy.mgrid[0:res, 0:res, 0:res]
# Xig = (1./(res - 1)) * numpy.array([g[2].flatten(), g[1].flatten(), g[0].flatten()]).T
# Xg = mesh.evaluate(1, Xig)

# fig.plot_points("Xg", Xg, size=0)
