import xviewer
import fieldscape.mesher
reload(fieldscape.mesher)

mesh = fieldscape.mesher.Mesh()
mesh.add_node(1, [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
mesh.add_node(2, [[1.2, 1, 0, 0], [0, 0, 1, 0], [0.3, 0, 0, 0]])
mesh.add_node(3, [[0.1, 1, 0, 0], [0.8, 0, 1, 0], [0.4, 0, 0, 0]])
mesh.add_node(4, [[0.9, 1, 0, 0], [0.7, 0, 1, 0], [0., 0, 0, 0]])
mesh.add_element(1, ['H3', 'H3'], [1, 2, 3, 4])

Xn = mesh.get_nodes()
Xs, Ts = mesh.get_surfaces(res=16)

try: S.clear()
except: S = xviewer.Scenes()
S.plotSurfaces('surface', Xs, Ts, color=(1,0.9,0.8))
S.plotPoints('nodes', Xn, size=0.02)
