import xviewer
import morphic.mesher
reload(morphic.mesher)


raise BaseException('Not Working, got to implement 3D')

mesh = morphic.mesher.Mesh()

# Prism element
mesh.add_node(1, [0, 0, 0])
mesh.add_node(2, [1.2, 0, 0.3])
mesh.add_node(3, [0.5, 0.8, 0.4])
mesh.add_node(4, [0, 0, 1.0])
mesh.add_node(5, [1.2, 0, 1.3])
mesh.add_node(6, [0.5, 0.8, 1.4])
mesh.add_element(1, ['T11', 'L1'], [1, 2, 3, 4, 5, 6])


Xn = mesh.get_nodes()
Xs, Ts = mesh.get_surfaces(res=16)

try: S.clear()
except: S = xviewer.Scenes()
S.plotSurfaces('surface', Xs, Ts, color=(1,0.9,0.8))
S.plotPoints('nodes', Xn, size=0.02)

