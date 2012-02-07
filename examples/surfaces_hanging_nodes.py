import xviewer
import morphic.mesher
reload(morphic.mesher)

mesh = morphic.mesher.Mesh()

# Bilinear element
mesh.add_node(1, [0, 0, 0])
mesh.add_node(2, [1.2, 0, 0.3])
mesh.add_node(3, [0.1, 0.8, 0.4])
mesh.add_node(4, [0.9, 0.7, 0.])
mesh.add_element(1, ['L1', 'L1'], [1, 2, 3, 4])

# Biquadratic element
mesh.add_node(-1, [1, 0.3], group='xi')
mesh.add_node(-2, [1, 0.6], group='xi')
mesh.add_node(5, [1.5, -0.1, 0.3])
mesh.add_node(6, [2, 0, 0.3])
mesh.add_node(7, 1, -1)
mesh.add_node(8, [1.5, 0.35, 0])
mesh.add_node(9, [2, 0.35, 0.1])
mesh.add_node(10, [1.5, 0.7, 0.3])
mesh.add_node(11, [2, 0.7, 0.3])
mesh.add_node(12, 1, -2)
mesh.add_element(2, ['L2', 'L2'], [2, 5, 6, 7, 8, 9, 12, 10, 11])

Xn = mesh.get_nodes()
Xs, Ts = mesh.get_surfaces(res=16)

try: S.clear()
except: S = xviewer.Scenes()
S.plotSurfaces('surface', Xs, Ts, color=(1,0.9,0.8))
S.plotPoints('nodes', Xn, size=0.02)

