import xviewer
import morphic.mesher
reload(morphic.mesher)

mesh = morphic.mesher.Mesh()
mesh.add_node(1, [0, 0, 0])
mesh.add_node(2, [1.2, 0, 0.3])
mesh.add_node(3, [0.1, 0.8, 0.4])
mesh.add_node(4, [0.9, 0.7, 0.])
mesh.add_element(1, ['L1', 'L1'], [1, 2, 3, 4])



mesh.generate()

Xs, Ts = mesh.get_surfaces(res=16)


try: S.clear()
except: S = xviewer.Scenes()
S.plotSurfaces('surface', Xs, Ts, color=(1,0.9,0.8))
