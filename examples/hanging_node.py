import morphic.mesher
reload(morphic.mesher)

mesh = morphic.mesher.Mesh()
mesh.add_stdnode(1, [0, 0])
mesh.add_stdnode(2, [2, 1])
mesh.add_element('elem1', ['L1'], [1, 2])
mesh.add_stdnode('xi', [0.5])
mesh.add_depnode(3, 'elem1', 'xi')
print mesh.nodes[3]

