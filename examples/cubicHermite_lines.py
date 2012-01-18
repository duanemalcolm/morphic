import fieldscape.mesher
reload(fieldscape.mesher)

mesh = fieldscape.mesher.Mesh()
mesh.add_node(1, [[0, 0.3], [0, 1.5]])
mesh.add_node(2, [[0.3, -0.3], [1.5, 1.5]])
mesh.add_node(4, [[0., -0.3], [0.4, 0.5]])
mesh.add_element(1, ['H3'], [1, 2])
mesh.add_element(2, ['H3'], [2, 4])
#~ mesh.add_element(3, ['L3'], [3, 4, 5, 6])
mesh.generate()

Xl = mesh.get_lines(res=32)
pylab.figure(1)
pylab.clf()
pylab.ion()
for xl in Xl:
    pylab.plot(xl[:,0], xl[:,1])
pylab.ioff()
pylab.draw()
pylab.show()

