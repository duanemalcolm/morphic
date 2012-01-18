import fieldscape.mesher
reload(fieldscape.mesher)

mesh = fieldscape.mesher.Mesh()
mesh.add_node(1, [0, 0, 0.])
mesh.add_node(2, [0.3, 0.5, 1.])
mesh.add_node(3, [0.4, 1.0, 0.])
mesh.add_node(4, [0.5, 1.5, 0.])
mesh.add_node(5, [0.3, 2.0, 0.])
mesh.add_node(6, [0.4, 2.5, 0.])
mesh.add_element(1, ['L2'], [1, 2, 3])
mesh.add_element(3, ['L3'], [3, 4, 5, 6])
mesh.generate()

Xl = mesh.get_lines(res=64)
pylab.figure(1)
pylab.clf()
pylab.ion()
for xl in Xl:
    pylab.plot(xl[:,0], xl[:,1])
pylab.ioff()
pylab.draw()
pylab.show()

