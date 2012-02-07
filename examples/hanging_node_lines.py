import morphic.mesher
reload(morphic.mesher)

mesh = morphic.mesher.Mesh()
mesh.add_node('hn1', [0.7], group='xi')
mesh.add_node('hn2', [0.4], group='xi')
mesh.add_node(1, [0, 0.])
mesh.add_node(2, [0.3, 1.])
mesh.add_node(5, [-0.3, 1.2])
mesh.add_node(3, 1, 'hn1')
mesh.add_node(6, 1, 'hn2')
mesh.add_node(4, [1, 0.5])
mesh.add_element(1, ['L2'], [1, 2, 5])
mesh.add_element(2, ['L2'], [3, 6, 4])
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

