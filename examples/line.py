from fieldscape import mesher

mesh = mesher.Mesh()
mesh.add_stdnode(1, [0, 0])
mesh.add_stdnode(2, [0.3, 1.5])
mesh.add_stdnode(3, [0.6, 1.0])
mesh.add_element(1, ['L1'], [1, 2])
mesh.add_element(2, ['L1'], [2, 3])

Xl = mesh.get_lines()
pylab.figure(1)
pylab.clf()
pylab.ion()
for xl in Xl:
    pylab.plot(xl[:,0], xl[:,1])
pylab.ioff()
pylab.draw()
pylab.show()

