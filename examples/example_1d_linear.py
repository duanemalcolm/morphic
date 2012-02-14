import os
import sys

example = 'example_1d_linear'
title = 'Sine Wave using four 1D Linear Lagrange Element'
testdatadir = 'data'
docimagedir = os.path.join('..', 'doc', 'images')

# sphinx tag start
import scipy
import morphic

x = scipy.linspace(0, 2 * scipy.pi, 5)
y = scipy.sin(x)
X = scipy.array([x, y]).T

mesh = morphic.Mesh()
mesh.add_stdnode(1, X[0,:])
mesh.add_stdnode(2, X[1,:])
mesh.add_stdnode(3, X[2,:])
mesh.add_stdnode(4, X[3,:])
mesh.add_stdnode(5, X[4,:])
mesh.add_element(1, ['L1'], [1, 2])
mesh.add_element(2, ['L1'], [2, 3])
mesh.add_element(3, ['L1'], [3, 4])
mesh.add_element(4, ['L1'], [4, 5])
# sphinx tag end

if len(sys.argv) > 1:
    action = sys.argv[1]
    
    from matplotlib import pyplot
    import pickle
    
    Xn = mesh.get_nodes()
    Xl = mesh.get_lines()
    
    if action == 'update':
        data = {'Xn': Xn, 'Xl': Xl}
        filepath = os.path.join(testdatadir, example+'.pkl')
        pickle.dump(data, open(filepath, 'w'))
    
    if action in ['update', 'plot']:
        Xl = mesh.get_lines(res=32)
        
        x = scipy.linspace(0, 2 * scipy.pi, 32)
        y = scipy.sin(x)
        Xs = scipy.array([x, y]).T

        pyplot.figure(1)
        pyplot.clf()
        pyplot.plot(Xs[:,0], Xs[:,1], 'xg')
        pyplot.plot(Xn[:,0], Xn[:,1], 'or')
        for xl in Xl:
            pyplot.plot(xl[:,0], xl[:,1], '-b')
        pyplot.title(title)
        pyplot.axis([-0.2, 6.4, -1.1, 1.1])
        pyplot.show()
        filepath = os.path.join(docimagedir, example+'.png')
        pyplot.savefig(filepath)
    
