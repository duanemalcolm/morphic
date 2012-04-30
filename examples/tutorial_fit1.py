# import modules
import scipy
from morphic import mesher
from morphic import fitter
reload(mesher)
reload(fitter)

def generate():
    x = scipy.linspace(0, 2 * scipy.pi, 7)
    y = scipy.cos(x)
    #~ y = scipy.zeros(x.shape[0])
    X = scipy.array([x, y]).T
    
    # Start Generate Mesh
    mesh = mesher.Mesh() # Instantiate a mesh
    
    # Add nodes
    for ind, x in enumerate(X):
        mesh.add_stdnode(ind + 1, x)
    
    # Add two quadratic elements each having 3 nodes
    mesh.add_element(1, ['L3'], [1, 2, 3, 4])
    mesh.add_element(2, ['L3'], [4, 5, 6, 7])
    
    fit = fitter.Fit()
    Xi = scipy.linspace(0, 1, 30)
    did = 0
    for elem in mesh.elements:
        for xi in Xi:
            if elem.id == 1:
                weight = 1 + xi**4 * 100
            else:
                weight = 1 + (1-xi)**4 * 100
            #~ weight = 1
            #~ fit.bind_element_point(elem.id, [xi], 'mydata', data_index=did, weight=weight)
            fit.bind_element_point(elem.id, [xi], 'mydata', weight=weight)
            did += 1
    
    Xi = scipy.linspace(0, 1, 100)
    Xd = mesh.evaluate(1, Xi)
    Xd = scipy.append(Xd, mesh.evaluate(2, Xi), axis=0)
    Xd[:,1] = scipy.cos(Xd[:,0])
    
    fit.set_data('mydata', Xd)
    
    XdA = scipy.array([[0, 0.], [scipy.pi, -1]])
    XdB = scipy.array([[2 * scipy.pi, 0.8]])
    fit.set_data('nodeA', XdA)
    fit.set_data('nodeB', XdB)
    
    fit.bind_node_point(1, 'nodeA', data_index=0, weight=10, param=None)
    #~ fit.bind_node_point(4, 'nodeA', data_index=1, weight=10)
    fit.bind_node_point(7, 'nodeB', data_index=0, weight=10, param=0)
    fit.bind_node_point(7, 'nodeB', data_index=0, weight=10, param=1)
    
    fit.update_from_mesh(mesh)
    fit.invert_matrix()
    mesh = fit.solve(mesh, niter=100, output=True)
    #~ mesh.generate(True)
    
    pylab.figure(1)
    pylab.clf()
    pylab.ion()
    
    pylab.plot(Xd[:,0], Xd[:,1], ',r') 
    
    Xn = mesh.get_nodes()
    pylab.plot(Xn[:,0], Xn[:,1], 'o') 
    
    Xl = mesh.get_lines(res=32)
    for xl in Xl:
        pylab.plot(xl[:,0], xl[:,1], 'b')
    
    pylab.ioff()
    pylab.axis([-0.3, 6.5, -1.2, 1.2])
    pylab.draw()
    pylab.show()
    
    return mesh, fit
    
    
if __name__ == '__main__':
    mesh, fit = generate()
