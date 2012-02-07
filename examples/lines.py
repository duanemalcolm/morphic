# import modules
import scipy
from fieldscape import mesher

def generate_data():
    # Start Generate Data
    # Create 25 nodes along a sine wave
    x = scipy.linspace(0, 2 * scipy.pi, 25)
    y = scipy.sin(x)
    X = scipy.array([x, y]).T
    # End Generate Data
    return X
    
def generate_mesh(X):
    # Start Generate Mesh
    mesh = mesher.Mesh() # Instantiate a mesh
    
    # NODES
    # add nodes starting the node ids at 0
    for i, x in enumerate(X):
        mesh.add_stdnode(i, x)
    
    # ELEMENTS
    # Add four linear elements each having 3 nodes
    # Note: uid=None will generate a unique id automatically
    mesh.add_element(None, ['L1'], [0, 6], group='4linear')
    mesh.add_element(None, ['L1'], [6, 12], group='4linear')
    mesh.add_element(None, ['L1'], [12, 18], group='4linear')
    mesh.add_element(None, ['L1'], [18, 24], group='4linear')
    
    # Add two quadratic elements each having 3 nodes
    mesh.add_element(None, ['L2'], [0, 6, 12], group='2quadratics')
    mesh.add_element(None, ['L2'], [12, 18, 24], group='2quadratics')
    
    # Add two cubic elements each having 4 nodes
    mesh.add_element(None, ['L3'], [0, 4, 8, 12], group='2cubics')
    mesh.add_element(None, ['L3'], [12, 16, 20, 24], group='2cubics')
    
    # Add two quartics elements each having 5 nodes
    mesh.add_element(None, ['L4'], [0, 3, 6, 9, 12], group='2quartics')
    mesh.add_element(None, ['L4'], [12, 15, 18, 21, 24], group='2quartics')
    
    # GROUPS
    mesh.nodes.add_to_group([0, 6, 12, 18, 24], '4linear')
    mesh.nodes.add_to_group([0, 6, 12, 18, 24], '2quadratics')
    mesh.nodes.add_to_group([0, 4, 8, 12, 16, 20, 24], '2cubics')
    mesh.nodes.add_to_group([0, 3, 6, 9, 12, 15, 18, 21, 24], '2quartics')
    
    # End Generate Mesh
    
    return mesh
    
def plot_mesh(X, mesh):
    # Sphinx Start Tag: Plotting
    Xn1 = mesh.get_nodes(group='4linear')
    Xn2 = mesh.get_nodes(group='2quadratics')
    Xn3 = mesh.get_nodes(group='2cubics')
    Xn4 = mesh.get_nodes(group='2quartics')
    
    Xl1 = mesh.get_lines(group='4linear')
    Xl2 = mesh.get_lines(res=32, group='2quadratics')
    Xl3 = mesh.get_lines(res=32, group='2cubics')
    Xl4 = mesh.get_lines(res=32, group='2quartics')
    
    pylab.figure(1)
    pylab.clf()
    pylab.ion()
    
    # Plot data points
    pylab.plot(X[:,0], X[:,1] + 0, 'rx') 
    pylab.plot(X[:,0], X[:,1] + 2, 'rx') 
    pylab.plot(X[:,0], X[:,1] + 4, 'rx') 
    pylab.plot(X[:,0], X[:,1] + 6, 'rx') 
    
    # Plot nodes
    pylab.plot(Xn1[:,0], Xn1[:,1] + 0, 'ro') 
    pylab.plot(Xn2[:,0], Xn2[:,1] + 2, 'ro') 
    pylab.plot(Xn3[:,0], Xn3[:,1] + 4, 'ro') 
    pylab.plot(Xn4[:,0], Xn4[:,1] + 6, 'ro') 
    
    #Plot lines
    for xl in Xl1:
        pylab.plot(xl[:,0], xl[:,1], 'b')
    for xl in Xl2:
        pylab.plot(xl[:,0], xl[:,1] + 2, 'b')
    for xl in Xl3:
        pylab.plot(xl[:,0], xl[:,1] + 4, 'b')
    for xl in Xl4:
        pylab.plot(xl[:,0], xl[:,1] + 6, 'b')
        
    pylab.ioff()
    pylab.axis([-0.5, 7, -1.5, 7.5])
    pylab.draw()
    pylab.show()
    # Sphinx End Tag: Plotting
    
    pylab.savefig('../doc/images/lines.png')
    
    
if __name__ == '__main__':
    X = generate_data()
    mesh = generate_mesh(X)
    plot_mesh(X, mesh)
    
