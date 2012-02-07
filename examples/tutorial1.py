# import modules
import scipy
from fieldscape import mesher
from fieldscape import fitter
reload(mesher)
reload(fitter)

def generate():
    # sphinx tag generate 0
    x = scipy.linspace(0, 2 * scipy.pi, 7)
    y = scipy.cos(x)
    X = scipy.array([x, y]).T
    
    # Start Generate Mesh
    mesh = mesher.Mesh() # Instantiate a mesh
    
    # Add nodes
    mesh.add_stdnode(1, X[0,:])
    mesh.add_stdnode(2, X[1,:])
    mesh.add_stdnode(3, X[2,:])
    mesh.add_stdnode(4, X[3,:])
    mesh.add_stdnode(5, X[4,:])
    mesh.add_stdnode(6, X[5,:])
    mesh.add_stdnode(7, X[6,:])
    # sphinx tag generate 1
    
    # sphinx tag generate 2
    # Add two cubic elements each having 3 nodes
    mesh.add_element(1, ['L3'], [1, 2, 3, 4])
    mesh.add_element(2, ['L3'], [4, 5, 6, 7])
    # sphinx tag generate 3
    
    # Calculate coordinates and derivatives at interpolated points
    # sphinx tag generate 4
    S = [0.2, 0.6]
    Xe = mesh.interpolate(1, S)
    dXe = mesh.interpolate(1, S, deriv=[1])
    # sphinx tag generate 5
    print Xe
    print dXe
    
    # Plot data
    # sphinx tag plotting 1
    x = scipy.linspace(0, 2 * scipy.pi, 20)
    y = scipy.sin(x)
    Xd = scipy.array([x, y]).T
    
    return mesh, Xd

def plotting(mesh, Xd=None):
    
    pylab.figure(1)
    pylab.clf()
    pylab.ion()
    
    # Plot data points
    # sphinx tag plotting 1
    x = scipy.linspace(0, 2 * scipy.pi, 20)
    y = scipy.sin(x)
    X = scipy.array([x, y]).T
    pylab.plot(X[:,0], X[:,1], 'rx') 
    # sphinx tag plotting 2
    
    # Plot nodes
    # sphinx tag plotting 3
    Xn = mesh.get_nodes()
    pylab.plot(Xn[:,0], Xn[:,1], 'ks') 
    # sphinx tag plotting 4
    
    # Plot lines
    # sphinx tag plotting 5
    Xl = mesh.get_lines(res=32)
    for xl in Xl:
        pylab.plot(xl[:,0], xl[:,1], 'b')
    # sphinx tag plotting 6
    
    # Plot interpolated coordinates and derivatives
    # sphinx tag plotting 7
    S = [0.25, 0.75] # interpolated location between 0 and 1 along the element
    Xe = mesh.interpolate([1, 2], S)
    dXe = mesh.interpolate([1, 2], S, deriv=[1])
    
    pylab.plot(Xe[:,0], Xe[:,1], 'go') # interpolated coordinates
    
    sc = 0.2 # scaling for the vectors
    for i, xe in enumerate(Xe):
        xv = scipy.array([xe, xe + sc*dXe[i,:]])
        pylab.plot(xv[:,0], xv[:,1], 'g-') # interpolated derivatives
    # sphinx tag plotting 8
    
    pylab.ioff()
    pylab.axis([-0.3, 6.5, -1.2, 1.2])
    pylab.draw()
    pylab.show()
    # Sphinx End Tag: Plotting
    
    pylab.savefig('../doc/images/tutorial1.png')


def fitting(mesh):
     
    x = scipy.linspace(0, 2 * scipy.pi, 200)
    y = 0.8 * scipy.cos(x)
    Xd = scipy.array([x, y]).T
    
    mesh.nodes[1].fix([True, False])
    mesh.nodes[7].fix([True, False])
    mesh._core.generate_fixed_index()
    
    fit = fitter.Fit('m2dc')
    
    res = 20
    fit.X = scipy.zeros((2*res, 2))
    fit.Xi = scipy.array([scipy.linspace(0, 1, res)]).T
    mesh = fit.optimize(mesh, Xd)
    
    plotting_fit(mesh, Xd)
    
    
def plotting_fit(mesh, X):
    
    pylab.figure(1)
    pylab.clf()
    pylab.ion()
    
    # Plot data points
    # sphinx tag plotting_fit 1
    #~ x = scipy.linspace(0, 2 * scipy.pi, 20)
    #~ y = scipy.sin(x)
    #~ X = scipy.array([x, y]).T
    pylab.plot(X[:,0], X[:,1], 'r.') 
    # sphinx tag plotting_fit 2
    
    # Plot nodes
    # sphinx tag plotting_fit 3
    Xn = mesh.get_nodes()
    pylab.plot(Xn[:,0], Xn[:,1], 'ks') 
    # sphinx tag plotting_fit 4
    
    # Plot lines
    # sphinx tag plotting_fit 5
    Xl = mesh.get_lines(res=32)
    for xl in Xl:
        pylab.plot(xl[:,0], xl[:,1], 'b')
    # sphinx tag plotting_fit 6
    
    pylab.ioff()
    pylab.axis([-0.3, 6.5, -1.2, 1.2])
    pylab.draw()
    pylab.show()
    # Sphinx End Tag: Plotting
    
    pylab.savefig('../doc/images/tutorial1b.png')
    
    
if __name__ == '__main__':
    mesh, Xd = generate()
    plotting(mesh, Xd)
    fitting(mesh)
