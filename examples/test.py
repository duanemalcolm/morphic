def project_tube_to_data(Xd, x0, vector, radius=None):
    '''
    Projects a tube through a data set to filter the data.
    The remaining data is returned as xd and it's xi value along the vector.
    '''
    dxd = Xd-x0
    dr = scipy.dot(vector.T, vector)
    u = scipy.absolute(scipy.array([scipy.dot(vector,dxd.T)/dr]).T)
    
    x = x0+u*scipy.array(vector)
    
    dx = x-Xd
    dr = scipy.sum(dx*dx, axis=1)
    
    if radius==None:
        idxmin = scipy.argmin(dr)
        xd = scipy.array(Xd[idxmin,:])
        xi = scipy.array(u[idxmin,0])
    else:
        radius = radius*radius
        ii = dr<=radius
        xd = Xd[ii,:]
        xi = u[ii,0]
    
    return xd, xi
    
    
    
x = scipy.linspace(0,6,100)
y = scipy.cos(x)
Xd = scipy.array([x, y, scipy.zeros(x.shape[0])]).T

x0 = scipy.array([0.6, 0, 0])
v = scipy.array([0, 1, 0])


xd, xi = project_tube_to_data(Xd, x0, v, radius=0.6)
