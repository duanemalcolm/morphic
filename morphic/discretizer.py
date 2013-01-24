import numpy    

def xi_grid(shape='quad', res=[8, 8], units='div', method='fit'):
    
    if units == 'div':
        if isinstance(res, int):
            divs = [res, res]
        else:
            divs = res
    elif units == 'xi':
        raise TypeError('Unimplemented units')
    
    nx = divs[0]+1
    xi = numpy.linspace(0,1,divs[0]+1)
    
    if shape == 'quad':
        NPQ = int(nx*nx)
        NTQ = int(2*(divs[0]*divs[0]))
        
        xi1,xi2 = numpy.meshgrid(xi, xi)
        xi1 = xi1.reshape([xi1.size])
        xi2 = xi2.reshape([xi2.size])
        XiQ = numpy.array([xi1, xi2]).T
        TQ = numpy.zeros((NTQ, 3), dtype='uint32')
        np = 0
        for row in range(divs[0]):
            for col in range(divs[0]):
                NPPR = row*nx
                TQ[np,:] = [NPPR+col,NPPR+col+1,NPPR+col+nx]
                np += 1
                TQ[np,:] = [NPPR+col+1,NPPR+col+nx+1,NPPR+col+nx]
                np += 1
        
        return XiQ, TQ
        
    elif shape == 'tri':
        NPT = int(0.5*nx*(nx-1)+nx)
        NTT = int(divs[0]*divs[0])
        
        XiT = numpy.zeros([NPT,2])
        TT = numpy.zeros((NTT, 3), dtype='uint32')
        NodesPerLine = range(divs[0], 0, -1)
        np = 0
        for row in range(nx):
            for col in range(nx-row):
                XiT[np,0] = xi[col]
                XiT[np,1] = xi[row]
                np += 1
        
        np = 0
        ns = 0
        for row in range(divs[0]):
            for col in range(divs[0]-row):
                TT[np,:] = [ns,ns+1,ns+nx-row]
                np += 1
                if col!=divs[0]-row-1:
                    TT[np,:] = [ns+1,ns+nx-row+1, ns+nx-row]
                    np += 1
                ns += 1
            ns += 1
        
        return XiT, TT
