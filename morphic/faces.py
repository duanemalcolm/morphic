import numpy

def dimensions(basis):
    dimensions = 0
    for base in basis:
        if base[0] == 'T':
            dimensions += 2
        else:
            dimensions += 1
    return dimensions

def element_face_nodes(basis, node_ids):
    dims = dimensions(basis)
    for base in basis:
        if base[0] == 'T':
            raise ValueError('Triangular basis are not supported')
    if dims is 1:
        return None
    elif dims is 2:
        return basis, node_ids
    elif dims is 3:
        shape = []
        for base in basis:
            if base[0] == 'L':
                shape.append(int(base[1:]) + 1)
            elif base[0] == 'H':
                if base[1:] is not '3':
                    ValueError('Only 3rd-order hermites are supported')
                shape.append(2)
            else:
                raise ValueError('Basis is not supported')
    else:
        raise ValueError('Dimensions >3 is not supported')
    
    face_basis = []
    face_basis.append([basis[0], basis[1]])
    face_basis.append([basis[0], basis[1]])
    face_basis.append([basis[0], basis[2]])
    face_basis.append([basis[0], basis[2]])
    face_basis.append([basis[1], basis[2]])
    face_basis.append([basis[1], basis[2]])
    
    shape.reverse()
    nids = numpy.array(node_ids).reshape(shape)
    
    faces = []
    faces.append(nids[0, :, :].flatten().tolist())
    faces.append(nids[shape[0] - 1, :, :].flatten().tolist())
    faces.append(nids[:, 0, :].flatten().tolist())
    faces.append(nids[:, shape[1] - 1, :].flatten().tolist())
    faces.append(nids[:, :, 0].flatten().tolist())
    faces.append(nids[:, :, shape[2] - 1].flatten().tolist())
    
    return faces