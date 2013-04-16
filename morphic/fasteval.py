import scipy.sparse
import numpy as np

class FEMatrix(object):
    """
    Generates a sparse matrix from mesh nodes and element points.
    This can be used for fast matrix evaluations of mesh values,
    particularly for fitting purposes.
    """

    def __init__(self, shape):
        self.mesh = None
        self.row_id = 0
        self._auto_increment_row_id = False
        self.A = scipy.sparse.dok_matrix(shape)
        self.rhs = np.zeros(shape[1])
    
    def tocsr(self):
        self.A = self.A.tocsr()
    
    def dot(self, other):
        return self.A.dot(other)
    
    def add_mesh(self, mesh):
        self.mesh = mesh
    
    def auto_increment_rows(self, state=True):
        self._auto_increment_row_id = state
    
    def set_row(self, row_index):
        self.row_id = row_index

    def prev_row(self):
        self.row_id -= 1

    def next_row(self):
        self.row_id += 1

    def add_element_point(self, eid, xi, field, deriv=None, scalar=1):
        if isinstance(xi, list):
            xi = np.array(xi)
        if len(xi.shape) == 1:
            xi = np.array([xi])
        cids = self.mesh.elements[eid].get_field_cids(field)
        weights = self.mesh.elements[eid].weights(np.array(xi), deriv=deriv)[0]
        for cid, weight in zip(cids, weights):
            self.A[self.row_id, cid] += scalar * weight
        if self._auto_increment_row_id:
            self.next_row()
