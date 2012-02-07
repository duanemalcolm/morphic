import time

import scipy
import scipy.optimize
import scipy.sparse
from scipy.spatial import cKDTree

from morphic import core

class BoundElementPoint:
    
    def __init__(self, element_id, xi, data_label, data_index=None,
            weight=1):
        self._class_ = 'elem'
        self.eid = element_id
        self.xi = xi
        self.data = data_label
        self.data_index = data_index
        self.bind_weight = weight
        
        self.param_ids = None
        self.param_weights = None
        self.num_fields = 0
    
    def get_field_id(self, field):
        return field
        
    def get_bind_weight(self):
        return self.bind_weight
        
    def get_param_ids(self, field):
        return self.param_ids[field]
        
    def get_param_weights(self, field):
        return self.param_weights * self.bind_weight
        
    def update_from_mesh(self, mesh):
        element = mesh.elements[self.eid]
        self.param_ids = element._get_param_indicies()
        self.param_weights = element.weights(self.xi)[0]
        self.num_fields = len(self.param_ids)
        
    def get_data(self, data, field, mesh):
        if self.data_index == None:
            #~ x = mesh.interpolate(self.eid, self.xi)[0]
            #~ xc = data[self.data].find_closest(x, 1)
            #~ return xc[field]
            return data[self.data].get_data(self.data_ids)[field]
        else:
            return data[self.data].values[self.data_index, field]
        
        

class BoundNodePoint:
    
    def __init__(self, node_id, data_label, data_index=None, weight=1,
            param=None):
        self._class_ = 'node'
        self.nid = node_id
        self.param = param
        self.data = data_label
        self.data_index = data_index
        self.bind_weight = weight
        
        self.param_ids = None
        self.param_weights = None
        self.num_fields = 0
        
    def get_field_id(self, field):
        if self.param != None:
            return self.param
        return field
        
    def get_bind_weight(self):
        return self.bind_weight
        
    def get_param_ids(self, field):
        return self.param_ids[field]
        
    def get_param_weights(self, field):
        return self.param_weights * self.bind_weight
        
    def update_from_mesh(self, mesh):
        node = mesh.nodes[self.nid]
        self.param_ids = node._get_param_indicies()
        self.param_weights = scipy.ones(len(self.param_ids))
        self.num_fields = len(self.param_ids)
        if self.param != None:
            self.param_ids = [self.param_ids[self.param]]
            self.param_weights = scipy.array([1])
            self.num_fields = 1
    
    def get_data(self, data, field, mesh):
        if self.data_index == None:
            x = mesh.nodes[self.nid].values[0]
            xc = data[self.data].find_closest(x, 1)
            return xc[field]
        else:
            return data[self.data].values[self.data_index, field]
        
           

class Data:
    
    def __init__(self, label, values):
        self.id = label
        self.values = values
        self.tree = cKDTree(self.values)
        
        self.row_ind = 0
        self.Phi = None
        self.ii = None
    
    def init_phi(self, M, N):
        self.row_ind = 0
        self.Phi = scipy.sparse.lil_matrix((M, N))
    
    def add_point(self, point):
        if point._class_ == 'elem':
            point.data_ids = []
            for pids in point.param_ids:
                self.Phi[self.row_ind, pids] = point.param_weights
                point.data_ids.append(self.row_ind)
                self.row_ind += 1
                
    def update_point_data(self, params):
        if self.Phi != None:
            xd = self.Phi.dot(params)
            rr, ii = self.tree.query(xd.reshape((xd.size/self.values.shape[1], self.values.shape[1])))
            self.xc = self.values[ii, :].reshape(xd.size)
    
    def get_data(self, ind):
        return self.xc[ind]
    
    def find_closest(self, x, num=1):
        r, ii = self.tree.query(list(x))
        #~ print ii, x, self.values[ii]
        return self.values[ii]
        
        

class Fit:
    
    def __init__(self, method='data_to_mesh_closest'):
        self.points = core.ObjectList()
        self.data = core.ObjectList()
        self.method = method
        
        self._objfns = {
            'd2mp': self.objfn_data_to_mesh_project,
            'd2mc': self.objfn_data_to_mesh_closest,
            'm2dc': self.objfn_mesh_to_data_closest,
            'data_to_mesh_project': self.objfn_data_to_mesh_project,
            'data_to_mesh_closest': self.objfn_data_to_mesh_closest,
            'mesh_to_data_closest': self.objfn_mesh_to_data_closest
            }
        
        if isinstance(method, str):
            self.objfn = self._objfns[method]
        
        
        self.X = None
        self.Xi = None
        self.A = None
        self.invA = None
        
        self.svd_UT, self.svd_S, self.svd_VT = None, None, None
        self.svd_invA = None
        
        
        self.use_sparse = True
        self.param_ids = []
        self.num_dof = 0
        self.num_rows = 0
        
    def bind_element_point(self, element_id, xi, data_label,
            data_index=None, weight=1):
        self.points.add(BoundElementPoint(element_id, xi, data_label,
                data_index=data_index, weight=weight))
    
    def bind_node_point(self, node_id, data_label, data_index=None,
            weight=1, param=None):
        self.points.add(BoundNodePoint(
                node_id, data_label, data_index=data_index,
                weight=weight, param=param))
    
    def set_data(self, label, values):
        self.data.add(Data(label, values))
    
    def get_data(self, mesh):
        #~ # Find bind points that require searching for the closest data
        #~ # point
        #~ find_closest = {}
        #~ for ind, dm in enumerate(self.data_map):
            #~ # if the data_index of the point is None then we must search
            #~ # for the data point.
            #~ pt = self.points[dm[0]]
            #~ if pt.data_index == None:
                
        Xd = scipy.zeros(self.num_rows)
        for ind, dm in enumerate(self.data_map):
            Xd[ind] = self.points[dm[0]].get_data(self.data, dm[1], mesh)
        return Xd
    
    def get_column_index(self, param_ids):
        return [self.param_ids.index(pid) for pid in param_ids]
    
    def update_from_mesh(self, mesh):
        for point in self.points:
            point.update_from_mesh(mesh)
        self.generate_matrix()
        # self.generate_fast_data()
    
    def generate_matrix(self):
        param_ids = []
        self.num_rows = 0
        for point in self.points:
            self.num_rows += point.num_fields
            param_ids.extend([item for sublist in point.param_ids
                    for item in sublist])
        
        self.param_ids = [pid for pid in set(param_ids)]
        self.param_ids.sort()
        self.num_dof = len(self.param_ids)
        self.W = scipy.ones(self.num_rows)
        self.data_map = []
        
        if self.use_sparse:
            self.A = scipy.sparse.lil_matrix((self.num_rows, self.num_dof))
        
        else:
            self.A = scipy.zeros((self.num_rows, self.num_dof))
        
        row_ind = -1
        for pid, point in enumerate(self.points):
            bind_weight = point.get_bind_weight()
            for field_ind in range(point.num_fields):
                field = point.get_field_id(field_ind)
                weights = point.get_param_weights(field_ind)
                param_ids = point.get_param_ids(field_ind)
                cols = self.get_column_index(param_ids)
                row_ind += 1
                self.data_map.append([pid, field])
                for col, weight in zip(cols, weights):
                    self.A[row_ind, col] += weight
                    self.W[row_ind] = bind_weight
        
        if self.use_sparse:
            self.A = self.A.tocsc()
    
    def generate_fast_data(self):
        num_rows = {}
        for point in self.points:
            if point._class_ == 'elem':
                if point.data_index == None:
                    if point.data not in num_rows.keys():
                        num_rows[point.data] = 0
                    num_rows[point.data] += point.num_fields
                    
        for key in num_rows.keys():
            self.data[key].init_phi(num_rows[key], self.num_dof)
        
        for point in self.points:
            if point._class_ == 'elem':
                self.data[point.data].add_point(point)
        
        
    def invert_matrix(self):
        from sparsesvd import sparsesvd
        self.svd_UT, self.svd_S, self.svd_VT = sparsesvd(self.A, self.A.shape[1])
        self.svd_invA = scipy.dot(\
            scipy.dot(self.svd_VT.T,scipy.linalg.inv(scipy.diag(self.svd_S))),self.svd_UT)
    
    def solve(self, mesh, niter=1, output=False):
        td, ts = 0, 0
        for iter in range(niter):
            
            for data in self.data:
                data.update_point_data(mesh._core.P[self.param_ids])
            
            t0 = time.time()
            Xd = self.get_data(mesh) * self.W
            t1 = time.time()
            
            if self.svd_invA==None:
                self.lsqr_result = scipy.sparse.linalg.lsqr(self.A, Xd)
                solved_x = self.lsqr_result[0]
            else:
                solved_x = scipy.dot(self.svd_invA, Xd)
                
            mesh.update_parameters(self.param_ids, solved_x)
                
            t2 = time.time()
            
            td += t1 - t0
            ts += t2 - t1
            
        if output:
            print 'Solve time: %4.2fs, (%4.2fs, %4.2fs)' % (ts+td, ts, td)
        
        return mesh
            
            
    def optimize(self, mesh, Xd, ftol=1e-9, xtol=1e-9, output=True):
        
        mesh.generate()
        
        Td = cKDTree(Xd)
        
        x0 = mesh.get_variables()
        t0 = time.time()
        x, success = scipy.optimize.leastsq(self.objfn, x0, args=[mesh, Xd, Td], ftol=ftol, xtol=xtol)
        if output: print 'Fit Time: ', time.time()-t0
        mesh.set_variables(x)
        return mesh
    
    def objfn_mesh_to_data_closest(self, x0, args):
        mesh, Xd, Td = args[0], args[1], args[2]
        mesh.set_variables(x0)
        NXi = self.Xi.shape[0]
        ind = 0
        for element in mesh.elements:
            self.X[ind:ind+NXi,:] = element.interpolate(self.Xi)
            ind += NXi
        err = Td.query(list(self.X))[0]
        return err*err
    
    def objfn_data_to_mesh_closest(self, x0, args):
        mesh, Xd, Td = args[0], args[1], args[2]
        mesh.set_variables(x0)
        NXi = self.Xi.shape[0]
        ind = 0
        for element in mesh.elements:
            self.X[ind:ind+NXi,:] = element.interpolate(self.Xi)
            ind += NXi
        Tm = cKDTree(self.X)
        err = Tm.query(list(Xd))[0]
        self.err = err
        return err*err
    
    def objfn_data_to_mesh_project(self, x0, args):
        mesh, Xd, Td = args[0], args[1], args[2]
        mesh.set_variables(x0)
        err = scipy.zeros(Xd.shape[0])
        ind = 0
        for xd in Xd:
            xi1 = mesh.elements[1].project(xd)
            xi2 = mesh.elements[2].project(xd)
            if 0<=xi1<=1:
                xi = xi1
            elif 0<=xi2<=1:
                xi = xi2
            else:
                Xi = scipy.array([xi1, xi1-1, xi2, xi2-1])
                Xi2 = Xi*Xi
                ii = Xi2.argmin()
                xi = Xi[ii]
                if ii < 2:
                    elem = 1
                else:
                    elem = 2
            dx = mesh.elements[elem].interpolate(scipy.array([xi]))[0] - xd
            err[ind] = scipy.sum(dx * dx)
            ind += 1
        return err
