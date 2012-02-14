from mayavi import mlab
import scipy

class Scene:
    
    def __init__(self, label):
        
        self.label = label
        self.text = []
        self.points = []
        self.lines = []
        self.surfaces = []
        
        self.autoRemove = True
        
    def clear(self):
        self.remove_points()
        self.remove_surfaces()

    def plot_text(self, X, text, color=None, size=None):
        
        if self.autoRemove: self.remove_text()
        
        if isinstance(X, list): X = scipy.asarray(X)
        if len(X.shape)==1: X = scipy.array([X])
        if isinstance(text, str): text = [text]
        
        if color==None: color=(1,0,0)
        if size==None: size=1.
        
        for i, x in enumerate(X):
            self.text.append(mlab.text3d(x[0], x[1], x[2], text[i], color=color, scale=size))
         
    def remove_text(self):
        for t in self.text:
            t.remove()
        self.text = []
        
        
    def plot_points(self, X, color=None, size=None, mode=None):
        
        if self.autoRemove: self.remove_points()
        
        if color==None: color=(1,0,0)
        
        if size==None and mode==None:
            size = 1
            mode = 'point'
        if size==None: size = 1
        if mode==None: mode='sphere'
        
        if len(X.shape)==1:
            X = scipy.array([X])
            if isinstance(color, tuple):
                if X.shape[0]>0: self.points.append(mlab.points3d(X[:,0], X[:,1], X[:,2], color=color, scale_factor=size, mode=mode))
            else:
                if X.shape[0]>0: self.points.append(mlab.points3d(X[:,0], X[:,1], X[:,2], color, scale_factor=size, scale_mode='none', mode=mode))
        elif len(X.shape)==2:
            if isinstance(color, tuple):
                self.points.append(mlab.points3d(X[:,0], X[:,1], X[:,2], color=color, scale_factor=size, mode=mode))
            else:
                if X.shape[0]>0: self.points.append(mlab.points3d(X[:,0], X[:,1], X[:,2], color, scale_factor=size, scale_mode='none', mode=mode))
                
        elif len(X.shape)==3:
            for x in X:
                if isinstance(color, tuple):
                    if X.shape[0]>0: self.points.append(mlab.points3d(x[:,0], x[:,1], x[:,2], color=color, scale_factor=size, mode=mode))
                else:
                    if X.shape[0]>0: self.points.append(mlab.points3d(x[:,0], x[:,1], x[:,2], color, scale_factor=size, scale_mode='none', mode=mode))
            
            
    def remove_points(self):
        for p in self.points:
            p.remove()
        self.points = []
        
        
    def plot_lines(self, X, color=None, size=None, scalars=None):
        
        if self.autoRemove: self.remove_lines()
        
        if isinstance(X, list):
            Ndim = 3
        else:
            Ndim = len(X.shape)
        
        if color==None: color=(1,0,0)
        
        #~ data.scene.disable_render = True
        #~ view = data.scene.mlab.view()
        #~ roll = data.scene.mlab.roll()
        #~ move = data.scene.mlab.move()
        
        if Ndim==2:
            if scalars==None:
                self.lines.append(mlab.plot3d(X[:,0], X[:,1], X[:,2], color=color, tube_radius=size))
            else:
                self.lines.append(mlab.plot3d(X[:,0], X[:,1], X[:,2], scalars, tube_radius=size))
        elif Ndim==3:
            for x in X:
                if scalars==None:
                    self.lines.append(mlab.plot3d(x[:,0], x[:,1], x[:,2], color=color, tube_radius=size))
                else:
                    self.lines.append(mlab.plot3d(x[:,0], x[:,1], x[:,2], scalars, tube_radius=size))
        
        #~ data.scene.mlab.view(view[0], view[1], view[2], view[3])
        #~ data.scene.mlab.roll(roll)
        #~ try: data.scene.mlab.move(move)
        #~ except: pass
        #~ data.scene.disable_render = False    
        

    def remove_lines(self):
        for l in self.lines:
            l.remove()
        self.lines = []
        
        
    
    def plot_surfaces(self, X, T, scalars=None, color=None, rep='surface'):
        
        if self.autoRemove: self.remove_surfaces()
        
        if color==None: color=(1,0,0)
        if scalars==None:
            self.surfaces.append(mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, color=color, representation=rep))
        else:
            self.surfaces.append(mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, scalars=scalars))
        #~ data.scene.disable_render = True
        #~ view = data.scene.mlab.view()
        #~ roll = data.scene.mlab.roll()
        #~ move = data.scene.mlab.move()
    #~ 
        #~ if scalars==None:
            #~ addFlower(data, data.scene.mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, color=color))
        #~ else:
            #~ addFlower(data, data.scene.mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, scalars=scalars))
        #~ 
        #~ data.scene.mlab.view(view[0], view[1], view[2], view[3])
        #~ data.scene.mlab.roll(roll)
        #~ try: data.scene.mlab.move(move)
        #~ except: pass
        #~ data.scene.disable_render = False
        
    def remove_surfaces(self):
        for s in self.surfaces:
            s.remove()
        self.surfaces = []
        
        
            
        
class Scenes:
    
    def __init__(self, figure='Default', bgcolor=(.9,.9,.9)):
        self.figure = mlab.figure(figure, bgcolor=bgcolor)
        self.scenes = {}
        
    def clear(self, label=None):
        if label==None:
            for label in self.scenes:
                self.scenes[label].clear()
            
            
    def get_scene(self, label):
        if label in self.scenes.keys():
            return self.scenes[label]
        else:
            self.scenes[label] = Scene(label)
            return self.scenes[label]
        
    def plot_text(self, label, X, text, color=None, size=None):
        
        s = self.get_scene(label)
        s.plot_text(X, text, color=color, size=size)
        
        return 1
        
    def remove_text(self, label):
        
        s = self.get_scene(label)
        s.remove_text()
        
        return 1
        
    def plot_points(self, label, X, color=None, size=None, mode=None):
        
        s = self.get_scene(label)
        s.plot_points(X, color=color, size=size, mode=mode)
        
        return 1
        
    def remove_points(self, label):
        
        s = self.get_scene(label)
        s.remove_points()
        
        return 1
        
    def plot_lines(self, label, X, color=None, size=None):
        
        s = self.get_scene(label)
        s.plot_lines(X, color=color, size=size)
        
        return 1
        
    def remove_lines(self, label):
        
        s = self.get_scene(label)
        s.remove_lines()
        
        return 1
        
    def plot_surfaces(self, label, X, T, scalars=None, color=None, rep='surface'):
        
        s = self.get_scene(label)
        s.plot_surfaces(X, T, scalars=scalars, color=color, rep=rep)
        
        return 1
        
    def remove_surfaces(self, label):
        
        s = self.get_scene(label)
        s.remove_surfaces()
        
        return 1
    
    def get_view(self):
        return mlab.view()
    
    def set_view(self, view):
        mlab.view(view[0], view[1], view[2], view[3])
        
    
