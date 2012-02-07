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
        self.removePoints()
        self.removeSurfaces()

    def plotText(self, X, text, color=None, size=None):
        
        if self.autoRemove: self.removeText()
        
        if isinstance(X, list): X = scipy.asarray(X)
        if len(X.shape)==1: X = scipy.array([X])
        if isinstance(text, str): text = [text]
        
        if color==None: color=(1,0,0)
        if size==None: size=1.
        
        for i, x in enumerate(X):
            self.text.append(mlab.text3d(x[0], x[1], x[2], text[i], color=color, scale=size))
         
    def removeText(self):
        for t in self.text:
            t.remove()
        self.text = []
        
        
    def plotPoints(self, X, color=None, size=None, mode=None):
        
        if self.autoRemove: self.removePoints()
        
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
            
            
    def removePoints(self):
        for p in self.points:
            p.remove()
        self.points = []
        
        
    def plotLines(self, X, color=None, size=None, scalars=None):
        
        if self.autoRemove: self.removeLines()
        
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
        

    def removeLines(self):
        for l in self.lines:
            l.remove()
        self.lines = []
        
        
    
    def plotSurfaces(self, X, T, scalars=None, color=None, rep='surface'):
        
        if self.autoRemove: self.removeSurfaces()
        
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
        
    def removeSurfaces(self):
        for s in self.surfaces:
            s.remove()
        self.surfaces = []
        
        
            
        
class Scenes:
    
    def __init__(self, figure='Default'):
        self.figure = mlab.figure(figure)
        self.scenes = {}
        
    def clear(self, label=None):
        if label==None:
            for label in self.scenes:
                self.scenes[label].clear()
            
            
    def getScene(self, label):
        if label in self.scenes.keys():
            return self.scenes[label]
        else:
            self.scenes[label] = Scene(label)
            return self.scenes[label]
        
    def plotText(self, label, X, text, color=None, size=None):
        
        s = self.getScene(label)
        s.plotText(X, text, color=color, size=size)
        
        return 1
        
    def removeText(self, label):
        
        s = self.getScene(label)
        s.removeText()
        
        return 1
        
    def plotPoints(self, label, X, color=None, size=None, mode=None):
        
        s = self.getScene(label)
        s.plotPoints(X, color=color, size=size, mode=mode)
        
        return 1
        
    def removePoints(self, label):
        
        s = self.getScene(label)
        s.removePoints()
        
        return 1
        
    def plotLines(self, label, X, color=None, size=None):
        
        s = self.getScene(label)
        s.plotLines(X, color=color, size=size)
        
        return 1
        
    def removeLines(self, label):
        
        s = self.getScene(label)
        s.removeLines()
        
        return 1
        
    def plotSurfaces(self, label, X, T, scalars=None, color=None, rep='surface'):
        
        s = self.getScene(label)
        s.plotSurfaces(X, T, scalars=scalars, color=color, rep=rep)
        
        return 1
        
    def removeSurfaces(self, label):
        
        s = self.getScene(label)
        s.removeSurfaces()
        
        return 1
    
