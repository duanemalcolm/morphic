import scipy

# Some systems have the mayavi2 module referenced by diffrent names.
try:
    from mayavi import mlab
    mlab_loaded = True
except:
    mlab_loaded = False

if not mlab_loaded :
    try:
        from enthought.mayavi import mlab
    except ImportError:
        print 'Enthought Mayavi mlab module not found'
        raise

class Figure:
    
    def __init__(self, figure='Default', bgcolor=(.5,.5,.5)):
        self.figure = mlab.figure(figure, bgcolor=bgcolor)
        self.plots = {}
        
    def clear(self, label=None):
        if label == None:
            labels = self.plots.keys()
        else:
            labels = [label]
            
        mlab.figure(self.figure.name)
        
        for label in labels:
            mlab_obj = self.plots.get(label)
            if mlab_obj != None:
                if mlab_obj.name == 'Surface':
                    mlab_obj.parent.parent.parent.remove()
                else:
                    mlab_obj.parent.parent.remove()
                self.plots.pop(label)

    def get_camera(self):
        return (mlab.view(), mlab.roll())

    def set_camera(self, camera):
        mlab.view(*camera[0])
        mlab.roll(camera[1])
    
    def plot_surfaces(self, label, X, T, scalars=None, color=None, rep='surface', opacity=1.0):
        
        mlab.figure(self.figure.name)
        
        if color == None:
            color = (1,0,0)
        
        mlab_obj = self.plots.get(label)
        if mlab_obj == None:
            if scalars==None:
                self.plots[label] = mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, color=color, opacity=opacity, representation=rep)
            else:
                self.plots[label] = mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, scalars=scalars, opacity=opacity)
        
        else:
            self.figure.scene.disable_render = True
            view = mlab.view()
            roll = mlab.roll()
            
            if X.shape[0] == mlab_obj.mlab_source.x.shape[0]:
                if scalars==None:
                    mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2])
                    mlab_obj.actor.property.color = color
                    mlab_obj.actor.property.opacity = opacity
                else:
                    mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2], scalars=scalars, opacity=opacity)
                
                
            else:
                self.clear(label)
                if scalars==None:
                    self.plots[label] = mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, color=color, opacity=opacity, representation=rep)
                else:
                    self.plots[label] = mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, scalars=scalars, opacity=opacity)
                
            mlab.view(*view)
            mlab.roll(roll)
            self.figure.scene.disable_render = False
            
    def plot_lines(self, label, X, color=None, size=0):
        
        nPoints = 0
        for x in X:
            nPoints += x.shape[0]
        
        Xl = scipy.zeros((nPoints, 3))
        connections = []
        
        ind = 0
        for x in X:
            Xl[ind:ind+x.shape[0],:] = x
            for l in range(x.shape[0]-1):
                connections.append([ind + l, ind + l + 1])
            ind += x.shape[0]
        connections = scipy.array(connections)
        
        mlab.figure(self.figure.name)
        
        if color == None:
            color = (1,0,0)
        if size == None:
            size = 1
        
        mlab_obj = self.plots.get(label)
        if mlab_obj == None:
            self.plots[label] = mlab.points3d(Xl[:,0], Xl[:,1], Xl[:,2], color=color, scale_factor=0)
            self.plots[label].mlab_source.dataset.lines = connections
            mlab.pipeline.surface(self.plots[label], color=(1, 1, 1),
                              representation='wireframe',
                              line_width=size,
                              name='Connections')
        else:
            self.figure.scene.disable_render = True
            self.clear(label)
            self.plots[label] = mlab.points3d(Xl[:,0], Xl[:,1], Xl[:,2], color=color, scale_factor=0)
            self.plots[label].mlab_source.dataset.lines = connections
            #~ self.plots[label].mlab_source.update()
            mlab.pipeline.surface(self.plots[label], color=color,
                              representation='wireframe',
                              line_width=size,
                              name='Connections')
            self.figure.scene.disable_render = False
        
            
    def plot_lines2(self, label, X, scalars=None, color=None, size=0):
        
        mlab.figure(self.figure.name)
        
        if color == None:
            color = (1,0,0)
        
        mlab_obj = self.plots.get(label)
        if mlab_obj == None:
            if scalars==None:
                self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], color=color, tube_radius=size)
            else:
                self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], scalars, tube_radius=size)
        
        else:
            self.figure.scene.disable_render = True
            #~ view = mlab.view()
            
            #~ if X.shape[0] == mlab_obj.mlab_source.x.shape[0]:
                #~ if scalars==None:
                    #~ mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2])
                    #~ mlab_obj.actor.property.color = color
                #~ else:
                    #~ mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2], scalars=scalars)
                #~ 
            #~ else:
                #~ self.clear(label)
                #~ if scalars==None:
                    #~ self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], color=color, line_width=size)
                #~ else:
                    #~ self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], scalars, line_width=size)
            
            self.clear(label)
            if scalars==None:
                self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], color=color, tube_radius=size, reset_zoom=False)
            else:
                self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], scalars, tube_radius=size, reset_zoom=False)
            
            #~ mlab.view(*view)
            self.figure.scene.disable_render = False
            
    def plot_points(self, label, X, color=None, size=None, mode=None):
        
        mlab.figure(self.figure.name)
        
        if color==None:
            color=(1,0,0)
        
        if size == None and mode == None or size == 0:
            size = 1
            mode = 'point'
        if size == None:
            size = 1
        if mode==None:
            mode='sphere'
        
        if isinstance(X, list):
            X = scipy.array(X)
        
        if len(X.shape) == 1:
            X = scipy.array([X])
        
        mlab_obj = self.plots.get(label)
        if mlab_obj == None:
            if isinstance(color, tuple):
                self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color=color, scale_factor=size, mode=mode)
            else:
                self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color, scale_factor=size, scale_mode='none', mode=mode)
        
        else:
            self.figure.scene.disable_render = True
            view = mlab.view()
            roll = mlab.roll()
            
            ### Commented out since VTK gives an error when using mlab_source.set
            #~ if X.shape[0] == mlab_obj.mlab_source.x.shape[0]:
                #~ if isinstance(color, tuple):
                    #~ mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2])
                    #~ mlab_obj.actor.property.color = color
                #~ else:
                    #~ mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2], scalars=color)
                #~ 
                #~ 
            #~ else:
                #~ self.clear(label)
                #~ if isinstance(color, tuple):
                    #~ self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color=color, scale_factor=size, mode=mode)
                #~ else:
                    #~ self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color, scale_factor=size, scale_mode='none', mode=mode)
            
            self.clear(label)
            if isinstance(color, tuple):
                self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color=color, scale_factor=size, mode=mode)
            else:
                self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color, scale_factor=size, scale_mode='none', mode=mode)
                
            mlab.view(*view)
            mlab.roll(roll)
            self.figure.scene.disable_render = False
            
            
    def plot_text(self, label, X, text, size=1):
        view = mlab.view()
        roll = mlab.roll()
        self.figure.scene.disable_render = True
        
        scale = (size, size, size)
        mlab_objs = self.plots.get(label)
        
        if mlab_objs != None:
            if len(mlab_objs) != len(text):
                for obj in mlab_objs:
                    obj.remove()
            self.plots.pop(label)
        
        mlab_objs = self.plots.get(label)
        if mlab_objs == None:
            text_objs = []
            for x, t in zip(X, text):
                text_objs.append(mlab.text3d(x[0], x[1], x[2], str(t), scale=scale))
            self.plots[label] = text_objs
        elif len(mlab_objs) == len(text):
            for i, obj in enumerate(mlab_objs):
                obj.position = X[i,:]
                obj.text = str(text[i])
                obj.scale = scale
        else:
            print "HELP, I shouldn\'t be here!!!!!"
        
        self.figure.scene.disable_render = False
        mlab.view(*view)
        mlab.roll(roll)