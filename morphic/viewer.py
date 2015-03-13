import numpy

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

    def hide(self, label):
        if label in self.plots.keys():
            self.plots[label].visible = False

    def show(self, label):
        if label in self.plots.keys():
            self.plots[label].visible = True

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
        
        Xl = numpy.zeros((nPoints, 3))
        connections = []
        
        ind = 0
        for x in X:
            Xl[ind:ind+x.shape[0],:] = x
            for l in range(x.shape[0]-1):
                connections.append([ind + l, ind + l + 1])
            ind += x.shape[0]
        connections = numpy.array(connections)
        
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
            X = numpy.array(X)
        
        if len(X.shape) == 1:
            X = numpy.array([X])
        
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

    def plot_dicoms(self, label, dicom_files):
        scan = self._load_dicom_attributes(dicom_files)

        mlab.figure(self.figure.name)
        
        mlab_objs = self.plots.get(label)
        if mlab_objs == None:
            src = mlab.pipeline.scalar_field(scan.values)
            src.origin = scan.origin
            src.spacing = scan.spacing
            plane = mlab.pipeline.image_plane_widget(src,
                                plane_orientation='z_axes',
                                slice_index=int(0.5 * scan.num_slices),
                                colormap='black-white')
            self.plots[label] = {}
            self.plots[label]['src'] = src
            self.plots[label]['plane'] = plane
            self.plots[label]['filepaths'] = scan.filepaths
        else:
            self.plots[label]['src'].origin = scan.origin
            self.plots[label]['src'].spacing = scan.spacing
            self.plots[label]['src'].scalar_data = scan.values
            self.plots[label]['plane'].update_pipeline()
            self.plots[label]['filepaths'] = scan.filepaths

    def _load_dicom_attributes(self, dicom_files):
        import dicom

        class Scan(object):

            def __init__(self):
                self.num_slices = 0
                self.origin = numpy.array([0.,0.,0.])
                self.spacing = numpy.array([1.,1.,1.])
                self.filepaths = []
                self.values = None

            def set_origin(self, values):
                self.origin = numpy.array([float(v) for v in values])

            def set_pixel_spacing(self, values):
                self.spacing[0] = float(values[0])
                self.spacing[1] = float(values[1])

            def set_slice_thickness(self, value):
                self.spacing[2] = float(value)

            def init_values(self, rows, cols, slices):
                self.num_slices = slices
                self.values = numpy.zeros((rows, cols, slices))

            def insert_slice(self, index, values):
                self.values[:,:,index] = values

        import os
        if isinstance(dicom_files, str):
            dicom_path = dicom_files
            dicom_files = os.listdir(dicom_path)
            for i, filename in enumerate(dicom_files):
                dicom_files[i] = os.path.join(dicom_path, dicom_files[i])

        scan = Scan()
        slice_location_tag = (0x0020, 0x1041)
        slice_thickness_tag = (0x0018, 0x0050)
        image_position_tag = (0x0020, 0x0032)
        image_orientation_tag = (0x0020, 0x0037)
        pixel_spacing_tag = (0x0028, 0x0030)
        rows_tag = (0x0028, 0x0010)
        cols_tag = (0x0028, 0x0011)
        slice_location = []
        slice_thickness = []
        image_position = []
        remove_files = []
        for i, dicom_file in enumerate(dicom_files):
            try:
                dcm = dicom.read_file(dicom_file)
                valid_dicom = True
            except dicom.filereader.InvalidDicomError:
                remove_files.append(i)
                valid_dicom = False

            if valid_dicom:
                dcmtags = dcm.keys()
                if slice_location_tag in dcmtags:
                    slice_location.append(float(dcm[slice_location_tag].value))
                else:
                    print 'No slice location found in ' + dicom_file
                    return
                if slice_thickness_tag in dcmtags:
                    slice_thickness.append(float(dcm[slice_thickness_tag].value))
                else:
                    print 'No slice thickness found in ' + dicom_file
                    return
                if image_position_tag in dcmtags:
                    image_position.append([float(v)
                        for v in dcm[image_position_tag].value])
                else:
                    print 'No image_position found in ' + dicom_file
                    return

        # Remove files that are not dicoms
        remove_files.reverse()
        for index in remove_files:
            dicom_files.pop(index)

        slice_location = numpy.array(slice_location)
        slice_thickness = numpy.array(slice_thickness)
        image_position = numpy.array(image_position)

        sorted_index = numpy.array(slice_location).argsort()
        dt = []
        for i,zi in enumerate(sorted_index[:-1]):
            i0 = sorted_index[i]
            i1 = sorted_index[i+1]
            dt.append(slice_location[i1] - slice_location[i0])
        dt = numpy.array(dt)

        if slice_thickness.std() > 1e-6 or dt.std() > 1e-6:
            print 'Warning: slices are not regularly spaced'

        scan.set_slice_thickness(slice_thickness[0])

        if pixel_spacing_tag in dcmtags:
            scan.set_pixel_spacing(dcm[pixel_spacing_tag].value)
        else:
            print 'No pixel spacing vlaues found in' + dicom_file
            return

        scan.set_origin(image_position.min(0))

        if rows_tag in dcmtags:
            rows = int(dcm[rows_tag].value)
        else:
            print 'Number of rows not found in ' + dicom_file
            return
        if cols_tag in dcmtags:
            cols = int(dcm[cols_tag].value)
        else:
            print 'Number of cols not found in ' + dicom_file
            return

        scan.init_values(rows, cols, slice_location.shape[0])
        for i, index in enumerate(sorted_index):
            scan.filepaths.append(dicom_files[index])
            scan.insert_slice(i, dicom.read_file(dicom_files[index]).pixel_array[:,::-1])

        return scan



