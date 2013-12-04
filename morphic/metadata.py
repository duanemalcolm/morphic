class Metadata(object):
    
    def set(self, name, value, overwrite=True):
        if not overwrite:
            if name in self.__dict__:
                return False
        self.__dict__[name] = value
        return True
        
    def get(self, name, default=None):
        if name not in self.__dict__:
            return default
        return self.__dict__[name]
    
    def delete(self, name):
        if name in self.__dict__:
            self.__dict__.pop(name)
    
    def get_dict(self):
        return self.__dict__
    
    def set_dict(self, data):
        for key, value in data.iteritems():
            self.__dict__[key] = value
            
    def save_pytables(self, node):
        for key, value in self.iteritems():
            node._v_attrs[key] = value
            
    def load_pytables(self, node):
        a = node._AttributeSet(node)
        for key in a._v_attrnamesuser:
            self.set(key, a[key])
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value
    
    def __getattr__(self, name):
        return self.__dict__[name]
    
    def __setitem__(self, name, value):
        self.__dict__[name] = value
    
    def __getitem__(self, name):
        return self.__dict__[name]
    
    def keys(self):
        return self.__dict__.keys()
    
    def has_key(self, name):
        return name in self.__dict__
    
    def values(self):
        return self.__dict__.values()
    
    def items(self):
        return self.__dict__.items()
    
    def iteritems(self):
        return self.__dict__.iteritems()
    
    def iterkeys(self):
        return self.__dict__.iterkeys()
    
    def itervalues(self):
        return self.__dict__.itervalues()
    
    def __contains__(self, name):
        return name in self.__dict__
    
    def __len__(self):
        return len(self.__dict__)
    
    def __str__(self):
        return self.__dict__.__str__()
    
