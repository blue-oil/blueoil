class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        super(EasyDict, self).__init__()
        self.update(d or {}, **kwargs)

    def update(self, d=None, **kwargs):
        d = d or {}
        d.update(kwargs)
        for key, value in d.items():
            self[key] = value

    def __setitem__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [
                self.__class__(x) if isinstance(x, dict) else x
                for x in value
            ]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setitem__(name, value)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return super(EasyDict, self).__getattr__(name)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            return super(EasyDict, self).__setattr__(name, value)
        self[name] = value
