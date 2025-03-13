class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"Key '{key}' not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"Key '{key}' not found")
