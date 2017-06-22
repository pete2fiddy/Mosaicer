
class NamedArgs:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getitem__(self, key):
        return self.kwargs[key]
