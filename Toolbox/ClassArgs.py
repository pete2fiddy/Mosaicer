from Toolbox.NamedArgs import NamedArgs

class ClassArgs:
    '''holds a class or function of any type that takes a **kwargs in initialization.
    useful for when both the function/class and their args need to be passed together (splitting
    adds another argument and is inconvenient)'''
    def __init__(self, inst, **kwargs):
        self.inst = inst
        self.kwargs = NamedArgs(**kwargs)

    def __getitem__(self, index):
        return self.kwargs[index]
