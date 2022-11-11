from .c_conv_kernel import CConvKernel
import numpy as np

class CConvolutionalCombo():
    def __init__(self, filters):
        self._filters = []
        self.filters = filters

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, value):
        for i in value:
            if not issubclass(type(i), CConvKernel):
                raise ValueError("Filters must be all subclass of CConvKernel")
        self._filters = value

    def kernel(self, x):
        xp = x.copy()
        for f in self.filters:
            xp = f.kernel(xp)
        return xp
