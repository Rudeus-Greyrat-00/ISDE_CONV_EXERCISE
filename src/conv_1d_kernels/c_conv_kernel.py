from abc import ABC, abstractmethod
import numpy as np


class CConvKernel(ABC):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        self.mask = None

    @abstractmethod
    def kernel_mask(self):
        raise NotImplementedError("Method not implemented yet")

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        if value % 2 == 0:
            raise ValueError("The kernel size must be an odd number")
        self._kernel_size = value
        self.kernel_mask()  # regenerate the mask

    @property
    def mask(self):
        return self._mask

    def kernel(self, x):
        xp = x.copy()
        k_range = int((self.kernel_size - 1) / 2)
        for i in range(k_range, x.size - k_range):
            xp[i] = np.dot(x[i-k_range:i+k_range+1], self.mask)
        return xp
