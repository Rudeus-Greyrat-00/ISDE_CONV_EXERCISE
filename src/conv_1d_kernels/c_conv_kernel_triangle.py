from .c_conv_kernel import CConvKernel
import numpy as np


class CConvKernelTriangle(CConvKernel):
    def kernel_mask(self):
        m = np.zeros(shape=(self.kernel_size,))
        half_index = int((self.kernel_size - 1) / 2)
        for i in range(half_index):
            m[i] = i + 1
            m[-i - 1] = i + 1
        m /= sum(m)
        self._mask = m
