from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time

up_ker = SourceModule("""
    __global__ void up_ker(double *x, double *x_old, int k)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int _2k = 1 << k;
        int _2k1 = 1 << (k+1);
        int j = tid * _2k1;
        x[j + _2k1 - 1] = x_old[j + _2k - 1] + x_old[j + _2k1 - 1];
    }
""")

up_gpu = up_ker.get_function("up_ker")

def up_sweep(x):
    x = np.float64(x)
    x_gpu = gpuarray.to_gpu(np.float64(x))
    x_old_gpu = x_gpu.copy()
    for k in range(int(np.log2(x.size))):
        num_threads = int(np.ceil(x.size/2**(k+1)))
        grid_size = int(np.ceil(num_threads/32))

        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads

        up_gpu(x_gpu, x_old_gpu, np.int32(k), block=(block_size,1,1), grid=(grid_size,1,1))
        x_old_gpu[:] = x_gpu[:]
    x_out = x_gpu.get()
    return x_out

