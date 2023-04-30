import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel

def log_example(idx):
    print("=========" + str(idx) + "=========")

log_example(1)
seq = np.array([1,2,3,4], dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
sum_gpu = InclusiveScanKernel(np.int32,'a+b')
print(sum_gpu(seq_gpu).get())
print(np.cumsum(seq))

log_example(2)
from pycuda.curandom import rand as curand
from pycuda.reduction import ReductionKernel
a = curand((10,200), dtype=np.float32)
red = ReductionKernel(np.float32, neutral=0,
                           reduce_expr="a+b",
                           arguments="float *in")

a_sum = gpuarray.empty(10, dtype=np.float32)
for i in range(10):
    red(a[i], out=a_sum[i])

assert(np.allclose(a_sum.get(), a.get().sum(axis=1)))