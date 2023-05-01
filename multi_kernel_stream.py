import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

num_arrays = 200
array_len = 1024 ** 2

ker = SourceModule("""       
__global__ void mult_ker(float * array, int array_len)
{
     int thd = blockIdx.x*blockDim.x + threadIdx.x;
     int num_iters = array_len / blockDim.x;

     for(int j=0; j < num_iters; j++)
     {
         int i = j * blockDim.x + thd;

         for(int k = 0; k < 50; k++)
         {
              array[i] *= 2.0;
              array[i] /= 2.0;
         }
     }

}
""")

mult_ker = ker.get_function("mult_ker")

data = []
data_gpu = []
gpu_out = []
streams = []

for _ in range(num_arrays):
    streams.append(drv.Stream())

for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

t_start = time()

for k in range(num_arrays):
    data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))

for k in range(num_arrays):
    gpu_out.append(data_gpu[k].get_async(stream=streams[k]))

t_end = time()

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

print("Total time: %f" % (t_end - t_start))