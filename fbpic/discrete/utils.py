
import functools
import numpy as np

from fbpic.utils.threading import (
  nthreads,
  njit_parallel,
  prange )

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
  from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d

from .array_op import ArrayOp

array_cache_size = 8

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@functools.lru_cache( maxsize = array_cache_size )
def tmp_ndarray( shape, dtype, acls ):
  """Generates a cached uninitialized array

  Note: This will share the same array amonge any method that calls this
  function with the same arguments. Array should not assumed to contained zeros,
  and should always be reinitialized.
  """
  return np.empty( shape, dtype = dtype )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@functools.lru_cache( maxsize = array_cache_size )
def tmp_numba_device_ndarray( shape, dtype ):
  """Generates a cached uninitialized numba DeviceNDArray array

  Note: This will share the same array amonge any method that calls this
  function with the same arguments. Array should not assumed to contained zeros,
  and should always be reinitialized.
  """
  return cuda.to_device( np.empty( shape, dtype = dtype ) )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NDArrayFill ( ArrayOp ):
  """Fills array with a value
  """

  #-----------------------------------------------------------------------------
  def exec( self,
    array,
    value,
    gpu = False ):
    """
    Parameters
    ----------
    array : array
    value : float, int
    """

    super().exec( array = array, fill = fill, gpu = gpu )

  #-----------------------------------------------------------------------------
  def init_numba_cuda( self ):

    @self.attr
    @cuda.jit
    def _gpu( array, value ):
      i = cuda.grid(1)

      if i < array.shape[0]:
        array[i] = value

  #-----------------------------------------------------------------------------
  def init_cpu( self ):

    @self.attr
    @njit_parallel
    def _cpu( array, value, np, nt, nf ):

      for i in prange( np ):
        for j in range(nt):
          array[i*nt + j] = value

      for j in range(nf):
        array[j] = value

  #-----------------------------------------------------------------------------
  def exec_numba_cuda ( self, array, value ):

    farray = array.ravel()

    bpg, tpb = cuda_tpb_bpg_1d( farray.shape[0] )

    self._gpu[bpg, tpb]( farray, value )

  #-----------------------------------------------------------------------------
  def exec_cpu( self, array, value ):

    farray = array.ravel()

    nt = farray.shape[0] // nthreads
    nf = farray.shape[0] % nthreads

    self._cpu( farray, value, nthreads, nt, nf )

ndarray_fill = ArrayFill()
