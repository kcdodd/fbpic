
import functools
import numpy as np

from fbpic.utils.threading import (
  nthreads,
  njit_parallel,
  prange )

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
  from numba import cuda
  from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d

from .array_op import ArrayOp

array_cache_size = 8

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@functools.lru_cache( maxsize = array_cache_size )
def tmp_ndarray( shape, dtype, key = 0 ):
  """Generates a cached uninitialized array

  Note: This will share the same array amonge any method that calls this
  function with the same arguments. Array should not assumed to contained zeros,
  and should always be reinitialized.

  Parameters
  ----------
  shape: tuple
  dtype : numpy.dtype
  key : hashable, optinoal
    a key used to allocate unique array if more than one temporary array of the same
    shape and dtype is required at the same time.
  """
  return np.empty( shape, dtype = dtype )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@functools.lru_cache( maxsize = array_cache_size )
def tmp_numba_device_ndarray( shape, dtype, key = 0 ):
  """Generates a cached uninitialized numba DeviceNDArray array

  Note: This will share the same array amonge any method that calls this
  function with the same arguments. Array should not assumed to contained zeros,
  and should always be reinitialized.

  Parameters
  ----------
  shape: tuple
  dtype : numpy.dtype
  key : hashable, optional
    a key used to allocate unique array if more than one temporary array of the same
    shape and dtype is required at the same time.
  """
  return cuda.to_device( np.empty( shape, dtype = dtype ) )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def empty_ndarray( shape, dtype, gpu = False ):
  """Generates an uninitialized array, numpy or DeviceNDArray
  """
  arr = np.empty( shape, dtype = dtype )

  if gpu:
    return cuda.to_device( arr )
  else:
    return arr

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

    super().exec( array = array, value = value, gpu = gpu )

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
    def _cpu( array, value, nthreads, nt, nf ):

      for i in prange( nthreads ):
        offset = i*nt

        for j in range(nt):
          array[offset + j] = value

      offset = nthreads * nt

      for j in range(nf):
        array[offset + j] = value

  #-----------------------------------------------------------------------------
  def exec_numba_cuda ( self, array, value ):

    if len(array.shape) > 1:
      array = array.ravel()

    bpg, tpb = cuda_tpb_bpg_1d( array.shape[0] )

    self._gpu[bpg, tpb]( array, value )

  #-----------------------------------------------------------------------------
  def exec_cpu( self, array, value ):
    if len(array.shape) > 1:
      array = array.ravel()

    nt = array.shape[0] // nthreads
    nf = array.shape[0] % nthreads

    self._cpu( array, value, nthreads, nt, nf )

ndarray_fill = NDArrayFill()
