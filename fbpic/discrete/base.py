
import numpy as np

from fbpic.utils.cuda import cuda_installed

if cuda_installed:
  from numba import cuda

class ArrayOp:
  """
  Parameters
  ----------
  to_gpu : bool
    Copies any supplied ndarray to gpu before executing gpu routine, and back to
    the ndarray after. Otherwise, presence of ndarray causes use of cpu routine,
    and presence of gpu array causes use of gpu routine
  """
  #-----------------------------------------------------------------------------
  def __init__( self,
    to_gpu = False ):

    self._to_gpu = to_gpu if cuda_installed else False

    if cuda_installed:
      self.init_gpu()

    self.init_cpu()

  #-----------------------------------------------------------------------------
  def init_cpu( self ):
    pass

  #-----------------------------------------------------------------------------
  def init_gpu( self ):
    pass

  #-----------------------------------------------------------------------------
  def exec( self, **kwargs ):

    kwargs_in = kwargs

    if self._to_gpu:
      # copy to gpu array

      kwargs_in = kwargs.copy()

      for k, v in kwargs.items():
        if type(v) is np.ndarray:
          kwargs_in[k] = cuda.to_device( v )

    test_array = next(iter(kwargs_in.items()))[1]

    if type( test_array ) is np.ndarray:
      self.exec_cpu( **kwargs_in )
    else:
      self.exec_gpu( **kwargs_in )

    if self._to_gpu:
      # copy array from device
      for k, v in kwargs.items():
        if type(v) is np.ndarray:
          kwargs_in[k].copy_to_host( ary = v )

  #-----------------------------------------------------------------------------
  def exec_cpu( self, **kwargs ):
    raise NotImplementedError("exec_cpu not implemented")

  #-----------------------------------------------------------------------------
  def exec_gpu ( self, **kwargs ):
    raise NotImplementedError("exec_gpu not implemented")
