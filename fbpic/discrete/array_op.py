
import numpy as np

from fbpic.utils.cuda import cuda_installed

if cuda_installed:
  from numba import cuda

class ArrayOp:
  """Abstraction of array operation that executes on either cpu or gpu

  Derive amd reimplement for new array operation.
  """
  #-----------------------------------------------------------------------------
  def __init__( self ):

    if cuda_installed:
      self.init_gpu()

    self.init_cpu()

  #-----------------------------------------------------------------------------
  def init_cpu( self ):
    """Initialize cpu implementation
    """
    pass

  #-----------------------------------------------------------------------------
  def init_gpu( self ):
    """Initialize gpu implementation
    """
    pass

  #-----------------------------------------------------------------------------
  def exec( self, gpu = False, **kwargs ):
    """Execute array operation

    Parameters
    ----------
    gpu : bool
      Use gpu implementation (if available) even when ndarrays are passed in as arguments
    **kwargs
      arguments passed to implementation exec function
    """

    kwargs_in = kwargs

    use_gpu = cuda_installed and (
      any(
        isinstance(arg, cuda.cudadrv.devicearray.DeviceNDArray)
        for kw,arg in kwargs.items() )
      or gpu )

    if use_gpu:
      self.exec_gpu( **kwargs )
    else:
      self.exec_cpu( **kwargs )



  #-----------------------------------------------------------------------------
  def exec_cpu( self, **kwargs ):
    """Execute cpu implementation
    """
    raise NotImplementedError("exec_cpu not implemented")

  #-----------------------------------------------------------------------------
  def exec_gpu ( self, **kwargs ):
    """Execute gpu implementation
    """
    raise NotImplementedError("exec_gpu not implemented")
