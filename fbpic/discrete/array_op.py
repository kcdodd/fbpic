
import numpy as np

from fbpic.utils.cuda import cuda_installed as numba_cuda_installed

if numba_cuda_installed:
  from numba import cuda as numba_cuda

class ArrayOp:
  """Abstraction of array operation that executes on either cpu or gpu

  Derive amd reimplement for new array operation.
  """
  #-----------------------------------------------------------------------------
  def __init__( self ):

    if numba_cuda_installed:
      self.init_numba_cuda()

    self.init_cpu()

  #-----------------------------------------------------------------------------
  def init_cpu( self ):
    """Initialize cpu implementation
    """
    pass

  #-----------------------------------------------------------------------------
  def init_numba_cuda( self ):
    """Initialize cuda implementation
    """
    pass

  #-----------------------------------------------------------------------------
  def attr ( self, func ):
    """Function decorator that adds the function to the instance of ArrayOp
    """
    if numba_cuda_installed and isinstance( func, numba_cuda.compiler.AutoJitCUDAKernel ):
      setattr( self, func.py_func.__name__, func )
    else:
      setattr( self, func.__name__, func )

      return func

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

    use_numba_cuda = numba_cuda_installed and (
      any(
        isinstance(arg, numba_cuda.cudadrv.devicearray.DeviceNDArray)
        for kw,arg in kwargs.items() )
      or gpu )

    if use_numba_cuda:
      self.exec_numba_cuda( **kwargs )
    else:
      self.exec_cpu( **kwargs )



  #-----------------------------------------------------------------------------
  def exec_cpu( self, **kwargs ):
    """Execute cpu implementation
    """
    raise NotImplementedError("exec_cpu not implemented")

  #-----------------------------------------------------------------------------
  def exec_numba_cuda ( self, **kwargs ):
    """Execute Numba cuda implementation
    """
    raise NotImplementedError("exec_numba_cuda not implemented")
