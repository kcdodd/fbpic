
import functools
import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@functools.lru_cache
def tmp_ndarray( shape, dtype ):
  """Generates a cached uninitialized array

  Note: This will share the same array amonge any method that calls this
  function with the same arguments. Array should not assumed to contained zeros,
  and should always be reinitialized.
  """
  return np.empty( shape, dtype = dtype )
