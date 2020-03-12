from fbpic.discrete import (
  ArrayOp,
  tmp_ndarray,
  tmp_numba_device_ndarray,
  ndarray_fill )

import math
import numpy as np

from fbpic.utils.threading import (
  nthreads,
  njit_parallel,
  prange,
  get_chunk_indices )

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
  from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, cuda_gpu_model

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class IntegratePotential ( ArrayOp ):
  """Integrates electric field along z direction with zero at infinity
  """

  #-----------------------------------------------------------------------------
  def exec (self,
    phi,
    Ez,
    kz,
    gpu = False ):

    super().exec(
      phi = phi,
      Ez = Ez,
      kz = kz,
      gpu = gpu )

  #-----------------------------------------------------------------------------
  def init_numba_cuda( self ):
    from numba import cuda

    @self.attr
    @cuda.jit
    def _gpu( phi, Ez, kz ):

      i = cuda.grid(1)

      if i < phi.shape[0]:
        if kz[i] == 0.0:
          phi[i] = 0.0
        else:
          phi[i] = Ez[i] / ( 1j * kz[i] )

  #-----------------------------------------------------------------------------
  def init_cpu( self ):

    @self.attr
    @njit_parallel
    def _cpu( phi, Ez, kz, np, nt, nf ):

      for i in prange( np ):
        offset = i*nt
        for j in range(nt):
          if kz[offset + j] == 0.0:
            phi[offset + j] = 0.0
          else:
            phi[offset + j] = Ez[offset + j] / ( 1j * kz[offset + j] )

      for j in range(nf):
        if kz[j] == 0.0:
          phi[j] = 0.0
        else:
          phi[j] = Ez[j] / ( 1j * kz[j] )

  #-----------------------------------------------------------------------------
  def exec_numba_cuda ( self, phi, Ez, kz ):

    if len(phi.shape) > 1:
      phi = phi.ravel()
      Ez = Ez.ravel()
      kz = kz.ravel()

    bpg, tpb = cuda_tpb_bpg_1d( phi.shape[0] )

    self._gpu[bpg, tpb]( phi, Ez, kz )

  #-----------------------------------------------------------------------------
  def exec_cpu( self, phi, Ez, kz ):
    if len(phi.shape) > 1:
      phi = phi.ravel()
      Ez = Ez.ravel()
      kz = kz.ravel()

    nt = phi.shape[0] // nthreads
    nf = phi.shape[0] % nthreads

    self._cpu( phi, Ez, kz, nthreads, nt, nf )

integrate_potential = IntegratePotential()
