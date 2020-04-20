# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the particles.
"""
import warnings
import numpy as np
from scipy.constants import e
from .tracking import ParticleTracker
from .elementary_process.ionization import Ionizer
from .elementary_process.compton import ComptonScatterer
from .injection import BallisticBeforePlane, ContinuousInjector, \
                        generate_evenly_spaced

# Load the numba methods
from .push.numba_methods import push_p_numba, push_p_ioniz_numba, \
                                push_p_after_plane_numba, push_x_numba


# use field methods for deposition routines
from fbpic.fields.numba_methods import sum_reduce_2d_array

from fbpic.fields.utility_methods import invvol

# Check if threading is enabled
from fbpic.utils.threading import nthreads, get_chunk_indices
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    # Load the CUDA methods

  from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, cuda_gpu_model
  from .push.cuda_methods import push_p_gpu, push_p_ioniz_gpu, \
                              push_p_after_plane_gpu, push_x_gpu

  from .utilities.cuda_sorting import write_sorting_buffer, \
      get_cell_idx_per_particle, sort_particles_per_cell, \
      prefill_prefix_sum, incl_prefix_sum

from .deposition import (
  deposit_moment_n,
  deposit_moment_nv )

from .gathering import (
  gather_vector )

from fbpic.discrete import (
  ndarray_fill )

class Particles(object) :
    """
    Class that contains the particles data of the simulation

    Main attributes
    ---------------
    - x, y, z : 1darrays containing the Cartesian positions
                of the macroparticles (in meters)
    - uz, uy, uz : 1darrays containing the unitless momenta
                (i.e. px/mc, py/mc, pz/mc)
    At the end or start of any PIC cycle, the momenta should be
    one half-timestep *behind* the position.
    """
    def __init__(self, q, m, n, Npz, zmin, zmax,
                    Npr, rmin, rmax, Nptheta, dt,
                    ux_m=0., uy_m=0., uz_m=0.,
                    ux_th=0., uy_th=0., uz_th=0.,
                    dens_func=None, continuous_injection=True,
                    grid_shape=None, particle_shape='linear',
                    use_cuda=False, dz_particles=None ):
        """
        Initialize a uniform set of particles

        Parameters
        ----------
        q : float (in Coulombs)
           Charge of the particle species

        m : float (in kg)
           Mass of the particle species

        n : float (in particles per m^3)
           Peak density of particles

        Npz : int
           Number of macroparticles along the z axis

        zmin, zmax : floats (in meters)
           z positions between which the particles are initialized

        Npr : int
           Number of macroparticles along the r axis

        rmin, rmax : floats (in meters)
           r positions between which the particles are initialized

        Nptheta : int
           Number of macroparticules along theta

        dt : float (in seconds)
           The timestep for the particle pusher

        ux_m, uy_m, uz_m: floats (dimensionless), optional
           Normalized mean momenta of the injected particles in each direction

        ux_th, uy_th, uz_th: floats (dimensionless), optional
           Normalized thermal momenta in each direction

        dens_func : callable, optional
           A function of the form :
           def dens_func( z, r ) ...
           where z and r are 1d arrays, and which returns
           a 1d array containing the density *relative to n*
           (i.e. a number between 0 and 1) at the given positions

        continuous_injection : bool, optional
           Whether to continuously inject the particles,
           in the case of a moving window

        grid_shape: tuple, optional
            Needed when running on the GPU
            The shape of the local grid (including guard cells), i.e.
            a tuple of the form (Nz, Nr). This is needed in order
            to initialize the sorting of the particles per cell.

        particle_shape: str, optional
            Set the particle shape for the charge/current deposition.
            Possible values are 'linear' and 'cubic' for first and third
            order particle shape factors.

        use_cuda : bool, optional
            Wether to use the GPU or not.

        dz_particles: float (in meter), optional
            The spacing between particles in `z` (for continuous injection)
            In most cases, the spacing between particles can be inferred
            from the arguments `zmin`, `zmax` and `Npz`. However, when
            there are no particles in the initial box (`Npz = 0`),
            `dz_particles` needs to be explicitly passed.
        """
        # Define whether or not to use the GPU
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            warnings.warn(
                'Cuda not available for the particles.\n'
                'Performing the particle operations on the CPU.')
            self.use_cuda = False

        # Generate evenly-spaced particles
        Ntot, x, y, z, ux, uy, uz, gamma_minus_1, w = generate_evenly_spaced(
            Npz, zmin, zmax, Npr, rmin, rmax, Nptheta, n, dens_func,
            ux_m, uy_m, uz_m, ux_th, uy_th, uz_th )

        # Register the properties of the particles
        # (Necessary for the pusher, and when adding more particles later, )
        self.Ntot = Ntot
        self.q = q
        self.m = m
        self.dt = dt

        # Register the particle arrarys
        self.x = x
        self.y = y
        self.z = z
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.gamma_minus_1 = gamma_minus_1
        self.w = w

        # Initialize the fields array (at the positions of the particles)
        self.Ez = np.zeros( Ntot )
        self.Ex = np.zeros( Ntot )
        self.Ey = np.zeros( Ntot )
        self.Bz = np.zeros( Ntot )
        self.Bx = np.zeros( Ntot )
        self.By = np.zeros( Ntot )

        # The particle injector stores information that is useful in order
        # continuously inject particles in the simulation, with moving window
        self.continuous_injection = continuous_injection
        if continuous_injection:
            self.injector = ContinuousInjector( Npz, zmin, zmax, dz_particles,
                                                Npr, rmin, rmax,
                                                Nptheta, n, dens_func,
                                                ux_m, uy_m, uz_m,
                                                ux_th, uy_th, uz_th )
        else:
            self.injector = None

        # By default, there is no particle tracking (see method track)
        self.tracker = None
        # By default, the species experiences no elementary processes
        # (see method make_ionizable and activate_compton)
        self.ionizer = None
        self.compton_scatterer = None
        # Total number of quantities (necessary in MPI communications)
        self.n_integer_quantities = 0
        self.n_float_quantities = 8 # x, y, z, ux, uy, uz, gamma_minus_1, w

        # Register particle shape
        self.particle_shape = particle_shape

        # Register boolean that records whether field array should
        # be rearranged whenever sorting particles
        # (gets modified during the main PIC loop, on GPU)
        self.keep_fields_sorted = False

        self.cell_idx = None
        self.prefix_sum = None
        self.sorted_idx = None
        self.sorting_buffer = None

        # Allocate arrays and register variables when using CUDA
        if self.use_cuda:
            if grid_shape is None:
                raise ValueError("A `grid_shape` is needed when running "
                "on the GPU.\nPlease provide it when initializing particles.")
            # Register grid shape
            self.grid_shape = grid_shape
            # Allocate arrays for the particles sorting when using CUDA
            # Most required arrays always stay on GPU
            Nz, Nr = grid_shape
            self.cell_idx = cuda.device_array( Ntot, dtype=np.int64)
            self.sorted_idx = cuda.device_array( Ntot, dtype=self.cell_idx.dtype )
            self.prefix_sum = cuda.device_array( Nz*(Nr+1), dtype=np.int32 )
            # sorting buffers are initialized on CPU like other particle arrays
            # (because they are swapped with these arrays during sorting)
            self.sorting_buffer = np.empty( Ntot, dtype=np.float64)

            # Register integer thta records shift in the indices,
            # induced by the moving window
            self.prefix_sum_shift = 0
            # Register boolean that records if the particles are sorted or not
            self.sorted = False
            # Define optimal number of CUDA threads per block for deposition
            # and gathering kernels (determined empirically)
            if particle_shape == "cubic":
                self.deposit_tpb = 32
                self.gather_tpb = 256
            else:
                self.deposit_tpb = 16 if cuda_gpu_model == "V100" else 8
                self.gather_tpb = 128



    def send_particles_to_gpu( self ):
        """
        Copy the particles to the GPU.
        Particle arrays of self now point to the GPU arrays.
        """
        if self.use_cuda:
            # Send positions, velocities, inverse gamma and weights
            # to the GPU (CUDA)
            self.x = cuda.to_device(self.x)
            self.y = cuda.to_device(self.y)
            self.z = cuda.to_device(self.z)
            self.ux = cuda.to_device(self.ux)
            self.uy = cuda.to_device(self.uy)
            self.uz = cuda.to_device(self.uz)
            self.gamma_minus_1 = cuda.to_device(self.gamma_minus_1)
            self.w = cuda.to_device(self.w)

            # Copy arrays on the GPU for the field
            # gathering and the particle push
            self.Ex = cuda.to_device(self.Ex)
            self.Ey = cuda.to_device(self.Ey)
            self.Ez = cuda.to_device(self.Ez)
            self.Bx = cuda.to_device(self.Bx)
            self.By = cuda.to_device(self.By)
            self.Bz = cuda.to_device(self.Bz)

            # Copy sorting buffers on the GPU
            self.sorting_buffer = cuda.to_device(self.sorting_buffer)
            if self.n_integer_quantities > 0:
                self.int_sorting_buffer = cuda.to_device(self.int_sorting_buffer)

            # Copy particle tracker data
            if self.tracker is not None:
                self.tracker.send_to_gpu()
            # Copy the ionizasuper().exec( )tion data
            if self.ionizer is not None:
                self.ionizer.send_to_gpu()

    def receive_particles_from_gpu( self ):
        """
        Receive the particles from the GPU.
        Particle arrays are accessible by the CPU again.
        """
        if self.use_cuda:
            # Copy the positions, velocities, inverse gamma and weights
            # to the GPU (CUDA)
            self.x = self.x.copy_to_host()
            self.y = self.y.copy_to_host()
            self.z = self.z.copy_to_host()
            self.ux = self.ux.copy_to_host()
            self.uy = self.uy.copy_to_host()
            self.uz = self.uz.copy_to_host()
            self.gamma_minus_1 = self.gamma_minus_1.copy_to_host()
            self.w = self.w.copy_to_host()

            # Copy arrays on the CPU for the field
            # gathering and the particle push
            self.Ex = self.Ex.copy_to_host()
            self.Ey = self.Ey.copy_to_host()
            self.Ez = self.Ez.copy_to_host()
            self.Bx = self.Bx.copy_to_host()
            self.By = self.By.copy_to_host()
            self.Bz = self.Bz.copy_to_host()

            # Copy arrays on the CPU
            # that represent the sorting arrays
            self.sorting_buffer = self.sorting_buffer.copy_to_host()
            if self.n_integer_quantities > 0:
                self.int_sorting_buffer = self.int_sorting_buffer.copy_to_host()

            # Copy particle tracker data
            if self.tracker is not None:
                self.tracker.receive_from_gpu()
            # Copy the ionization data
            if self.ionizer is not None:
                self.ionizer.receive_from_gpu()

    def generate_continuously_injected_particles( self, time ):
        """
        Generate particles at the right end of the simulation boundary.
        (Typically, in the presence of a moving window.)

        Note that the `ContinuousInjector` object keeps track of the
        positions and number of macroparticles to be injected.
        """
        # This function should only be called if continuous injection is activated
        assert self.continuous_injection == True

        # Have the continuous injector generate the new particles
        Ntot, x, y, z, ux, uy, uz, gamma_minus_1, w = \
                            self.injector.generate_particles( time )

        # Convert them to a particle buffer
        # - Float buffer
        float_buffer = np.empty((self.n_float_quantities,Ntot),dtype=np.float64)
        float_buffer[0,:] = x
        float_buffer[1,:] = y
        float_buffer[2,:] = z
        float_buffer[3,:] = ux
        float_buffer[4,:] = uy
        float_buffer[5,:] = uz
        float_buffer[6,:] = gamma_minus_1
        float_buffer[7,:] = w
        if self.ionizer is not None:
            # All new particles start at the default ionization level
            float_buffer[8,:] = w * self.ionizer.level_start
        # - Integer buffer
        uint_buffer = np.empty((self.n_integer_quantities,Ntot),dtype=np.uint64)
        i_int = 0
        if self.tracker is not None:
            uint_buffer[i_int,:] = self.tracker.generate_new_ids( Ntot )
            i_int += 1
        if self.ionizer is not None:
            # All new particles start at the default ionization level
            uint_buffer[i_int,:] = self.ionizer.level_start

        return( float_buffer, uint_buffer )


    def track( self, comm ):
        """
        Activate particle tracking for the current species
        (i.e. allocates an array of unique IDs for each macroparticle;
        these IDs are written in the openPMD file)

        Parameters
        ----------
        comm: an fbpic.BoundaryCommunicator object
            Contains information about the number of processors
        """
        self.tracker = ParticleTracker( comm.size, comm.rank, self.Ntot )
        # Update the number of integer quantities
        self.n_integer_quantities += 1
        # Allocate the integer sorting buffer if needed
        if hasattr( self, 'int_sorting_buffer' ) is False and self.use_cuda:
            self.int_sorting_buffer = np.empty( self.Ntot, dtype=np.uint64 )

    def activate_compton( self, target_species, laser_energy, laser_wavelength,
        laser_waist, laser_ctau, laser_initial_z0, ratio_w_electron_photon=1,
        boost=None ):
        """
        Activate Compton scattering.

        This considers a counterpropagating Gaussian laser pulse (which is not
        represented on the grid, for compatibility with the boosted-frame,
        but is instead assumed to propagate rigidly along the z axis).
        Interaction between this laser and the current species results
        in the generation of photons, according to the Klein-Nishina formula.

        See the docstring of the class `ComptonScatterer` for more information
        on the physical model used, and its domain of validity.

        The API of this function is not stable, and may change in the future.

        Parameters:
        -----------
        target_species: a `Particles` object
            The photons species, to which new macroparticles will be added.

        laser_energy: float (in Joules)
            The energy of the counterpropagating laser pulse (in the lab frame)

        laser_wavelength: float (in meters)
            The wavelength of the laser pulse (in the lab frame)

        laser_waist, laser_ctau: floats (in meters)
            The waist and duration of the laser pulse (in the lab frame)
            Both defined as the distance, from the laser peak, where
            the *field* envelope reaches 1/e of its peak value.

        laser_initial_z0: float (in meters)
            The initial position of the laser pulse (in the lab frame)

        ratio_w_electron_photon: float
            The ratio of the weight of an electron macroparticle to the
            weight of the photon macroparticles that it will emit.
            Increasing this ratio increases the number of photon macroparticles
            that will be emitted and therefore improves statistics.
        """
        self.compton_scatterer = ComptonScatterer(
            self, target_species, laser_energy, laser_wavelength,
            laser_waist, laser_ctau, laser_initial_z0,
            ratio_w_electron_photon, boost )


    def make_ionizable(self, element, target_species,
                       level_start=0, level_max=None):
        """
        Make this species ionizable.

        The implemented ionization model is the **ADK model**
        (using the **instantaneous** electric field, i.e. **without** averaging
        over the laser period).

        The expression of the ionization rate can be found in
        `Chen, JCP 236 (2013), equation 2
        <https://www.sciencedirect.com/science/article/pii/S0021999112007097>`_.

        Note that the implementation in FBPIC evaluates this ionization rate
        *in the reference frame of each macroparticle*, and is thus valid
        in lab-frame simulations as well as boosted-frame simulation.

        Parameters
        ----------
        element: string
            The atomic symbol of the considered ionizable species
            (e.g. 'He', 'N' ;  do not use 'Helium' or 'Nitrogen')

        target_species: a `Particles` object, or a dictionary of `Particles`
            Stores the electron macroparticles that are created in
            the ionization process. If a single `Particles` object is passed,
            then electrons from all ionization levels are stored into this
            object. If a dictionary is passed, then its keys should be integers
            (corresponding to the ionizable levels of `element`, starting
            at `level_start`), and its values should be `Particles` objects.
            In this case, the electrons from each distinct ionizable level
            will be stored into these separate objects. Note that using
            separate objects will typically require longer computing time.

        level_start: int
            The ionization level at which the macroparticles are initially
            (e.g. 0 for initially neutral atoms)

        level_max: int, optional
            If not None, defines the maximum ionization level that
            macroparticles can reach. Should not exceed the physical
            limit for the chosen element.
        """
        # Initialize the ionizer module
        self.ionizer = Ionizer( element, self, target_species,
                                level_start, level_max=level_max )
        # Set charge to the elementary charge e (assumed by deposition kernel,
        # when using self.ionizer.w_times_level as the effective weight)
        self.q = e

        # Update the number of float and int arrays
        self.n_float_quantities += 1 # w_times_level
        self.n_integer_quantities += 1 # ionization_level
        # Allocate the integer sorting buffer if needed
        if hasattr( self, 'int_sorting_buffer' ) is False and self.use_cuda:
            self.int_sorting_buffer = np.empty( self.Ntot, dtype=np.uint64 )


    def handle_elementary_processes( self, t ):
        """
        Handle elementary processes for this species (e.g. ionization,
        Compton scattering) at simulation time t.
        """
        # Ionization
        if self.ionizer is not None:
            self.ionizer.handle_ionization( self )
        # Compton scattering
        if self.compton_scatterer is not None:
            self.compton_scatterer.handle_scattering( self, t )




    def rearrange_particle_arrays( self ):
        """
        Rearranges the particle data arrays to match with the sorted
        cell index array. The sorted index array is used to resort the
        arrays. A particle buffer is used to temporarily store
        the rearranged data.
        """
        # Get the threads per block and the blocks per grid
        dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
        # Iterate over (float) particle attributes
        attr_list = [ (self,'x'), (self,'y'), (self,'z'), \
                        (self,'ux'), (self,'uy'), (self,'uz'), \
                        (self, 'w'), (self,'gamma_minus_1') ]
        if self.keep_fields_sorted:
            attr_list += [ (self, 'Ex'), (self, 'Ey'), (self, 'Ez'), \
                            (self, 'Bx'), (self, 'By'), (self, 'Bz') ]
        if self.ionizer is not None:
            attr_list += [ (self.ionizer,'w_times_level') ]
        for attr in attr_list:
            # Get particle GPU array
            particle_array = getattr( attr[0], attr[1] )
            # Write particle data to particle buffer array while rearranging
            write_sorting_buffer[dim_grid_1d, dim_block_1d](
                self.sorted_idx, particle_array, self.sorting_buffer)
            # Assign the particle buffer to
            # the initial particle data array
            setattr( attr[0], attr[1], self.sorting_buffer)
            # Assign the old particle data array to the particle buffer
            self.sorting_buffer = particle_array
        # Iterate over (integer) particle attributes
        attr_list = [ ]
        if self.tracker is not None:
            attr_list += [ (self.tracker,'id') ]
        if self.ionizer is not None:
            attr_list += [ (self.ionizer,'ionization_level') ]
        for attr in attr_list:
            # Get particle GPU array
            particle_array = getattr( attr[0], attr[1] )
            # Write particle data to particle buffer array while rearranging
            write_sorting_buffer[dim_grid_1d, dim_block_1d](
                self.sorted_idx, particle_array, self.int_sorting_buffer)
            # Assign the particle buffer to
            # the initial particle data array
            setattr( attr[0], attr[1], self.int_sorting_buffer)
            # Assign the old particle data array to the particle buffer
            self.int_sorting_buffer = particle_array

    def push_p( self, t ) :
        """
        Advance the particles' momenta over one timestep, using the Vay pusher
        Reference : Vay, Physics of Plasmas 15, 056701 (2008)

        This assumes that the momenta (ux, uy, uz) are initially one
        half-timestep *behind* the positions (x, y, z), and it brings
        them one half-timestep *ahead* of the positions.

        Parameters
        ----------
        t: float
            The current simulation time
            (Useful for particles that are ballistic before a given plane)
        """
        # Skip push for neutral particles (e.g. photons)
        if self.q == 0:
            return
        # For particles that are ballistic before a plane,
        # get the current position of the plane
        if isinstance( self.injector, BallisticBeforePlane ):
            z_plane = self.injector.get_current_plane_position( t )
            if self.ionizer is not None:
                raise NotImplementedError('Ballistic injection before a plane '
                    'is not implemented for ionizable particles.')
        else:
            z_plane = None

        # GPU (CUDA) version
        if self.use_cuda:
            # Get the threads per block and the blocks per grid
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
            # Call the CUDA Kernel for the particle push
            if self.ionizer is not None:
                # Ionizable species can have a charge that depends on the
                # macroparticle, and hence require a different function
                push_p_ioniz_gpu[dim_grid_1d, dim_block_1d](
                    self.ux, self.uy, self.uz, self.gamma_minus_1,
                    self.Ex, self.Ey, self.Ez,
                    self.Bx, self.By, self.Bz,
                    self.m, self.Ntot, self.dt, self.ionizer.ionization_level )
            elif z_plane is not None:
                # Particles that are ballistic before a plane also
                # require a different pusher
                push_p_after_plane_gpu[dim_grid_1d, dim_block_1d](
                    self.z, z_plane,
                    self.ux, self.uy, self.uz, self.gamma_minus_1,
                    self.Ex, self.Ey, self.Ez,
                    self.Bx, self.By, self.Bz,
                    self.q, self.m, self.Ntot, self.dt )
            else:
                # Standard pusher
                push_p_gpu[dim_grid_1d, dim_block_1d](
                    self.ux, self.uy, self.uz, self.gamma_minus_1,
                    self.Ex, self.Ey, self.Ez,
                    self.Bx, self.By, self.Bz,
                    self.q, self.m, self.Ntot, self.dt )

        # CPU version
        else:
            if self.ionizer is not None:
                # Ionizable species can have a charge that depends on the
                # macroparticle, and hence require a different function
                push_p_ioniz_numba(self.ux, self.uy, self.uz, self.gamma_minus_1,
                    self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz,
                    self.m, self.Ntot, self.dt, self.ionizer.ionization_level )
            elif z_plane is not None:
                # Particles that are ballistic before a plane also
                # require a different pusher
                push_p_after_plane_numba(
                    self.z, z_plane,
                    self.ux, self.uy, self.uz, self.gamma_minus_1,
                    self.Ex, self.Ey, self.Ez,
                    self.Bx, self.By, self.Bz,
                    self.q, self.m, self.Ntot, self.dt )
            else:
                # Standard pusher
                push_p_numba(self.ux, self.uy, self.uz, self.gamma_minus_1,
                    self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz,
                    self.q, self.m, self.Ntot, self.dt )


    def push_x( self, dt, x_push=1., y_push=1., z_push=1. ) :
        """
        Advance the particles' positions over `dt` using the current
        momenta (ux, uy, uz).

        Parameters:
        -----------
        dt: float, seconds
            The timestep that should be used for the push
            (This can be typically be half of the simulation timestep)

        x_push, y_push, z_push: float, dimensionless
            Multiplying coefficient for the momenta in x, y and z
            e.g. if x_push=1., the particles are pushed forward in x
                 if x_push=-1., the particles are pushed backward in x
        """
        # GPU (CUDA) version
        if self.use_cuda:
            # Get the threads per block and the blocks per grid
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
            # Call the CUDA Kernel for push in x
            push_x_gpu[dim_grid_1d, dim_block_1d](
                self.x, self.y, self.z,
                self.ux, self.uy, self.uz,
                self.gamma_minus_1, dt, x_push, y_push, z_push )
            # The particle array is unsorted after the push in x
            self.sorted = False
        # CPU version
        else:
            push_x_numba( self.x, self.y, self.z,
                self.ux, self.uy, self.uz,
                self.gamma_minus_1, self.Ntot,
                dt, x_push, y_push, z_push )

    #---------------------------------------------------------------------------
    def gather_clear( self ):
        ndarray_fill.exec( self.Ex, 0.0 )
        ndarray_fill.exec( self.Ey, 0.0 )
        ndarray_fill.exec( self.Ez, 0.0 )
        ndarray_fill.exec( self.Bx, 0.0 )
        ndarray_fill.exec( self.By, 0.0 )
        ndarray_fill.exec( self.Bz, 0.0 )

    #---------------------------------------------------------------------------
    def gather( self, grid, comm ) :
        """
        Gather the fields onto the macroparticles

        This assumes that the particle positions are currently at
        the same timestep as the field that is to be gathered.

        Parameter
        ----------
        grid : a list of InterpolationGrid objects
             (one InterpolationGrid object per azimuthal mode)
             Contains the field values on the interpolation grid

        comm: an fbpic.BoundaryCommunicator object
            Contains information about the number of processors
            and the local and global box dimensions.
        """
        # Skip gathering for neutral particles (e.g. photons)
        if self.q == 0:
            return

        # Number of modes
        Nm = len(grid)

        # Restrict field gathering to physical domain
        rmax_gather = comm.get_rmax( with_damp=False )

        gather_vector.exec(
          vector = [ self.Ex, self.Ey, self.Ez ],
          x = self.x, y = self.y, z = self.z,
          grid = [ [ grid[m].Er, grid[m].Et, grid[m].Ez ] for m in range(len(grid)) ],
          dz = dz, zmin = zmin, dr = dr, rmin = rmin, rmax = rmax_gather,
          ptcl_shape = self.particle_shape )

        gather_vector.exec(
          vector = [ self.Bx, self.By, self.Bz ],
          x = self.x, y = self.y, z = self.z,
          grid = [ [ grid[m].Br, grid[m].Bt, grid[m].Bz ] for m in range(len(grid)) ],
          dz = dz, zmin = zmin, dr = dr, rmin = rmin, rmax = rmax_gather,
          ptcl_shape = self.particle_shape )

    #---------------------------------------------------------------------------
    def deposit( self,
        moment,
        grid,
        zmin,
        dz,
        rmin,
        dr,
        coeff = None ) :
        """
        Deposit the particles distribution moments onto 2d grid for each
        azimuthal mode.

        Parameter
        ----------
        moment : str
          Indicates the moment of the particle distribution to compute
          ( density = 1/cell. multiply by cell/m^3 to get physical density )

          'n' = coeff * density
          'nke' = coeff * density * ( gamma - 1 )
          'nv' = coeff * density * v / c
          'np' = coeff * density * gamma * v / c

        grid : array
          arrays to deposite computed moment for each m mode
          ( nm = number of azimuthal modes )

          'n', 'nke' : (nm, nz, nr)
          'np', 'nv' : (nm, 3, nz, nr)

        coeff : float, str, optional
          coefficient to multiply before adding to grid

        zmin : float
        dz : float
        rmin : float
        dr : float
        """
        # Shortcuts and safe-guards

        assert moment in [ 'n', 'nke', 'nv', 'np' ]
        assert self.particle_shape in [ 'linear', 'cubic' ]

        if coeff is None:
          coeff = 1.0

        if coeff == 0.0:
          return

        if moment in [ 'n', 'nke' ]:
          nz = grid[0].shape[0]
          nr = grid[0].shape[1]
        else:
          nz = grid[0][0].shape[0]
          nr = grid[0][0].shape[1]

        # When running on GPU: first sort the arrays of particles
        if self.use_cuda:
          # Sort the particles
          if not self.sorted:
            self.sort_particles(
              zmin = zmin,
              dz = dz,
              nz = nz,
              rmin = rmin,
              dr = dr,
              nr = nr )

            # The particles are now sorted and rearranged
            self.sorted = True

        # For ionizable atoms: set the effective weight to the weight
        # times the ionization level (on GPU, this needs to be done *after*
        # sorting, otherwise `weight` is not equal to the corresponding array)
        # if self.ionizer is not None:
        #     weight = self.ionizer.w_times_level
        # else:
        #     weight = self.w

        if moment in [ 'n', 'nke' ]:

          if moment == 'n':
            gamma_minus_1 = None
          else:
            gamma_minus_1 = self.gamma_minus_1

          deposit_moment_n.exec(
            grid = grid,
            coeff = coeff,
            weight = self.w,
            cell_idx = self.cell_idx,
            prefix_sum = self.prefix_sum,
            x = self.x, y = self.y, z = self.z,
            gamma_minus_1 = gamma_minus_1,
            dz = dz, zmin = zmin, dr = dr, rmin = rmin,
            ptcl_shape = self.particle_shape )

        elif moment in [ 'nv', 'np' ]:

          if moment == 'np':
            gamma_minus_1 = None
          else:
            gamma_minus_1 = self.gamma_minus_1

          deposit_moment_nv.exec(
            grid = grid,
            coeff = coeff,
            weight = self.w,
            cell_idx = self.cell_idx,
            prefix_sum = self.prefix_sum,
            x = self.x, y = self.y, z = self.z,
            ux = self.ux, uy = self.uy, uz = self.uz,
            gamma_minus_1 = gamma_minus_1,
            dz = dz, zmin = zmin, dr = dr, rmin = rmin,
            ptcl_shape = self.particle_shape )

    #---------------------------------------------------------------------------
    def sort_particles(self,
        zmin,
        dz,
        nz,
        rmin,
        dr,
        nr ):
        """
        Sort the particles by performing the following steps:
        1. Get fied cell index
        2. Sort field cell index
        3. Parallel prefix sum
        4. Rearrange particle arrays

        Parameter
        ----------
        zmin : float
        dz : float
        nz : int
        rmin : float
        dr : float
        nr : int
        """
        # Shortcut for interpolation grids

        # Get the threads per block and the blocks per grid
        dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
        dim_grid_2d_flat, dim_block_2d_flat = \
                cuda_tpb_bpg_1d( self.prefix_sum.shape[0] )

        # ------------------------
        # Sorting of the particles
        # ------------------------
        # Get the cell index of each particle
        # (defined by iz_lower and ir_lower)
        get_cell_idx_per_particle[dim_grid_1d, dim_block_1d](
            self.cell_idx,
            self.sorted_idx,
            self.x, self.y, self.z,
            1/dz, zmin, nz,
            1/dr, rmin, nr )
        # Sort the cell index array and modify the sorted_idx array
        # accordingly. The value of the sorted_idx array corresponds
        # to the index of the sorted particle in the other particle
        # arrays.
        sort_particles_per_cell(self.cell_idx, self.sorted_idx)
        # Reset the old prefix sum
        self.prefix_sum_shift = 0
        prefill_prefix_sum[dim_grid_2d_flat, dim_block_2d_flat](
            self.cell_idx, self.prefix_sum, self.Ntot )
        # Perform the inclusive parallel prefix sum
        incl_prefix_sum[dim_grid_1d, dim_block_1d](
            self.cell_idx, self.prefix_sum)
        # Rearrange the particle arrays
        self.rearrange_particle_arrays()
