import numpy as np

from body import Body
from util import pairwise_difference


class SPHSystem():
    '''
    Encapsulates a full Smoothed Particle Hydrodynamics (SPH) system and related functions.
    For simplicity the simulation is bound to x,y ∈ [0,1]

    For a concise summary on SPH systems from 2016 see:
    https://eprints.bournemouth.ac.uk/23384/1/2016%20Fluid%20simulation.pdf
    Or for a more in depth explanation:
    https://interactivecomputergraphics.github.io/SPH-Tutorial/pdf/SPH_Tutorial.pdf
    This Python example was also referenced:
    https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1
    '''

    def __init__(self, masses, positions, bodies, smoothing_length = 0.005, stiffness_constant = 0.1, timestep = 0.005):
        '''
        Initializer for SPHSystem.

        Parameters
        ----------
        masses : numpy.ndarray
            A 1*N vector with the masses of every particle.
        positions : numpy.ndarray
            A 2*N matrix with the positions of every particle.
        bodies : ?
            TODO Doc
        
        All particle parameters describe a single particle at a given index.
        '''

        # make sure inputs are of correct dimensions
        assert(masses.shape[1] == positions.shape[1])
        assert(masses.shape[0] == 1 and positions.shape[0] == 2)

        self.mass = masses
        self.inverse_mass = 1.0 / self.mass
        self.position = positions
        self.body = bodies
        self.smoothing_length = smoothing_length
        self.stiffness_constant = stiffness_constant

        # start at t=0 with no velocity
        self.time = 0
        self.timestep = timestep
        self.velocity = np.zeros((2, masses.shape[1]))

    def update(self):
        '''
        Move the SPH system in time by Δt = self.timestep.
        '''

        # get the net forces acting on every particle
        internal_forces = self.getInternalForces(self.position)
        external_forces = self.getExternalForces(self.position)
        net_forces      = internal_forces + external_forces

        # integrate velocity and position
        self.velocity += net_forces * self.timestep * self.inverse_mass
        self.position += self.velocity * self.timestep

        # solve boundary breaks
        self.velocity *= np.ones(self.velocity.shape) - 1.25 * ((self.position < 0) | (self.position > 1)) # TODO cleanup
        self.position  = self.position.clip(0, 1)

        self.time += self.timestep

        # update bodies
        # TODO cleanup
        for body in self.body:
            vol   = 0
            direc = np.zeros(2)
            for part in range(self.position.shape[1]):
                if abs(np.linalg.norm(body.c - self.position[:,part])) < 0.1:
                    vol += 1
                    direc += body.c - self.position[:,part]
            if np.linalg.norm(direc) != 0:
                direc = direc / np.linalg.norm(direc)

            force_grav = np.array([0, 9.81 * body.m])
            force_buoy = 1 * body.m * 9.81 * vol * direc
            net_forces = force_grav + force_buoy

            body.v += net_forces * self.timestep * (1.0 / body.m)
            body.c += body.v * self.timestep

            body.v *= np.array([1, 1]) - 1.25 * ((body.c < 0.1) | (body.c > 0.9))
            body.c = body.c.clip(0.1, 0.9)

    def getInternalForces(self, points):
        '''
        Obtains all the internal forces acting on the given points.

        Parameters
        ----------
        points : numpy.ndarray
            A 2*N vector with the sampling points.
        
        Returns
        -------
        force_total : numpy.ndarray
            The 2*N internal forces working on the points. 
        '''

        density = self.sampleDensity(points)
        pressure = self.samplePressure(density)

        force_pressure = 1.0 / density * np.sum(self.mass * (pressure / density) * self.smoothingKernelGrad(points), axis=1) # TODO alt formula
        force_viscosity = 0 # TODO

        force_total = force_pressure + force_viscosity

        return force_total

    def getExternalForces(self, points):
        '''
        Obtains all the external forces acting on the given points.

        Parameters
        ----------
        points : numpy.ndarray
            A 2*N vector with the sampling points.
        
        Returns
        -------
        force_total : numpy.ndarray
            The 2*N external forces working on the points. 
        '''

        force_gravity = np.vstack((np.zeros(self.mass.shape[1]), self.mass * -9.81))
        force_object = 0 # TODO

        force_total = force_gravity + force_object

        return force_total

    def sampleDensity(self, points):
        '''
        ρ: Density at sampling points.

        Parameters
        ----------
        points : numpy.ndarray
            A 2*N vector with the sampling points.

        Returns
        -------
        density : numpy.ndarray
            The 1*N density samples.
        '''

        density = np.sum(self.mass * self.smoothingKernel(points), axis=1) # TODO axis?

        return density
    
    def samplePressure(self, density):
        '''
        p: Pressure at sampling points.

        Parameters
        ----------
        density : numpy.ndarray
            A 1*N vector with density samples.
        
        Returns
        -------
        pressure
            The 1*N pressure samples.
        '''

        pressure = self.stiffness_constant * density # TODO (p-p0)

        return pressure

    def smoothingKernel(self, points):
        '''
        W: Smoothing kernel for sampling points.

        Parameters
        ----------
        points : numpy.ndarray
            A 2*N vector with the sampling points.

        Returns
        -------
        smoothing : numpy.ndarray
            The N*M smoothing scalars, where M is the particle count.
        '''

        dx, dy = pairwise_difference(points, self.position)
        distances = np.sqrt(dx**2 + dy**2)

        smoothing = (1.0 / (self.smoothing_length * np.sqrt(np.pi)))**3 * np.exp( -distances**2 / self.smoothing_length**2) # TODO correct?

        return smoothing

    def smoothingKernelGrad(self, points):
        '''
        ∇W: Gradient of smoothing kernel for particles.

        Parameters
        ----------
        points : numpy.ndarray
            A 2*N vector with the sampling points.
        
        Returns
        -------
        smoothingX, smoothingY : numpy.ndarray
            The N*M smoothing gradients, where M is the particle count.
        '''

        dx, dy = pairwise_difference(points, self.position)
        distances = np.sqrt(dx**2 + dy**2)

        smoothing = (-2 * np.exp(-distances**2 / self.smoothing_length**2) / self.smoothing_length**5 / np.pi**(3/2)) * distances # TODO correct?
        smoothingX = smoothing * dx
        smoothingY = smoothing * dy

        return smoothingX, smoothingY


if __name__ == '__main__':
    # render the SPH
    from render import render_sph
    render_sph()

