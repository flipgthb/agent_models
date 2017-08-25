# =---------------------------------------------------------------------------=
# HEAD: Homo Entropicus Agents, Dude
# Version: 0.1.0
# Author: Felippe Alves
# E-Mail: flipgm@gmail.com
# =---------------------------------------------------------------------------=
# This module contains the basic dynamics generator, the class for Entropic
# Agents Societies (and the particular case of Teacher-Student learning
# scenario), and some utility funtions used in the analysis of such
# systems.
#
# I separated it in some sections:
#  1. Simulation Interface
#    1.1 Utility functions
#    1.2 Dynamics Interface
#  2. Agents Society Model
#    2.1 Theory
#    2.2 Model
#  3. Teacher-Student Scenario
#  4. Restricted Universe of Discorse Scenario
#  5. Analysis Interface
#
# Everything is pretty straight forward, and should be no problem to understand
# what the code does from reading it.

import functools
from itertools import count , islice
from copy import deepcopy
from collections import defaultdict
from numpy import (cos, sin, sqrt, r_, ones, tile, vstack, array, ndindex, ndim,
                   log, outer, sign, eye, infty as oo, einsum, zeros, stack,
                   zeros_like, fill_diagonal, clip, pi, split, arange, empty, NaN,
                   linspace)
from numpy.random import choice, randn, rand
from numpy.linalg import norm, qr
from scipy.stats.distributions import norm as normal

__all__ = ['triangle_indices', 'rand_sphere', 'random_at_angle',
           'dynamics', 'n_steps', 'record_trajectory',
           'BOMAgentSociety', 'random_initialization',
           'TeacherStutdentScneario', 'RUDBOMAgentSociety']

# # Simulation Interface

# ## Utility functions
Phi = normal.cdf
G = normal.pdf
row_norm = functools.partial(norm, keepdims=True, axis=1)


def triangle_indices(N):
    """Compute all triples of integer indices and, as consequence,
    all the closed oriented triangles in a complete graph with N
    vertices.
    Input:
    -----
      N: int - the number of vertices in a complete graph

    Output:
    -------
      N(N-1)(N-2) x 3 array where each line is a triple of
      indices in a triangle
    """
    tri_idx = array([(a, b, c) for
                     (a, b, c) in ndindex((N, N, N))
                     if (a!=b and a!=c and b!=c)])
    return tri_idx


def rand_sphere(K, size=1):
    """Computes K-dimensional vectors distributed uniformly
    in the unit sphere.
    Input:
    ------
      K: int - the dimension of the vector space
      size (default 1): int - the number of vectors to sample
    Output:
    -------
      array with shape (K,) if size==1 or (size, K) if size>1
    """
    v = randn(size, K)
    v /= row_norm(v)
    if size == 1:
        v = v.reshape(K)
    return v


def random_at_angle(B, theta):
    """Computes a random vector with angle theta from a vector B.
    Input:
    ------
      B: 1d array - the reference vector
      theta: float - the angle from the reference vector B
    """
    K = B.shape[0]
    B0, v = (qr(vstack([B, randn(K)]).T)[0]*r_[-1, 1]).T
    w = cos(theta)*B0 + sin(theta)*v
    w /= norm(w)
    w *= sign(cos(theta)*w@B0)
    return w


def random_from_cosine(B, cosine):
    """Computes a random vector with a given cosine with the vector B.
    Input:
    ------
      B: 1d array - the reference vector
      cosine: float - the cosine with the reference vector B
    """
    K = B.shape[0]
    B0, v = (qr(vstack([B, randn(K)]).T)[0]).T
    B0 *= sign(B0@B)
    w = cosine*B0 + sin(arccos(cosine))*v
    assert abs(norm(w)-1) < 1e-5
    return w


## Dynamics interface
def dynamics(system, *, in_place=True, **params):
    """Generator to loop over systems (obejcts) with
    a update and a get_state methods.
    Input:
    ------
      system: any object with the following interface:
        system.update(**params) - a method for updating the system state
        system.get_state(**params) - a method to retrieve the system state
      in_place (default False): bool - if True, the updates are done to the
      system passed in the call. If false, a deepcopy is made before looping
      **params: any other parameter to be passed to the system methods
    Output:
    -------
      an iterator over the system dynamics
    """
    if not in_place:
        system = deepcopy(system)
    for _ in count():
        yield system.get_state(**params)
        system.update(**params)


def n_steps(infinite_iterator, n):
    """An alias to enumerate(itertools.islice(ifinite_iteretor, n))"""
    return enumerate(islice(infinite_iterator, n))


def record_trajectory(system, number_of_steps, *,
                      record_step=lambda n: True, **params):
    """A simple materialization of the dynamics iterator for a finite
    number of steps, possibly filtered.
    Input:
    ------
      system: object following the dynamics interface
      (see sas.dyamics docstring)
      number_of_steps: int - the number of steps to iterate the dynamics
      record_step: callable accepting an integer argument - funtion to
      decide when to record a given interger step
      **params: all other parameters to pass to system's methods
    Output:
    -------
      array of type(system.get_state()) with shape (number_of_steps,)
    """
    trajectory = [s for (n, s) in n_steps(dynamics(system, **params),
                                          number_of_steps)
                  if record_step(n)]
    return array(trajectory)

# # Agents Society Model

# ## Theory

def Z(phi_w, phi_mu):
    """Evidence for the Bayesian Opinion-Trust learning model
    Input:
    ------
      phi_mu, phi_w: floats in [0,1] range"""
    return phi_mu + phi_w - 2*phi_mu*phi_w


def derivatives_lnZ(arg, const, z):
    """Compute the value of the first and second derivatives of ln(z) for a given `arg`, `const` and value of `z`
    Input:
    ------
      arg: float - the argument in the derivative
      const: float - the constant argument in the derivative
      z: float - the value of Z(arg, const)
    """
    dlnZ_darg =  (1-2*const)*G(arg)/z
    d2lnZ_darg = -dlnZ_darg*(dlnZ_darg + arg)
    return dlnZ_darg, d2lnZ_darg


def compute_deltas_w_and_C(w, C, x, sigma, phi_mu_l, z):
    gamma = sqrt(x @ (C @ x)) / norm(x)
    hs_g = w @ x * sigma / gamma 
    dlnZ_dhs_g, d2lnZ_dhs_g = derivatives_lnZ(hs_g, phi, z)
    dw = dlnZ_dhs_g*(C@x)*sigma/gamma
    dC = d2lnZ_dhs_g*C@outer(x, x)@C/(gamma*gamma)
    return [dw, dC]


def compute_deltas_mu_and_s2(mu, s2, phi_hs_g, z):
    lmbda = sqrt(1+s2)
    mu_l = mu/lmbda
    dlnZ_dmu_l, d2lnZ_dmu_l = derivatives_lnZ(mu_l, phi_w, z)
    dmu = dlnZ_dmu_l*s2/lmbda
    ds2 = d2lnZ_dmu_l*s2*s2/(1+s2)
    return [dmu, ds2]


def compute_deltas(w, C, mu, s2, x, sigma):
    """Compute the update values for w, C, mu and s2 given the
    example (x,sigma). The values correspond to the Bayesian
    Opinion-Trust model optimal updates.
    Input:
    ------
      w: array with shape (K,) - agent's opinion vector
      C: array with shape (K, K) [positive definite] - agent's opinion
      uncertainty
      mu: float - agent's distrust value
      s2: float [greater than 0] - agent's distrust uncertainty
    Output:
    -------
      list with the update values for w, C, mu and s2
    """
    K = w.shape[0]
    h = x@w
    gamma = sqrt(x@C@x)/norm(x)
    lmbda = sqrt(1+s2)
    hs_g = h*sigma/gamma
    mu_l = mu/lmbda
    phi_w = Phi(hs_g)
    phi_mu = Phi(mu_l)
    z = Z(phi_w, phi_mu)
    dlnZ_dmu_l, d2lnZ_dmu_l = derivatives_lnZ(mu_l, phi_w, z)
    dlnZ_dhs_g, d2lnZ_dhs_g = derivatives_lnZ(hs_g, phi_mu, z)
    delta_mu = dlnZ_dmu_l*s2/lmbda
    delta_s2 = d2lnZ_dmu_l*s2*s2/(1+s2)
    delta_w = dlnZ_dhs_g*sigma*C@x/gamma
    delta_C = d2lnZ_dhs_g*C@outer(x, x)@C/(gamma*gamma)
    return [delta_w, delta_C, delta_mu, delta_s2]


def learning_cost(w, C, mu, s2, x, sigma):
    """Compute the learning cost for w, C, mu and s2 given the
    example (x,sigma). The values correspond to the Bayesian
    Opinion-Trust model learning cost.
    Input:
    ------
      w: array with shape (K,) - agent's opinion vector
      C: array with shape (K, K) [positive definite] - agent's opinion
      uncertainty
      mu: float - agent's distrust value
      s2: float [greater than 0] - agent's distrust uncertainty
    Output:
    -------
      float value of learning cost
    """
    h = x@w
    gamma = sqrt(x@C@x)/norm(x)
    lmbda = sqrt(1+s2)
    phi_w = Phi(h*sigma/gamma)
    phi_mu = Phi(mu/lmbda)
    return -log(Z(phi_mu, phi_w))


# ## Model
class HEODAgentSociety(object):  
    def __init__(self, w0, C0, mu0, s20, *args, **kwargs):
        """Society of "Homo Entropicus" Opinion-Distrust agents.
        Input:
        ------
          w0: array with shape (N, K) - opinion vector with dimension K for
          N agents - representing the opinion weight vector;
          C0: array with shape (N, K, K) - opinion uncertainty for each 
          agent - representing the opinion uncertainty; 
          mu0: array with shape (N, N) - distrust array for each agent
          - representing the distrust proxy;
          s20: array with shape (N, N) - distrust uncertainty for each
          agent - reprsenting the distrust uncertainty.
        """
        self.w = w0.copy()
        self.N, self.K = w0.shape
        assert C0.shape == (self.N, self.K, self.K)
        self.C = C0.copy()
        assert mu0.shape == s20.shape == (self.N, self.N)
        self.mu = mu0.copy()
        self.s2 = s20.copy()
        self.initial_state = [w0.copy(), C0.copy(), mu0.copy(), s20.copy()]
        self._state_struct = [('w', 'f8', self.K),
                              ('C', 'f8', (self.K, self.K)),
                              ('mu', 'f8', self.N),
                              ('s2', 'f8', self.N)]
        self.interaction_counter = 0
        
    @classmethod
    def random(cls, N, K, C0, s20, *args, **kwargs):
        """Provides a simple and random set of initial values for BOTAgentSociety.
        Input:
        ------
        N: int - the number of agents
        K: int - the agents internal dimension
        
        Output:
        -------
        list with initial values for w, C, mu, and s2"""
        w0 = randn(N, K)
        w0 /= row_norm(w0)
        C0 = tile(C0*eye(K)/K, (N, 1, 1))
        mu0 = randn(N, N)
        fill_diagonal(mu0, -100)
        s20 = s20*ones((N, N))
        return cls(w0, C0, mu0, s20, *args, **kwargs)

    def agent_answer(self, i, x, *, real_epsilon=0.0, **params):
        sigma = sign(self.w[i]@x)*choice([-1, 1], p=[real_epsilon,
                                                     1-real_epsilon])
        return sigma

    def learning_amplitude(self, i, j, x, *args, constants=(), scales=(1.,1.,1.,1.), **params):
        wi, Ci, mui, s2i = self[i]
        sigma_j = self.agent_answer(j, x, **params)
        Dw, DC, Dmuij, Ds2ij = compute_deltas(wi, Ci, mui[j], s2i[j], x, sigma_j)
        sw, sC, smuij, ss2ij = scales
        if 'C' in constants:
            DC[:] = 0.0
        if 's2' in constants:
            Ds2ij = 0.0
        return [Dw/sw, DC/sC, (j, Dmuij/smuij, Ds2ij/ss2ij)]

    def move_agent(self, i, deltas, *args, constants=(), **params):
        self[i] = deltas
        if 'norm' in constants:
            self.normalize_agent_opinion(i, *args, **params)
        if 'bounds' in constants:
            self.normalize_agent_distrust(i, *args, **params)

    def interaction(self, i, j, x, *args, symmetric=False, **params):
        Deltas_i = self.learning_amplitude(i, j, x, *args, **params)
        if symmetric:
            Deltas_j = self.learning_amplitude(j, i, x, *args, **params)
            self.move_agent(j, Deltas_j, *args, **params)
        self.move_agent(i, Deltas_i, *args, **params)
        self.interaction_counter += 1

    def get_state(self, *args, **kwargs):
        state = zeros(self.N, self._state_struct)
        for n in state.dtype.names:
            state[n] = getattr(self, n)[:]
        return state

    def normalize_agent_opinion(self, i, *, opinion_norm=1., **params):
        self.w[i] *= opinion_norm/norm(self.w[i])

    def normalize_agent_distrust(self, i, *, distrust_bound=1., **params):
        clip(self.mu[i], -distrust_bound, distrust_bound, self.mu[i])

    def __getitem__(self, i):
        """Just a convinience to access agents properties"""
        values = self.w[i], self.C[i], self.mu[i], self.s2[i]
        return values

    def update(self, *args, **params):
        x = rand_sphere(self.K)
        i, j = choice(self, size=2, replace=False)
        self.interaction(0, 1, x, *args, **params)

    def __setitem__(self, i, deltas):
        """Conviniece to set agents properties values"""
        Dw, DC, (j, Dmu, Ds2) = deltas
        self.w[i] += Dw[:]
        self.C[i] += DC[:]
        self.mu[i, j] += Dmu
        self.s2[i, j] += Ds2

    def reset(self):
        """Convinience to restore the initial state"""
        w0, C0, mu0, s20 = self._initial_state
        self.w = w0.copy()
        self.C = C0.copy()
        self.mu = mu0.copy()
        self.s2 = s20.copy()
        self.interaction_counter = 0

## Teacher-Student Scenario
class TeacherStutdentScneario(HEODAgentSociety):
    def __init__(self, teacher, theta_0, C0, mu0, s20, *args, **kwargs):
        """Teacher/Student learning scenario
        Input:
        ------
          teacher: array - the teacher vector
          theta_0: flaot - initial angle between teacher and student
          C0 - 2d array - initial student opinion uncertainty
          mu0, s20: flaot, float > 0 - intial distrust and respective
        uncertainty for the student.
        """
        assert ndim(teacher) == 1
        student = random_at_angle(teacher, theta_0)
        w0 = vstack([student, teacher])
        K = teacher.shape[0]
        w0 /= row_norm(w0)
        super().__init__(w0, C0, mu0, s20, *args, **kwargs)

    def update(self, *args, **params):
        x = rand_sphere(self.K)
        self.interaction(0, 1, x, *args, **params)


## Restricted Universe of Discorse Scenario
class RUDHEODAgentSociety(HEODAgentSociety):
    """Restricted Universe of Discourse for the "Homo Entropicus" Opinion-Distrust
    Agent Society."""
    def pick_issue(self, *args, issue_list=None, **params):
        if issue_list is not None:
            k = choice(len(issue_list))
            x = issue_list[k]
        else:
            x = rand_sphere(K)
        return x

    def pick_agents(self, *args, **params):
        i, j = choice(self.N, size=2, replace=False)
        return i, j

    def update(self, *args, **params):
        x = self.pick_issue(*args, **params)
        i, j = self.pick_agents(*args, **params)
        self.interaction(i, j, x, *args, **params)


## Analysis Interface
class HEODAgentSocietyTrajectory(object):
    _observables = 'overlap distrust trust balance mean_balance frustration mean_frustration coherence coherence_mean'.split()

    def __init__(self, trajectory_array):
        self.w = trajectory_array['w']
        self.C = trajectory_array['C']
        self.mu = trajectory_array['mu']
        self.s2 = trajectory_array['s2']
        self.T, self.N, self.K = self.w.shape
        self._trajectory_array = trajectory_array

    @property
    def triangle_indices(self):
        if not hasattr(self, '_triangle_indices'):
            self._triangle_indices = triangle_indices(self.N)
        return self._triangle_indices.T

    @property
    def normalized_w(self):
        return self.w/norm(self.w, axis=2, keepdims=True)

    @property
    def overlap(self):
        """Compute the normalized overlap between agents"""
        u = self.normalized_w #self.w/norm(self.w, axis=2, keepdims=True)
        return einsum('...ij,...kj->...ik', u, u)

    @property
    def distrust(self):
        """Compute the distrust estimate"""
        return Phi(self.mu/sqrt(1+self.s2))

    @property
    def trust(self):
        """Compute the trust as 1 - 2*distrust"""
        return 1-2*self.distrust

    @property
    def balance_naive(self):
        t = self.trust
        return einsum('...ij,...jk,...ik->...ijk', t, t, t)

    @property
    def balance(self):
        I, J, K  = self.triangle_indices
        return self.balance_naive[:, I, J, K]

    @property
    def mean_balance(self):
        return self.balance.mean(axis=1)

    @property
    def frustration_naive(self):
        o = self.overlap
        return einsum('...ij,...jk,...ik->...ijk', o, o, o)

    @property
    def frustration(self):
        I, J, K = self.triangle_indices
        return self.frustration_naive[:, I, J, K]

    @property
    def mean_frustration(self):
        return self.frustration.mean(axis=1)

    @property
    def coherence_naive(self):
        c = (self.overlap * self.trust)
        return einsum('...ij,...jk,...ik->...ijk', c, c, c)

    @property
    def coherence(self):
        I, J, K = self.triangle_indices
        return self.coherence_naive[:, I, J, K]

    @property
    def coherence_mean(self):
        return self.coherence.mean(axis=1)

    @property
    def observables(self):
        """Creates the dict of observables"""
        return {n: getattr(self, n) for n in self._observables}