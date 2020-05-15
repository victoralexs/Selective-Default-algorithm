import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
import random

from interpolation import interp
from quantecon.optimize import brentq
from numba import jit, jitclass, int64, float64




# First, defining data information for jitclass.

selective_data = [
    ('b_d', float64[:]), ('b_f',float64[:]), ('P', float64[:, :]), ('y', float64[:]),
    ('β', float64), ('σ', float64), ('r', float64),
    ('ρ', float64), ('η', float64), ('θ_d', float64), ('θ_f', float64),
    ('dom_y', float64[:]),('ext_y', float64[:]),('tot_y', float64[:]),('g', float64[:]), ('μ', float64[:])
]
# Define utility function: a CRRA

    @jit(nopython=True)
    def u(c, σ):
        return c ** (1 - σ) / (1 - σ)

# Now, we create a class that states the initial parameters and creates the Bellman Equations as determine the saving policy given prices and value functions:

@jitclass(selective_data)
class Selective_Economy:
    """

    Selective Economy is a small open economy whose government invest in both domestic and foreign assets.
    Foreign assets is borrowed from international investors, while domestic bonds are from households.
    The government objective is to smooth consumption of domestic households.
    Domestic households receive a stochastic path of income, which will differ depending on default state.

    Parameters
    ----------
    b_d : vector(float64)
        A grid for domestic bond holdings
    b_f : vector(float64)
        A grid for foreign bond holdings
    P : matrix(float64)
        The transition matrix for a country's output
    y : vector(float64)
        The possible output states
    β : float
        Time discounting parameter
    σ : float
        Risk-aversion parameter
    r : float
        int lending rate
    ρ : float
        Persistence in the income process
    η : float
        Standard deviation of the income process
    θ_d : float
        Probability of re-entering domestic financial markets in each period
    θ_f : float
        Probability of re-entering foreign financial markets in each period
    """

    def __init__(
            self, b_d, b_f, P, y,
            β=0.825, σ=2.0, r=0.017,
            ρ=0.897, η=0.053, θ_d=0.5, θ_f = 0.5
    ):

        # Save parameters
        self.b_d, self.b_f, self.P, self.y = b_d, b_f, P, y
        self.β, self.σ, self.r, = β, σ, r
        self.ρ, self.η, self.θ_d, self.θ_f = ρ, η, θ_d, θ_f

        # Compute the mean outputs
        self.dom_y,self.ext_y,self.tot_y = np.minimum(0.955 * np.mean(y), y),np.minimum(0.905 * np.mean(y), y),np.minimum(0.955 * 0.905 * np.mean(y), y)

    def u_prime(self, c):
        "Derivative of u"
        return c**(-self.σ)

    def u_prime_inv(self, c):
        "Inverse of u'"
        return c**(-1 / self.σ)

    def bellman_ext_default(self, iy, EVfd, EV):
        """
        The RHS of the Bellman equation when the country is in a external
        defaulted state on their debt
        """
        # Unpack certain parameters for simplification
        β, σ, θ_f = self.β, self.σ, self.θ_f

        # Compute continuation value
        zero_ind_ext = len(self.b_f) // 2
        cont_value_ext = θ_f * EV[iy, zero_ind_ext] + (1 - θ_f) * EVfd[iy]

        return u(self.ext_y[iy], σ) + β*cont_value_ext

    def bellman_dom_default(self, iy, EVdd, EV):
        """
        The RHS of the Bellman equation when the country is in a domestic
        defaulted state on their debt
        """
        # Unpack certain parameters for simplification
        β, σ, θ_d = self.β, self.σ, self.θ_d

        # Compute continuation value
        zero_ind_dom = len(self.b_d) // 2
        cont_value_dom = θ_d * EV[iy, zero_ind_dom] + (1 - θ_d) * EVdd[iy]

        return u(self.dom_y[iy], σ) + β*cont_value_dom

    def bellman_tot_default(self, iy, EVtd, EVdd, EVfd, EV):
        """
        The RHS of the Bellman equation when the country is in a non-selective
        defaulted state on their debt
        """
        # Unpack certain parameters for simplification
        β, σ, θ_f, θ_f = self.β, self.σ, self.θ_d, self.θ_f

        # Compute continuation value
        zero_ind_tot = len(self.b_d + self.b_f) // 2
        cont_value_tot = θ_d * θ_f * EV[iy, zero_ind_tot] + θ_f * (1 - θ_d) * EVdd[iy] + (1 - θ_f) * θ_d * EVfd[iy] + (1 - θ_f) * (1 - θ_d) * Evtd[iy]

        return u(self.tot_y[iy], σ) + β*cont_value_tot

    def bellman_nondefault(self, iy, ib_d, ib_f, q_d_r, q_f_r, EV, ib_d_tp1_star=-1, ib_f_tp1_star=-1):
        """
        The RHS of the Bellman equation when the country is not in a
        defaulted state on their debt
        """
        # Unpack certain parameters for simplification
        β, σ, θ_d, θ_f  = self.β, self.σ, self.θ_d, self.θ_f
        b_d, b_f, y = self.b_d, self.b_f, self.y

        # Compute the RHS of Bellman equation
        if ib_d_tp1_star < 0 and ib_f_tp1_star < 0 :
            ib_d_tp1_star = self.compute_savings_policy(iy, ib_d, q_d_r, EV)
            ib_f_tp1_star = self.compute_savings_policy(iy, ib_f, q_f_r, EV)
        c = max(y[iy] - q_d_r[iy, ib_d_tp1_star]*b_d[ib_d_tp1_star] + b_d[ib_d], 1e-14)

        return u(c, σ) + β*EV[iy, ib_d_tp1_star, ib_f_tp1_star]

    def compute_savings_policy(self, iy, ib_d, ib_f, q_d_r, EV):
        """
        Finds the debt/savings that maximizes the value function
        for a particular state given prices and a value function
        """
        # Unpack certain parameters for simplification
        β, σ, θ_d, θ_f = self.β, self.σ, self.θ_d, self.θ_f
        b_d, b_f, y = self.b_d, self.b_f, self.y

        # Compute the RHS of Bellman equation
        current_max = -1e14
        ib_d_tp1_star = 0
        ib_f_tp1_star = 0
        for ib_d_tp1, b_d_tp1, ib_f_tp1, b_f_tp1 in enumerate(b_d + b_f):
            c = max(y[iy] - q_d_r[iy, ib_d_tp1] * b_d[ib_d_tp1] + b_d[ib_d], 1e-14)
            j = u(c, σ) + β * EV[iy, ib_d_tp1, ib_f_tp1]

            if j > current_max:
                ib_d_tp1_star = ib_d_tp1
                ib_f_tp1_star = ib_f_tp1
                current_max = j

        return ib_d_tp1_star, ib_f_tp1_star


# Now we define a function solve that solves the solution of the model with maximum of 100000 iterations, with "model" being the instance to be defined later:

@jit(nopython=True)
def solve(model, tol=1e-8, maxiter=10_000):
    """
    Given an Selective_Economy type, this function computes the optimal
    policy and value functions
    """
    # Unpack certain parameters for simplification
    β, σ, r, θ_d, θ_f = model.β, model.σ, model.r, model.θ_d, model.θ_f
    b_d = np.ascontiguousarray(model.b_d)
    b_f = np.ascontiguousarray(model.b_f)
    P, y = np.ascontiguousarray(model.P), np.ascontiguousarray(model.y)
    nb_d, nb_f, ny = b_d.size, b_f.size, y.size

    # Allocate space
    ib_d_star = np.zeros((ny, nb_d, nb_f), int64)
    ib_f_star = np.zeros((ny, ,nb_d, nb_f), int64)
    dom_default_prob = np.zeros((ny, nb_d))
    ext_default_prob = np.zeros((ny, nb_f))
    dom_default_states = np.zeros((ny, nb_d))
    ext_default_states = np.zeros((ny, nb_f))
    tot_default_states = np.zeros((ny, nb_d, nb_f))
    q_f_r = np.ones((ny, nb_f)) * 0.95
    q_f_dd = np.ones((ny, nb_f)) * 0.95
    q_d_r = np.ones((ny, nb_d)) * 0.95
    q_d_fd = np.ones((ny, nb_d)) * 0.95
    Vfd = np.zeros(ny, nb_d)
    Vdd = np.zeros(ny, nb_f)
    Vtd = np.zeros(ny)
    Vc, V, Vupd = np.zeros((ny, nb_d,nb_f)), np.zeros((ny, nb_d,nb_f)), np.zeros((ny, nb_d, nb_f))

    it = 0
    dist = 10.0
    while (it < maxiter) and (dist > tol):

        # Compute expectations used for this iteration
        EV = P@V
        EVfd = P@Vfd
        EVdd = P@Vdd
        EVtd = P@Vtd

        for iy in range(ny):
            # Update value function for default state
            Vfd[iy] = model.bellman_ext_default(iy, EVfd, EV)
            Vdd[iy] = model.bellman_dom_default(iy, EVdd, EV)
            Vtd[iy] = model.bellman_tot_default(iy, EVtd, EV)

            # Update value function for non-default state
            for ib_d in range(nb_d):
                ib_d_star[iy, ib_d, ib_f] = model.compute_savings_policy(iy, ib_d, q_d_r, EV)

            for ib_f in range(nb_f):
                ib_f_star[iy, ib_d, ib_f] = model.compute_savings_policy(iy, ib_f, q_f_r, EV)

            for ib_d, ib_f in range(nb_d + nb_f):
                Vc[iy, ib_d, ib_f] = model.bellman_nondefault(iy, ib_d, ib_f, q_d_r, q_f_r, EV, ib_d_star[iy, ib_d, ib_f], ib_f_star[iy, ib_d, ib_f])

        # Once value functions are updated, can combine them to get
        # the full value function
        Vfd_compat = np.reshape(np.repeat(Vfd, nb_f), (ny, nb_f))
        Vdd_compat = np.reshape(np.repeat(Vdd, nb_d), (ny, nb_d))
        Vtd_compat = np.reshape(np.repeat(Vtd, nb_d, nb_f), (ny, nb_d, nb_f))
        Vupd[:, :] = np.maximum(Vc, Vfd_compat, Vdd_compat, Vtd_compat)

        # Can also compute default states and update prices
        ext_default_states[:, :] = 1.0 * (Vfd_compat > max(Vc, Vdd, Vtd))
        dom_default_states[:, :] = 1.0 * (Vdd_compat > max(Vc, Vfd, Vtd))
        tot_default_states[:, :] = 1.0 * (Vtd_compat > max(Vc, Vfd, Vdd))
        ext_default_prob[:, :] = P @ ext_default_states
        dom_default_prob[:, :] = P @ dom_default_states

        q_f_r[:, :] = (1 - ext_default_prob) / (1 + r)
        q_f_dd[:, :] = θ_f * (1 - ext_default_prob) / (1 + r) + (1 - θ_f) (1 - ext_default_prob) / (1 + r)
        q_d_r[:, :] = β * (1 - ext_default_prob) * (1 - dom_default_prob) * u_prime(y[iy]) + ext_default_prob * (1 - dom_default_prob) * u_prime(ext_y[iy]) / u_prime(y[iy])
        q_d_dd[:, :] = β * θ_f * (1 - dom_default_prob) * u_prime(y[iy]) + (1 - θ_f) * (1 - dom_default_prob) * u_prime(ext_y[iy]) / u_prime(ext_y[iy])

        # Check tolerance etc...
        dist = np.max(np.abs(Vupd - V))
        V[:, :] = Vupd[:, :]
        it += 1

    return V, Vc, Vfd, Vdd, Vtd, ib_d_star, ib_f_star, ext_default_prob, dom_default_prob, ext_default_states, dom_default_states, tot_default_states, q_f_r, q_f_dd, q_d_r, q_d_fd

# Then, writing a function that allow us to simulate the economy once we have the policy functions:

def simulate(model, T, ext_default_states, dom_default_states, tot_default_states, ib_d_star, ib_f_star, q_d_r, q_d_fd, q_f_r, q_f_dd, y_init=None, b_d_init=None, b_f_init=None):
    """
    Simulates the Selective Default model of sovereign debt

    Parameters
    ----------
    model: Selective_Economy
        An instance of the Selective Default model with the corresponding parameters
    T: integer
        The number of periods that the model should be simulated
    ext_default_states: array(float64, 2)
        A matrix of 0s and 1s that denotes whether the country was in
        external default on their debt in that period (ext_default = 1)
    dom_default_states: array(float64, 2)
        A matrix of 0s and 1s that denotes whether the country was in
        domestic default on their debt in that period (dom_default = 1)
    tot_default_states: array(float64, 2)
        A matrix of 0s and 1s that denotes whether the country was in
        total default on their debt in that period (ext_default & dom_default = 1)
    ib_f_star: array(float64, 2)
        A matrix which specifies the external debt/savings level that a country holds
        during a given state
    ib_d_star: array(float64, 2)
        A matrix which specifies the domestic debt/savings level that a country holds
        during a given state
    q_f_r: array(float64, 2)
        A matrix that specifies the price, in case of total repayment, at which a country can borrow/save externally
        for a given state
    q_f_dd: array(float64, 2)
        A matrix that specifies the price, in case of domestic default, at which a country can borrow/save externally
        for a given state
    q_d_r: array(float64, 2)
        A matrix that specifies the price, in case of total repayment, at which a country can borrow/save domestically
        for a given state
    q_d_fd: array(float64, 2)
        A matrix that specifies the price, in case of foreign default, at which a country can borrow/save domestically
        for a given state
    y_init: integer
        Specifies which state the income process should start in
    b_f_init: integer
        Specifies which state the external debt/savings state should start
    b_d_init: integer
        Specifies which state the domestic debt/savings state should start

    Returns
    -------
    y_sim: array(float64, 1)
        A simulation of the country's income
    b_d_sim: array(float64, 1)
        A simulation of the country's domestic debt/savings
    b_f_sim: array(float64, 1)
        A simulation of the country's foreign debt/savings
    q_f_r_sim: array(float64, 1)
        A simulation of the external price, in case of total repayment, required to have an extra unit of
        consumption in the following period
    q_f_dd_sim: array(float64, 1)
        A simulation of the external price, in case of domestic default, required to have an extra unit of
        consumption in the following period
    q_d_r_sim: array(float64, 1)
        A simulation of the domestic price, in case of total repayment, required to have an extra unit of
        consumption in the following period
   q_d_fd_sim: array(float64, 1)
        A simulation of the domestic price, in case of foreign default, required to have an extra unit of
        consumption in the following period
    ext_default_sim: array(bool, 1)
        A simulation of whether the country was in external default or not
    dom_default_sim: array(bool, 1)
        A simulation of whether the country was in domestic default or not
    tot_default_sim: array(bool, 1)
        A simulation of whether the country was in total default or not
    """
    # Find index i such that Bgrid[i] is approximately 0
    zero_b_f_index = np.searchsorted(model.b_f, 0.0)
    zero_b_d_index = np.searchsorted(model.b_d, 0.0)

    # Set initial conditions
    ext_in_default = False
    dom_in_default = False
    tot_in_default = False

    ext_max_y_default = 0.905 * np.mean(model.y)
    dom_max_y_default = 0.955 * np.mean(model.y)
    tot_max_y_default = 0.905 * 0.955 * np.mean(model.y)

    if y_init == None:
        y_init = np.searchsorted(model.y, model.y.mean())
    if b_d_init == None:
        b_d_init = zero_b_d_index
    if b_f_init == None:
        b_f_init = zero_b_f_index

    # Create Markov chain and simulate income process
    mc = se.MarkovChain(model.P, model.y)
    y_sim_indices = mc.simulate_indices(T+1, init=y_init)

    # Allocate memory for remaining outputs
    b_di = b_d_init
    b_fi = b_f_init
    b_d_sim = np.empty(T)
    b_f_sim = np.empty(T)
    y_sim = np.empty(T)
    q_d_r_sim = np.empty(T)
    q_d_fd_sim = np.empty(T)
    q_f_r_sim = np.empty(T)
    q_f_dd_sim = np.empty(T)
    dom_default_sim = np.empty(T, dtype=bool)
    ext_default_sim = np.empty(T, dtype=bool)
    tot_default_sim = np.empty(T, dtype=bool)

    # Perform simulation
    for t in range(T):
        yi = y_sim_indices[t]

        # Fill y/B for today
        if not ext_in_default and dom_in_default:
            y_sim[t] = model.y[yi]

        if ext_in_default and not dom_in_default:
            y_sim[t] = np.minimum(model.y[yi], ext_max_y_default)

        if dom_in_default and not ext_in_default:
            y_sim[t] = np.minimum(model.y[yi], dom_max_y_default)

        else:
            y_sim[t] = np.minimum(model.y[yi], tot_max_y_default)

        b_d_sim[t] = model.b_d[b_di]
        b_f_sim[t] = model.b_f[b_fi]

        ext_default_sim[t] = ext_in_default
        dom_default_sim[t] = dom_in_default
        tot_default_sim[t] = tot_in_default

        # Check whether in default and branch depending on that state
        if not ext_in_default and not dom_in_default:
            if ext_default_states[yi, b_di] > 1e-4 and dom_default_states[yi, b_fi] > 1e-4:
                tot_in_default=True
                b_di_next = zero_b_f_index
                b_fi_next = zero_b_d_index

            if ext_default_states[yi, b_fi] > 1e-4 and not dom_default_states[yi, b_fi] > 1e-4:
                ext_in_default=True
                b_fi_next = zero_b_f_index
                b_di_next = ib_d_star[yi,b_di]

            if dom_default_states[yi, b_di] > 1e-4 and not ext_default_states[yi, b_di] > 1e-4:
                dom_in_default=True
                b_di_next = zero_b_d_index
                b_fi_next = ib_f_star[yi,b_fi]

            else:
                b_fi_next = ib_f_star[yi, b_fi]
                b_di_next = ib_d_star[yi, b_di]
        else:
            b_fi_next = zero_b_f_index
            b_di_next = zero_b_d_index
            if np.random.rand() < model.θ_f and np.random.rand() < model.θ_d:
                tot_in_default=False

            if np.random.rand() < model.θ_f and not np.random.rand() < model.θ_d:
                dom_in_default=False

            if np.random.rand() < model.θ_d and not np.random.rand() < model.θ_f:
                ext_in_default=False

        # Fill in states
        q_f_r_sim[t] = q_f_r[yi, b_fi_next]
        q_f_dd_sim[t] = q_f_dd[yi,b_fi_next]
        q_d_r_sim[t] = q_d_r[yi, b_di_next]
        q_d_fd_sim[t] = q_d_fd_sim[yi,b_di_next]
        b_fi = b_fi_next
        b_di = b_di_next

    return y_sim, b_d_sim, b_f_sim, q_d_r_sim, q_d_fd_sim, q_f_r_sim, q_f_dd_sim, ext_default_sim, dom_default_sim, tot_default_sim


# Now, let's see some results. Firstly, let us compute the value function, policy and equilibrium prices using Arellano (2008) parameters:

β, σ, r = 0.825, 2.0, 0.017
ρ, η, θ_f, θ_d  = 0.897, 0.053, 0.5, 0.5
ny = 21
nb_f = 251
nb_d = 251
b_f_grid = np.linspace(-0.45, 0.45, nb_f)
b_d_grid = np.linspace(-0.45, 0.45, nb_d)
mc = qe.markov.tauchen(ρ, η, 0, 3, ny) # Tauchen is the method to do the output process, by Tauchen's quadrature method (https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/markov/approximation.py)
ygrid, P = np.exp(mc.state_values), mc.P

se = Selective_Economy(
    b_f_grid, b_d_grid, P, ygrid, β=β, σ=σ, r=r, ρ=ρ, η=η, θ_f=θ_f, θ_d=θ_d
)

# Now, solving it:

V, Vc, Vfd, Vdd, Vtd, ib_d_star, ib_f_star, ext_default_prob, dom_default_prob, ext_default_states, dom_default_states, tot_default_states, q_f_r = solve(se), q_f_dd = solve(se), q_d_r = solve(se), q_d_fd = solve(se)