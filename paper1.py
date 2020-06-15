import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
import random

from interpolation import interp
from quantecon.optimize import brentq
from numba import jit, jitclass, int64, float64

# The model first deals with an economy that spend an amount "g" in government spending and collect lump sum taxes "T".
    # Later, we will add money supply and money holdings. After we be able to run it.

# It also decides on borrowing: b_d and b_f, for domestic and foreign investors, respectively.

########## Glossary

# iy is y', the next period y.

# ib_d and ib_f are b'_d and b'_f, the debt for the next period.

# b_d and b_f are the outstanding debt.

# ib_d_star and ib_f_star are the final optimal policy functions for savings, in each jurisdiction.
    # ib_d_tp1_star and ib_f_tp1_star are the optimal policy functions for savings when assessing the functions.

# EQUILIBRIUNM:

# An equilibrium is
                    # 1) 4 sets of default schedules:

                        # delta_f_r(iy, ib_d, ib_f)
                        # delta_d_r(iy, ib_d, ib_f)
                        # delta_d_fd(iy, ib_d)
                        # delta_f_dd(iy, ib_f)

                    # 2) 4 sets of debt discount prices:

                        # q_d_fd(iy, ib_d, b_d)
                        # q_f_dd(iy, ib_f)
                        # q_d_r(iy, ib_d, ib_f, b_d, b_f) -> To avoid dimensionality problem, use policy functions from previous iteration:
                                                           # iib_d(iy, ib_d, ib_f) -> Think the best way to do it.
                        # q_f_r(iy, ib_d, ib_f)

                    # 3) 4 sets of government debt issuance policies:

                        # ib_d_rep_star(iy, ib_d, ib_f)
                        # ib_f_rep_star(iy, ib_d, ib_f)
                        # ib_d_fd_star(iy, ib_d)
                        # ib_f_dd_star(iy, ib_f)

                    # 4) 4 sets of consumption:

                        # c_r
                        # c_fd
                        # c_dd
                        # c_td

# Conditional on exogenous process {y} s.t.

    # a) Taking prices {q_d_r, q_f_r, q_d_fd, q_f_dd} and government debt issues {ib_d_rep_star, ib_f_rep_star, ib_d_fd_star, ib_f_dd_star} as given,
    #    households consumption {c_r, c_dd, c_fd, c_td} satisfy households' budget constraint

    # b) Taking prices {q_d_r, q_f_r, q_d_fd, q_f_dd} as given, gov. default schedules
                        # delta_f_r(iy, ib_d, ib_f)
                        # delta_d_r(iy, ib_d, ib_f)
                        # delta_d_fd(iy, ib_d)
                        # delta_f_dd(iy, ib_f)

    #   and debt issuance policies {ib_d_rep_star, ib_f_rep_star, ib_d_fd_star, ib_f_dd_star} solve government's budget constraint

    # c) Given foreign default schedules {delta_f_r, delta_f_dd}, foreign debt prices {q_f_r, q_f_dd} satisfy foreign investors expected zero payoffs.

    # d) Given domestic default schedules {delta_d_r, delta_d_fd}, households' consumptions {c_r, c_fd} and future expected default schedules,
    #    debt issuances and households' consumptions, domestic debt prices {q_d_r, q_f_fd} satisfy households' FOC.

######################

# ALGORITHM:

# Step 1: Guesses
            # 1.1. Guess value functions Vfd, Vdd, Vtd, V (V being the decision value)
            # 1.2. Guess price functions q_d_fd, q_f_dd, q_d_r, q_f_r

# Step 2: At each truple (y, b_d, b_f):

            # 2.1. Update Vfd, Vdd, Vtd (default value functions)
            # 2.2. Update V0 as Vc (continuation value)

# Step 3:
            # 3.1. Update the value of decision V(y, b_d, b_f) # inside loop
            # 3.2. Update default rules by comparing value functions V, Vfd, Vdd, Vtd
            # 3.3. Update implied ex ante default probabilities delta_f_r, delta_d_r, delta_d_fd, delta_f_dd
            # 3.4. Update price functions q_d_fd, q_f_dd, q_d_r, q_f_r # outside loop

# Step 4: Check for convergence. If converged, stop - if not, get back to step 2.

# Output process discretized using Tauchen's quadrature method.



# First, defining data information for jitclass.

selective_data = [
    ('b_d', float64[:]), ('b_f',float64[:]), ('P', float64[:, :, :]), ('y', float64[:]),
    ('g', float64), ('T', float64[:]),
    ('β', float64), ('σ', float64), ('r', float64),
    ('ρ', float64), ('η', float64), ('θ_d', float64), ('θ_f', float64),
    ('dom_y', float64[:]), ('ext_y', float64[:]), ('tot_y', float64[:])
]

# Define utility function: a CRRA and its derivative.

    @jit(nopython=True)
    def u(c, σ):
        return c ** (1 - σ) / (1 - σ)

    @jit(nopython=True)
    def u_prime(c, σ):
        return c ** (-σ)

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
    T : vector(float64)
        Lump-sum tax
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
            self, b_d, b_f, P, y, g, T,
            β=0.825, σ=2.0, r=0.017,
            ρ=0.897, η=0.053, θ_d=0.5, θ_f = 0.5
    ):

        # Save parameters
        self.b_d, self.b_f, self.P, self.y, self.g, self.T = b_d, b_f, P, y, g, T
        self.β, self.σ, self.r, = β, σ, r
        self.ρ, self.η, self.θ_d, self.θ_f = ρ, η, θ_d, θ_f

        # Compute the mean outputs
        self.dom_y = np.minimum(0.955 * np.mean(y), y)
        self.ext_y = np.minimum(0.905 * np.mean(y), y)
        self.tot_y = np.minimum(0.955 * 0.905 * np.mean(y), y)

    def bellman_ext_default(self, iy, ib_d, EVfd, q_d_fd, EV, ib_d_tp1_fd_star=-1):  # Since domestic debt is still on, we need to maximize it.
        """
        The RHS of the Bellman equation when the country is in a external
        defaulted state on their debt
        """
        # Unpack certain parameters for simplification
        β, σ, θ_f = self.β, self.σ, self.θ_f
        b_d, b_f, ext_y, g, T = self.b_d, self.b_f, self.ext_y, self.g, self.T

        # Compute continuation value
        zero_ind_ext = len(self.b_f) // 2                                               # Should it be // 3?

        if ib_d_tp1_fd_star < 0:
            ib_d_tp1_fd_star = self.compute_dom_savings_policy_ext_default(iy, ib_d, q_d_fd, EV) # For now, we are taking q_d_fd as given. Later we will compute it.

        T = g + b_d[ib_d] - q_d_fd[iy, ib_d_tp1_fd_star, b_d] * b_d[ib_d_tp1_fd_star]

        c = max(ext_y[iy] - T - q_d_fd[iy, ib_d_tp1_fd_star, b_d] * b_d[ib_d_tp1_fd_star] + b_d[ib_d], 1e-14) # Since q_d_fd(b_d,b_d', y)

        cont_value_ext = θ_f * EV[iy, ib_d_tp1_fd_star, zero_ind_ext] + (1 - θ_f) * EVfd[iy, ib_d_tp1_fd_star]  # It keeps b'_d since is only a foreign default.

        return u(c, σ) + β*cont_value_ext

    def compute_dom_savings_policy_ext_default(self, iy, ib_d, q_d_fd, EV):
        """
        Finds the DOMESTIC debt/savings that maximizes the value function
        for a particular state given prices and a value function
        in case of a FOREIGN DEFAULT.
        """

        # Unpack certain parameters for simplification
        β, σ, θ_d, θ_f = self.β, self.σ, self.θ_d, self.θ_f
        b_d, ext_y, g, T = self.b_d, self.ext_y, self.g, self.T

        # Compute the RHS pf Bellman equation
        current_max = -1e24
        ib_d_tp1_fd_star = 0 # First guess for b'_f_fd

        for ib_d_tp1, b_d_tp1 in enumerate(b_d):
            T = g + b_d[ib_d] - q_d_fd[iy, ib_d_tp1, b_d_tp1] * b_d[ib_d_tp1]
            c = max(ext_y[iy] - T - q_d_fd[iy, ib_d_tp1, b_d_tp1]*b_d[ib_d_tp1] + b_d[ib_d], 1e-14)
            j = u(c, σ) + β*EV[iy, ib_d_tp1]

            if j > current_max:
                ib_d_tp1_fd_star = ib_d_tp1
                current_max = j

        return ib_d_tp1_fd_star

    def bellman_dom_default(self, iy, ib_f, q_f_dd, EVdd, EV, ib_f_tp1_dd_star=-1):  # Since external debt is still on, we need to maximize it.
        """
        The RHS of the Bellman equation when the country is in a domestic
        defaulted state on their debt
        """
        # Unpack certain parameters for simplification
        β, σ, θ_d = self.β, self.σ, self.θ_d
        b_f, dom_y, g, T = self.b_f, self.dom_y, self.g, self.T

        # Compute continuation value
        zero_ind_dom = len(self.b_d) // 2                                               # Should it be // 3?

        if ib_f_tp1_dd_star < 0:
            ib_f_tp1_dd_star = self.compute_ext_savings_policy_dom_default(iy, ib_f, q_f_dd, EV) # For now, we are taking q_f_dd as given. Later we will compute it.

        T = g + b_f[ib_f] - q_f_dd[iy, ib_f_dd_star] * b_f[ib_f_dd_star]

        c = max(dom_y[iy] - T, -1e-14)

        cont_value_dom = θ_d * EV[iy, zero_ind_dom, ib_f_tp1_dd_star] + (1 - θ_d) * EVdd[iy, ib_f_tp1_dd_star]

        return u(c, σ) + β*cont_value_dom

    def compute_ext_savings_policy_dom_default(self, iy, ib_f, q_f_dd, EV):
        """
        Finds the EXTERNAL debt/savings that maximizes the value function
        for a particular state given prices and a value function
        in case of a EXTERNAL DEFAULT.
        """
        # Unpack certain parameters for simplification
        β, σ, θ_d, θ_f = self.β, self.σ, self.θ_d, self.θ_f
        b_f, dom_y, g, T = self.b_f, self.dom_y, self.g, self.T

        # Compute the RHS of Bellman equation
        current_max = -1e14
        ib_f_tp1_dd_star = 0                # First guess for b_f'_dd
        for ib_f_tp1, b_f_tp1 in enumerate(b_f):

            T = g + b_f[ib_f] - q_f_dd[iy, ib_f_tp1] * b_f[ib_f_tp1]

            c = max(dom_y[iy] - T, -1e-14)

            j = u(c,σ) + β*EV[iy, ib_f_tp1]

            if j > current_max:
                ib_f_tp1_dd_star = ib_f_tp1
                current_max = j

        return ib_f_tp1_dd_star


    def bellman_tot_default(self, iy, EVtd, EVdd, EVfd, EV):
        """
        The RHS of the Bellman equation when the country is in a non-selective
        defaulted state on their debt
        """
        # Unpack certain parameters for simplification
        β, σ, θ_f, θ_f = self.β, self.σ, self.θ_d, self.θ_f
        T, g, tot_y = self.T, self.g, self.tot_y

        # Compute continuation value
        zero_ind_ext = len(self.b_f) // 2
        zero_ind_dom = len(self.b_d) // 2
        cont_value_tot = θ_d * θ_f * EV[iy, zero_ind_dom, zero_ind_ext] + θ_f * (1 - θ_d) * EVdd[iy, zero_ind_ext] + (1 - θ_f) * θ_d * EVfd[iy, zero_ind_dom] + (1 - θ_f) * (1 - θ_d) * EVtd[iy]

        T = g

        return u(self.tot_y[iy] - T, σ) + β*cont_value_tot

    def bellman_nondefault(self, iy, ib_d, ib_f, q_d_r, q_f_r, EV, ib_d_tp1_rep_star=-1, ib_f_tp1_rep_star=-1):
        """
        The RHS of the Bellman equation when the country is not in a
        defaulted state on their debt
        """
        # Unpack certain parameters for simplification
        β, σ, θ_d, θ_f  = self.β, self.σ, self.θ_d, self.θ_f
        b_d, b_f, y, g, T = self.b_d, self.b_f, self.y, self.g, self.T

        # Compute the RHS of Bellman equation
        if ib_d_tp1_rep_star < 0 and ib_f_tp1_rep_star < 0:
            ib_d_tp1_rep_star = self.compute_dom_savings_policy_rep(iy, ib_d, ib_f, q_d_r, q_f_r, EV)
            ib_f_tp1_rep_star = self.compute_ext_savings_policy_rep(iy, ib_d, ib_f, q_d_r, q_f_r, EV)

        T = g + b_d[ib_d] + b_f[ib_f] - q_d_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star, b_d, b_f] * b_d[ib_d_tp1_rep_star] - q_f_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star] * b_f[ib_f_tp1_rep_star]
        c = max(y[iy] - T - q_d_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star, b_d, b_f]*b_d[ib_d_tp1_rep_star] + b_d[ib_d, ib_f], 1e-14)

        return u(c, σ) + β*EV[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star]

    def compute_dom_savings_policy_rep(self, iy, ib_d, ib_f, q_d_r, q_f_r, EV):
        """
        Finds the DOMESTIC debt/savings that maximizes the value function
        for a particular state given prices and a value function
        in case of a REPAYMENT.
        """

        # Unpack certain parameters for simplification
        β, σ, θ_d, θ_f  = self.β, self.σ, self.θ_d, self.θ_f
        b_d, b_f, y, g, T = self.b_d, self.b_f, self.y, self.g, self.T

        # Compute the RHS of Bellman Equation
        current_max = -1e24
        ib_d_tp1_rep_star = 0 # First guess for b'_d_r
        for ib_d_tp1, b_d_tp1 in enumerate(b_d):

            T = g + b_d[ib_d] + b_f[ib_f] - q_d_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star, b_d, b_f] * b_d[ib_d_tp1_rep_star] - q_f_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star] * b_f[ib_f_tp1_rep_star]

            c = max(y[iy] - T + b_d[ib_d] + b_f[ib_f] - q_d_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star, b_d, b_f] * b_d[ib_d_tp1_rep_star] - q_f_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star] * b_f[ib_f_tp1_rep_star], 1e-14)
            j = u(c, σ) + β*EV[iy, ib_d_tp1, ib_f]

            if j > current_max:
                ib_d_tp1_rep_star = ib_d_tp1
                current_max = j

        return ib_d_tp1_rep_star

    def compute_ext_savings_policy_rep(self, iy, ib_d, ib_f, q_d_r, q_f_r, EV):
        """
        Finds the EXTERNAL debt/savings that maximizes the value function
        for a particular state given prices and a value function
        in case of a REPAYMENT.
        """

        # Unpack certain parameters for simplification
        β, σ, θ_d, θ_f  = self.β, self.σ, self.θ_d, self.θ_f
        b_d, b_f, y, g, T = self.b_d, self.b_f, self.y, self.g, self.T

        # Compute the RHS of Bellman Equation
        current_max = -1e24
        ib_f_tp1_rep_star = 0 # First guess for b'_f_r
        for ib_f_tp1, b_f_tp1 in enumerate(b_f):

            T = g + b_d[ib_d] + b_f[ib_f] - q_d_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star, b_d, b_f] * b_d[ib_d_tp1_rep_star] - q_f_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star] * b_f[ib_f_tp1_rep_star]

            c = max(y[iy] - T + b_d[ib_d] + b_f[ib_f] - q_d_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star, b_d, b_f] * b_d[ib_d_tp1_rep_star] - q_f_r[iy, ib_d_tp1_rep_star, ib_f_tp1_rep_star] * b_f[ib_f_tp1_rep_star], 1e-14)
            j = u(c, σ) + β*EV[iy, ib_d, ib_f_tp1]

            if j > current_max:
                ib_f_tp1_rep_star = ib_f_tp1
                current_max = j

        return ib_f_tp1_rep_star

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
    P, y, g, T = np.ascontiguousarray(model.P), np.ascontiguousarray(model.y), np.ascontiguousarray(model.g), np.ascontiguousarray(model.T)
    nb_d, nb_f, ny = b_d.size, b_f.size, y.size # Grids.

    # Allocate space
    ib_d_fd_star = np.zeros((ny, nb_d, nb_f), int64)
    ib_f_dd_star = np.zeros((ny, nb_d, nb_f), int64)
    ib_d_rep_star = np.zeros((ny, nb_d, nb_f), int64)
    ib_f_rep_star = np.zeros((ny, nb_d, nb_f), int64)
    dom_default_prob_rep = np.zeros((ny, nb_d, nb_f))
    ext_default_prob_rep = np.zeros((ny, nb_d, nb_f))
    dom_default_prob_fd = np.zeros((ny, nb_d, nb_f))
    ext_default_prob_dd = np.zeros((ny, nb_f, nb_f))
    dom_default_states = np.zeros((ny, nb_d, nb_f))
    ext_default_states = np.zeros((ny, nb_d, nb_f))
    tot_default_states = np.zeros((ny, nb_d, nb_f))
    q_d_r = np.ones((ny, nb_d, nb_f, nb_d, nb_f)) * 0.95 # 5 grids, since it depends on resource constraint as well
    q_f_r = np.ones((ny, nb_d, nb_f)) * 0.95
    q_f_dd = np.ones((ny, nb_f)) * 0.95
    q_d_fd = np.ones((ny, nb_d, nb_d)) * 0.95
    Vfd = np.zeros(ny, nb_d)
    Vdd = np.zeros(ny, nb_f)
    Vtd = np.zeros(ny)
    Vc, V, Vupd = np.zeros((ny, nb_d, nb_f)), np.zeros((ny, nb_d, nb_f)), np.zeros((ny, nb_d, nb_f))

    it = 0
    dist = 10.0
    while (it < maxiter) and (dist > tol):

        # Compute expectations used for this iteration
        EV = P@V
        EVfd = P@Vfd
        EVdd = P@Vdd
        EVtd = P@Vtd

        for iy in range(ny):
            # Update value function for total default state
            Vtd[iy] = model.bellman_tot_default(iy, EVtd, EVdd, EVfd, EV)

            for ib_d in range(nb_d):
                # Update the value function for external default state and also optimal domestic savings:
                ib_d_fd_star[iy, ib_d, np.zeros(nb_f)] = model.compute_dom_savings_policy_ext_default(iy, ib_d, q_d_fd, EV)
                Vc[iy, ib_d, np.zeros(nb_f)] = model.bellman_ext_default(iy, ib_d, EVfd, q_d_fd, EV, ib_d_fd_star[iy, ib_d, np.zeros(nb_f)])

            for ib_f in range(nb_f):
                # Update the value function for domestic default state and also optimal domestic savings
                ib_f_dd_star[iy, np.zeros(nb_d), ib_f] = model.compute_ext_savings_policy_dom_default(iy, ib_f, q_f_dd, EV)
                Vc[iy, np.zeros(nb_d), ib_f] = model.bellman_dom_default(iy, ib_f, q_f_dd, EVdd, EV, ib_f_dd_star[iy, np.zeros(nb_d), ib_f])

            for ib_d, ib_f in range(nb_d + nb_f):
                # Update the value function for repayment and also optimal savings.
                ib_d_rep_star[iy, ib_d, ib_f] = model.compute_dom_savings_policy_rep(iy, ib_d, ib_f, q_d_r, q_f_r, EV)
                ib_f_rep_star[iy, ib_d, ib_f] = model.compute_ext_savings_policy_rep(iy, ib_d, ib_f, q_d_r, q_f_r, EV)
                Vc[iy, ib_d, ib_f] = model.bellman_nondefault(iy, ib_d, ib_f, q_d_r, q_f_r, EV, ib_d_rep_star[iy, ib_d, ib_f], ib_f_rep_star[iy, ib_d, ib_f])

        # Once value functions are updated, can combine them to get
        # the full value function
        Vfd_compat = np.reshape(np.repeat(Vfd, nb_d, nb_f), (ny, nb_d, nb_f))
        Vdd_compat = np.reshape(np.repeat(Vdd, nb_d, nb_f), (ny, nb_d, nb_f))
        Vtd_compat = np.reshape(np.repeat(Vtd, nb_d, nb_f), (ny, nb_d, nb_f))
        Vupd[:, :, :] = np.maximum(Vc, Vfd_compat, Vdd_compat, Vtd_compat)

        # Can also compute default states and update prices
        ext_default_states[:, :, :] = 1.0 * (Vfd_compat > max(Vc, Vdd, Vtd))
        dom_default_states[:, :, :] = 1.0 * (Vdd_compat > max(Vc, Vfd, Vtd))
        tot_default_states[:, :, :] = 1.0 * (Vtd_compat > max(Vc, Vfd, Vdd))
        ext_default_prob_dd[:, :, :] = P @ ext_default_states
        dom_default_prob_fd[:, :, :] = P @ dom_default_states
        ext_default_prob_rep[:, :, :] = P @ ext_default_states
        dom_default_prob_rep[:, :, :] = P @ dom_default_states

        # Prices
        q_f_r[:, :, :] = (1 - ext_default_prob_rep) / (1 + r)
        q_f_dd[:, :] = θ_f * (1 - ext_default_prob_rep) / (1 + r) + (1 - θ_f) (1 - ext_default_prob_dd) / (1 + r)
        q_d_r[:, :, :, :, :] = β * ((1 - ext_default_prob_rep) * (1 - dom_default_prob_rep) * u_prime(y[iy] - T) + ext_default_prob_rep * (1 - dom_default_prob_rep) * u_prime(ext_y[iy] - T)) / u_prime(y[iy] - T)
        q_d_fd[:, :, :] = β * (θ_f * (1 - dom_default_prob_rep) * u_prime(y[iy] - T) + (1 - θ_f) * (1 - dom_default_prob_fd) * u_prime(ext_y[iy] - T)) / u_prime(ext_y[iy] - T)

        # Check tolerance etc...
        dist = np.max(np.abs(Vupd - V))
        V[:, :] = Vupd[:, :]
        it += 1

    return V, Vc, Vfd, Vdd, Vtd, ib_d_rep_star, ib_f_rep_star, ib_d_fd_star, ib_f_dd_star, ext_default_prob_rep, ext_default_prob_dd, dom_default_prob_rep, dom_default_prob_fd, ext_default_states, dom_default_states, tot_default_states, q_f_r, q_f_dd, q_d_r, q_d_fd

# Then, writing a function that allow us to simulate the economy once we have the policy functions:

def simulate(model, J, ext_default_states, dom_default_states, tot_default_states, ib_d_rep_star, ib_f_rep_star, ib_d_fd_star, ib_f_dd_star, q_d_r, q_d_fd, q_f_r, q_f_dd, y_init=None, b_d_init=None, b_f_init=None):
    """
    Simulates the Selective Default model of sovereign debt

    Parameters
    ----------
    model: Selective_Economy
        An instance of the Selective Default model with the corresponding parameters
    J: integer
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
    ib_f_rep_star: array(float64, 2)
        A matrix which specifies the external debt/savings level that a country holds
        during a given state in case of REPAYMENT
    ib_d__rep_star: array(float64, 2)
        A matrix which specifies the domestic debt/savings level that a country holds
        during a given state in case of REPAYMENT
    ib_f_dd_star: array(float64, 2)
        A matrix which specifies the external debt/savings level that a country holds
        during a given state in case of DOMESTIC DEFAULT
    ib_d_fd_star: array(float64, 2)
        A matrix which specifies the DOMESTIC debt/savings level that a country holds
        during a given state in case of FOREIGN DEFAULT
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
    mc = qe.MarkovChain(model.P, model.y)
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
    for t in range(J):
        yi = y_sim_indices[t]

        # Fill y/B for today
        if not ext_in_default and not dom_in_default:
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
            if ext_default_states[yi, b_di, b_fi] > 1e-4 and dom_default_states[yi, b_di, b_fi] > 1e-4:
                tot_in_default=True
                b_di_next = zero_b_f_index
                b_fi_next = zero_b_d_index

            if ext_default_states[yi, b_di, b_fi] > 1e-4 and not dom_default_states[yi, b_di, b_fi] > 1e-4:
                ext_in_default=True
                b_fi_next = zero_b_f_index
                b_di_next = ib_d_fd_star[yi, b_di, b_fi]

            if dom_default_states[yi, b_di, b_fi] > 1e-4 and not ext_default_states[yi, b_di, b_fi] > 1e-4:
                dom_in_default=True
                b_di_next = zero_b_d_index
                b_fi_next = ib_f_dd_star[yi, b_di, b_fi]

            else:
                b_fi_next = ib_f_rep_star[yi, b_di, b_fi]
                b_di_next = ib_d_rep_star[yi, b_di, b_fi]
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
        q_f_r_sim[t] = q_f_r[yi, b_fi, b_fi_next]
        q_f_dd_sim[t] = q_f_dd[yi, b_fi_next]
        q_d_r_sim[t] = q_d_r[yi, b_di, b_di_next, b_fi, b_fi_next]
        q_d_fd_sim[t] = q_d_fd_sim[yi, b_di, b_di_next]
        b_fi = b_fi_next
        b_di = b_di_next

    return y_sim, b_d_sim, b_f_sim, q_d_r_sim, q_d_fd_sim, q_f_r_sim, q_f_dd_sim, ext_default_sim, dom_default_sim, tot_default_sim


# Now, let's see some results. Firstly, let us compute the value function, policy and equilibrium prices using Arellano (2008) parameters:

β, σ, r = 0.825, 2.0, 0.017
g, T = 0.13, 0.05
ρ, η, θ_f, θ_d  = 0.897, 0.053, 0.5, 0.5
ny = 21
nb_f = 251
nb_d = 251
b_f_grid = np.linspace(-0.45, 0.45, nb_f)
b_d_grid = np.linspace(-0.45, 0.45, nb_d)
mc = qe.markov.tauchen(ρ, η, 0, 3, ny) # Tauchen is the method to do the output process, by Tauchen's quadrature method (https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/markov/approximation.py)
ygrid, P = np.exp(mc.state_values), mc.P

# Until here, it is ok.

# The code has a problem from here. It does not work. It says that there are multiple values for "g". Why is that?

se = Selective_Economy(b_f_grid, b_d_grid, P, T, ygrid, β=β, σ=σ, r=r, ρ=ρ, η=η, θ_f=θ_f, θ_d=θ_d, g=g)

# Now, solving it:

V, Vc, Vfd, Vdd, Vtd, ib_d_rep_star, ib_d_fd_star, ib_f_rep_star, ib_f_dd_star,  ext_default_prob_rep, ext_default_prob_dd, dom_default_prob_rep, dom_default_prob_fd, ext_default_states, dom_default_states, tot_default_states