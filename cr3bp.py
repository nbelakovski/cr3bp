from dataclasses import dataclass
import numpy as np
from scipy.optimize import root_scalar


@dataclass
class System:
    '''A dataclass representing a 3 body system with 2 massive primary bodies
    and a third body of negligible mass. Several systems are provided, including
    Earth-Moon, Sun-Earth, and Sun-Jupiter. To instantiate your own, construct
    a new System using System(mass_primary, mass_secondary, distance) where
    the first 2 values are expected in kg and the last in m (not km)'''

    mass_primary: float  # kg
    mass_secondary: float  # kg
    l: float  # m
    G: float = 6.67408e-11  # m^3 / (kg s^2)

    @property
    def mu(self):
        return self.mass_secondary / (self.mass_primary + self.mass_secondary)

    @property
    def theta_dot(self):
        return np.sqrt(self.G*(self.mass_primary+self.mass_secondary)
                       / (self.l**3))

    @property
    def total_mass(self):
        return self.mass_primary + self.mass_secondary

    def convert_to_dimensional_state(self, state):
        '''Pass in a dimensionless state and you will be given back
        dimensional values'''
        return state * self.conversion_vector

    @property
    def conversion_vector(self):
        '''Multiply this with the state to get dimensional values'''
        return np.array([self.l,
                         self.l,
                         self.l,
                         self.l/self.s,
                         self.l/self.s,
                         self.l/self.s])

    @property
    def seconds(self):
        '''Number of seconds in a non-dimensional second'''
        return 1 / self.theta_dot

    @property
    def s(self):
        ''' Alias for number of seconds in a nondimensional second'''
        return self.seconds

    def x_eqn(self):
        '''Equation for equilibrium points along the x axis, used for finding L1,
        L2, L3'''
        def func(x):
            return (-(1 - self.mu)/((x + self.mu) * abs(x + self.mu))
                    - self.mu/((x - (1 - self.mu)) * abs(x - (1 - self.mu)))
                    + x)
        return func

    @property
    def L1(self):
        eps = np.finfo(float).eps
        return root_scalar(self.x_eqn(),
                           bracket=[-self.mu + eps, 1 - self.mu - eps]).root

    @property
    def L2(self):
        eps = np.finfo(float).eps
        return root_scalar(self.x_eqn(),
                           bracket=[1 - self.mu + eps, 2]).root

    @property
    def L3(self):
        eps = np.finfo(float).eps
        return root_scalar(self.x_eqn(),
                           bracket=[-2, -self.mu - eps]).root

    @property
    def L4(self):
        return (0.5, np.sqrt(3) / 2)

    @property
    def L5(self):
        return (0.5, -np.sqrt(3) / 2)


mass_earth = 5.972e24  # kg
mass_moon = 7.35e22  # kg
earth_moon_distance = 384400000  # m
EarthMoon = System(mass_earth, mass_moon, earth_moon_distance)

mass_sun = 1.989e30  # kg
sun_earth_distance = 149.6e9  # m
SunEarth = System(mass_sun, mass_earth, sun_earth_distance)

mass_jupiter = 1.898e27  # kg
sun_jupiter_distance = 778.34e9  # m

SunJupiter = System(mass_sun, mass_jupiter, sun_jupiter_distance)


def r13(x, y, z, mu):
    return np.sqrt((x+mu)**2 + y**2 + z**2)


def r23(x, y, z, mu):
    return np.sqrt((x-(1-mu))**2 + y**2 + z**2)


# For those who want to work directly in meters and seconds, use this
def DimensionalEOMs(t, X, system=EarthMoon):
    G = system.G
    Xdot = np.zeros(len(X))
    Xdot[:3] = X[3:6]
    x = X[0]
    y = X[1]
    z = X[2]
    xdot = X[3]
    ydot = X[4]
    x1 = mu_13 = system.mass_secondary * system.l / system.total_mass
    mu_23 = 1 - system.mass_primary * system.l / system.total_mass
    x2 = 1 - mu_23
    r13eval = r13(x, y, z, mu_13)
    r23eval = r23(x, y, z, mu_23)
    xdoubledot = (system.theta_dot**2 * x
                  + 2 * ydot * system.theta_dot
                  - G*system.mass_primary * (x + x1) / (r13eval**3)
                  - G*system.mass_secondary * (x - x2) / (r23eval**3))
    ydoubledot = (system.theta_dot**2 * y
                  - 2 * xdot * system.theta_dot
                  - G*system.mass_primary * y / (r13eval**3)
                  - G*system.mass_secondary * y / (r23eval**3))
    zdoubledot = (-G*system.mass_primary * z / (r13eval**3)
                  - G*system.mass_secondary * z / (r23eval**3))
    Xdot[3] = xdoubledot
    Xdot[4] = ydoubledot
    Xdot[5] = zdoubledot
    return Xdot


# Pass the mu of the system in here and you will get back a function which
# can be passed to RK45 and the like
def EOMConstructor(mu, STM=False):
    def NonLinearEOMs(t, X, mu=mu):
        Xdot = np.zeros(6)
        Xdot[:3] = X[3:6]
        (x, y, z) = (X[0], X[1], X[2])
        (xdot, ydot) = (X[3], X[4])
        xdoubledot = (x
                      + 2 * ydot
                      - (1 - mu)*(x + mu)/(r13(x, y, z, mu)**3)
                      - mu*(x - (1 - mu))/(r23(x, y, z, mu)**3))
        ydoubledot = (y
                      - 2 * xdot
                      - (1 - mu)*y/(r13(x, y, z, mu)**3)
                      - mu*y/(r23(x, y, z, mu)**3))
        zdoubledot = (-(1 - mu)*z/(r13(x, y, z, mu)**3)
                      - mu*z/(r23(x, y, z, mu)**3))
        Xdot[3] = xdoubledot
        Xdot[4] = ydoubledot
        Xdot[5] = zdoubledot
        return Xdot

    if STM:
        def EOMs(t, X, mu=mu):
            Xdot = np.zeros(len(X))
            Xdot[:6] = NonLinearEOMs(t, X, mu)
            STM = X[6:].reshape((6, 6))
            (x, y, z) = (X[0], X[1], X[2])
            STMdot = A(x, y, z, mu) @ STM
            Xdot[6:] = STMdot.reshape(36)
            return Xdot
    else:
        def EOMs(t, X, mu=mu):
            return NonLinearEOMs(t, X, mu)
    return EOMs


def A(x, y, z, mu):
    Fxx = fxx(x, y, z, mu)
    Fyy = fyy(x, y, z, mu)
    Fzz = fzz(x, y, z, mu)
    Fxy = fxy(x, y, z, mu)
    Fyz = fyz(x, y, z, mu)
    Fxz = fxz(x, y, z, mu)
    A = np.array([[0,     0,   0,  1, 0, 0],
                  [0,     0,   0,  0, 1, 0],
                  [0,     0,   0,  0, 0, 1],
                  [Fxx, Fxy, Fxz,  0, 2, 0],
                  [Fxy, Fyy, Fyz, -2, 0, 0],
                  [Fxz, Fyz, Fzz,  0, 0, 0],
                  ])
    return A


def fxx(x, y, z, mu):
    return (1 - ((1-mu) / r13(x, y, z, mu)**3 - 3*(x + mu)**2 * (1 - mu) / r13(x, y, z, mu)**5)
              - (mu / r23(x, y, z, mu)**3 - 3*(x - (1 - mu))**2 * mu / r23(x, y, z, mu)**5))


def fyy(x, y, z, mu):
    return (1 - ((1-mu) / r13(x, y, z, mu)**3 - 3 * (1 - mu) * y**2 / r13(x, y, z, mu)**5)
              - (mu / r23(x, y, z, mu)**3 - 3 * mu * y**2 / r23(x, y, z, mu)**5))


def fzz(x, y, z, mu):
    return -((1 - mu) / r13(x, y, z, mu)**3 + mu / r23(x, y, z, mu)**3) + 3 * z**2 * (
            (1 - mu) / r13(x, y, z, mu)**5 + mu / r23(x, y, z, mu)**5)


def fxy(x, y, z, mu):
    return (3 * (1 - mu) * (x + mu) * y / r13(x, y, z, mu)**5
            + 3 * mu * (x - (1 - mu)) * y / r23(x, y, z, mu)**5)


def fxz(x, y, z, mu):
    return 3 * z * ((1-mu)*(x+mu)/r13(x, y, z, mu)**5 + mu*(x-(1-mu))/r23(x, y, z, mu)**5)


def fyz(x, y, z, mu):
    return 3 * y * z * ((1 - mu) / r13(x, y, z, mu)**5 + mu / r23(x, y, z, mu)**5)


def Lambda2(fxx, fyy, fxy=0):
    return (-(4 - fxx - fyy) - np.sqrt((4 - fxx - fyy)**2 - 4 * (fxx*fyy - fxy**2)))/2


def initial_velocity(initial_position: tuple[float, float], lpoint: float, mu: float) -> tuple[float, float]:
    fxxL = fxx(lpoint, 0, 0, mu)
    fyyL = fyy(lpoint, 0, 0, mu)
    Lamb2 = Lambda2(fxxL, fyyL)
    xdot = (-2 * Lamb2)/(fxxL - Lamb2) * initial_position[1]
    ydot = -(fxxL - Lamb2)/2 * (initial_position[0] - lpoint)
    return (xdot, ydot)
