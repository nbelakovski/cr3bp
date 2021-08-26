![](https://github.com/nbelakovski/cr3bp/actions/workflows/build.yml/badge.svg)


# Intro

This is a small library meant to facilitate investigation of trajectories
within the Circular Restricted 3 Body Problem (CR3BP)

# Usage

The library consists of a `System` dataclass representing a 3 body system,
a couple of functions which encode the equations of motion of the CR3BP, and
a function for setting initial conditions to get onto a periodic trajectory
around the collinear Lagrange points. Below we'll explain the `System` class
as well as the relevant functions

# System class


You can either use one of the provided 3 body systems

```python
from cr3bp import EarthMoon
```

Or you can create your own system as follows:

```python
from cr3bp import System
mass_primary = 2
mass_secondary = 1
distance = 100
MySystem = System(mass_primary, mass_secondary, distance)
```

The following systems are provided:

EarthMoon  
SunEarth  
SunJupiter  

### API

### Properties

`mass_primary` - kg  
`mass_seconary` - kg  
`l` - m  
`G` - 6.67408e-11 m^3 / (kg s^2) (this is constant across all Systems)  
`mu`  
`theta_dot` - radians/sec  
`total_mass` - kg  
`seconds` - number of seconds in a non-dimensional second (s*)  
`L1`  
`L2`  
`L3`  
`L4`  
`L5`  

All of the L* functions return dimensionless values. L1-3 return a float,
L4 and L5 return a tuple with x and y values (but they just return
(0.5, +/- sqrt(3)/2) for all systems). L1-3 perform a root-finding calculation
so it's best not to use them in a performance sensitive loop - store the
returned value elsewhere and use that

### Functions

`convert_to_dimensional_state(state)`

Accepts a dimensionless 6-element state vector and returns a 6-element state
vector with dimensions added in


`x_eqn`

This is used for calculating the location of the collinear Lagrange points. It
returns a function that bakes in the mu value of the system and which can then
be used for plotting or root finding.

# Other functions

- `dimensional_eoms(t, X, system=EarthMoon)`  
This function encodes the EOMs for the CR3BP using dimensional values (meters, seconds, etc.). It is meant to be used with a numerical integrator from scipy. For example:
`scipy.integrate.solve_ivp(cr3bp.dimensional_eoms, [0, 3600*24], IC, args=[cr3bp.SunEarth], atol=1e-6, rtol=1e-12)`  
In this case we are starting from some initial condition `IC` (a 6 element state vector) and integrating for 1 day using the SunEarth system with a tolerance down to the micrometer level (`atol=1e-6`)
  
- `EOMConstructor(mu)`  
This encodes the EOMs in dimensionless form but it's a little different from `dimensional_eoms` in that it takes a `mu` value and returns a function that can be used with numerical integrators. This just makes it a little smoother to use the numerical integrators since you specify the system when creating the EOMs as opposed to passing it in every time the EOMs are evaluated. Use this function like this:
```python
EM = cr3bp.EarthMoon
eoms = cr3bp.EOMConstructor(EM.mu)
# We set the absolute tolerance to be at the micrometer level and keep the rtol small enough so that the atol dominates. rtol cannot be set lower than machine precision, so there may be some precision issues here which make it preferable to use the dimensional EOMs, since atol for EarthMoon here is 2.6e-15
solution = scipy.integrate.solve_ivp(eoms, [0, 3600*24 / EM.seconds], IC, atol=0.000001/EM.l, rtol=2.3e-14)
```   
- `initial_velocity(initial_position: tuple[float, float], lpoint: float, mu: float) -> tuple[float, float]:`  
This function takes a dimensionless initial position consisting of x and y position relative to a Lagrange point, the dimensionless value of the Lagrange point to which that x and y are relative (this function is meant to be used for collinear points, hence why it only takes a single float value) and the mu of the system and returns the necessary initial x and y velocity to get onto an oscillatory trajectory around the Lagrange point
