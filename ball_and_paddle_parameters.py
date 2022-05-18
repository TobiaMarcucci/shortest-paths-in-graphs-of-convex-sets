# external imports
import numpy as np

# internal imports
from spp.convex_sets import Polyhedron

# numeric parameters of the system
m = 1. # mass
r = .1 # radius
j = .4*m*r**2. # moment of inertia
d = .4 # nominal floor-ceiling distance
l = .3 # floor and ceiling width
mu = .2 # friction coefficient
g = 10. # gravity acceleration
h = .05 # discretization time step

# state bounds
x_max = np.array([
    l, d - 2.*r, 1.2*np.pi, # ball configuration
    l, d - 2.*r - .05, # floor configuration
    2., 2., 10., # ball velocity
    2., 2. # floor velocity
])
x_min = - x_max

# input bounds
u_max = np.array([
    30., 30., # floor acceleration
])
u_min = - u_max

# controller parameters

# time steps
K = 20

# weight matrices (*2 to cancel out the 1/2 in the controller code)
Q = np.diag([
    1., 1., .01,
    1., 1.,
    1., 1., .01,
    1., 1.
])*2.
R = np.diag([
    .01, .001
])*2.
S = np.zeros((10, 10))
cost_matrices = (Q, R, S)

# terminal set
Z = Polyhedron.from_bounds(*[np.zeros(10)]*2)
