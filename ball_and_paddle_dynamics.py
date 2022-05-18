# external imports
import numpy as np
import sympy as sp

# internal imports
from spp.convex_sets import Polyhedron
from spp.pwa_systems import PieceWiseAffineSystem
import ball_and_paddle_parameters as params

def get_affine_expression(x, expr):
    """
    Extracts from the symbolic affine expression the matrices such that expr(x) = C x - d.

    Arguments
    ----------
    x : sympy matrix filled with sympy symbols
        Variables.
    expr : sympy matrix filled with sympy symbolic affine expressions
        Left hand side of the inequality constraint.
    """

    # state transition matrices
    C = np.array(expr.jacobian(x)).astype(np.float64)

    # offset term
    d = - np.array(expr.subs({xi:0 for xi in x})).astype(np.float64).flatten()
    
    return C, d

def get_transition_matrices(x, u, x_next):
    """
    Extracts from the symbolic expression of the state at the next time step the matrices A, B, and c.

    Arguments
    ----------
    x : sympy matrix filled with sympy symbols
        Symbolic state of the system.
    u : sympy matrix filled with sympy symbols
        Symbolic input of the system.
    x_next : sympy matrix filled with sympy symbolic linear expressions
        Symbolic value of the state update.
    """

    # state transition matrices
    A = np.array(x_next.jacobian(x)).astype(np.float64)
    B = np.array(x_next.jacobian(u)).astype(np.float64)

    # offset term
    origin = {xi:0 for xi in x}
    origin.update({ui:0 for ui in u})
    c = np.array(x_next.subs(origin)).astype(np.float64).flatten()
    
    return A, B, c

# symbolic state
xb, yb, tb = sp.symbols('xb yb tb') # position of the ball
xf, yf = sp.symbols('xf yf') # position of the floor
xdb, ydb, tdb = sp.symbols('xdb ydb tdb') # velocity of the ball
xdf, ydf = sp.symbols('xdf ydf') # velocity of the floor
x = sp.Matrix([
    xb, yb, tb,
    xf, yf,
    xdb, ydb, tdb,
    xdf, ydf
])

# symbolic input
xd2f, yd2f = sp.symbols('xd2f yd2f') # acceleration of the floor
u = sp.Matrix([
    xd2f, yd2f
])

# symbolic contact forces
ftf, fnf = sp.symbols('ftf fnf') # floor force
ftc, fnc = sp.symbols('ftc fnc') # ceiling force

# symbolic ball velocity update
xdb_next = xdb + params.h*ftf/params.m - params.h*ftc/params.m
ydb_next = ydb + params.h*fnf/params.m - params.h*fnc/params.m - params.h*params.g
tdb_next = tdb + params.r*params.h*ftf/params.j + params.r*params.h*ftc/params.j

# symbolic ball position update
xb_next = xb + params.h*xdb_next
yb_next = yb + params.h*ydb_next
tb_next = tb + params.h*tdb_next

# symbolic floor velocity update
xdf_next = xdf + params.h*xd2f
ydf_next = ydf + params.h*yd2f

# symbolic floor position update
xf_next = xf + params.h*xdf_next
yf_next = yf + params.h*ydf_next

# symbolic state update
x_next = sp.Matrix([
    xb_next, yb_next, tb_next,
    xf_next, yf_next,
    xdb_next, ydb_next, tdb_next,
    xdf_next, ydf_next
])

# symbolic relative tangential velocity (ball wrt floor and ceiling)
sliding_velocity_floor = xdb_next + params.r*tdb_next - xdf_next
sliding_velocity_ceiling = xdb_next - params.r*tdb_next

# symbolic gap functions (ball wrt floor and ceiling)
gap_floor = yb_next - yf_next
gap_ceiling = params.d - 2.*params.r - yb_next

# symbolic ball distance to floor and ceiling boundaries
ball_on_floor = sp.Matrix([
    xb_next - xf_next - params.l,
    xf_next - xb_next - params.l
])
ball_on_ceiling = sp.Matrix([
    xb_next - params.l,
    - xb_next - params.l
])

# state + input bounds
xu = x.col_join(u)
xu_min = np.concatenate((params.x_min, params.u_min))
xu_max = np.concatenate((params.x_max, params.u_max))

# discrete time dynamics in mode 1
# (ball in the air)

# set forces to zero
f_m1 = {ftf: 0., fnf: 0., ftc: 0., fnc: 0.}

# get dynamics
x_next_m1 = x_next.subs(f_m1)
S1 = get_transition_matrices(x, u, x_next_m1)

# build domain
C1 = [-np.eye(12), np.eye(12)]
d1 = [-xu_min, xu_max]

# - gap <= 0 with floor and ceiling
gap_floor_m1 = gap_floor.subs(f_m1)
gap_ceiling_m1 = gap_ceiling.subs(f_m1)
C, d = get_affine_expression(xu, sp.Matrix([- gap_floor_m1]))
C1.append(C)
d1.append(d)
C, d = get_affine_expression(xu, sp.Matrix([- gap_ceiling_m1]))
C1.append(C)
d1.append(d)

C1 = np.vstack(C1)
d1 = np.concatenate(d1)
D1 = Polyhedron(ineq_matrices=(C1, d1))

# discrete time dynamics in mode 2
# (ball sticking with the floor, not in contact with the ceiling)

# enforce sticking
fc_m2 = {ftc: 0., fnc: 0.}
ftf_m2 = sp.solve(sp.Eq(sliding_velocity_floor.subs(fc_m2), 0), ftf)[0]
fnf_m2 = sp.solve(sp.Eq(gap_floor.subs(fc_m2), 0), fnf)[0]
f_m2 = fc_m2.copy()
f_m2.update({ftf: ftf_m2, fnf: fnf_m2})

# get dynamics
x_next_m2 = x_next.subs(f_m2)
S2 = get_transition_matrices(x, u, x_next_m2)

# build domain
C2 = [-np.eye(12), np.eye(12)]
d2 = [-xu_min, xu_max]

# gap <= 0 with floor
C, d = get_affine_expression(xu, sp.Matrix([gap_floor_m1]))
C2.append(C)
d2.append(d)

# - gap <= 0 with ceiling
C, d = get_affine_expression(xu, sp.Matrix([- gap_ceiling_m1]))
C2.append(C)
d2.append(d)

# ball not falling down the floor
C, d = get_affine_expression(xu, ball_on_floor.subs(f_m2))
C2.append(C)
d2.append(d)

# friction cone
C, d = get_affine_expression(xu, sp.Matrix([ftf_m2 - params.mu*fnf_m2]))
C2.append(C)
d2.append(d)
C, d = get_affine_expression(xu, sp.Matrix([- ftf_m2 - params.mu*fnf_m2]))
C2.append(C)
d2.append(d)

C2 = np.vstack(C2)
d2 = np.concatenate(d2)
D2 = Polyhedron(ineq_matrices=(C2, d2))

# discrete time dynamics in mode 3
# (ball sliding right on the floor, not in contact with the ceiling)

# enforce sticking
f_m3 = {ftf: -params.mu*fnf_m2, fnf: fnf_m2, ftc: 0., fnc: 0.}

# get dynamics
x_next_m3 = x_next.subs(f_m3)
S3 = get_transition_matrices(x, u, x_next_m3)

# build domain
C3 = [-np.eye(12), np.eye(12)]
d3 = [-xu_min, xu_max]

# gap <= 0 with floor
C, d = get_affine_expression(xu, sp.Matrix([gap_floor_m1]))
C3.append(C)
d3.append(d)

# - gap <= 0 with ceiling
C, d = get_affine_expression(xu, sp.Matrix([- gap_ceiling_m1]))
C3.append(C)
d3.append(d)

# ball not falling down the floor
C, d = get_affine_expression(xu, ball_on_floor.subs(f_m3))
C3.append(C)
d3.append(d)

# positive relative velocity
C, d = get_affine_expression(xu, sp.Matrix([- sliding_velocity_floor.subs(f_m3)]))
C3.append(C)
d3.append(d)

C3 = np.vstack(C3)
d3 = np.concatenate(d3)
D3 = Polyhedron(ineq_matrices=(C3, d3))

# discrete time dynamics in mode 4
# (ball sliding left on the floor, not in contact with the ceiling)

# enforce sticking
f_m4 = {ftf: params.mu*fnf_m2, fnf: fnf_m2, ftc: 0., fnc: 0.}

# get dynamics
x_next_m4 = x_next.subs(f_m4)
S4 = get_transition_matrices(x, u, x_next_m4)

# build domain
C4 = [-np.eye(12), np.eye(12)]
d4 = [-xu_min, xu_max]

# gap <= 0 with floor
C, d = get_affine_expression(xu, sp.Matrix([gap_floor_m1]))
C4.append(C)
d4.append(d)

# - gap <= 0 with ceiling
C, d = get_affine_expression(xu, sp.Matrix([- gap_ceiling_m1]))
C4.append(C)
d4.append(d)

# ball not falling down the floor
C, d = get_affine_expression(xu, ball_on_floor.subs(f_m4))
C4.append(C)
d4.append(d)

# negative relative velocity
C, d = get_affine_expression(xu, sp.Matrix([sliding_velocity_floor.subs(f_m4)]))
C4.append(C)
d4.append(d)

C4 = np.vstack(C4)
d4 = np.concatenate(d4)
D4 = Polyhedron(ineq_matrices=(C4, d4))

# discrete time dynamics in mode 5
# (ball sticking on the ceiling, not in contact with the floor)

# enforce sticking
ff_m5 = {ftf: 0., fnf: 0.}
ftc_m5 = sp.solve(sp.Eq(sliding_velocity_ceiling.subs(ff_m5), 0), ftc)[0]
fnc_m5 = sp.solve(sp.Eq(gap_ceiling.subs(ff_m5), 0), fnc)[0]
f_m5 = ff_m5.copy()
f_m5.update({ftc: ftc_m5, fnc: fnc_m5})

# get dynamics
x_next_m5 = x_next.subs(f_m5)
S5 = get_transition_matrices(x, u, x_next_m5)

# build domain
C5 = [-np.eye(12), np.eye(12)]
d5 = [-xu_min, xu_max]

# - gap <= 0 with floor
C, d = get_affine_expression(xu, sp.Matrix([- gap_floor_m1]))
C5.append(C)
d5.append(d)

# gap <= 0 with ceiling
C, d = get_affine_expression(xu, sp.Matrix([gap_ceiling_m1]))
C5.append(C)
d5.append(d)

# ball in contact with the ceiling
C, d = get_affine_expression(xu, ball_on_ceiling.subs(f_m5))
C5.append(C)
d5.append(d)

# friction cone
C, d = get_affine_expression(xu, sp.Matrix([ftc_m5 - params.mu*fnc_m5]))
C5.append(C)
d5.append(d)
C, d = get_affine_expression(xu, sp.Matrix([- ftc_m5 - params.mu*fnc_m5]))
C5.append(C)
d5.append(d)

C5 = np.vstack(C5)
d5 = np.concatenate(d5)
D5 = Polyhedron(ineq_matrices=(C5, d5))

# discrete time dynamics in mode 6
# (ball sliding right on the ceiling, not in contact with the floor)

# enforce sticking
f_m6 = {ftc: -params.mu*fnc_m5, fnc: fnc_m5, ftf: 0., fnf: 0.}

# get dynamics
x_next_m6 = x_next.subs(f_m6)
S6 = get_transition_matrices(x, u, x_next_m6)

# build domain
C6 = [-np.eye(12), np.eye(12)]
d6 = [-xu_min, xu_max]

# - gap <= 0 with floor
C, d = get_affine_expression(xu, sp.Matrix([- gap_floor_m1]))
C6.append(C)
d6.append(d)

# gap <= 0 with ceiling
C, d = get_affine_expression(xu, sp.Matrix([gap_ceiling_m1]))
C6.append(C)
d6.append(d)

# ball in contact with the ceiling
C, d = get_affine_expression(xu, ball_on_ceiling.subs(f_m6))
C6.append(C)
d6.append(d)

# positive relative velocity
C, d = get_affine_expression(xu, sp.Matrix([- sliding_velocity_ceiling.subs(f_m6)]))
C6.append(C)
d6.append(d)

C6 = np.vstack(C6)
d6 = np.concatenate(d6)
D6 = Polyhedron(ineq_matrices=(C6, d6))

# discrete time dynamics in mode 7
# (ball sliding left on the ceiling, not in contact with the floor)

# enforce sticking
f_m7 = {ftc: params.mu*fnc_m5, fnc: fnc_m5, ftf: 0., fnf: 0.}

# get dynamics
x_next_m7 = x_next.subs(f_m7)
S7 = get_transition_matrices(x, u, x_next_m7)

# build domain
C7 = [-np.eye(12), np.eye(12)]
d7 = [-xu_min, xu_max]

# - gap <= 0 with floor
C, d = get_affine_expression(xu, sp.Matrix([- gap_floor_m1]))
C7.append(C)
d7.append(d)

# gap <= 0 with ceiling
C, d = get_affine_expression(xu, sp.Matrix([gap_ceiling_m1]))
C7.append(C)
d7.append(d)

# ball in contact with the ceiling
C, d = get_affine_expression(xu, ball_on_ceiling.subs(f_m7))
C7.append(C)
d7.append(d)

# negative relative velocity
C, d = get_affine_expression(xu, sp.Matrix([sliding_velocity_ceiling.subs(f_m7)]))
C7.append(C)
d7.append(d)

C7 = np.vstack(C7)
d7 = np.concatenate(d7)
D7 = Polyhedron(ineq_matrices=(C7, d7))

# list of dynamics
S_list = [S1, S2, S3, S4, S5, S6, S7]

# list of domains
D_list = [D1, D2, D3, D4, D5, D6, D7]

# PWA system
pwa = PieceWiseAffineSystem(S_list, D_list)
