import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import product
from scipy.spatial import ConvexHull, HalfspaceIntersection
from pydrake.all import Expression, MathematicalProgram, MosekSolver, eq, le

class ConvexSet():

    def __init__(self):
        raise NotImplementedError

    def contains(self, x):
        raise NotImplementedError

    def translate(self, x):
        raise NotImplementedError

    def scale(self, s):
        assert s > 0
        return self._scale(s)

    def add_perspective_constraint(self, prog, scale, x):
        raise NotImplementedError

    def _cheb_constraint(self, prog, x, r):
        raise NotImplementedError

    def add_membership_constraint(self, prog, x):
        return self.add_perspective_constraint(prog, 1, x)

    def plot(self, **kwargs):
        if self.dimension != 2:
            raise NotImplementedError
        options = {'facecolor':'lightcyan', 'edgecolor':'k', 'zorder':1}
        options.update(kwargs)
        self._plot(**options)

    @property
    def dimension(self):
        return self._dimension

    @property
    def center(self):
        '''Chebyshev center (center of largest inscribed ball).'''
        if self._center is None:
            self._center = self._compute_center()
        return self._center

class Singleton(ConvexSet):
    '''Singleton set {x}.'''

    def __init__(self, x):
        self._center = np.array(x).astype('float64')
        self._dimension = self.center.size

    def contains(self, x):
        return np.isclose(self.center, x)

    def translate(self, x):
        self._center + x

    def _scale(self, s):
        pass

    def add_perspective_constraint(self, prog, scale, x):
        return prog.AddLinearConstraint(eq(x, self.center * scale))

    def _plot(self, **kwargs):
        plt.scatter(*self.center, c='k')

    def _cheb_constraint(self, prog, x, r):
        prog.AddLinearConstraint(eq(x, self._center))
        prog.AddLinearConstraint(r == 0)

class Polyhedron(ConvexSet):
    '''Polyhedron in halfspace representation {x : A x = b, C x <= d}.'''

    def __init__(self, eq_matrices=None, ineq_matrices=None):
        '''Arguments are eq_matrices = (A, b) and ineq_matrices = (C, d).'''

        if eq_matrices is not None:
            self.A, self.b = [M.astype('float64') for M in eq_matrices]
            self._dimension = self.A.shape[1]
        if ineq_matrices is not None:
            self.C, self.d = [M.astype('float64') for M in ineq_matrices]
            self._dimension = self.C.shape[1]

        if eq_matrices is None:
            self.A = np.zeros((0, self._dimension))
            self.b = np.zeros(0)
        if ineq_matrices is None:
            self.C = np.zeros((0, self._dimension))
            self.d = np.zeros(0)

        self._center = None
        self._vertices = None

    def contains(self, x):

        residual = self.A.dot(x) - self.b
        eq_matrices = np.allclose(residual, 0)

        residual = self.C.dot(x) - self.d
        ineq_matrices = np.isclose(max(max(residual, default=0), 0), 0)

        return eq_matrices and ineq_matrices

    def translate(self, x):

        self.b += self.A.dot(x)
        self.d += self.C.dot(x)

        if self._center is not None:
            self._center += x

        if self._vertices is not None:
            self._vertices += x

    def _scale(self, s):

        center = self.center.copy()
        self.translate(- center)
        self.b *= s
        self.d *= s
        if self._vertices is not None:
            self._vertices *= s
        self.translate(center)

    def add_perspective_constraint(self, prog, scale, x):

        if self.A.shape[0] == 0:
            eq_matrices = None
        else:
            residual = self.A.dot(x) - self.b * scale
            eq_matrices = prog.AddLinearConstraint(eq(residual, 0))

        if self.C.shape[0] == 0:
            ineq_matrices = None
        else:
            residual = self.C.dot(x) - self.d * scale
            ineq_matrices = prog.AddLinearConstraint(le(residual, 0))

        return eq_matrices, ineq_matrices

    def _plot(self, **kwargs):

        if self.vertices.shape[0] < 3:
            raise NotImplementedError

        hull = ConvexHull(self.vertices) # orders vertices counterclockwise
        vertices = self.vertices[hull.vertices]
        plt.fill(*vertices.T, **kwargs)

    def _compute_center(self):

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dimension)
        r = prog.NewContinuousVariables(1)[0]
        prog.AddLinearConstraint(r >= 0)
        prog.AddLinearCost(-r)

        self._cheb_constraint(prog, x, r)
        result = MosekSolver().Solve(prog)

        return result.GetSolution(x)

    def _cheb_constraint(self, prog, x, r):

        if self.A.shape[0] > 0:
            prog.AddLinearConstraint(eq(self.A.dot(x), self.b))

        if self.C.shape[0] > 0:
            C_row_norm = np.linalg.norm(self.C, axis=1)
            lhs = self.C.dot(x) + C_row_norm * r
            prog.AddLinearConstraint(le(lhs, self.d))

    @property
    def vertices(self):

        if self._vertices is not None:
            return self._vertices

        elif self.A.shape[0] > 0:
            raise NotImplementedError

        else:
            halfspaces = np.column_stack((self.C, -self.d))
            P = HalfspaceIntersection(halfspaces, self.center)
            self._vertices = P.intersections
            return self._vertices

    @staticmethod
    def from_bounds(x_min, x_max):

        I = np.eye(len(x_min))
        C = np.vstack((I, -I))
        d = np.concatenate((x_max, -np.array(x_min)))
        P = Polyhedron(ineq_matrices=(C, d))
        P._vertices = np.array(list(product(*zip(x_min, x_max))))

        return P

    @staticmethod
    def from_vertices(vertices):

        vertices = np.array(vertices).astype('float64')
        m, dimension = vertices.shape

        if m <= dimension:
            raise NotImplementedError
        else:
            ch = ConvexHull(vertices)
            ineq_matrices = (ch.equations[:, :-1], - ch.equations[:, -1])

        P = Polyhedron(ineq_matrices=ineq_matrices)
        P._vertices = vertices

        return P

class Ellipsoid(ConvexSet):
    '''Ellipsoid in the form {x : (x - center)' A (x - center) <= 1}.
    The matrix A is assumed to be PSD (and symmetric).'''

    def __init__(self, center, A):

        self._center = np.array(center)
        self.A = np.array(A).astype('float64')
        self._dimension = self._center.size

    def contains(self, x):

        d = np.array(x) - self.center
        ineq_matrices = d.dot(self.A).dot(d) - 1
        
        return np.isclose(max(ineq_matrices, 0), 0)

    def translate(self, x):
        self._center += x

    def _scale(self, s):
        self.A *= 1 / s ** 2

    def add_perspective_constraint(self, prog, scale, x):

        R = sp.linalg.sqrtm(self.A)
        v = np.concatenate(([scale], R.dot(x - self.center * scale)))
        cone_constraint = prog.AddLorentzConeConstraint(v)

        return cone_constraint

    def _cheb_constraint(self, prog, x, r):
        '''Section 8.5.1 of Boyd and Vandenberghe - Convex Optimization.'''
        
        l = prog.NewContinuousVariables(1)[0]
        I = np.eye(self.dimension)
        M11 = np.array([[1 - l]])
        M12 = np.zeros((1, self.dimension))
        M13 = np.array([x - self.center])
        M22 = l * I
        M23 = r * I
        M33 = np.linalg.inv(self.A)
        M = np.block([[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]])
        prog.AddPositiveSemidefiniteConstraint(M * Expression(1))

    def polyhedral_approximation(self, n=100):

        assert self.dimension == 2
        thetas = np.linspace(0, 2 * np.pi, n)
        vertices = np.zeros((n, 2))
        for i, t in enumerate(thetas):
            d = np.array([np.cos(t), np.sin(t)])
            scale = 1 / np.sqrt(d.dot(self.A).dot(d))
            vertices[i] = self.center + scale * d

        return Polyhedron.from_vertices(vertices)

    def _plot(self, **kwargs):

        l, v = np.linalg.eig(self.A)
        angle = 180 * np.arctan2(*v[0]) / np.pi + 90
        ellipse = (self.center, 2 * l[0] ** -.5, 2 * l[1] ** -.5, angle)
        patch = patches.Ellipse(*ellipse, **kwargs)
        plt.gca().add_artist(patch)

class Intersection(ConvexSet):

    def __init__(self, sets):

        self.sets = sets
        assert len(set(X.dimension for X in sets)) == 1
        self._dimension = sets[0].dimension
        self._center = None

    def contains(self, x):
        return all(X.contains(x) for X in self.sets)

    def translate(self, x):
        return Intersection([X.translate(x) for X in self.sets])

    def _scale(self, s):
        return Intersection([X.scale(s) for X in self.sets])

    def add_perspective_constraint(self, prog, scale, x):
        return [X.add_perspective_constraint(prog, scale, x) for X in self.sets]

    def _compute_center(self):

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dimension)
        r = prog.NewContinuousVariables(1)[0]
        prog.AddLinearConstraint(r >= 0)
        prog.AddLinearCost(-r)

        for X in self.sets:
            X._cheb_constraint(prog, x, r)
        result = MosekSolver().Solve(prog)

        return result.GetSolution(x)

    def _plot(self, **kwargs):

        vertices = []
        for X in self.sets:

            if isinstance(X, Singleton):
                raise NotImplementedError

            if not isinstance(X, Polyhedron):
                X = X.polyhedral_approximation()

            for v in X.vertices:
                if self.contains(v):
                    vertices.append(v)

        P = Polyhedron.from_vertices(np.vstack(vertices))
        P.plot()

class CartesianProduct(ConvexSet):

    def __init__(self, sets):

        self.sets = sets
        self._split_at = np.cumsum([X.dimension for X in sets])
        self._dimension = self._split_at[-1]
        self._center = None

    def split(self, x):
        return np.split(x, self._split_at[:-1])

    def contains(self, x):
        return all(X.contains(p) for p, X in zip(self.split(x), self.sets))

    def translate(self, x):
        return CartesianProduct([X.translate(p) for p, X in zip(self.split(x), self.sets)])

    def _scale(self, s):
        return CartesianProduct([X.scale(s) for X in self.sets])

    def add_perspective_constraint(self, prog, scale, x):
        return [X.add_perspective_constraint(prog, scale, p) for p, X in zip(self.split(x), self.sets)]

    def _compute_center(self):
        return np.concatenate([X.center for X in self.sets])

    def _plot(self, **kwargs):
        raise NotImplementedError
