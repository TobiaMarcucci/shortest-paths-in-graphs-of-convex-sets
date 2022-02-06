import numpy as np
from spp.convex_functions import Constant
from spp.convex_sets import Singleton, Polyhedron, CartesianProduct
from spp.graph import GraphOfConvexSets
from spp.shortest_path import ShortestPathProblem
from scipy.special import binom

class MotionPlanner():

    def __init__(self, safe_sets, qs, qt, degree, continuity=0, relaxation=0):

        self.dimension = len(qs)
        assert self.dimension == len(qt)
        for S in safe_sets:
            assert self.dimension == S.dimension

        self.qs = qs
        self.qt = qt
        self.safe_sets = safe_sets
        self.extended_safe_sets = [Singleton(qs), Singleton(qt)] + list(safe_sets)
        self.degree = degree

        spp_sets = [CartesianProduct([S] * (self.degree + 1)) for S in self.extended_safe_sets]
        spp_edges = []
        for i, S in enumerate(self.extended_safe_sets):
            for j, T in enumerate(self.extended_safe_sets[i+1:]):
                if S.intersects_with(T):
                    j += i + 1
                    spp_edges += [(i, j), (j, i)]

        Z = np.zeros((self.dimension, self.dimension * self.degree))
        I = np.eye(self.dimension)
        A = np.hstack((Z, I, -I, Z))
        b = np.zeros(self.dimension)
        D = Polyhedron(eq_matrices=(A, b))
        spp_edge_lengths = [Constant(0, D) for e in spp_edges]

        graph = GraphOfConvexSets()
        graph.add_sets(spp_sets)
        graph.set_source(0)
        graph.set_target(1)
        for e, l in zip(spp_edges, spp_edge_lengths):
            graph.add_edge(*e, l)

        self.spp = ShortestPathProblem(graph, relaxation=relaxation, two_cycle_elimination=True)

    def solve(self):

        sol = self.spp.solve()

        return sol.primal.x[sol.path].reshape(len(sol.path), self.degree + 1, self.dimension)

class Bezier():

    def __init__(self, points):
        self.points = points
        self.n = points.shape[0] - 1

    def evaluate(self, t):
        c = np.array([self._berstein(t, i, self.n) for i in range(self.n + 1)])
        return c.T.dot(self.points)

    def derivative(self):
        return Bezier(self.n * (self.points[1:] - self.points[:-1]))

    @staticmethod
    def _berstein(t, i, n):
        return binom(n, i) * t ** i * (1 - t) ** (n - i) 
        
