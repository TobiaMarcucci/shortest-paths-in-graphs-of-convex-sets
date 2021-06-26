import numpy as np
import scipy as sp
from utils.convex_sets import Singleton, Polyhedron, CartesianProduct
from utils.convex_functions import Constant, SquaredTwoNorm
from utils.graph import GraphOfConvexSets
from utils.shortest_path import ShortestPathProblem


class PieceWiseAffineSystem():
    '''Dynamical system of the form
        z(k+1) = Ai z(k) + Bi u(k) + ci if (z(k), u(k)) in Di
    where Di := {(z, u): Fi z + Gi u <= hi}.'''

    def __init__(self, dynamics, domains):
        '''Arguments must be:
            dynamics: list of triples (Ai, Bi, ci)
            domains: list of instances of Polyhedron (must be bounded)
        '''
        self.dynamics = dynamics
        self.domains = domains
        self.nm = len(dynamics)
        self.nz, self.nu = dynamics[0][1].shape

class RegulationSolution():

    def __init__(self, z, u, ms, spp):

        self.z = z
        self.u = u
        self.ms = ms
        self.spp = spp

class ShortestPathRegulator():

    def __init__(self, pwa, K, z1, Z, cost_matrices, relaxation=False):

        self.pwa = pwa
        self.K = K
        self.z1 = z1
        self.Z = Z
        self.Q, self.R, self.S = cost_matrices
        graph = self._construct_graph()
        self.spp = ShortestPathProblem(graph, relaxation)

    def _construct_graph(self):

        # initialize graph
        graph = GraphOfConvexSets()

        # ficticious source set
        Zs = Singleton(self.z1)
        U = Singleton(np.zeros(self.pwa.nu))
        graph.add_set(CartesianProduct((Zs, U)), 0)
        graph.set_source(0)

        # vertices for time steps k = 1, ..., K - 1
        for k in range(1, self.K):
            for i in range(self.pwa.nm):
                graph.add_set(self.pwa.domains[i], (k, i))

        # target vertex
        graph.add_set(CartesianProduct((self.Z, U)), self.K)
        graph.set_target(self.K)

        # time step zero
        for i in range(self.pwa.nm):

            # force initial conditions
            I = np.eye(self.pwa.nz)
            zero = np.zeros((self.pwa.nz, self.pwa.nu))
            A = np.hstack((I, zero, -I, zero))
            b = np.zeros(self.pwa.nz)
            D = Polyhedron(eq_matrices=(A, b))
            graph.add_edge(0, (1, i), Constant(0, D))

        # domains for the edge lengths
        D = []
        for i in range(self.pwa.nm):
            Ai, Bi, ci = self.pwa.dynamics[i]
            A = np.hstack((Ai, Bi, - np.eye(self.pwa.nz), np.zeros((self.pwa.nz, self.pwa.nu))))
            b = - ci
            D.append(Polyhedron(eq_matrices=(A, b)))

        # edges for time steps k = 1, ..., K - 2
        H = sp.linalg.block_diag(self.Q, self.R, np.zeros((self.pwa.nz + self.pwa.nu,) * 2))
        for k in range(1, self.K - 1):
            for i in range(self.pwa.nm):
                for j in range(self.pwa.nm):
                    graph.add_edge((k, i), (k + 1, j), SquaredTwoNorm(H, D[i]))

        # edges for time step K - 1
        HT = sp.linalg.block_diag(self.Q, self.R, self.S, np.zeros((self.pwa.nu, self.pwa.nu)))
        for i in range(self.pwa.nm):
            graph.add_edge((self.K - 1, i), (self.K), SquaredTwoNorm(HT, D[i]))

        return graph

    def solve(self):
        '''if relaxation returns approximate value for states and controls.'''

        # solve shortest path problem
        sol = self.spp.solve()

        # initialize state, controls, and mode sequence
        z = np.full((self.K, self.pwa.nz), np.nan)
        u = np.full((self.K - 1, self.pwa.nu), np.nan)
        ms = np.full(self.K - 1, np.nan)

        # time step 1
        E_out = self.spp.graph.outgoing_edges(0)[1]
        zu = sum(sol.primal.z[E_out])
        z[0], u[0] = np.split(zu, (self.pwa.nz,))

        # all the remaining time steps
        for k in range(1, self.K):
            E_out = [e for i in range(self.pwa.nm) for e in self.spp.graph.outgoing_edges((k, i))[1]]
            zu = sum(sol.primal.z[E_out])
            if k < self.K - 1:
                z[k], u[k] = np.split(zu, (self.pwa.nz,))
            z[self.K - 1] = zu[:self.pwa.nz]

        # reconstruct mode sequence
        if not self.spp.relaxation:
            for j, edge in enumerate(self.spp.graph.edges):
                if edge[1] != self.spp.graph.target:
                    if np.isclose(sol.primal.phi[j], 1):
                        ms[k - 1] = edge[1][1]

        return RegulationSolution(z, u, ms, sol)
