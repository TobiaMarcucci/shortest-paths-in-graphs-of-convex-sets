import numpy as np
from pydrake.all import MathematicalProgram, MosekSolver

class ShortestPathVariables():

    def __init__(self, phi, y, z, l, x):

        self.phi = phi
        self.y = y
        self.z = z
        self.l = l
        self.x = x

    @staticmethod
    def populate_program(prog, graph, relaxation=False):

        phi_type = prog.NewContinuousVariables if relaxation else prog.NewBinaryVariables
        phi = phi_type(graph.n_edges)
        y = prog.NewContinuousVariables(graph.n_edges, graph.dimension)
        z = prog.NewContinuousVariables(graph.n_edges, graph.dimension)
        l = prog.NewContinuousVariables(graph.n_edges)
        x = prog.NewContinuousVariables(graph.n_sets, graph.dimension)

        return ShortestPathVariables(phi, y, z, l, x)

    @staticmethod
    def from_result(result, vars):

        phi = result.GetSolution(vars.phi)
        y = result.GetSolution(vars.y)
        z = result.GetSolution(vars.z)
        l = result.GetSolution(vars.l)
        x = result.GetSolution(vars.x)

        return ShortestPathVariables(phi, y, z, l, x)

class ShortestPathConstraints():

    @staticmethod
    def populate_program(prog, graph, vars):

        for vertex, set in graph.sets.items():

            edges_in = graph.incoming_edges(vertex)[1]
            edges_out = graph.outgoing_edges(vertex)[1]

            phi_in = sum(vars.phi[edges_in])
            phi_out = sum(vars.phi[edges_out])

            delta_sv = 1 if vertex == graph.source else 0
            delta_tv = 1 if vertex == graph.target else 0

            # conservation of flow
            if len(edges_in) > 0 or len(edges_out) > 0:
                residual = phi_out + delta_tv - phi_in - delta_sv
                prog.AddLinearConstraint(residual == 0)

            # degree constraints
            if len(edges_out) > 0:
                residual = phi_out + delta_tv - 1
                prog.AddLinearConstraint(residual <= 0)

        for k, edge in enumerate(graph.edges):

            # spatial nonnegativity
            graph.sets[edge[0]].add_perspective_constraint(prog, vars.phi[k], vars.y[k])
            graph.sets[edge[1]].add_perspective_constraint(prog, vars.phi[k], vars.z[k])

            # spatial upper bound
            xu = vars.x[graph.vertices.index(edge[0])]
            xv = vars.x[graph.vertices.index(edge[1])]
            graph.sets[edge[0]].add_perspective_constraint(prog, 1 - vars.phi[k], xu - vars.y[k])
            graph.sets[edge[1]].add_perspective_constraint(prog, 1 - vars.phi[k], xv - vars.z[k])

            # slack constraints for the objetive (not stored)
            yz = np.concatenate((vars.y[k], vars.z[k]))
            graph.lengths[edge].add_perspective_constraint(prog, vars.l[k], vars.phi[k], yz)

class ShortestPathSolution():

    def __init__(self, cost, time, primal):

        self.cost = cost
        self.time = time
        self.primal = primal
        self.dual = None

class ShortestPathProblem():

    def __init__(self, graph, relaxation=False):

        self.graph = graph
        self.relaxation = relaxation

        self.prog = MathematicalProgram()
        self.vars = ShortestPathVariables.populate_program(self.prog, graph, relaxation)
        self.constraints = ShortestPathConstraints.populate_program(self.prog, graph, self.vars)
        self.prog.AddLinearCost(sum(self.vars.l))

    def solve(self): # relaxation should really be an argument of this

        result = MosekSolver().Solve(self.prog)
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time
        primal = ShortestPathVariables.from_result(result, self.vars)

        return ShortestPathSolution(cost, time, primal)
