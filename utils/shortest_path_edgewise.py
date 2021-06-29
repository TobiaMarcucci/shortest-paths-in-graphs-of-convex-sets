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

        # loop through the vertices
        for v, Xv in graph.sets.items():

            # indices of the edges incident with this vertex
            edges_in = graph.incoming_edges(v)[1]
            edges_out = graph.outgoing_edges(v)[1]

            # incident flow variables
            phi_in = sum(vars.phi[edges_in])
            phi_out = sum(vars.phi[edges_out])

            # indicators for source and target
            delta_sv = 1 if v == graph.source else 0
            delta_tv = 1 if v == graph.target else 0

            # conservation of flow
            if len(edges_in) > 0 or len(edges_out) > 0:
                residual = phi_out + delta_tv - phi_in - delta_sv
                prog.AddLinearConstraint(residual == 0)

            # degree constraint
            if len(edges_out) > 0:
                residual = phi_out + delta_tv - 1
                prog.AddLinearConstraint(residual <= 0)

        # loop through the edges
        for k, e in enumerate(graph.edges):

            # spatial nonnegativity
            Xu, Xv = [graph.sets[v] for v in e]
            Xu.add_perspective_constraint(prog, vars.phi[k], vars.y[k])
            Xv.add_perspective_constraint(prog, vars.phi[k], vars.z[k])

            # spatial upper bound
            xu, xv = vars.x[graph.vertex_indices(e)]
            Xu.add_perspective_constraint(prog, 1 - vars.phi[k], xu - vars.y[k])
            Xv.add_perspective_constraint(prog, 1 - vars.phi[k], xv - vars.z[k])

            # slack constraints for the objetive
            yz = np.concatenate((vars.y[k], vars.z[k]))
            graph.lengths[e].add_perspective_constraint(prog, vars.l[k], vars.phi[k], yz)

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

    def solve(self):

        result = MosekSolver().Solve(self.prog)
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time
        primal = ShortestPathVariables.from_result(result, self.vars)

        return ShortestPathSolution(cost, time, primal)
