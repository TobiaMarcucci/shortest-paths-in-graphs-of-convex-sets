import numpy as np
from pydrake.all import MathematicalProgram, MosekSolver, eq

class ShortestPathVariables():

    def __init__(self, phi, x, y, z, l):

        self.phi = phi
        self.x = x
        self.y = y
        self.z = z
        self.l = l

    @staticmethod
    def populate_program(prog, graph, relaxation=False):

        phi_type = prog.NewContinuousVariables if relaxation else prog.NewBinaryVariables
        phi = phi_type(graph.n_edges)
        x = prog.NewContinuousVariables(graph.n_sets, graph.dimension)
        y = prog.NewContinuousVariables(graph.n_edges, graph.dimension)
        z = prog.NewContinuousVariables(graph.n_edges, graph.dimension)
        l = prog.NewContinuousVariables(graph.n_edges)

        return ShortestPathVariables(phi, x, y, z, l)

    @staticmethod
    def from_result(result, vars):

        phi = result.GetSolution(vars.phi)
        x = result.GetSolution(vars.x)
        y = result.GetSolution(vars.y)
        z = result.GetSolution(vars.z)
        l = result.GetSolution(vars.l)

        return ShortestPathVariables(phi, x, y, z, l)

def populate_constraints(prog, graph, vars):

    bounding_boxes = {}

    for vertex, set in graph.sets.items():

        v = graph.vertex_index(vertex)
        bounding_boxes[vertex] = set.bounding_box()
        set.add_membership_constraint(prog, vars.x[v])

        edges_in, k_in = graph.incoming_edges(vertex)
        edges_out, k_out = graph.outgoing_edges(vertex)

        phi_in = sum(vars.phi[k_in])
        phi_out = sum(vars.phi[k_out])
        y_out = sum(vars.y[k_out])
        z_in = sum(vars.z[k_in])

        delta_sv = 1 if vertex == graph.source else 0
        delta_tv = 1 if vertex == graph.target else 0

        # conservation of flow
        if len(edges_in) > 0 or len(edges_out) > 0:
            residual = phi_out + delta_tv - phi_in - delta_sv
            prog.AddLinearConstraint(residual == 0)

    for k, edge in enumerate(graph.edges):

        Bu = bounding_boxes[edge[0]]
        Bv = bounding_boxes[edge[1]]
        # Bu = graph.sets[edge[0]]
        # Bv = graph.sets[edge[1]]
        u = graph.vertex_index(edge[0])
        v = graph.vertex_index(edge[1])

        Bu.add_perspective_constraint(prog, vars.phi[k], vars.y[k])
        Bv.add_perspective_constraint(prog, vars.phi[k], vars.z[k])
        Bu.add_perspective_constraint(prog, 1 - vars.phi[k], vars.x[u] - vars.y[k])
        Bv.add_perspective_constraint(prog, 1 - vars.phi[k], vars.x[v] - vars.z[k])

        # slack constraints for the objetive (not stored)
        yz = np.concatenate((vars.y[k], vars.z[k]))
        graph.lengths[edge].add_perspective_constraint(prog, vars.l[k], vars.phi[k], yz)

class ShortestPathSolution():

    def __init__(self, cost, time, primal):

        self.cost = cost
        self.time = time
        self.primal = primal

class ShortestPathProblem():

    def __init__(self, graph, relaxation=False):

        self.graph = graph
        self.relaxation = relaxation

        self.prog = MathematicalProgram()
        self.vars = ShortestPathVariables.populate_program(self.prog, graph, relaxation)
        populate_constraints(self.prog, graph, self.vars)
        self.prog.AddLinearCost(sum(self.vars.l))

    def solve(self):

        from pydrake.solvers.mathematicalprogram import CommonSolverOption, SolverOptions
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintFileName, 'mosek_log.txt')

        result = MosekSolver().Solve(self.prog, solver_options=solver_options)
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time
        primal = ShortestPathVariables.from_result(result, self.vars)

        return ShortestPathSolution(cost, time, primal)
