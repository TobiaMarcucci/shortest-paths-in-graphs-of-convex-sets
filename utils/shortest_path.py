import numpy as np
from pydrake.all import MathematicalProgram, MosekSolver, eq

class ShortestPathVariables():

    def __init__(self, phi, y, z, l, x=None):

        self.phi = phi
        self.y = y
        self.z = z
        self.l = l
        self.x = x

    def reconstruct_x(self, graph):

        self.x = np.zeros((graph.n_sets, graph.dimension))
        for i, vertex in enumerate(graph.sets):

            if vertex == graph.target:
                edges_in = graph.incoming_edges(vertex)[1]
                self.x[i] = sum(self.z[edges_in])

            else:
                edges_out = graph.outgoing_edges(vertex)[1]
                self.x[i] = sum(self.y[edges_out])

                if vertex != graph.source:
                    center = graph.sets[vertex].center
                    self.x[i] += (1 - sum(self.phi[edges_out])) * center

    @staticmethod
    def populate_program(prog, graph, relaxation=False):

        phi_type = prog.NewContinuousVariables if relaxation else prog.NewBinaryVariables
        phi = phi_type(graph.n_edges)
        y = prog.NewContinuousVariables(graph.n_edges, graph.dimension)
        z = prog.NewContinuousVariables(graph.n_edges, graph.dimension)
        l = prog.NewContinuousVariables(graph.n_edges)

        return ShortestPathVariables(phi, y, z, l)

    @staticmethod
    def from_result(result, vars):

        phi = result.GetSolution(vars.phi)
        y = result.GetSolution(vars.y)
        z = result.GetSolution(vars.z)
        l = result.GetSolution(vars.l)

        return ShortestPathVariables(phi, y, z, l)

class ShortestPathConstraints():

    def __init__(self, cons, deg, sp_cons, obj=None):

        # not all constraints of the spp are stored here
        # only the ones we care of (the linear ones)
        self.conservation = cons
        self.degree = deg
        self.spatial_conservation = sp_cons
        self.objective = obj

    @staticmethod
    def populate_program(prog, graph, vars):

        # containers for the constraints we want to keep track of
        cons = []
        deg = []
        sp_cons = []

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
                cons.append(prog.AddLinearConstraint(residual == 0))

                # spatial conservation of flow
                if vertex not in (graph.source, graph.target):
                    y_out = sum(vars.y[edges_out])
                    z_in = sum(vars.z[edges_in])
                    residual = y_out - z_in
                    sp_cons.append(prog.AddLinearConstraint(eq(residual, 0)))

            # degree constraints
            if len(edges_out) > 0:
                residual = phi_out + delta_tv - 1
                deg.append(prog.AddLinearConstraint(residual <= 0))

        # spatial nonnegativity (not stored)
        for k, edge in enumerate(graph.edges):
            graph.sets[edge[0]].add_perspective_constraint(prog, vars.phi[k], vars.y[k])
            graph.sets[edge[1]].add_perspective_constraint(prog, vars.phi[k], vars.z[k])

            # slack constraints for the objetive (not stored)
            yz = np.concatenate((vars.y[k], vars.z[k]))
            graph.lengths[edge].add_perspective_constraint(prog, vars.l[k], vars.phi[k], yz)

        return ShortestPathConstraints(cons, deg, sp_cons)

    @staticmethod
    def from_result(result, constraints):

        def get_dual(result, constraints):
            dual = np.array([result.GetDualSolution(c) for c in constraints])
            if dual.shape[1] == 1:
                return dual.flatten()
            return dual

        cons = get_dual(result, constraints.conservation)
        deg = get_dual(result, constraints.degree)
        sp_cons = get_dual(result, constraints.spatial_conservation)
        obj = cons[0] - cons[-1] + sum(deg[:-1])

        return ShortestPathConstraints(cons, deg, sp_cons, obj)

class ShortestPathSolution():

    def __init__(self, cost, time, primal, dual):

        self.cost = cost
        self.time = time
        self.primal = primal
        self.dual = dual

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
        primal.reconstruct_x(self.graph)
        dual = ShortestPathConstraints.from_result(result, self.constraints) if self.relaxation else None

        return ShortestPathSolution(cost, time, primal, dual)
