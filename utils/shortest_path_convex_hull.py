import numpy as np
from pydrake.all import MathematicalProgram, MosekSolver, eq, ge

class ShortestPathVariables():

    def __init__(self, phi, y, z, l, x):

        self.phi = phi
        self.y = y
        self.z = z
        self.l = l
        self.x = x

    @staticmethod
    def populate_program(prog, graph):

        phi = prog.NewContinuousVariables(graph.n_edges)
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
    def populate_program(prog, graph, vars, relaxation=False):

        for vertex, set in graph.sets.items():

            if vertex != graph.source and vertex != graph.target:

                edges_in = graph.incoming_edges(vertex)[1]
                edges_out = graph.outgoing_edges(vertex)[1]

                # auxiliary flow variables
                if relaxation:
                    alpha = prog.NewContinuousVariables(len(edges_in), len(edges_out))
                    prog.AddLinearConstraint(ge(alpha.flatten(), 0))
                else:
                    alpha = prog.NewBinaryVariables(len(edges_in), len(edges_out))
                prog.AddLinearConstraint(alpha.sum() <= 1)

                # reconstruct flows
                for j, k in enumerate(edges_in):
                    prog.AddLinearConstraint(vars.phi[k] == alpha[j].sum())
                for j, k in enumerate(edges_out):
                    prog.AddLinearConstraint(vars.phi[k] == alpha[:, j].sum())

                # auxiliary spatial variables
                x_aux = np.array([prog.NewContinuousVariables(len(edges_out), graph.dimension) for _ in edges_in])
                for j, k in enumerate(edges_in):
                    prog.AddLinearConstraint(eq(vars.z[k], x_aux[j].sum(axis=0)))
                for j, k in enumerate(edges_out):
                    prog.AddLinearConstraint(eq(vars.y[k], x_aux[:, j].sum(axis=0)))

                # spatial constraints
                i = graph.vertex_index(vertex)
                argument = vars.x[i] - x_aux.sum(axis=0).sum(axis=0)
                scaling = 1 - alpha.sum()
                set.add_perspective_constraint(prog, scaling, argument)

                for j in range(len(edges_in)):
                    for k in range(len(edges_out)):
                        set.add_perspective_constraint(prog, alpha[j, k], x_aux[j, k])

        # source
        edges_in = graph.incoming_edges(graph.source)[1]
        edges_out = graph.outgoing_edges(graph.source)[1]

        for k in edges_in:
            prog.AddLinearConstraint(vars.phi[k] == 0)
            prog.AddLinearConstraint(eq(vars.z[k], 0))

        s = graph.vertex_index(graph.source)
        prog.AddLinearConstraint(sum(vars.phi[edges_out]) == 1)
        prog.AddLinearConstraint(eq(vars.x[s], sum(vars.y[edges_out])))
        for k in edges_out:
            graph.source_set.add_perspective_constraint(prog, vars.phi[k], vars.y[k])

        # target
        edges_in = graph.incoming_edges(graph.target)[1]
        edges_out = graph.outgoing_edges(graph.target)[1]

        for k in edges_out:
            prog.AddLinearConstraint(vars.phi[k] == 0)
            prog.AddLinearConstraint(eq(vars.y[k], 0))

        t = graph.vertex_index(graph.target)
        prog.AddLinearConstraint(sum(vars.phi[edges_in]) == 1)
        prog.AddLinearConstraint(eq(vars.x[t], sum(vars.z[edges_in])))
        for k in edges_in:
            graph.target_set.add_perspective_constraint(prog, vars.phi[k], vars.z[k])

        # cost function
        for k, edge in enumerate(graph.edges):
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

        print('using ch')

        self.graph = graph
        self.relaxation = relaxation

        self.prog = MathematicalProgram()
        self.vars = ShortestPathVariables.populate_program(self.prog, graph)
        self.constraints = ShortestPathConstraints.populate_program(self.prog, graph, self.vars, relaxation)
        self.prog.AddLinearCost(sum(self.vars.l))

    def solve(self):

        result = MosekSolver().Solve(self.prog)
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time
        primal = ShortestPathVariables.from_result(result, self.vars)

        return ShortestPathSolution(cost, time, primal)
