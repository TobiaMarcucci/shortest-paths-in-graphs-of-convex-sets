import numpy as np
from pydrake.all import MathematicalProgram, MosekSolver, GurobiSolver, eq, ge
from pydrake.solvers.mathematicalprogram import CommonSolverOption, SolverOptions

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

    def __init__(self, cons, sp_cons, sp_y, sp_z, obj=None):

        self.conservation = cons
        self.spatial_conservation = sp_cons
        self.objective = obj
        self.spatial_y = sp_y
        self.spatial_z = sp_z

    @staticmethod
    def populate_program(prog, graph, vars):

        # containers for the constraints we want to keep track of
        cons = []
        sp_cons = []
        sp_y = []
        sp_z = []

        prog.AddLinearConstraint(ge(vars.phi, 0))

        for vertex, set in graph.sets.items():

            edges_in = graph.incoming_edges(vertex)[1]
            edges_out = graph.outgoing_edges(vertex)[1]

            phi_in = sum(vars.phi[edges_in])
            phi_out = sum(vars.phi[edges_out])

            delta_sv = 1 if vertex == graph.source else 0

            # conservation of flow
            if len(edges_in) > 0 or len(edges_out) > 0:
                if vertex != graph.target:
                    residual = phi_out - phi_in - delta_sv
                    cons.append(prog.AddLinearConstraint(residual == 0))

                # spatial conservation of flow
                if vertex not in (graph.source, graph.target):
                    y_out = sum(vars.y[edges_out])
                    z_in = sum(vars.z[edges_in])
                    residual = y_out - z_in
                    sp_cons.append(prog.AddLinearConstraint(eq(residual, 0)))

        # spatial nonnegativity
        for k, edge in enumerate(graph.edges):

            yu = prog.NewContinuousVariables(graph.dimension)
            phiu = prog.NewContinuousVariables(1)[0]
            sp_y.append(
                prog.AddLinearConstraint(eq(
                    np.append(vars.y[k], vars.phi[k]),
                    np.append(yu, phiu)
                    ))
                )
            graph.sets[edge[0]].add_perspective_constraint(prog, phiu, yu)

            zv = prog.NewContinuousVariables(graph.dimension)
            phiv = prog.NewContinuousVariables(1)[0]
            sp_z.append(
                prog.AddLinearConstraint(eq(
                    np.append(vars.z[k], vars.phi[k]),
                    np.append(zv, phiv)
                    ))
                )
            graph.sets[edge[1]].add_perspective_constraint(prog, phiv, zv)

            # slack constraints for the objetive (not stored)
            yz = np.concatenate((vars.y[k], vars.z[k]))
            graph.lengths[edge].add_perspective_constraint(prog, vars.l[k], vars.phi[k], yz)

        return ShortestPathConstraints(cons, sp_cons, sp_y, sp_z)

    @staticmethod
    def from_result(result, constraints):

        def get_dual(result, constraints):
            dual = np.array([result.GetDualSolution(c) for c in constraints])
            if dual.shape[1] == 1:
                return dual.flatten()
            return dual

        cons = get_dual(result, constraints.conservation)
        np.concatenate([cons,[0]])
        sp_cons = get_dual(result, constraints.spatial_conservation)
        sp_y = get_dual(result, constraints.spatial_y)
        sp_z = get_dual(result, constraints.spatial_z)
        obj = cons[0]

        return ShortestPathConstraints(cons, sp_cons, sp_y, sp_z, obj)

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

        # import mosek
        
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)

        solver = MosekSolver()
        # self.prog.SetSolverOption(solver.solver_id(), 'MSK_IPAR_INTPNT_SOLVE_FORM', mosek.solveform.primal)
        # self.prog.SetSolverOption(solver.solver_id(), 'intpntCoTolDfeas', 1.0e-8)
        self.prog.SetSolverOption(solver.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-12)
        self.prog.SetSolverOption(solver.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_INFEAS", 1e-12)
        self.prog.SetSolverOption(solver.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-12)
        self.prog.SetSolverOption(solver.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-12)
        self.prog.SetSolverOption(solver.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-12)
        result = solver.Solve(self.prog, solver_options=options)

        # solver = GurobiSolver()
        # self.prog.SetSolverOption(solver.solver_id(), 'QCPDual', 1)
        # result = solver.Solve(self.prog, solver_options=options)

        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time
        primal = ShortestPathVariables.from_result(result, self.vars)
        primal.reconstruct_x(self.graph)
        dual = ShortestPathConstraints.from_result(result, self.constraints) if self.relaxation else None

        return ShortestPathSolution(cost, time, primal, dual)
