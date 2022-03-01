import numpy as np

class ConvexFunction():
    """Parent class for all the convex functions."""

    def __call__(self, x):

        if self.D is not None and not self.D.contains(x):
            return np.inf
        else:
            return self._evaluate(x)
            
    def add_perspective_constraint(self, prog, slack, scale, x):

        cost = self._add_perspective_constraint(prog, slack, scale, x)
        domain = self.enforce_domain(prog, scale, x)

        return cost, domain

    def enforce_domain(self, prog, scale, x):
        if self.D is not None:
            return self.D.add_perspective_constraint(prog, scale, x)

class Constant(ConvexFunction):
    """Function of the form c for x in D, where D is a ConvexSet."""

    def __init__(self, c, D=None):

        self.c = c
        self.D = D

    def _evaluate(self, x):

        return self.c

    def _add_perspective_constraint(self, prog, slack, scale, x):

        return prog.AddLinearConstraint(slack >= self.c * scale)

class TwoNorm(ConvexFunction):
    """Function of the form ||H x||_2 for x in D, where D is a ConvexSet."""

    def __init__(self, H, D=None):

        self.H = H
        self.D = D

    def _evaluate(self, x):

        return np.linalg.norm(self.H.dot(x))

    def _add_perspective_constraint(self, prog, slack, scale, x):

        Hx = self.H.dot(x)
        return prog.AddLorentzConeConstraint(slack, Hx.dot(Hx))

class SquaredTwoNorm(ConvexFunction):
    """Function of the form ||H x||_2^2 for x in D, where D is a ConvexSet."""

    def __init__(self, H, D=None):

        self.H = H
        self.D = D

    def _evaluate(self, x):

        Hx = self.H.dot(x)
        return Hx.dot(Hx)

    def _add_perspective_constraint(self, prog, slack, scale, x):

        Hx = self.H.dot(x)
        return prog.AddRotatedLorentzConeConstraint(slack, scale, Hx.dot(Hx))
