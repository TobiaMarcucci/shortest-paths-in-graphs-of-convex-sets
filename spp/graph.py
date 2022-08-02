import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from spp.convex_sets import ConvexSet
from spp.convex_functions import ConvexFunction, Constant
from graphviz import Digraph

class GraphOfConvexSets():

    def __init__(self):

        self.sets = {}
        self.lengths = {}

        self._source = None
        self._target = None
        
    def add_set(self, set, vertex=None):

        assert isinstance(set, ConvexSet)
        if vertex is None:
            vertex = len(self.sets)

        assert vertex not in self.sets
        self.sets[vertex] = set

        return vertex

    def add_sets(self, sets, vertices=None):

        if vertices is None:
            vertices = [None] * len(sets)
        else:
            assert len(sets) == len(vertices)

        for i, set in enumerate(sets):
            vertices[i] = self.add_set(set, vertices[i])

        return vertices

    def remove_set(self, vertex):

        self.sets.pop(vertex)
        for edge in self.edges:
            if vertex in edge:
                self.remove_edge(edge)
        
    def add_edge(self, u, v, length=None):

        if length is None:
            length = Constant(0)

        assert u in self.sets and v in self.sets
        assert isinstance(length, ConvexFunction)
        self.lengths[(u, v)] = length

    def add_edges(self, us, vs, lengths=None):

        assert len(us) == len(vs)
        if lengths is None:
            lengths = [None] * len(us)
        else:
            assert len(us) == len(lengths)

        for u, v, length in zip(us, vs, lengths):
            self.add_edge(u, v, length)

    def remove_edge(self, edge):

        self.lengths.pop(edge)

    def set_source(self, vertex):

        assert vertex in self.sets
        self._source = vertex

    def set_target(self, vertex):

        assert vertex in self.sets
        self._target = vertex

    def self_transition(self, vertex, self_length, vertex_copy=None):

        vertex_copy = self.add_set(self.sets[vertex], vertex_copy)
        self.add_edge(vertex_copy, vertex, self_length)

        incomings = [edge for edge in self.edges if edge[1] == vertex]
        for edge in incomings:
            self.add_edge(edge[0], vertex_copy, self.lengths[edge])

        return vertex_copy

    def double_visit(self, vertex, self_length, vertex_copy=None):
        vertex_copy = self.self_transition(vertex, self_length, vertex_copy)
        for edge, length in self.lengths.items():
            if e[0] == vertex:
                self.add_edge(vertex_copy, e[1], length)
        return vertex_copy

    def set_edge_length(self, edge, length):
        self.lengths[edge] = length

    def edge_index(self, edge):
        return self.edges.index(edge)

    def edge_indices(self, edges):
        return [self.edges.index(edge) for edge in edges]

    def vertex_index(self, vertex):
        return self.vertices.index(vertex)

    def vertex_indices(self, vertices):
        return [self.vertices.index(vertex) for vertex in vertices]

    def incoming_edges(self, vertex):
        assert vertex in self.vertices
        edges = [edge for edge in self.edges if edge[1] == vertex]
        return edges, self.edge_indices(edges)

    def outgoing_edges(self, vertex):
        assert vertex in self.vertices
        edges = [edge for edge in self.edges if edge[0] == vertex]
        return edges, self.edge_indices(edges)

    def incident_edges(self, vertex):
        incomings = self.incoming_edges(vertex)
        outgoings = self.outgoing_edges(vertex)
        return [i + o for i, o in zip(incomings, outgoings)]

    def scale(self, s):
        for convex_set in set(self.sets.values()):
            convex_set.scale(s)

    def draw_sets(self, **kwargs):
        plt.rc('axes', axisbelow=True)
        plt.gca().set_aspect('equal')
        for set in self.sets.values():
            set.plot(**kwargs)

    def draw_edges(self, **kwargs):
        options = {'color':'k', 'zorder':2,
            'arrowstyle':'->, head_width=3, head_length=8'}
        options.update(kwargs)
        for edge in self.edges:
            tail = self.sets[edge[0]].center
            head = self.sets[edge[1]].center
            arrow = patches.FancyArrowPatch(tail, head, **options)
            plt.gca().add_patch(arrow)

    def label_sets(self, labels=None, **kwargs):

        options = {'c':'b'}
        options.update(kwargs)
        if labels is None:
            labels = self.vertices

        for set, label in zip(self.sets.values(), labels):
            plt.text(*set.center, label, **options)

    def label_edges(self, labels, **kwargs):
        options = {'c':'r', 'va':'top'}
        options.update(kwargs)
        for edge, label in zip(self.edges, labels):
            center = (self.sets[edge[0]].center + self.sets[edge[1]].center) / 2
            plt.text(*center, label, **options)

    def draw_vertices(self, x):
        options = {'marker':'o', 'facecolor':'w', 'edgecolor':'k', 'zorder':3}
        options.update(kwargs)
        plt.scatter(*x.T, **options)

    def draw_path(self, phis, x, **kwargs):
        options = {'color':'g', 'marker': 'o', 'markeredgecolor': 'k', 'markerfacecolor': 'w'}
        options.update(kwargs)
        for k, phi in enumerate(phis):
            if phi > 1 - 1e-3:
                edge = [self.vertices.index(vertex) for vertex in self.edges[k]]
                plt.plot(*x[edge].T, **options)

    def graphviz(self, vertex_labels=None, edge_labels=None):

        if vertex_labels is None:
            vertex_labels = self.vertices
        if edge_labels is None:
            edge_labels = [''] * len(self.edges)

        G = Digraph()
        for label in vertex_labels:
            G.node(str(label))
        for k, edge in enumerate(self.edges):
            u = vertex_labels[self.vertices.index(edge[0])]
            v = vertex_labels[self.vertices.index(edge[1])]
            G.edge(str(u), str(v), str(edge_labels[k]))
        return G

    @property
    def vertices(self):
        return list(self.sets.keys())

    @property
    def edges(self):
        return list(self.lengths.keys())

    @property
    def source(self):
        return self.sets[self.vertices[0]] if self._source is None else self._source

    @property
    def target(self):
        return self.sets[self.vertices[-1]] if self._target is None else self._target

    @property
    def source_set(self):
        return self.sets[self.source]

    @property
    def target_set(self):
        return self.sets[self.target]

    @property
    def dimension(self):
        assert len(set(S.dimension for S in self.sets.values())) == 1
        return self.sets[self.vertices[0]].dimension

    @property
    def n_sets(self):
        return len(self.sets)

    @property
    def n_edges(self):
        return len(self.lengths)
