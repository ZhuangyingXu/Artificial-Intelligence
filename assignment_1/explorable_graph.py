# coding=utf-8
from networkx import Graph

class ExplorableGraph(object):
    """
    Keeps track of "explored nodes" i.e. nodes that have been queried from the
    graph.

    Delegates graph operations to a networkx.Graph
    """

    def __init__(self, graph):
        """
        :type graph: Graph
        """
        self.__graph = graph
        self._explored_nodes = dict([(node, 0) for node in self.__graph.nodes()])

    def explored_nodes(self):
        return self._explored_nodes

    def __getattr__(self, item):
        return getattr(self.__graph, item)

    def reset_search(self):
        self._explored_nodes = dict([(node, 0) for node in self.__graph.nodes()])

    def __iter__(self):
        return self.__graph.__iter__()

    def __getitem__(self, n):
        #self._explored_nodes |= {n}
        if n in self.__graph.nodes():
            self._explored_nodes[n] += 1
        return self.__graph.__getitem__(n)

    def nodes_iter(self, data=False):
        self._explored_nodes = set(self.__graph.nodes_iter())
        return self.__graph.nodes_iter(data)

    def neighbors(self, n):
        if n in self.__graph.nodes():
            self._explored_nodes[n] += 1
        return self.__graph.neighbors(n)
    
    def get_edge_weight(self, u, v):
        return self.__graph.get_edge_data(u, v)['weight']
