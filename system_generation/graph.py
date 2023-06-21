from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from system_generation import notation


class Graph:

    def __init__(self, last_node_id=None):
        self.graph = nx.Graph()
        self.root_id = None
        self.last_node_id = last_node_id

    def add_node(self, node_for_adding, **attr):
        self.graph.add_node(node_for_adding, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        self.graph.add_nodes_from(nodes_for_adding, **attr)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self.graph.add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        self.graph.add_edges_from(ebunch_to_add, **attr)

    def join_node(self, root: int = None):
        if root is None:
            root = self.last_node_id

        id = self.last_node_id + 1 if self.last_node_id is not None else 0
        self.add_node(id)

        if self.root_id is None:
            self.root_id = id

        self.last_node_id = id

        if root:
            self.add_edge(root, id)

        return self.last_node_id

    def join_nodes(self, n: int, root: int = None):
        if root is None:
            root = self.last_node_id

        base = self.last_node_id + 1 if self.last_node_id is not None else 0
        ids = list(range(base, base + n))

        self.add_nodes_from(ids)

        if self.root_id is None:
            self.root_id = ids[0]

        if root is not None:
            ids = [root] + ids
        self.add_edges_from([(ids[idx], ids[idx + 1]) for idx in range(len(ids) - 1)])

        self.last_node_id = ids[-1]

        return self.last_node_id

    def from_notation(self, notation: notation.Notation):
        self.join_node()

        elem = notation.start
        if elem is None:
            return
        elem.add_to_graph(self)

        return self

    def draw(self):
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()


if __name__ == '__main__':
    text = '-<->-'
    notation = notation.Notation().parse(text)
    graph = Graph().from_notation(notation)
    graph.draw()
