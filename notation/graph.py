from functools import cached_property

import matplotlib.pyplot as plt
import networkx as nx


class Graph:

    def __init__(self, last_node_id=None):
        self.graph = nx.DiGraph()
        self.root_id = None
        self.last_node_id = last_node_id
        self.anchor_nodes = {}

    @cached_property
    def nodes(self, **kwargs):
        return self.graph.nodes(**kwargs)

    @cached_property
    def edges(self, *args, **kwargs):
        return self.graph.edges(*args, **kwargs)

    def add_node(self, node_for_adding, **attr):
        self.graph.add_node(node_for_adding, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        self.graph.add_nodes_from(nodes_for_adding, **attr)

    def remove_node(self, n):
        self.graph.remove_node(n)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self.graph.add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        self.graph.add_edges_from(ebunch_to_add, **attr)

    def join_node(self, root: int = None, data=None):
        if root is None:
            root = self.last_node_id

        id = self.last_node_id + 1 if self.last_node_id is not None else 0
        self.add_node(id)

        if self.root_id is None:
            self.root_id = id

        self.last_node_id = id

        if root is not None:
            self.add_edge(root, id, data=data)

        return self.last_node_id

    def join_nodes(self, n: int, root: int = None, data=None):
        if root is None:
            root = self.last_node_id

        base = self.last_node_id + 1 if self.last_node_id is not None else 0
        ids = list(range(base, base + n))

        self.add_nodes_from(ids)

        if self.root_id is None:
            self.root_id = ids[0]

        if root is not None:
            ids = [root] + ids
        self.add_edges_from([(ids[idx], ids[idx + 1]) for idx in range(len(ids) - 1)], data=data)

        self.last_node_id = ids[-1]

        return self.last_node_id

    def get_node_by_id(self, id):
        return next(
            (node for node, data in self.graph.nodes(data=True) if data.get('id') == id),
            None)

    def draw(self):
        pos = nx.spring_layout(self.graph)
        plt.figure()
        nx.draw(
            self.graph, pos,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=500,
            node_color=['lightblue', *['pink'] * (self.last_node_id - 1), 'lightblue'],
            alpha=0.9,
            with_labels=True
        )

        edge_labels = {(e[0], e[1]): str(e[2]['data'])
                       for e in self.graph.edges(data=True) if e[2] and e[2]['data']}

        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_color='red'
        )
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    pass
