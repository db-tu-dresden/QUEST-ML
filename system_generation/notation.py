from __future__ import annotations
import re

import regex

from system_generation import graph


class NotationElement:
    VALUES = []

    def __init__(self, value: str, next: NotationElement):
        self.value = value
        self.next = next

    @classmethod
    def represents(cls, text: str):
        values = r'|'.join(cls.VALUES)
        if not values:
            return False
        return regex.match(values, text) is not None

    @classmethod
    def parse(cls, text: str, next: NotationElement = None, notation: Notation = None):
        pass

    def add_to_graph(self, graph: graph.Graph, root: int = None):
        pass


class Line(NotationElement):
    VALUES = [r'-(\d+)?-', r'-']

    def __init__(self, value: str, next: NotationElement, count: int):
        super().__init__(value, next)
        self.count = count

    @classmethod
    def parse(cls, text: str, next: NotationElement = None, notation: Notation = None):
        MATCH = '|'.join(cls.VALUES)

        match = re.match(MATCH, text)

        if not match:
            raise Exception('Line could not be parsed!')

        count = int(match.groups()[0]) + 2 if match.groups()[0] else len(text)

        return cls(text, next, count)

    def add_to_graph(self, graph: graph.Graph, root: int = None):
        last_id = graph.join_nodes(self.count, root)
        if self.next:
            return self.next.add_to_graph(graph)
        return last_id


class Fork(NotationElement):
    VALUES = [r'<(?:-*(?:(?:-+\d*-*)?|(?R)?)-*)>']

    def __init__(self, value: str, next: NotationElement, upper: NotationElement, lower: NotationElement):
        super().__init__(value, next)
        self.upper = upper
        self.lower = lower

    @classmethod
    def parse(cls, text: str, next: NotationElement = None, notation: Notation = None):
        MATCH = r'<(.*)>'
        match = re.match(MATCH, text)

        if not match:
            raise Exception('Fork could not be parsed!')

        path = notation._parse(match.groups()[0], None)

        return cls(text, next, path, path)

    def add_to_graph(self, graph: graph.Graph, root: int = None):
        if root is None:
            root = graph.last_node_id

        last_idx_upper = graph.join_node(root)
        if self.upper:
            last_idx_upper = self.upper.add_to_graph(graph, last_idx_upper)
        last_idx_lower = graph.join_node(root)
        if self.lower:
            last_idx_lower = self.lower.add_to_graph(graph, last_idx_lower)

        graph.add_node(graph.last_node_id + 1)
        graph.last_node_id += 1
        graph.add_edge(last_idx_upper, graph.last_node_id)
        graph.add_edge(last_idx_lower, graph.last_node_id)

        if self.next:
            return self.next.add_to_graph(graph)
        return graph.last_node_id


class Notation:

    def __init__(self):
        self.start = None
        self.string = None

    def parse_string(self, string: str, next: NotationElement = None):
        if Line.represents(string):
            return Line.parse(string, next=next, notation=self)

        if Fork.represents(string):
            return Fork.parse(string, next=next, notation=self)

    def _parse(self, string: str, next: NotationElement = None):
        self.string = string
        splitter = '(' + '|'.join(Fork.VALUES + Line.VALUES) + ')'
        elements = [elem for elem in regex.split(splitter, string) if elem]

        if not elements:
            return None

        for elem in reversed(elements):
            next = self.parse_string(elem, next=next)

        self.start = next

        return next

    def parse(self, string: str):
        self._parse(string)
        return self
