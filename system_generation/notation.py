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
    VALUES = [r'<\[(?:\$\d+(?:, \$\d+)+)\]>', r'<(?:-*(?:(?:-+\d*-*)?|(?R)?)-*)>']

    def __init__(self, value: str, next: NotationElement, children: [NotationElement]):
        super().__init__(value, next)
        self.children = children

    @classmethod
    def parse(cls, text: str, next: NotationElement = None, notation: Notation = None):
        FORK = r'<(.*)>'
        match = re.match(FORK, text)

        if not match:
            raise Exception('Fork could not be parsed!')

        group = match.groups()[0]

        COMPLEX_PATH = r'\[(?:\$\d+(?:, \$\d+)+)\]'
        complex_fork = re.match(COMPLEX_PATH, group)

        if complex_fork:
            REF_PATTERN = r'\$\d+'
            refs = re.findall(REF_PATTERN, group)
            notation.refs |= dict.fromkeys(refs)
            return cls(text, next, refs)

        path = notation._parse(group, None) if group else None
        return cls(text, next, [path, path])

    def add_to_graph(self, graph: graph.Graph, root: int = None):
        if root is None:
            root = graph.last_node_id

        last_child_ids = []
        for child in self.children:
            child_id = graph.join_node(root)
            if child:
                child_id = child.add_to_graph(graph, child_id)
            last_child_ids.append(child_id)

        graph.add_node(graph.last_node_id + 1)
        graph.last_node_id += 1

        for child_id in last_child_ids:
            graph.add_edge(child_id, graph.last_node_id)

        if self.next:
            return self.next.add_to_graph(graph)
        return graph.last_node_id


class Notation:

    def __init__(self):
        self.start = None
        self.string = None
        self.refs = {}

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

    def _parse_ref(self, string: str):
        REF_PATTERN = r'^(\$\d+):(.*)'
        match = re.match(REF_PATTERN, string)

        if not match:
            return

        ref = match.groups()[0]
        string = match.groups()[1].strip()

        self.refs[ref] = Notation().parse(string)

    def _resolve_refs(self, refs: {} = {}):
        self.refs |= refs
        if not self.refs:
            return
        elem = self.start

        while elem:
            if isinstance(elem, Fork):
                for i, child in enumerate(elem.children):
                    if isinstance(child, str):
                        self.refs[child]._resolve_refs(self.refs)
                        elem.children[i] = self.refs[child].start
            elem = elem.next

    def parse(self, string: str):
        base, *ref_strs = string.split('\n')
        self._parse(base)

        for ref_str in ref_strs:
            self._parse_ref(ref_str)

        if any(1 for v in self.refs.values() if v is None):
            raise Exception('Path referenced but never defined')

        self._resolve_refs()

        return self
