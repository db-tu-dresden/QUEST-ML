from __future__ import annotations

import re
from typing import Any

from notation.graph import Graph


class Parseable:
    PATTERN = r'(?!x)x'  # never matches anything

    class Exception(Exception):
        pass

    def __init__(self, value: Any, node_id: int = None):
        self.value = value
        self.node_id = node_id
        self.next = None
        self.data = None

    @classmethod
    def match(cls, string: str):
        return re.match(cls.PATTERN, string)

    @classmethod
    def _force_match(cls, string: str):
        match = re.match(cls.PATTERN, string)
        if not match:
            raise cls.Exception(f'Given string can not be parsed as {cls.__name__}. String is: {string}')
        return match

    @classmethod
    def parse(cls, string: str, notation: Notation):
        match = cls._force_match(string)
        return cls(match.group())

    def validate_data_flow(self, incoming_data: DataFlowElement):
        if self.data:
            self.data.value.update(incoming_data.value)
        else:
            self.data = DataFlowElement(set(incoming_data.value))
        if self.next:
            self.next.validate_data_flow(self.data)

    def add_to_graph(self, graph: Graph, root: int) -> [Parseable]:
        pass


class Line(Parseable):
    PATTERN = r'-(\d+)-|-+'

    class LineParseException(Exception):
        pass

    def __init__(self, value: str, count: int):
        super().__init__(value)
        self.count = count

    @classmethod
    def parse(cls, string: str, notation: Notation):
        match = cls._force_match(string)

        count = int(match.group(1)) + 2 if match.group(1) else len(string)

        return cls(string, count)

    def add_to_graph(self, graph: Graph, root: int) -> [Parseable]:
        self.node_id = graph.join_nodes(self.count, root, data=self.data.value)
        if self.next:
            return self.next.add_to_graph(graph, self.node_id)
        return [self]


class Fork(Parseable):
    PATTERN = r'<(\[[^->]*\])(>)?'

    class ForkParseException(Exception):
        pass

    def __init__(self, value: str, ref_list: ReferenceList, end: bool):
        super().__init__(value)
        self.ref_list = ref_list
        self.end = end

    @classmethod
    def parse(cls, string: str, notation: Notation):
        match = cls._force_match(string)

        ref_list = ReferenceList.parse(match.group(1), notation)
        end = match.group(2) is not None

        return cls(string, ref_list, end)

    def validate_data_flow(self, incoming_data: DataFlowElement):
        self.data = incoming_data
        self.ref_list.validate_data_flow(incoming_data)
        if self.next:
            self.next.validate_data_flow(incoming_data)

    def add_to_graph(self, graph: Graph, root: int) -> [Parseable]:
        last_nodes = self.ref_list.add_to_graph(graph, root)

        if self.end:
            graph.last_node_id += 1
            end_id = graph.last_node_id
            self.node_id = end_id
            graph.add_node(end_id, data=self.data.value)

            for node in last_nodes:
                node_id = node.node_id
                graph.add_edge(node_id, end_id, data=graph.nodes(data=True)[node_id]['data'])
            last_nodes = [self]
            if self.next:
                return self.next.add_to_graph(graph, end_id)
        elif self.next:
            raise Exception('Fork not closed but next element specified.')
        return last_nodes


class ReferenceList(Parseable):
    PATTERN = r'\[((?:[^->]+,\s?)+[^->]+)\]'

    class ReferenceListParseException(Exception):
        pass

    def __init__(self, value: str, refs: [Reference]):
        super().__init__(value)
        self.refs = refs

    @classmethod
    def parse(cls, string: str, notation: Notation):
        match = cls._force_match(string)

        strings = [string[m.start():m.end()] for m in re.finditer(Reference.PATTERN, string)]
        return cls(string, [Reference.parse(string, notation) for string in strings])

    def add_to_graph(self, graph: Graph, root: int) -> [Parseable]:
        last_nodes = []
        for ref in self.refs:
            node = graph.join_node(root, data=ref.data.value)
            last_nodes.extend(ref.add_to_graph(graph, node))

        return list(set(last_nodes))

    def validate_data_flow(self, incoming_data: DataFlowElement):
        self.data = incoming_data

        refs_data = DataFlowElement.from_exclusives(*(ref.data for ref in self.refs))

        if self.data != refs_data:
            raise Exception(f'DataFlowException: Data flow not consistent, element(s) added or removed. '
                            f'Incoming elements: {self.data}; '
                            f'Outgoing elements: {refs_data}')

        for ref in self.refs:
            ref.validate_data_flow(self.data)


class Reference(Parseable):
    PATTERN = r'\((\$\d+): ?(\[(?:\w+(?:, ?)?)*\])\)'

    class ReferenceParseException(Exception):
        pass

    def __init__(self, value: str, data: DataFlowElement):
        super().__init__(value)
        self.data = data

    @classmethod
    def parse(cls, string: str, notation: Notation):
        match = cls._force_match(string)

        value = match.group(1)
        data = DataFlowElement.parse(match.group(2), notation)

        return cls(value, data)

    def validate_data_flow(self, incoming_data: DataFlowElement):
        if self.next:
            self.next.validate_data_flow(self.data)

    def add_to_graph(self, graph: Graph, root: int) -> [Parseable]:
        if self.next:
            return self.next.add_to_graph(graph, root)
        return [Parseable(None, node_id=root)]


class ReferenceDefinition(Parseable):
    PATTERN = r'(\$\d+):(.*)'

    class ReferenceDefinitionParseException(Exception):
        pass

    def __init__(self, value: str, key: str, seq: Sequence):
        super().__init__(value)
        self.key = key
        self.seq = seq

    @classmethod
    def parse(cls, string: str, notation: Notation):
        match = cls._force_match(string)

        key = match.group(1)
        seq = Sequence.parse(match.group(2).strip(), notation)

        return cls(string, key, seq)


class Anchor(Parseable):
    PATTERN = r'(!\d+)'

    class AnchorParseException(Exception):
        pass

    @classmethod
    def parse(cls, string: str, notation: Notation):
        match = cls._force_match(string)
        anchor = next((anchor for anchor in notation.anchors if anchor.value == match.group()), None)
        if anchor:
            return anchor
        anchor = cls(match.group())
        notation.anchors.update([anchor])
        return anchor

    def add_to_graph(self, graph: Graph, root: int):
        edges = []
        node_data = set()
        root_data = graph.nodes[root]['data']

        old_node_id = graph.get_node_by_id(self.value)
        if old_node_id:
            node_data.update(graph.nodes[old_node_id]['data'])
            edges = list(graph.in_edges(old_node_id, data=True))
            graph.remove_edges_from(edges)
            graph.remove_node(old_node_id)

        graph.last_node_id += 1
        self.node_id = graph.last_node_id
        graph.add_node(self.node_id, id=self.value, data=node_data.union(root_data))
        graph.add_edges_from([(n, self.node_id, data) for n, _, data in edges])
        graph.add_edge(root, self.node_id, data=root_data)
        if self.next:
            return self.next.add_to_graph(graph, self.node_id)
        return [self]


class Sequence(Parseable):
    PATTERN = r'.*'
    elements = [Line, Fork, Reference, Anchor]

    class Exception(Exception):
        pass

    def __init__(self, value, next: Parseable):
        super().__init__(value)
        self.next = next

    @classmethod
    def parse(cls, string: str, notation: Notation):
        original_string = string
        first = None
        last = None

        while string:
            did_match = False
            for parseable in cls.elements:
                match = parseable.match(string)
                if match:
                    did_match = True
                    elem = string[:match.end()]
                    string = string[match.end():]
                    elem = parseable.parse(elem, notation)

                    if not first:
                        first = elem
                    if last:
                        last.next = elem

                    last = elem
                    break
            if not did_match:
                raise cls.Exception(
                    f'Given string can not be parsed. String is: {string}')

        return cls(original_string, first)

    def add_to_graph(self, graph: Graph, root: int):
        if not self.next:
            return
        self.next.add_to_graph(graph, root)


class DataFlowElement(Parseable):
    PATTERN = r'\[(?:\w+(?:, ?)?)*\]'

    @classmethod
    def parse(cls, string: str, notation: Notation):
        match = cls._force_match(string)
        return cls(set(re.findall(r'\w+', string)))

    @classmethod
    def from_exclusives(cls, *ls):
        value = set()
        for elem in ls:
            if not isinstance(elem, DataFlowElement):
                raise Exception(f'DataFlowException: Can not create exclusive DataFlowElement instance '
                                f'from list containing non DataFlowElements')
            if value.intersection(elem.value):
                raise Exception(f'DataFlowException: Can not create exclusive DataFlowElement instance from list. '
                                f'Same element found multiple times.'
                                f'Existing elements: {value}; '
                                f'Tried joining elements: {elem.value}')
            value = value.union(elem.value)
        return cls(value)

    def __eq__(self, other):
        return self.value == other.value


class Notation:
    def __init__(self):
        self.elements = [Line, Fork, Reference, Anchor]
        self.refs = {}
        self.anchors = set()
        self.seq = None
        self.data = None
        self.graph = None

    def _resolve_ref(self, seq: Sequence):
        elem = seq.next

        while elem:
            if isinstance(elem, Fork):
                for ref in elem.ref_list.refs:
                    if not ref.next:
                        ref.next = self.refs[ref.value].seq.next
            elem = elem.next

    def _resolve_refs(self):
        for ref in self.refs.values():
            self._resolve_ref(ref.seq)

        self._resolve_ref(self.seq)

    def validate_data_flow(self):
        if self.data:
            self.seq.validate_data_flow(self.data)

    @classmethod
    def parse(cls, string: str):
        self = cls()
        data_str, string, *ref_defs = [elem for elem in string.split('\n') if elem]

        self.data = DataFlowElement.parse(data_str, self)
        self.seq = Sequence.parse(string, self)

        for ref_def in ref_defs:
            ref = ReferenceDefinition.parse(ref_def, self)
            self.refs.update({ref.key: ref})

        self._resolve_refs()

        self.validate_data_flow()

        self.generate_graph()

        return self

    def generate_graph(self):
        if self.seq is None:
            return
        self.graph = Graph()
        root = self.graph.join_node(data=self.data.value)
        self.seq.add_to_graph(self.graph, root)

    def draw(self, path: str = None, show: bool = True):
        self.graph.draw(path=path, show=show)
