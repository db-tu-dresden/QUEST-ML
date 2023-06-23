from __future__ import annotations

import re

from system_generation import graph


class Parseable:
    PATTERN = r'(?!x)x'  # never matches anything

    class Exception(Exception):
        pass

    def __init__(self, value: str):
        self.value = value
        self.next = None
        self.data = None

    @classmethod
    def match(cls, string: str):
        return re.match(cls.PATTERN, string)

    @classmethod
    def _force_match(cls, string: str):
        match = re.match(cls.PATTERN, string)
        if not match:
            raise cls.Exception(f'Given string can not be parsed as Reference element. String is: {string}')
        return match

    @classmethod
    def parse(cls, string: str):
        match = cls._force_match(string)
        return cls(match.group())

    def validate_data_flow(self, incoming_data: DataFlowElement):
        self.data = incoming_data
        if self.next:
            self.next.validate_data_flow(self.data)

    def add_to_graph(self, graph: graph.Graph, root: int) -> [int]:
        pass


class Line(Parseable):
    PATTERN = r'-(\d+)-|-+'

    class LineParseException(Exception):
        pass

    def __init__(self, value: str, count: int):
        super().__init__(value)
        self.count = count

    @classmethod
    def parse(cls, string: str):
        match = cls._force_match(string)

        count = int(match.group(1)) + 2 if match.group(1) else len(string)

        return cls(string, count)

    def add_to_graph(self, graph: graph.Graph, root: int) -> [int]:
        last_id = graph.join_nodes(self.count, root, data=self.data)
        if self.next:
            return self.next.add_to_graph(graph, last_id)
        return [last_id]


class Fork(Parseable):
    PATTERN = r'<(\[.*\])(>)?'

    class ForkParseException(Exception):
        pass

    def __init__(self, value: str, ref_list: ReferenceList, end: bool):
        super().__init__(value)
        self.ref_list = ref_list
        self.end = end

    @classmethod
    def parse(cls, string: str):
        match = cls._force_match(string)

        ref_list = ReferenceList.parse(match.group(1))
        end = match.group(2) is not None

        return cls(string, ref_list, end)

    def validate_data_flow(self, incoming_data: DataFlowElement):
        self.data = incoming_data
        self.ref_list.validate_data_flow(incoming_data)
        if self.next:
            self.next.validate_data_flow(incoming_data)

    def add_to_graph(self, graph: graph.Graph, root: int) -> [int]:
        last_ids = self.ref_list.add_to_graph(graph, root)

        if self.end:
            graph.last_node_id += 1
            end_id = graph.last_node_id
            graph.add_node(end_id)

            for ref_id in last_ids:
                graph.add_edge(ref_id, end_id)
            last_ids = [end_id]
            if self.next:
                return self.next.add_to_graph(graph, end_id)
        elif self.next:
            raise Exception('Fork not closed but next element specified.')
        return last_ids


class ReferenceList(Parseable):
    PATTERN = r'\[((?:.+,\s?)+.+)\]'

    class ReferenceListParseException(Exception):
        pass

    def __init__(self, value: str, refs: [Reference]):
        super().__init__(value)
        self.refs = refs

    @classmethod
    def parse(cls, string: str):
        match = cls._force_match(string)

        strings = [string[m.start():m.end()] for m in re.finditer(Reference.PATTERN, string)]
        return cls(string, [Reference.parse(string) for string in strings])

    def add_to_graph(self, graph: graph.Graph, root: int) -> [int]:
        last_ids = []
        for ref in self.refs:
            node = graph.join_node(root, data=self.data)
            last_ids.extend(ref.add_to_graph(graph, node))

        return set(last_ids)

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
    def parse(cls, string: str):
        match = cls._force_match(string)

        value = match.group(1)
        data = DataFlowElement.parse(match.group(2))

        return cls(value, data)

    def validate_data_flow(self, incoming_data: DataFlowElement):
        if self.next:
            self.next.validate_data_flow(self.data)

    def add_to_graph(self, graph: graph.Graph, root: int) -> [int]:
        if self.next:
            return self.next.add_to_graph(graph, root)
        return []


class ReferenceDefinition(Parseable):
    PATTERN = r'(\$\d+):(.*)'

    class ReferenceDefinitionParseException(Exception):
        pass

    def __init__(self, value: str, key: str, seq: Sequence):
        super().__init__(value)
        self.key = key
        self.seq = seq

    @classmethod
    def parse(cls, string: str):
        match = cls._force_match(string)

        key = match.group(1)
        seq = Sequence.parse(match.group(2).strip())

        return cls(string, key, seq)


class Anchor(Parseable):
    PATTERN = r'(!\d+)'

    class AnchorParseException(Exception):
        pass

    def add_to_graph(self, graph: graph.Graph, root: int):
        node_id = graph.get_node_by_id(self.value)
        if not node_id:
            graph.last_node_id += 1
            node_id = graph.last_node_id
            graph.add_node(node_id, id=self.value)
        graph.add_edge(root, node_id, data=self.data)
        if self.next:
            return self.next.add_to_graph(graph, node_id)
        return [node_id]


class Sequence(Parseable):
    PATTERN = r'.*'
    elements = [Line, Fork, Reference, Anchor]

    class Exception(Exception):
        pass

    def __init__(self, value, next: Parseable):
        super().__init__(value)
        self.next = next

    @classmethod
    def parse(cls, string: str):
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
                    elem = parseable.parse(elem)

                    if not first:
                        first = elem
                    if last:
                        last.next = elem

                    last = elem
                    break
            if not did_match:
                raise cls.Exception(
                    f'Given string can not be parsed as ElementSequenceException. String is: {string}')

        return cls(original_string, first)

    def add_to_graph(self, graph: graph.Graph, root: int):
        if not self.next:
            return
        self.next.add_to_graph(graph, root)


class DataFlowElement(Parseable):
    PATTERN = r'\[(?:\w+(?:, ?)?)*\]'

    def __init__(self, value, data: {str}):
        super().__init__(value)
        self.data = data

    @classmethod
    def parse(cls, string: str):
        match = cls._force_match(string)
        return cls(string, set(re.findall(r'\w+', string)))

    @classmethod
    def from_exclusives(cls, *ls):
        value = ''
        data = set()
        for elem in ls:
            if not isinstance(elem, DataFlowElement):
                raise Exception(f'DataFlowException: Can not create exclusive DataFlowElement instance '
                                f'from list containing non DataFlowElements')
            value += elem.value
            if data.intersection(elem.data):
                raise Exception(f'DataFlowException: Can not create exclusive DataFlowElement instance from list. '
                                f'Same element found multiple times.'
                                f'Existing elements: {data}; '
                                f'Tried joining elements: {elem.data}')
            data = data.union(elem.data)
        return cls(value, data)

    def __eq__(self, other):
        return self.data == other.data

    def __str__(self):
        return str(self.data) if self.data else ''


class Notation:
    def __init__(self):
        self.elements = [Line, Fork, Reference, Anchor]
        self.refs = {}
        self.seq = None
        self.data = None

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

        self.data = DataFlowElement.parse(data_str)
        self.seq = Sequence.parse(string)

        for ref_def in ref_defs:
            ref = ReferenceDefinition.parse(ref_def)
            self.refs.update({ref.key: ref})

        self._resolve_refs()

        self.validate_data_flow()

        return self

    def to_graph(self, graph: graph.Graph, root: int):
        if self.seq is None:
            return
        self.seq.add_to_graph(graph, root)


def main():
    text = '-<[$1, $2]>-\n' \
           '$1: -<[$3, $2]\n' \
           '$2: -!1\n' \
           '$3: -!1'

    notation = Notation()
    notation.parse(text)
    print()


if __name__ == '__main__':
    main()
