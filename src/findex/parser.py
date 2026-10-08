import re
from dataclasses import dataclass
from typing import List, Set, Union


class Node:
    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def evaluate(self, index) -> Set[int]:
        raise NotImplementedError


@dataclass(frozen=True)
class Term(Node):
    value: str

    def evaluate(self, index) -> Set[int]:
        normalized = self.value.strip().lower()
        if normalized in index:
            return {p.doc_id for p in index[normalized]}
        return set()


@dataclass(frozen=True)
class Phrase(Node):
    terms: List[str]

    def evaluate(self, index) -> Set[int]:
        if not self.terms:
            return set()
        clean_terms = [t.strip().lower() for t in self.terms if t.strip()]
        if not clean_terms:
            return set()

        first_term = clean_terms[0]
        if first_term not in index:
            return set()

        candidate_docs = {p.doc_id for p in index[first_term]}
        for t in clean_terms[1:]:
            if t not in index:
                return set()
            candidate_docs &= {p.doc_id for p in index[t]}

        if not candidate_docs:
            return set()

        matching_docs = set()
        for doc_id in candidate_docs:
            positions_by_term = []
            for t in clean_terms:
                postings = index[t]
                pos_list = []
                for p in postings:
                    if p.doc_id == doc_id:
                        pos_list = getattr(p, "positions", ())
                        break
                positions_by_term.append(pos_list)

            if all(positions_by_term):
                for p0 in positions_by_term[0]:
                    match = True
                    for offset, pos_list in enumerate(positions_by_term[1:], start=1):
                        if (p0 + offset) not in pos_list:
                            match = False
                            break
                    if match:
                        matching_docs.add(doc_id)
                        break
            else:
                matching_docs.add(doc_id)

        return matching_docs


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index) -> Set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index) -> Set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index) -> Set[int]:
        all_docs = set(index.documents.keys())
        return all_docs - self.child.evaluate(index)


class QueryParser:
    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str) -> List[str]:
        token_spec = [
            ("PHRASE", r'"[^"]*"'),
            ("LPAREN", r"\("),
            ("RPAREN", r"\)"),
            ("AND", r"\b(AND|and|&)\b"),
            ("OR", r"\b(OR|or|\|)\b"),
            ("NOT", r"\b(NOT|not|~)\b"),
            ("WORD", r"[^\s()\"&|~]+"),
        ]
        tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_spec)
        tokens = []
        for mo in re.finditer(tok_regex, text):
            kind = mo.lastgroup
            val = mo.group()
            if kind == "PHRASE":
                tokens.append(val)
            elif kind in ("AND", "OR", "NOT"):
                tokens.append(val.upper())
            elif kind == "LPAREN":
                tokens.append("(")
            elif kind == "RPAREN":
                tokens.append(")")
            elif kind == "WORD":
                tokens.append(val)
        return tokens

    def peek(self) -> Union[str, None]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self) -> Node:
        if not self.tokens:
            return Term("")
        node = self.parse_or()
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.peek() in ("OR", "|"):
            self.consume()
            right = self.parse_and()
            node = Or(node, right)
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while self.peek() not in ("OR", "|", ")", None):
            if self.peek() in ("AND", "&"):
                self.consume()
            right = self.parse_not()
            node = And(node, right)
        return node

    def parse_not(self) -> Node:
        if self.peek() in ("NOT", "~"):
            self.consume()
            child = self.parse_not()
            return Not(child)
        return self.parse_primary()

    def parse_primary(self) -> Node:
        token = self.peek()
        if token == "(":
            self.consume()
            node = self.parse_or()
            if self.peek() == ")":
                self.consume()
            return node
        elif token and token.startswith('"') and token.endswith('"'):
            self.consume()
            inner = token[1:-1]
            words = inner.strip().split()
            return Phrase(words)
        elif token and token != ")":
            self.consume()
            return Term(token)
        return Term("")


def parse_query(query_str: str) -> Node:
    return QueryParser(query_str).parse()