"""
GAMS (.gms) parser for MINLP models.

Hand-written recursive-descent parser that reads GAMS source and builds
a discopt Model.  Targets the subset of GAMS needed for MINLP:

  Sets, Scalars, Parameters, Tables, Variables (positive/binary/integer/free),
  Equations with =e=/=l=/=g=, bounds (.lo/.up/.fx), sum/prod over indexed
  domains, nonlinear functions (exp, log, sin, cos, sqrt, power, sqr, ...),
  Model, Solve ... using MINLP minimizing/maximizing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np


# ── Token types ────────────────────────────────────────────────
class _Tok:
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    SYMBOL = "SYMBOL"
    EOF = "EOF"


@dataclass
class Token:
    kind: str
    value: str
    line: int = 0
    col: int = 0


# ── Lexer ──────────────────────────────────────────────────────
_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING_RE = re.compile(r'"[^"]*"|\'[^\']*\'')


def _tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(src)
    while i < n:
        ch = src[i]
        # newline
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        # whitespace
        if ch in " \t\r":
            i += 1
            col += 1
            continue
        # line comment  * at start-ish or $
        if ch == "*" and (col == 1 or (i > 0 and src[i - 1] == "\n")):
            while i < n and src[i] != "\n":
                i += 1
            continue
        if ch == "$":
            # dollar control line — skip unless it's a dollar condition
            # inside an expression (handled by parser).  At line start it's
            # a control directive we skip.
            if col == 1:
                while i < n and src[i] != "\n":
                    i += 1
                continue
            # otherwise treat $ as a symbol for dollar conditions
            tokens.append(Token(_Tok.SYMBOL, "$", line, col))
            i += 1
            col += 1
            continue
        # string literal
        m = _STRING_RE.match(src, i)
        if m and ch in ('"', "'"):
            tokens.append(Token(_Tok.STRING, m.group()[1:-1], line, col))
            col += m.end() - m.start()
            i = m.end()
            continue
        # two-char symbols  =e= =l= =g= .. **
        if i + 2 < n and src[i : i + 3] in ("=e=", "=l=", "=g=", "=E=", "=L=", "=G="):
            tokens.append(Token(_Tok.SYMBOL, src[i : i + 3].lower(), line, col))
            i += 3
            col += 3
            continue
        if i + 1 < n and src[i : i + 2] == "**":
            tokens.append(Token(_Tok.SYMBOL, "**", line, col))
            i += 2
            col += 2
            continue
        if i + 1 < n and src[i : i + 2] == "..":
            tokens.append(Token(_Tok.SYMBOL, "..", line, col))
            i += 2
            col += 2
            continue
        # number (but not if preceded by ident char — that's part of ident)
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            m = _NUM_RE.match(src, i)
            if m:
                tokens.append(Token(_Tok.NUMBER, m.group(), line, col))
                col += m.end() - m.start()
                i = m.end()
                continue
        # identifier
        if ch.isalpha() or ch == "_":
            m = _IDENT_RE.match(src, i)
            if m:
                tokens.append(Token(_Tok.IDENT, m.group(), line, col))
                col += m.end() - m.start()
                i = m.end()
                continue
        # single char symbols
        if ch in "()[]{},.;:+-*/=<>":
            tokens.append(Token(_Tok.SYMBOL, ch, line, col))
            i += 1
            col += 1
            continue
        # slash
        if ch == "/":
            tokens.append(Token(_Tok.SYMBOL, "/", line, col))
            i += 1
            col += 1
            continue
        # skip unknown
        i += 1
        col += 1
    tokens.append(Token(_Tok.EOF, "", line, col))
    return tokens


# ── AST node types ─────────────────────────────────────────────
@dataclass
class GamsSet:
    name: str
    elements: list[str]
    description: str = ""


@dataclass
class GamsScalar:
    name: str
    value: float = 0.0
    description: str = ""


@dataclass
class GamsParameter:
    name: str
    domain: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)  # tuple-key -> float
    description: str = ""


@dataclass
class GamsTable:
    name: str
    domain: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class GamsVariable:
    name: str
    var_type: str = "free"  # free, positive, negative, binary, integer
    domain: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class GamsEquation:
    name: str
    domain: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class GamsEquationDef:
    name: str
    domain: list[str] = field(default_factory=list)
    dollar_cond: object = None  # optional filter expression
    lhs: object = None  # expression AST
    sense: str = "=e="  # =e=, =l=, =g=
    rhs: object = None  # expression AST


@dataclass
class GamsBound:
    var_name: str
    suffix: str  # lo, up, fx, l
    domain: list[str] = field(default_factory=list)
    dollar_cond: object = None
    expr: object = None


@dataclass
class GamsModel:
    name: str
    equations: list[str] = field(default_factory=list)  # or ["all"]


@dataclass
class GamsSolve:
    model_name: str
    model_type: str = "minlp"
    sense: str = "minimizing"
    objective_var: str = ""


# Expression AST nodes
@dataclass
class ExprNum:
    value: float


@dataclass
class ExprRef:
    name: str


@dataclass
class ExprIndex:
    name: str
    indices: list  # list of expression ASTs


@dataclass
class ExprBinOp:
    op: str
    left: object
    right: object


@dataclass
class ExprUnaryMinus:
    operand: object


@dataclass
class ExprFunc:
    func: str
    args: list


@dataclass
class ExprSum:
    index_names: list[str]
    dollar_cond: object  # optional
    body: object


@dataclass
class ExprProd:
    index_names: list[str]
    dollar_cond: object
    body: object


@dataclass
class ExprCard:
    set_name: str


@dataclass
class ExprOrd:
    set_name: str


# ── Recursive-descent parser ──────────────────────────────────
class GamsParseError(Exception):
    pass


class _Parser:
    """Recursive-descent parser for GAMS .gms files."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        # Collected declarations
        self.sets: dict[str, GamsSet] = {}
        self.aliases: dict[str, str] = {}  # alias -> original set
        self.scalars: dict[str, GamsScalar] = {}
        self.parameters: dict[str, GamsParameter] = {}
        self.tables: dict[str, GamsTable] = {}
        self.variables: dict[str, GamsVariable] = {}
        self.equations: dict[str, GamsEquation] = {}
        self.equation_defs: list[GamsEquationDef] = []
        self.bounds: list[GamsBound] = []
        self.models: dict[str, GamsModel] = {}
        self.solves: list[GamsSolve] = []
        self.options: dict[str, str] = {}
        self.param_assigns: list[tuple] = []  # (name, domain, dollar, expr)

    # ── Helpers ──

    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return t

    def _expect(self, kind: str, value: str | None = None) -> Token:
        t = self._cur()
        if t.kind != kind:
            raise GamsParseError(
                f"Expected {kind}({value!r}) at line {t.line}, got {t.kind}({t.value!r})"
            )
        if value is not None and t.value.lower() != value.lower():
            raise GamsParseError(f"Expected {value!r} at line {t.line}, got {t.value!r}")
        return self._advance()

    def _match_ident(self, value: str) -> bool:
        t = self._cur()
        return t.kind == _Tok.IDENT and t.value.lower() == value.lower()

    def _match_sym(self, value: str) -> bool:
        t = self._cur()
        return t.kind == _Tok.SYMBOL and t.value == value

    def _at_end(self) -> bool:
        return self._cur().kind == _Tok.EOF

    def _skip_semi(self):
        if self._match_sym(";"):
            self._advance()

    # ── Keywords ──

    _SET_KW = {"set", "sets"}
    _SCALAR_KW = {"scalar", "scalars"}
    _PARAM_KW = {"parameter", "parameters"}
    _TABLE_KW = {"table", "tables"}
    _VAR_KW = {"variable", "variables"}
    _EQ_KW = {"equation", "equations"}
    _MODEL_KW = {"model"}
    _SOLVE_KW = {"solve"}
    _OPTION_KW = {"option", "options"}
    _DISPLAY_KW = {"display"}
    _ABORT_KW = {"abort"}
    _ALIAS_KW = {"alias"}
    _VARTYPE_KW = {
        "positive",
        "negative",
        "free",
        "binary",
        "integer",
        "semicont",
        "semiint",
        "sos1",
        "sos2",
    }

    def _cur_kw(self) -> str | None:
        t = self._cur()
        if t.kind != _Tok.IDENT:
            return None
        return t.value.lower()

    # ── Top-level parse ──

    def parse(self):
        while not self._at_end():
            kw = self._cur_kw()
            if kw in self._SET_KW:
                self._parse_set_decl()
            elif kw in self._ALIAS_KW:
                self._parse_alias()
            elif kw in self._SCALAR_KW:
                self._parse_scalar_decl()
            elif kw in self._PARAM_KW:
                self._parse_param_decl()
            elif kw in self._TABLE_KW:
                self._parse_table_decl()
            elif kw in self._VARTYPE_KW:
                self._parse_variable_decl()
            elif kw in self._VAR_KW:
                self._parse_variable_decl()
            elif kw in self._EQ_KW:
                self._parse_equation_decl()
            elif kw in self._MODEL_KW:
                self._parse_model_decl()
            elif kw in self._SOLVE_KW:
                self._parse_solve_stmt()
            elif kw in self._OPTION_KW:
                self._parse_option_stmt()
            elif kw in self._DISPLAY_KW:
                self._skip_to_semi()
            elif kw in self._ABORT_KW:
                self._skip_to_semi()
            elif self._is_equation_def():
                self._parse_equation_def()
            elif self._is_bound_assign():
                self._parse_bound_assign()
            elif self._is_param_assign():
                self._parse_param_assign_stmt()
            else:
                # skip unrecognized token
                self._advance()

    def _skip_to_semi(self):
        while not self._at_end() and not self._match_sym(";"):
            self._advance()
        self._skip_semi()

    # ── Lookahead helpers ──

    def _is_equation_def(self) -> bool:
        """name(..) .. expr =e= expr ;"""
        if self._cur().kind != _Tok.IDENT:
            return False
        # scan ahead for ".."
        j = self.pos + 1
        while j < len(self.tokens) and j < self.pos + 30:
            t = self.tokens[j]
            if t.kind == _Tok.SYMBOL and t.value == "..":
                return True
            if t.kind == _Tok.SYMBOL and t.value == ";":
                return False
            j += 1
        return False

    def _is_bound_assign(self) -> bool:
        """name.lo/up/fx(...) = expr ;"""
        if self._cur().kind != _Tok.IDENT:
            return False
        j = self.pos + 1
        if j < len(self.tokens) and self.tokens[j].value == ".":
            j2 = j + 1
            if j2 < len(self.tokens) and self.tokens[j2].value.lower() in (
                "lo",
                "up",
                "fx",
                "l",
                "m",
                "prior",
                "scale",
            ):
                return True
        return False

    def _is_param_assign(self) -> bool:
        """name(...) = expr ; where name is a known param or scalar."""
        if self._cur().kind != _Tok.IDENT:
            return False
        name = self._cur().value.lower()
        if name in {k.lower() for k in self.scalars} | {k.lower() for k in self.parameters}:
            return True
        return False

    # ── Set declarations ──

    def _parse_set_decl(self):
        self._advance()  # skip 'set'/'sets'
        while not self._at_end() and not self._match_sym(";"):
            name_tok = self._expect(_Tok.IDENT)
            name = name_tok.value
            # optional domain
            if self._match_sym("("):
                self._parse_domain()
            # optional description string
            desc = ""
            if self._cur().kind == _Tok.STRING:
                desc = self._advance().value
            # data in /.../
            elements: list[str] = []
            if self._match_sym("/"):
                self._advance()  # skip /
                elements = self._parse_set_elements()
                self._expect(_Tok.SYMBOL, "/")
            self.sets[name] = GamsSet(name, elements, desc)
            # optional comma between items
            if self._match_sym(","):
                self._advance()

        self._skip_semi()

    def _parse_set_elements(self) -> list[str]:
        elements: list[str] = []
        while not self._at_end() and not self._match_sym("/"):
            tok = self._cur()
            if tok.kind == _Tok.IDENT or tok.kind == _Tok.NUMBER:
                name = self._advance().value
                # check for range  name*name
                if self._match_sym("*"):
                    self._advance()  # skip *
                    end_tok = self._advance()
                    elements.extend(self._expand_range(name, end_tok.value))
                else:
                    elements.append(name)
            else:
                self._advance()  # skip commas etc
            # skip comma
            if self._match_sym(","):
                self._advance()
        return elements

    def _expand_range(self, start: str, end: str) -> list[str]:
        """Expand 1*5 or i1*i5 style ranges."""
        try:
            return [str(x) for x in range(int(start), int(end) + 1)]
        except ValueError:
            pass
        # try prefix+number pattern
        m1 = re.match(r"([A-Za-z_]*)(\d+)$", start)
        m2 = re.match(r"([A-Za-z_]*)(\d+)$", end)
        if m1 and m2 and m1.group(1) == m2.group(1):
            prefix = m1.group(1)
            return [f"{prefix}{x}" for x in range(int(m1.group(2)), int(m2.group(2)) + 1)]
        return [start, end]

    # ── Alias ──

    def _parse_alias(self):
        self._advance()  # skip 'alias'
        self._expect(_Tok.SYMBOL, "(")
        names: list[str] = []
        while not self._match_sym(")"):
            if self._cur().kind == _Tok.IDENT:
                names.append(self._advance().value)
            elif self._match_sym(","):
                self._advance()
            else:
                self._advance()
        self._expect(_Tok.SYMBOL, ")")
        self._skip_semi()
        if len(names) >= 2:
            original = names[0]
            for alias_name in names[1:]:
                self.aliases[alias_name] = original

    # ── Scalars ──

    def _parse_scalar_decl(self):
        self._advance()  # skip 'scalar'/'scalars'
        while not self._at_end() and not self._match_sym(";"):
            name = self._expect(_Tok.IDENT).value
            desc = ""
            if self._cur().kind == _Tok.STRING:
                desc = self._advance().value
            val = 0.0
            if self._match_sym("/"):
                self._advance()
                if self._cur().kind == _Tok.NUMBER:
                    val = float(self._advance().value)
                elif self._match_sym("-"):
                    self._advance()
                    val = -float(self._expect(_Tok.NUMBER).value)
                elif self._match_sym("+"):
                    self._advance()
                    val = float(self._expect(_Tok.NUMBER).value)
                self._expect(_Tok.SYMBOL, "/")
            self.scalars[name] = GamsScalar(name, val, desc)
            if self._match_sym(","):
                self._advance()
        self._skip_semi()

    # ── Domain parsing ──

    def _parse_domain(self) -> list[str]:
        """Parse (i, j, k) domain list, consuming parens."""
        self._expect(_Tok.SYMBOL, "(")
        names: list[str] = []
        while not self._match_sym(")"):
            if self._cur().kind == _Tok.IDENT:
                names.append(self._advance().value)
            elif self._match_sym(","):
                self._advance()
            else:
                self._advance()
        self._expect(_Tok.SYMBOL, ")")
        return names

    # ── Parameters ──

    def _parse_param_decl(self):
        self._advance()  # skip 'parameter'/'parameters'
        while not self._at_end() and not self._match_sym(";"):
            name = self._expect(_Tok.IDENT).value
            domain: list[str] = []
            if self._match_sym("("):
                domain = self._parse_domain()
            desc = ""
            if self._cur().kind == _Tok.STRING:
                desc = self._advance().value
            data: dict = {}
            if self._match_sym("/"):
                self._advance()
                data = self._parse_param_data()
                self._expect(_Tok.SYMBOL, "/")
            self.parameters[name] = GamsParameter(name, domain, data, desc)
            if self._match_sym(","):
                self._advance()
        self._skip_semi()

    def _parse_param_data(self) -> dict:
        data: dict = {}
        while not self._at_end() and not self._match_sym("/"):
            keys: list[str] = []
            # read keys (ident or number tokens before the numeric value)
            while self._cur().kind == _Tok.IDENT or (
                self._cur().kind == _Tok.NUMBER
                and self._peek(1).kind in (_Tok.IDENT, _Tok.NUMBER)
                and not self._peek(1).value.startswith("-")
            ):
                keys.append(self._advance().value)
                if self._match_sym("."):
                    self._advance()  # multi-dim key separator
            # read value
            neg = False
            if self._match_sym("-"):
                self._advance()
                neg = True
            elif self._match_sym("+"):
                self._advance()
            if self._cur().kind == _Tok.NUMBER:
                val = float(self._advance().value)
                if neg:
                    val = -val
            else:
                val = 0.0
            key = tuple(keys) if len(keys) > 1 else (keys[0] if keys else ("",))
            data[key] = val
            if self._match_sym(","):
                self._advance()
        return data

    # ── Tables ──

    def _parse_table_decl(self):
        self._advance()  # skip 'table'
        name = self._expect(_Tok.IDENT).value
        domain: list[str] = []
        if self._match_sym("("):
            domain = self._parse_domain()
        desc = ""
        if self._cur().kind == _Tok.STRING:
            desc = self._advance().value
        # Read column headers and rows until semicolon
        col_headers: list[str] = []
        data: dict = {}
        # First collect column headers (identifiers before first row data)
        while not self._at_end() and not self._match_sym(";"):
            if self._cur().kind == _Tok.IDENT:
                # could be col header or row label
                # heuristic: if next token is also IDENT or NUMBER and we haven't
                # started rows yet, these are col headers
                if not col_headers or (
                    self._peek(1).kind in (_Tok.IDENT, _Tok.NUMBER)
                    and not any(isinstance(v, dict) for v in data.values())
                ):
                    # check if this starts a data row
                    # a data row starts with ident followed by numbers
                    j = self.pos + 1
                    has_numbers = False
                    while j < len(self.tokens) and self.tokens[j].kind == _Tok.NUMBER:
                        has_numbers = True
                        j += 1
                    if has_numbers and col_headers:
                        # This is a row
                        row_label = self._advance().value
                        row_data: list[float] = []
                        while self._cur().kind == _Tok.NUMBER:
                            row_data.append(float(self._advance().value))
                        for ci, col in enumerate(col_headers):
                            if ci < len(row_data):
                                key = (row_label, col)
                                data[key] = row_data[ci]
                    else:
                        col_headers.append(self._advance().value)
                else:
                    # row
                    row_label = self._advance().value
                    row_data = []
                    while self._cur().kind == _Tok.NUMBER:
                        row_data.append(float(self._advance().value))
                    for ci, col in enumerate(col_headers):
                        if ci < len(row_data):
                            data[(row_label, col)] = row_data[ci]
            elif self._cur().kind == _Tok.SYMBOL and self._cur().value == "+":
                # continuation line with + for additional columns
                self._advance()
                # read more column headers
                while self._cur().kind == _Tok.IDENT:
                    col_headers.append(self._advance().value)
            else:
                self._advance()
        self._skip_semi()
        self.tables[name] = GamsTable(name, domain, data, desc)

    # ── Variables ──

    def _parse_variable_decl(self):
        var_type = "free"
        kw = self._cur_kw()
        if kw in self._VARTYPE_KW:
            var_type = kw
            self._advance()
        # optional 'variable'/'variables' keyword
        if self._cur_kw() in self._VAR_KW:
            self._advance()
        while not self._at_end() and not self._match_sym(";"):
            name = self._expect(_Tok.IDENT).value
            domain: list[str] = []
            if self._match_sym("("):
                domain = self._parse_domain()
            desc = ""
            if self._cur().kind == _Tok.STRING:
                desc = self._advance().value
            self.variables[name] = GamsVariable(name, var_type, domain, desc)
            if self._match_sym(","):
                self._advance()
        self._skip_semi()

    # ── Equations declaration ──

    def _parse_equation_decl(self):
        self._advance()  # skip 'equation'/'equations'
        while not self._at_end() and not self._match_sym(";"):
            name = self._expect(_Tok.IDENT).value
            domain: list[str] = []
            if self._match_sym("("):
                domain = self._parse_domain()
            desc = ""
            if self._cur().kind == _Tok.STRING:
                desc = self._advance().value
            self.equations[name] = GamsEquation(name, domain, desc)
            if self._match_sym(","):
                self._advance()
        self._skip_semi()

    # ── Equation definitions ──

    def _parse_equation_def(self):
        name = self._expect(_Tok.IDENT).value
        domain: list[str] = []
        if self._match_sym("("):
            domain = self._parse_domain()
        dollar_cond = None
        if self._match_sym("$"):
            dollar_cond = self._parse_dollar_cond()
        self._expect(_Tok.SYMBOL, "..")
        lhs = self._parse_expr()
        sense_tok = self._advance()
        sense = sense_tok.value.lower()
        rhs = self._parse_expr()
        self._skip_semi()
        self.equation_defs.append(GamsEquationDef(name, domain, dollar_cond, lhs, sense, rhs))

    # ── Bound assignments ──

    def _parse_bound_assign(self):
        var_name = self._expect(_Tok.IDENT).value
        self._expect(_Tok.SYMBOL, ".")
        suffix = self._expect(_Tok.IDENT).value.lower()
        domain: list[str] = []
        if self._match_sym("("):
            domain = self._parse_domain()
        dollar_cond = None
        if self._match_sym("$"):
            dollar_cond = self._parse_dollar_cond()
        self._expect(_Tok.SYMBOL, "=")
        expr = self._parse_expr()
        self._skip_semi()
        self.bounds.append(GamsBound(var_name, suffix, domain, dollar_cond, expr))

    # ── Parameter assignment ──

    def _parse_param_assign_stmt(self):
        name = self._expect(_Tok.IDENT).value
        domain: list[str] = []
        if self._match_sym("("):
            domain = self._parse_domain()
        dollar_cond = None
        if self._match_sym("$"):
            dollar_cond = self._parse_dollar_cond()
        self._expect(_Tok.SYMBOL, "=")
        expr = self._parse_expr()
        self._skip_semi()
        self.param_assigns.append((name, domain, dollar_cond, expr))

    # ── Model / Solve / Option ──

    def _parse_model_decl(self):
        self._advance()  # skip 'model'
        name = self._expect(_Tok.IDENT).value
        if self._cur().kind == _Tok.STRING:
            self._advance()  # skip description
        self._expect(_Tok.SYMBOL, "/")
        eqs: list[str] = []
        if self._match_ident("all"):
            eqs = ["all"]
            self._advance()
        else:
            while not self._match_sym("/"):
                if self._cur().kind == _Tok.IDENT:
                    eqs.append(self._advance().value)
                elif self._match_sym(","):
                    self._advance()
                else:
                    self._advance()
        self._expect(_Tok.SYMBOL, "/")
        self._skip_semi()
        self.models[name] = GamsModel(name, eqs)

    def _parse_solve_stmt(self):
        self._advance()  # skip 'solve'
        model_name = self._expect(_Tok.IDENT).value
        self._expect(_Tok.IDENT, "using")
        model_type = self._expect(_Tok.IDENT).value.lower()
        sense_tok = self._expect(_Tok.IDENT)
        sense = sense_tok.value.lower()
        obj_var = self._expect(_Tok.IDENT).value
        self._skip_semi()
        self.solves.append(GamsSolve(model_name, model_type, sense, obj_var))

    def _parse_option_stmt(self):
        self._advance()  # skip 'option'
        key = self._expect(_Tok.IDENT).value
        self._expect(_Tok.SYMBOL, "=")
        val = self._advance().value
        self._skip_semi()
        self.options[key] = val

    # ── Dollar condition ──

    def _parse_dollar_cond(self):
        self._advance()  # skip $
        if self._match_sym("("):
            self._advance()
            expr = self._parse_expr()
            self._expect(_Tok.SYMBOL, ")")
            return expr
        return self._parse_atom()

    # ── Expression parser (precedence climbing) ──

    def _parse_expr(self):
        return self._parse_add()

    def _parse_add(self):
        left = self._parse_mul()
        while self._match_sym("+") or self._match_sym("-"):
            op = self._advance().value
            right = self._parse_mul()
            left = ExprBinOp(op, left, right)
        return left

    def _parse_mul(self):
        left = self._parse_unary()
        while self._match_sym("*") or self._match_sym("/"):
            # don't consume ** here
            if self._match_sym("*") and self._peek(1).value == "*":
                break
            op = self._advance().value
            right = self._parse_unary()
            left = ExprBinOp(op, left, right)
        return left

    def _parse_unary(self):
        if self._match_sym("-"):
            self._advance()
            operand = self._parse_power()
            return ExprUnaryMinus(operand)
        if self._match_sym("+"):
            self._advance()
            return self._parse_power()
        return self._parse_power()

    def _parse_power(self):
        base = self._parse_atom()
        if self._match_sym("**"):
            self._advance()
            exp = self._parse_unary()
            return ExprBinOp("**", base, exp)
        return base

    # GAMS built-in functions
    _GAMS_FUNCS = {
        "exp",
        "log",
        "log2",
        "log10",
        "sqrt",
        "sqr",
        "abs",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "power",
        "sign",
        "min",
        "max",
        "ceil",
        "floor",
        "round",
        "mod",
        "uniform",
        "normal",
        "errorf",
        "sigmoid",
    }

    _INDEXED_OPS = {"sum", "prod", "smin", "smax"}

    def _parse_atom(self):
        t = self._cur()

        # number literal
        if t.kind == _Tok.NUMBER:
            self._advance()
            return ExprNum(float(t.value))

        # parenthesized expression
        if t.kind == _Tok.SYMBOL and t.value == "(":
            self._advance()
            expr = self._parse_expr()
            self._expect(_Tok.SYMBOL, ")")
            return expr

        # identifier: could be func, indexed op, indexed ref, or simple ref
        if t.kind == _Tok.IDENT:
            name_lower = t.value.lower()

            # indexed operations: sum, prod, smin, smax
            if name_lower in self._INDEXED_OPS:
                return self._parse_indexed_op(name_lower)

            # card(set), ord(set)
            if name_lower == "card":
                self._advance()
                self._expect(_Tok.SYMBOL, "(")
                sname = self._expect(_Tok.IDENT).value
                self._expect(_Tok.SYMBOL, ")")
                return ExprCard(sname)
            if name_lower == "ord":
                self._advance()
                self._expect(_Tok.SYMBOL, "(")
                sname = self._expect(_Tok.IDENT).value
                self._expect(_Tok.SYMBOL, ")")
                return ExprOrd(sname)

            # built-in functions
            if name_lower in self._GAMS_FUNCS and self._peek(1).value == "(":
                return self._parse_func_call()

            # indexed reference: name(i,j)
            if self._peek(1).value == "(":
                # check if this is a function call on expressions
                # vs an indexed variable/param reference
                # heuristic: if name is a known variable/param/set, it's indexing
                return self._parse_indexed_ref()

            # simple name reference
            self._advance()
            return ExprRef(t.value)

        raise GamsParseError(f"Unexpected token {t.kind}({t.value!r}) at line {t.line}")

    def _parse_indexed_op(self, op_name: str):
        self._advance()  # skip sum/prod/smin/smax
        self._expect(_Tok.SYMBOL, "(")
        # parse index list
        index_names = self._parse_index_list()
        # optional dollar condition
        dollar_cond = None
        if self._match_sym("$"):
            dollar_cond = self._parse_dollar_cond()
        self._expect(_Tok.SYMBOL, ",")
        body = self._parse_expr()
        self._expect(_Tok.SYMBOL, ")")
        if op_name == "sum":
            return ExprSum(index_names, dollar_cond, body)
        elif op_name == "prod":
            return ExprProd(index_names, dollar_cond, body)
        else:
            # smin/smax: treat as func for now
            return ExprFunc(op_name, [body])

    def _parse_index_list(self) -> list[str]:
        names: list[str] = []
        if self._match_sym("("):
            self._advance()
            while not self._match_sym(")"):
                if self._cur().kind == _Tok.IDENT:
                    names.append(self._advance().value)
                elif self._match_sym(","):
                    self._advance()
                else:
                    self._advance()
            self._expect(_Tok.SYMBOL, ")")
        else:
            names.append(self._expect(_Tok.IDENT).value)
        return names

    def _parse_func_call(self):
        func_name = self._advance().value.lower()
        self._expect(_Tok.SYMBOL, "(")
        args: list = []
        while not self._match_sym(")"):
            args.append(self._parse_expr())
            if self._match_sym(","):
                self._advance()
        self._expect(_Tok.SYMBOL, ")")
        return ExprFunc(func_name, args)

    def _parse_indexed_ref(self):
        name = self._advance().value
        self._expect(_Tok.SYMBOL, "(")
        indices: list = []
        while not self._match_sym(")"):
            indices.append(self._parse_expr())
            if self._match_sym(","):
                self._advance()
        self._expect(_Tok.SYMBOL, ")")
        return ExprIndex(name, indices)


# ── Model builder: parsed GAMS AST → discopt Model ────────────
class _ModelBuilder:
    """Convert parsed GAMS declarations into a discopt Model."""

    def __init__(self, parser: _Parser):
        self.p = parser
        # resolved set elements as ordered lists
        self.set_elements: dict[str, list[str]] = {}
        # resolved parameter values: name -> {tuple_key: float}
        self.param_values: dict[str, dict] = {}
        # scalar values: name -> float
        self.scalar_values: dict[str, float] = {}
        # discopt variable references: (var_name, *indices) -> Expression
        self.dvar_map: dict[str, object] = {}  # var_name -> discopt Variable
        self.model: object = None  # discopt Model

    def build(self):
        from discopt.modeling.core import Model

        # 1. Resolve sets (including aliases)
        self._resolve_sets()
        # 2. Resolve scalars and parameters
        self._resolve_scalars()
        self._resolve_parameters()
        # 3. Create discopt model
        solve = self.p.solves[0] if self.p.solves else None
        model_name = solve.model_name if solve else "gams_model"
        m = Model(model_name)
        self.model = m
        # 4. Create variables
        self._create_variables(m)
        # 5. Build equations (constraints + objective)
        self._build_equations(m, solve)
        # 6. Apply bounds
        self._apply_bounds(m)
        return m

    def _resolve_sets(self):
        for name, gs in self.p.sets.items():
            self.set_elements[name] = gs.elements
        # aliases point to the same elements
        for alias, original in self.p.aliases.items():
            if original in self.set_elements:
                self.set_elements[alias] = self.set_elements[original]

    def _resolve_scalars(self):
        for name, gs in self.p.scalars.items():
            self.scalar_values[name] = gs.value

    def _resolve_parameters(self):
        for name, gp in self.p.parameters.items():
            self.param_values[name] = dict(gp.data)
        for name, gt in self.p.tables.items():
            self.param_values[name] = dict(gt.data)
        # Process assignment statements for known params/scalars
        for pname, domain, dollar, expr in self.p.param_assigns:
            if pname in self.scalar_values and not domain:
                val = self._eval_const_expr(expr)
                if val is not None:
                    self.scalar_values[pname] = val

    def _eval_const_expr(self, expr) -> float | None:
        """Evaluate a constant expression (no variables)."""
        if isinstance(expr, ExprNum):
            return expr.value
        if isinstance(expr, ExprRef):
            if expr.name in self.scalar_values:
                return self.scalar_values[expr.name]
            return None
        if isinstance(expr, ExprUnaryMinus):
            v = self._eval_const_expr(expr.operand)
            return -v if v is not None else None
        if isinstance(expr, ExprBinOp):
            lv = self._eval_const_expr(expr.left)
            rv = self._eval_const_expr(expr.right)
            if lv is None or rv is None:
                return None
            if expr.op == "+":
                return lv + rv
            if expr.op == "-":
                return lv - rv
            if expr.op == "*":
                return lv * rv
            if expr.op == "/":
                return lv / rv if rv != 0 else None
            if expr.op == "**":
                return lv**rv
        if isinstance(expr, ExprFunc):
            args = [self._eval_const_expr(a) for a in expr.args]
            if any(a is None for a in args):
                return None
            fn = expr.func.lower()
            if fn == "exp":
                return math.exp(args[0])
            if fn == "log":
                return math.log(args[0])
            if fn == "sqrt":
                return math.sqrt(args[0])
            if fn == "sqr":
                return args[0] ** 2
            if fn == "abs":
                return abs(args[0])
            if fn == "power":
                return args[0] ** args[1]
            if fn == "sin":
                return math.sin(args[0])
            if fn == "cos":
                return math.cos(args[0])
        if isinstance(expr, ExprCard):
            elems = self.set_elements.get(expr.set_name, [])
            return float(len(elems))
        return None

    def _eval_dollar_cond(self, expr, env: dict) -> float | None:
        """Evaluate a dollar condition with loop index substitution.

        Returns 1.0 (true) or 0.0 (false) if evaluable, None otherwise.
        Handles common patterns like set membership checks.
        """
        if isinstance(expr, ExprRef):
            name = expr.name
            if name in env:
                return 1.0  # index variable exists → true
            # Check if it's a set used as a membership test
            if name in self.set_elements:
                return 1.0
            return None
        if isinstance(expr, ExprIndex):
            # e.g., connected(i,j) — look up in parameter data
            if expr.name in self.param_values:
                indices = []
                for idx in expr.indices:
                    if isinstance(idx, ExprRef) and idx.name in env:
                        indices.append(env[idx.name])
                    else:
                        return None
                key = tuple(indices) if len(indices) > 1 else (indices[0],)
                val = self.param_values[expr.name].get(key, 0.0)
                return float(val)
            return None
        # For other expressions, try constant evaluation
        return self._eval_const_expr(expr)

    def _create_variables(self, m):
        for name, gv in self.p.variables.items():
            shape = self._domain_shape(gv.domain)
            vtype = gv.var_type.lower()
            if vtype == "binary":
                var = m.binary(name, shape=shape)
            elif vtype == "integer":
                var = m.integer(name, shape=shape, lb=0, ub=1e6)
            elif vtype == "positive":
                var = m.continuous(name, shape=shape, lb=0.0)
            elif vtype == "negative":
                var = m.continuous(name, shape=shape, ub=0.0)
            else:  # free
                var = m.continuous(name, shape=shape)
            self.dvar_map[name] = var

    def _domain_shape(self, domain: list[str]) -> tuple:
        if not domain:
            return ()
        dims = []
        for d in domain:
            # resolve to set elements
            resolved = d
            if d in self.p.aliases:
                resolved = self.p.aliases[d]
            if resolved in self.set_elements:
                dims.append(len(self.set_elements[resolved]))
            elif d in self.set_elements:
                dims.append(len(self.set_elements[d]))
            else:
                # unknown set dimension — try equation/variable domain
                # fallback: check if any set has this name (case insensitive)
                found = False
                for sn, se in self.set_elements.items():
                    if sn.lower() == d.lower():
                        dims.append(len(se))
                        found = True
                        break
                if not found:
                    raise GamsParseError(f"Unknown set '{d}' in domain")
        return tuple(dims)

    def _domain_set_names(self, domain: list[str]) -> list[str]:
        """Resolve domain names to actual set names."""
        result = []
        for d in domain:
            if d in self.p.aliases:
                result.append(self.p.aliases[d])
            else:
                result.append(d)
        return result

    def _build_equations(self, m, solve):
        """Build constraints and objective from equation definitions."""
        obj_var_name = solve.objective_var if solve else None
        obj_sense = solve.sense if solve else "minimizing"

        for eqdef in self.p.equation_defs:
            # Build the constraint for each index combination
            domain = eqdef.domain
            if domain:
                set_names = self._domain_set_names(domain)
                index_sets = []
                for sn in set_names:
                    if sn in self.set_elements:
                        index_sets.append(self.set_elements[sn])
                    else:
                        for k, v in self.set_elements.items():
                            if k.lower() == sn.lower():
                                index_sets.append(v)
                                break
                        else:
                            raise GamsParseError(f"Unknown set '{sn}'")
                # iterate over cartesian product
                import itertools

                for combo in itertools.product(*index_sets):
                    env = {}
                    for dn, val in zip(domain, combo):
                        env[dn] = val
                    lhs_expr = self._build_expr(eqdef.lhs, env)
                    rhs_expr = self._build_expr(eqdef.rhs, env)
                    self._add_constraint(m, lhs_expr, eqdef.sense, rhs_expr, eqdef.name)
            else:
                lhs_expr = self._build_expr(eqdef.lhs, {})
                rhs_expr = self._build_expr(eqdef.rhs, {})
                # Check if this equation defines the objective
                # (single equation with obj_var on one side)
                if self._is_obj_equation(eqdef, obj_var_name):
                    obj_expr = self._extract_objective(
                        lhs_expr, rhs_expr, eqdef.sense, obj_var_name
                    )
                    if obj_sense.startswith("min"):
                        m.minimize(obj_expr)
                    else:
                        m.maximize(obj_expr)
                else:
                    self._add_constraint(m, lhs_expr, eqdef.sense, rhs_expr, eqdef.name)

    def _is_obj_equation(self, eqdef, obj_var_name: str | None) -> bool:
        if obj_var_name is None:
            return False
        # Check if lhs or rhs is just the objective variable reference
        lhs = eqdef.lhs
        rhs = eqdef.rhs
        if isinstance(lhs, ExprRef) and lhs.name == obj_var_name and eqdef.sense == "=e=":
            return True
        if isinstance(rhs, ExprRef) and rhs.name == obj_var_name and eqdef.sense == "=e=":
            return True
        return False

    def _extract_objective(self, lhs_expr, rhs_expr, sense, obj_var_name):
        """Extract the objective expression from z =e= expr."""
        from discopt.modeling.core import Variable

        # If lhs is the objective variable, return rhs
        if isinstance(lhs_expr, Variable) and lhs_expr.name == obj_var_name:
            return rhs_expr
        # If rhs is the objective variable, return lhs
        if isinstance(rhs_expr, Variable) and rhs_expr.name == obj_var_name:
            return lhs_expr
        # fallback: return rhs - lhs or similar
        return rhs_expr

    def _add_constraint(self, m, lhs, sense, rhs, name):
        if sense == "=e=":
            m.subject_to(lhs == rhs, name=name)
        elif sense == "=l=":
            m.subject_to(lhs <= rhs, name=name)
        elif sense == "=g=":
            m.subject_to(lhs >= rhs, name=name)

    def _build_expr(self, ast_node, env: dict):
        """Convert a GAMS expression AST node to a discopt Expression."""
        from discopt.modeling import core as dm

        if isinstance(ast_node, ExprNum):
            return dm.Constant(ast_node.value)

        if isinstance(ast_node, ExprRef):
            name = ast_node.name
            # check if it's a loop index variable
            if name in env:
                # it's an index — resolve to the element name (used for param lookup)
                return env[name]  # returns the string element name
            # check discopt variables
            if name in self.dvar_map:
                var = self.dvar_map[name]
                if var.shape == () or var.shape == (1,):
                    return var
                return var
            # check scalars
            if name in self.scalar_values:
                return dm.Constant(self.scalar_values[name])
            # check parameters (0-dim)
            if name in self.param_values:
                pdata = self.param_values[name]
                if len(pdata) == 1:
                    return dm.Constant(list(pdata.values())[0])
            raise GamsParseError(f"Unresolved reference: '{name}'")

        if isinstance(ast_node, ExprIndex):
            name = ast_node.name
            # Resolve indices to integer positions
            indices = []
            for idx_ast in ast_node.indices:
                idx_val = self._build_expr(idx_ast, env)
                indices.append(idx_val)
            # If all indices resolved to strings (set element names), map to ints
            if all(isinstance(v, str) for v in indices):
                return self._resolve_indexed(name, indices)
            # If it's a variable with expression indices
            if name in self.dvar_map:
                var = self.dvar_map[name]
                int_indices = []
                for iv in indices:
                    if isinstance(iv, int):
                        int_indices.append(iv)
                    elif isinstance(iv, str):
                        int_indices.append(self._element_index(name, iv, len(int_indices)))
                    else:
                        int_indices.append(iv)
                if len(int_indices) == 1:
                    return var[int_indices[0]]
                return var[tuple(int_indices)]
            raise GamsParseError(f"Cannot resolve indexed ref: {name}")

        if isinstance(ast_node, ExprBinOp):
            left = self._build_expr(ast_node.left, env)
            right = self._build_expr(ast_node.right, env)
            # handle string returns (index values) — convert to constants
            left = self._ensure_expr(left)
            right = self._ensure_expr(right)
            if ast_node.op == "+":
                return left + right
            if ast_node.op == "-":
                return left - right
            if ast_node.op == "*":
                return left * right
            if ast_node.op == "/":
                return left / right
            if ast_node.op == "**":
                return left**right
            raise GamsParseError(f"Unknown operator: {ast_node.op}")

        if isinstance(ast_node, ExprUnaryMinus):
            operand = self._build_expr(ast_node.operand, env)
            operand = self._ensure_expr(operand)
            return -operand

        if isinstance(ast_node, ExprFunc):
            args = [self._build_expr(a, env) for a in ast_node.args]
            args = [self._ensure_expr(a) for a in args]
            return self._map_func(ast_node.func, args)

        if isinstance(ast_node, ExprSum):
            return self._build_sum(ast_node, env)

        if isinstance(ast_node, ExprProd):
            return self._build_prod(ast_node, env)

        if isinstance(ast_node, ExprCard):
            elems = self.set_elements.get(ast_node.set_name, [])
            return dm.Constant(float(len(elems)))

        if isinstance(ast_node, ExprOrd):
            # ord(i) returns the 1-based ordinal position of element i in its set
            sname = ast_node.set_name
            if sname in env:
                elem_val = env[sname]
                # find which set this belongs to and get position
                for set_name, elems in self.set_elements.items():
                    if elem_val in elems:
                        return dm.Constant(float(elems.index(elem_val) + 1))
            return dm.Constant(1.0)

        raise GamsParseError(f"Unknown AST node: {type(ast_node)}")

    def _ensure_expr(self, val):
        """Convert non-Expression values (strings, ints, floats) to Constant."""
        from discopt.modeling.core import Constant, Expression

        if isinstance(val, Expression):
            return val
        if isinstance(val, (int, float)):
            return Constant(float(val))
        if isinstance(val, str):
            # possibly a scalar or param
            if val in self.scalar_values:
                return Constant(self.scalar_values[val])
            return Constant(0.0)
        return Constant(float(val))

    def _resolve_indexed(self, name: str, str_indices: list[str]):
        """Resolve name(elem1, elem2) to either a param value or variable element."""
        from discopt.modeling.core import Constant

        # check params/tables
        if name in self.param_values:
            key = tuple(str_indices) if len(str_indices) > 1 else (str_indices[0],)
            val = self.param_values[name].get(key, 0.0)
            return Constant(float(val))
        # check variables
        if name in self.dvar_map:
            var = self.dvar_map[name]
            int_indices = []
            for dim, elem in enumerate(str_indices):
                int_indices.append(self._element_index(name, elem, dim))
            if len(int_indices) == 1:
                return var[int_indices[0]]
            return var[tuple(int_indices)]
        raise GamsParseError(f"Cannot resolve '{name}' with indices {str_indices}")

    def _element_index(self, var_or_param_name: str, element: str, dim: int) -> int:
        """Find the integer position of a set element for a given dimension."""
        # figure out which set this dimension belongs to
        # check variable domain first
        domain = None
        if var_or_param_name in self.p.variables:
            domain = self.p.variables[var_or_param_name].domain
        elif var_or_param_name in self.p.parameters:
            domain = self.p.parameters[var_or_param_name].domain
        elif var_or_param_name in self.p.tables:
            domain = self.p.tables[var_or_param_name].domain
        if domain and dim < len(domain):
            set_name = domain[dim]
            if set_name in self.p.aliases:
                set_name = self.p.aliases[set_name]
            elems = self.set_elements.get(set_name, [])
            if element in elems:
                return elems.index(element)
        # fallback: search all sets
        for sn, elems in self.set_elements.items():
            if element in elems:
                return elems.index(element)
        raise GamsParseError(f"Element '{element}' not found in any set")

    def _build_sum(self, node: ExprSum, env: dict):
        """Build a sum over indexed set elements."""
        import itertools

        from discopt.modeling.core import Constant

        index_names = node.index_names
        index_sets = []
        for iname in index_names:
            resolved = iname
            if iname in self.p.aliases:
                resolved = self.p.aliases[iname]
            if resolved in self.set_elements:
                index_sets.append(self.set_elements[resolved])
            elif iname in self.set_elements:
                index_sets.append(self.set_elements[iname])
            else:
                # might be an alias or same-name set
                for sn, se in self.set_elements.items():
                    if sn.lower() == iname.lower():
                        index_sets.append(se)
                        break
                else:
                    raise GamsParseError(f"Unknown set '{iname}' in sum")

        terms = []
        for combo in itertools.product(*index_sets):
            new_env = dict(env)
            for iname, val in zip(index_names, combo):
                new_env[iname] = val
            # Evaluate dollar condition — skip this combination if false
            if node.dollar_cond is not None:
                cond_val = self._eval_const_expr(node.dollar_cond)
                if cond_val is None:
                    # Try evaluating with env substitution
                    cond_val = self._eval_dollar_cond(node.dollar_cond, new_env)
                if cond_val is not None and cond_val == 0.0:
                    continue
            term = self._build_expr(node.body, new_env)
            term = self._ensure_expr(term)
            terms.append(term)

        if not terms:
            return Constant(0.0)
        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    def _build_prod(self, node: ExprProd, env: dict):
        """Build a product over indexed set elements."""
        import itertools

        from discopt.modeling.core import Constant

        index_names = node.index_names
        index_sets = []
        for iname in index_names:
            resolved = iname
            if iname in self.p.aliases:
                resolved = self.p.aliases[iname]
            if resolved in self.set_elements:
                index_sets.append(self.set_elements[resolved])
            elif iname in self.set_elements:
                index_sets.append(self.set_elements[iname])
            else:
                raise GamsParseError(f"Unknown set '{iname}' in prod")

        terms = []
        for combo in itertools.product(*index_sets):
            new_env = dict(env)
            for iname, val in zip(index_names, combo):
                new_env[iname] = val
            if node.dollar_cond is not None:
                cond_val = self._eval_const_expr(node.dollar_cond)
                if cond_val is None:
                    cond_val = self._eval_dollar_cond(node.dollar_cond, new_env)
                if cond_val is not None and cond_val == 0.0:
                    continue
            term = self._build_expr(node.body, new_env)
            term = self._ensure_expr(term)
            terms.append(term)

        if not terms:
            return Constant(1.0)
        result = terms[0]
        for t in terms[1:]:
            result = result * t
        return result

    def _map_func(self, func_name: str, args: list):
        """Map a GAMS function name to a discopt expression."""
        from discopt.modeling import core as dm

        fn = func_name.lower()
        if fn == "exp":
            return dm.exp(args[0])
        if fn == "log":
            return dm.log(args[0])
        if fn == "log2":
            return dm.log2(args[0])
        if fn == "log10":
            return dm.log10(args[0])
        if fn == "sqrt":
            return dm.sqrt(args[0])
        if fn == "sqr":
            return args[0] ** 2
        if fn == "abs":
            return dm.abs_(args[0])
        if fn == "sin":
            return dm.sin(args[0])
        if fn == "cos":
            return dm.cos(args[0])
        if fn == "tan":
            return dm.tan(args[0])
        if fn == "arcsin":
            return dm.FunctionCall("asin", args[0])
        if fn == "arccos":
            return dm.FunctionCall("acos", args[0])
        if fn == "arctan":
            return dm.FunctionCall("atan", args[0])
        if fn == "sinh":
            return dm.FunctionCall("sinh", args[0])
        if fn == "cosh":
            return dm.FunctionCall("cosh", args[0])
        if fn == "tanh":
            return dm.tanh(args[0])
        if fn == "sigmoid":
            return dm.sigmoid(args[0])
        if fn == "power":
            return args[0] ** args[1]
        if fn == "sign":
            return dm.sign(args[0])
        if fn == "min":
            return dm.minimum(args[0], args[1])
        if fn == "max":
            return dm.maximum(args[0], args[1])
        if fn == "errorf":
            return dm.erf(args[0])
        # fallback
        return dm.FunctionCall(fn, *args)

    def _apply_bounds(self, m):
        """Apply .lo, .up, .fx bound assignments."""
        for b in self.p.bounds:
            if b.var_name not in self.dvar_map:
                continue
            var = self.dvar_map[b.var_name]
            val = self._eval_const_expr(b.expr)
            if val is None:
                continue
            if b.domain:
                # indexed bound: apply to specific elements
                import itertools

                set_names = self._domain_set_names(b.domain)
                index_sets = []
                for sn in set_names:
                    if sn in self.set_elements:
                        index_sets.append(self.set_elements[sn])
                    else:
                        for k, v in self.set_elements.items():
                            if k.lower() == sn.lower():
                                index_sets.append(v)
                                break
                for combo in itertools.product(*index_sets):
                    idx = []
                    for dim, elem in enumerate(combo):
                        idx.append(self._element_index(b.var_name, elem, dim))
                    flat_idx = tuple(idx) if len(idx) > 1 else idx[0]
                    if b.suffix == "lo":
                        var.lb = np.array(var.lb, dtype=np.float64)
                        if isinstance(flat_idx, int):
                            var.lb[flat_idx] = val
                        else:
                            var.lb[flat_idx] = val
                    elif b.suffix == "up":
                        var.ub = np.array(var.ub, dtype=np.float64)
                        if isinstance(flat_idx, int):
                            var.ub[flat_idx] = val
                        else:
                            var.ub[flat_idx] = val
                    elif b.suffix == "fx":
                        var.lb = np.array(var.lb, dtype=np.float64)
                        var.ub = np.array(var.ub, dtype=np.float64)
                        if isinstance(flat_idx, int):
                            var.lb[flat_idx] = val
                            var.ub[flat_idx] = val
                        else:
                            var.lb[flat_idx] = val
                            var.ub[flat_idx] = val
            else:
                # scalar bound
                if b.suffix == "lo":
                    var.lb = np.full(var.shape, val) if var.shape else np.asarray(val)
                elif b.suffix == "up":
                    var.ub = np.full(var.shape, val) if var.shape else np.asarray(val)
                elif b.suffix == "fx":
                    var.lb = np.full(var.shape, val) if var.shape else np.asarray(val)
                    var.ub = np.full(var.shape, val) if var.shape else np.asarray(val)


# ── Public API ─────────────────────────────────────────────────
def parse_gams(source: str):
    """Parse GAMS source text and return a discopt Model.

    Parameters
    ----------
    source : str
        GAMS (.gms) source code as a string.

    Returns
    -------
    discopt.modeling.core.Model
        A discopt Model ready to solve.

    Raises
    ------
    GamsParseError
        If the GAMS source cannot be parsed.
    """
    tokens = _tokenize(source)
    parser = _Parser(tokens)
    parser.parse()
    builder = _ModelBuilder(parser)
    return builder.build()


def parse_gams_file(path: str):
    """Parse a GAMS .gms file and return a discopt Model.

    Parameters
    ----------
    path : str
        Path to the .gms file.

    Returns
    -------
    discopt.modeling.core.Model

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    GamsParseError
        If the GAMS source cannot be parsed.
    """
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"GAMS file not found: {path}")
    with open(path) as f:
        source = f.read()
    return parse_gams(source)
