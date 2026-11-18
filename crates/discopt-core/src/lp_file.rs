//! Provenance: derived from HiGHS 1.16.0 (https://github.com/ERGO-Code/HiGHS, Copyright (c) HiGHS,
//! MIT License): a translation of its LP-file reader `highs/io/LpReader.cpp` (the lexer and
//! recursive-descent parser) plus the parts of `Highs::readModel` -> `passModel` that a read model
//! passes through (`lp_data/Highs.cpp`, `lp_data/HighsLpUtils.cpp::assessLp`,
//! `util/HighsMatrixUtils.cpp::assessMatrix`, `model/HighsHessianUtils.cpp`), cited below. See
//! `THIRD_PARTY_NOTICES.md` at the repository root for the license text.
//!
//! CPLEX-LP-format (`.lp`) reader that reproduces HiGHS 1.16.0's `readModel` ->
//! `getLp` (and `getModel().hessian_`) exactly (highs-parity P6: the in-house
//! replacement for HiGHS's reader in the benchmark loader, and the reader users get
//! for `.lp` models).
//!
//! "Exactly" is the contract, tested by an exact differential against highspy >= 1.16
//! (`python/tests/test_lp_reader.py`; against a frozen fixture of HiGHS 1.16.0's
//! readings on an older highspy).
//!
//! **Why 1.16.0 and not 1.15.1 (#2221).** Up to 1.15.1 HiGHS's reader was a staged
//! pipeline (lexer, token classifier, section splitter, per-section parsers in a FIXED
//! order) and this module ported it quirks included. Several of those quirks silently
//! read a valid file as a DIFFERENT model: an objective cost was ASSIGNED per term
//! (`2 x + 3 y - x` gave `c_x = -1`), `minimise`/`maximise` objectives were dropped,
//! `x + inf` in an objective became an infinite offset, keywords were recognised
//! anywhere on a line. HiGHS 1.16.0 (HiGHS PRs #3355, #3446) replaced it with a
//! parser for a formal grammar that rejects invalid files instead. A reader that
//! misreads a valid file is a correctness bug, so this module ports the 1.16.0 reader.
//!
//! The grammar (LpReader.cpp:13-39; keywords case-insensitive):
//!
//! ```text
//! <file>        := { <section> } "end" | <empty>
//! <objective>   := ("minimize" | "maximize") [<name>] <expression>
//! <constraints> := "subject to" { [<name>] <expression> <comparison> <number> <newline> }
//! <bounds>      := "bounds" { <identifier> "free" | <identifier> <cmp> <number>
//!                           | <number> <cmp> <identifier> [<cmp> <number>] } (one per line)
//! general | binary | semi-continuous := { <identifier> }
//! <term>        := <constant> [["*"] <identifier>] | <identifier> | "[" quad "]" ["/" "2"]
//! ```
//!
//! What is not in the grammar (LpReader.cpp:41-67), all reproduced here:
//!
//! 1. **Lexer** ([`lex`], LpReader.cpp:388-508): `\` comments to end of line; C
//!    `isspace` separates; a section keyword is one only as the FIRST token of its line
//!    and not followed by `:`; identifiers are any printable byte except whitespace and
//!    `\ : + - * ^ < > = [ ]`, not starting with a digit, `/`, or `.digit`; a number
//!    is `digits[.digits][e[+-]digits]` (so `2e3x` is `2000` then `x`, `0x10` is `0`
//!    then `x10`); `inf`/`infinity` are numbers only where a number is expected (a
//!    bound or right-hand side), elsewhere they are variable names; a control byte
//!    is an error.
//! 2. **Parser** ([`Parser`], LpReader.cpp:510-1150): sections are parsed in FILE
//!    order and columns are created on first mention in file order; objective
//!    coefficients are SUMMED; a repeated variable in a row is summed (zero sums
//!    dropped); a constant on a constraint's left-hand side is ignored (a warning in
//!    HiGHS); the objective may appear once; `sos`, indicator constraints and
//!    quadratic constraints are errors; `[ ... ]` without `/ 2` is doubled; a binary
//!    column gets upper bound 1 only when its upper bound is still infinite; unnamed
//!    rows are `HiGHS_R<i>`, and all row names are cleared when an explicit name also
//!    starts with `HiGHS_R`.
//! 3. **`passModel`** (H:455-540): no rows or no columns means an empty matrix;
//!    `assessLp` (costs/bounds at or past 1e20 become infinite, NaN or wrong-side
//!    infinite bounds are errors; the matrix entries with `|a| >= 1e15` an error,
//!    `|a| <= 1e-9` dropped, NaN KEPT, which is 1.16's `assessMatrix`) and
//!    `assessHessian` (HU:22-89): duplicates summed, `normaliseHessian` ported
//!    literally (in the square case the gather overwrites its own input, which fixes
//!    the entry order), small entries dropped, missing diagonals filled with explicit
//!    zeros; a Hessian with no nonzeros is cleared.
//!
//! A file HiGHS rejects (`readModel` returns `kError`) is rejected with
//! [`LpFileError`]; a file HiGHS reads with `kWarning` (repeated variables, a
//! left-hand-side constant, crossed bounds, tiny entries) is read the same way.
//!
//! One known limit of "bit-exact": HiGHS sums repeated Hessian terms after a
//! `std::sort` on (column, row), which is not stable; this port uses a stable sort.
//! The two can differ only in the last bit of a Hessian entry written three or more
//! times in one objective (floating-point addition is not associative), and the C++
//! order itself depends on the platform's standard library.
//!
//! Sources: `highs/io/LpReader.cpp` (LpReader.cpp), `lp_data/Highs.cpp` (H),
//! `util/HighsMatrixUtils.cpp` (MU), `model/HighsHessianUtils.cpp` (HU), all HiGHS
//! 1.16.0.

use std::collections::HashMap;

const INF: f64 = f64::INFINITY;

/// A model read from an LP file, as HiGHS's `getLp` (and `hessian_`) returns it.
#[derive(Debug, Clone, PartialEq)]
pub struct LpFileModel {
    /// Rows.
    pub num_row: usize,
    /// Columns, in creation order (see the module documentation).
    pub num_col: usize,
    /// Column-wise matrix (CSC): `a_start` has `num_col + 1` entries.
    pub a_start: Vec<usize>,
    /// Row index of each matrix entry (ascending within a column).
    pub a_index: Vec<usize>,
    /// Value of each matrix entry.
    pub a_value: Vec<f64>,
    /// Objective coefficients.
    pub col_cost: Vec<f64>,
    /// Objective constant.
    pub offset: f64,
    /// The objective section is `maximize` (and no `minimize` section exists).
    pub maximize: bool,
    /// Column lower bounds (`-inf` when unbounded).
    pub col_lower: Vec<f64>,
    /// Column upper bounds (`inf` when unbounded).
    pub col_upper: Vec<f64>,
    /// Row lower sides.
    pub row_lower: Vec<f64>,
    /// Row upper sides.
    pub row_upper: Vec<f64>,
    /// Empty when every column is continuous (HiGHS's convention); else one entry
    /// per column: 0 continuous, 1 integer, 2 semi-continuous, 3 semi-integer.
    pub integrality: Vec<u8>,
    /// Column names (never empty for a non-empty model: names are unique keys).
    pub col_names: Vec<String>,
    /// Row names; automatic ones are `HiGHS_R<row>`; empty when HiGHS clears them.
    pub row_names: Vec<String>,
    /// The objective's name (`obj` in `obj: x + y`), empty when none.
    pub objective_name: String,
    /// The objective Hessian, when it has a nonzero.
    pub hessian: Option<LpHessian>,
}

/// HiGHS's normalised objective Hessian: the lower triangle, column-wise
/// (`HessianFormat::kTriangular`), entry order exactly as `assessHessian` leaves it.
#[derive(Debug, Clone, PartialEq)]
pub struct LpHessian {
    /// Dimension (= number of columns).
    pub dim: usize,
    /// Column starts, `dim + 1` entries.
    pub start: Vec<usize>,
    /// Row index of each entry (the diagonal first in each column).
    pub index: Vec<usize>,
    /// Value of each entry (explicit zeros on otherwise empty diagonals).
    pub value: Vec<f64>,
}

/// Why a file could not be read: HiGHS's `readModel` would return `kError` too.
#[derive(Debug, Clone, PartialEq)]
pub struct LpFileError(pub String);

impl std::fmt::Display for LpFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LP file error: {}", self.0)
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, LpFileError> {
    Err(LpFileError(msg.into()))
}

// ---------------------------------------------------------------------------
// Lexer (LpReader.cpp `Lexer`).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Identifier,
    Number,
    Plus,
    Minus,
    Times,
    Divide,
    Power,
    OpenBracket,
    CloseBracket,
    Colon,
    Less,
    Greater,
    Equal,
    Implies,
    EndOfFile,
}

/// A token: its kind, location (byte offset, length, line), whether it is the first
/// token on its line, and a number's value.
#[derive(Clone, Copy, Debug)]
struct Token {
    kind: Kind,
    pos: usize,
    len: usize,
    line: usize,
    line_start: bool,
    value: f64,
}

fn is_digit(c: Option<u8>) -> bool {
    matches!(c, Some(b'0'..=b'9'))
}

/// `isIdentifierChar`: printable, not DEL, not one of the operators.
fn is_identifier_char(c: Option<u8>) -> bool {
    match c {
        Some(c) => c > 0x20 && c != 0x7F && !b"\\:+-*^<>=[]".contains(&c),
        None => false,
    }
}

/// C `isspace` in the "C" locale.
fn c_isspace(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

/// The whole file as tokens, ending with one `EndOfFile` token. HiGHS lexes lazily,
/// but a successful parse always reaches the end of the file, so lexing eagerly
/// accepts and rejects exactly the same files.
fn lex(text: &[u8]) -> Result<Vec<Token>, LpFileError> {
    let at = |p: usize| text.get(p).copied();
    let mut toks = Vec::new();
    let (mut pos, mut line, mut line_start) = (0usize, 1usize, true);
    let (mut last_end, mut last_line) = (0usize, 1usize);
    loop {
        // Skip whitespace and comments.
        while let Some(c) = at(pos) {
            if c == b'\\' {
                while matches!(at(pos), Some(c) if c != b'\n') {
                    pos += 1;
                }
                continue;
            }
            if c == b'\n' {
                line += 1;
                line_start = true;
            } else if (0x21..=0x7E).contains(&c) || !c_isspace(c) {
                break;
            }
            pos += 1;
        }
        let Some(c) = at(pos) else {
            toks.push(Token {
                kind: Kind::EndOfFile,
                pos: last_end,
                len: 0,
                line: last_line,
                line_start: false,
                value: 0.0,
            });
            return Ok(toks);
        };
        let mut tok = Token {
            kind: Kind::EndOfFile,
            pos,
            len: 1,
            line,
            line_start,
            value: 0.0,
        };
        line_start = false;
        let next = at(pos + 1);
        if is_digit(Some(c)) || (c == b'.' && is_digit(next)) {
            // <constant> := <digits> ["." [<digits>]] [<exponent>] | "." <digits> [<exponent>]
            let mut end = pos;
            while is_digit(at(end)) {
                end += 1;
            }
            if at(end) == Some(b'.') {
                end += 1;
                while is_digit(at(end)) {
                    end += 1;
                }
            }
            if matches!(at(end), Some(b'e' | b'E')) {
                let mut exponent = end + 1;
                if matches!(at(exponent), Some(b'+' | b'-')) {
                    exponent += 1;
                }
                if is_digit(at(exponent)) {
                    end = exponent;
                    while is_digit(at(end)) {
                        end += 1;
                    }
                }
            }
            tok.kind = Kind::Number;
            tok.len = end - pos;
            // The text is ASCII digits/./e/sign, so this is `strtod` (correctly rounded,
            // overflow to inf, underflow to 0).
            let s = std::str::from_utf8(&text[pos..end]).expect("ASCII number");
            tok.value = s
                .parse::<f64>()
                .map_err(|e| LpFileError(format!("line {line}: bad number {s:?}: {e}")))?;
        } else if is_identifier_char(Some(c)) && c != b'/' {
            tok.kind = Kind::Identifier;
            while is_identifier_char(at(pos + tok.len)) {
                tok.len += 1;
            }
        } else {
            tok.kind = match c {
                b'+' => Kind::Plus,
                b'-' if next == Some(b'>') => {
                    tok.len = 2;
                    Kind::Implies
                }
                b'-' => Kind::Minus,
                b'*' => Kind::Times,
                b'/' => Kind::Divide,
                b'^' => Kind::Power,
                b'[' => Kind::OpenBracket,
                b']' => Kind::CloseBracket,
                b':' => Kind::Colon,
                b'<' | b'>' => {
                    if next == Some(b'=') {
                        tok.len = 2;
                    }
                    if c == b'<' {
                        Kind::Less
                    } else {
                        Kind::Greater
                    }
                }
                b'=' => {
                    if matches!(next, Some(b'=' | b'<' | b'>')) {
                        tok.len = 2;
                    }
                    match next {
                        Some(b'<') => Kind::Less,
                        Some(b'>') => Kind::Greater,
                        _ => Kind::Equal,
                    }
                }
                _ => {
                    return err(format!(
                        "line {line}: unexpected control character 0x{c:02X}"
                    ))
                }
            };
        }
        pos += tok.len;
        last_end = pos;
        last_line = line;
        toks.push(tok);
    }
}

// ---------------------------------------------------------------------------
// Parser (LpReader.cpp `Parser`).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    None,
    Minimize,
    Maximize,
    Constraints,
    Bounds,
    General,
    Binary,
    Semi,
    Sos,
    End,
}

/// The section keywords that are one token long (`kKeywords`).
const KEYWORDS: &[(&[u8], Section)] = &[
    (b"minimize", Section::Minimize),
    (b"minimise", Section::Minimize),
    (b"minimum", Section::Minimize),
    (b"min", Section::Minimize),
    (b"maximize", Section::Maximize),
    (b"maximise", Section::Maximize),
    (b"maximum", Section::Maximize),
    (b"max", Section::Maximize),
    (b"st", Section::Constraints),
    (b"s.t.", Section::Constraints),
    (b"st.", Section::Constraints),
    (b"bounds", Section::Bounds),
    (b"bound", Section::Bounds),
    (b"general", Section::General),
    (b"generals", Section::General),
    (b"gen", Section::General),
    (b"integer", Section::General),
    (b"integers", Section::General),
    (b"binary", Section::Binary),
    (b"binaries", Section::Binary),
    (b"bin", Section::Binary),
    (b"semi", Section::Semi),
    (b"semis", Section::Semi),
    (b"sos", Section::Sos),
    (b"end", Section::End),
];

const ROW_PREFIX: &[u8] = b"HiGHS_R";

fn is_comparison(t: &Token) -> bool {
    matches!(t.kind, Kind::Less | Kind::Greater | Kind::Equal)
}

fn is_sign(t: &Token) -> bool {
    matches!(t.kind, Kind::Plus | Kind::Minus)
}

/// HiGHS's `HessianEntry`: `coef * x[row]` in column `col` of the square Hessian.
#[derive(Clone, Copy)]
struct HessianEntry {
    col: usize,
    row: usize,
    value: f64,
}

/// An entry of the row being parsed (`RowEntry`).
struct RowEntry {
    col: usize,
    value: f64,
}

struct Parser<'a> {
    text: &'a [u8],
    toks: Vec<Token>,
    next_tok: usize,
    // The model, as `HighsLp` fields.
    maximize: bool,
    offset: f64,
    objective_name: Vec<u8>,
    col_names: Vec<Vec<u8>>,
    col_index: HashMap<Vec<u8>, usize>,
    col_cost: Vec<f64>,
    col_lower: Vec<f64>,
    col_upper: Vec<f64>,
    integrality: Vec<u8>,
    row_names: Vec<Vec<u8>>,
    row_lower: Vec<f64>,
    row_upper: Vec<f64>,
    // Row-wise matrix.
    ar_start: Vec<usize>,
    ar_index: Vec<usize>,
    ar_value: Vec<f64>,
    has_integrality: bool,
    binary_cols: Vec<usize>,
    hessian_entries: Vec<HessianEntry>,
    used_prefix: bool,
    prefix_ok: bool,
    row_entries: Vec<RowEntry>,
    col_slot: Vec<Option<usize>>,
}

/// HighsVarType codes.
const CONTINUOUS: u8 = 0;
const INTEGER: u8 = 1;
const SEMI_CONTINUOUS: u8 = 2;
const SEMI_INTEGER: u8 = 3;

impl<'a> Parser<'a> {
    fn new(text: &'a [u8], toks: Vec<Token>) -> Self {
        Parser {
            text,
            toks,
            next_tok: 0,
            maximize: false,
            offset: 0.0,
            objective_name: Vec::new(),
            col_names: Vec::new(),
            col_index: HashMap::new(),
            col_cost: Vec::new(),
            col_lower: Vec::new(),
            col_upper: Vec::new(),
            integrality: Vec::new(),
            row_names: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            ar_start: vec![0],
            ar_index: Vec::new(),
            ar_value: Vec::new(),
            has_integrality: false,
            binary_cols: Vec::new(),
            hessian_entries: Vec::new(),
            used_prefix: false,
            prefix_ok: true,
            row_entries: Vec::new(),
            col_slot: Vec::new(),
        }
    }

    /// The token `n` places ahead (the end-of-file token repeats).
    fn peek(&self, n: usize) -> Token {
        self.toks[(self.next_tok + n).min(self.toks.len() - 1)]
    }

    fn next(&mut self) -> Token {
        let t = self.peek(0);
        if self.next_tok < self.toks.len() - 1 {
            self.next_tok += 1;
        }
        t
    }

    fn bytes(&self, t: &Token) -> &'a [u8] {
        &self.text[t.pos..t.pos + t.len]
    }

    fn eq_ci(&self, t: &Token, lower: &[u8]) -> bool {
        t.len == lower.len()
            && self
                .bytes(t)
                .iter()
                .zip(lower)
                .all(|(a, b)| a.to_ascii_lowercase() == *b)
    }

    fn describe(&self, t: &Token) -> String {
        if t.kind == Kind::EndOfFile {
            "end of file".to_string()
        } else {
            format!("`{}`", String::from_utf8_lossy(self.bytes(t)))
        }
    }

    fn error<T>(&self, t: &Token, message: impl Into<String>) -> Result<T, LpFileError> {
        err(format!("line {}: {}", t.line, message.into()))
    }

    /// `keywordAt`: the section starting at token `i`, and its length in tokens.
    fn keyword_at(&self, i: usize) -> (Section, usize) {
        let token = self.peek(i);
        if token.kind != Kind::Identifier || !token.line_start {
            return (Section::None, 0);
        }
        // A keyword followed by ":" is a name.
        let second = self.peek(i + 1);
        if second.kind == Kind::Colon {
            return (Section::None, 0);
        }
        if (self.eq_ci(&token, b"subject") && self.eq_ci(&second, b"to"))
            || (self.eq_ci(&token, b"such") && self.eq_ci(&second, b"that"))
        {
            return if second.line_start {
                (Section::None, 0)
            } else {
                (Section::Constraints, 2)
            };
        }
        // "semi-continuous", possibly with spaces, but on one line.
        if self.eq_ci(&token, b"semi") && second.kind == Kind::Minus && !second.line_start {
            let third = self.peek(i + 2);
            if third.kind == Kind::Identifier
                && !third.line_start
                && self.eq_ci(&third, b"continuous")
            {
                return (Section::Semi, 3);
            }
        }
        for &(kw, section) in KEYWORDS {
            if self.eq_ci(&token, kw) {
                return (section, 1);
            }
        }
        (Section::None, 0)
    }

    fn at_keyword(&self) -> bool {
        self.keyword_at(0).0 != Section::None
    }

    fn at_section_end(&self) -> bool {
        self.peek(0).kind == Kind::EndOfFile || self.at_keyword()
    }

    /// `getColumn`: the column named by the token, created `[0, inf]` if new.
    fn get_column(&mut self, t: &Token) -> usize {
        let name = self.bytes(t);
        if let Some(&col) = self.col_index.get(name) {
            return col;
        }
        let col = self.col_names.len();
        self.col_names.push(name.to_vec());
        self.col_index.insert(name.to_vec(), col);
        self.col_cost.push(0.0);
        self.col_lower.push(0.0);
        self.col_upper.push(INF);
        self.integrality.push(CONTINUOUS);
        self.col_slot.push(None);
        col
    }

    fn parse(&mut self) -> Result<(), LpFileError> {
        // An empty file (or one with only comments) is an empty model.
        if self.peek(0).kind == Kind::EndOfFile {
            return Ok(());
        }
        let mut objective_line: Option<usize> = None;
        loop {
            let token = self.peek(0);
            let (section, num_tokens) = self.keyword_at(0);
            if section == Section::None {
                if token.kind == Kind::EndOfFile {
                    return self.error(&token, "expected `end`, found end of file");
                }
                return self.error(
                    &token,
                    format!(
                        "expected a section keyword, found {}",
                        self.describe(&token)
                    ),
                );
            }
            for _ in 0..num_tokens {
                self.next();
            }
            match section {
                Section::Minimize | Section::Maximize => {
                    if let Some(first) = objective_line {
                        return self.error(
                            &token,
                            format!(
                                "the objective is defined more than once (first on line {first})"
                            ),
                        );
                    }
                    objective_line = Some(token.line);
                    self.maximize = section == Section::Maximize;
                    self.parse_objective()?;
                }
                Section::Constraints => {
                    while !self.at_section_end() {
                        self.parse_constraint()?;
                    }
                }
                Section::Bounds => {
                    while !self.at_section_end() {
                        self.parse_bound()?;
                    }
                }
                Section::General | Section::Binary | Section::Semi => {
                    self.parse_variable_list(section)?;
                }
                Section::Sos => {
                    return self.error(&token, "SOS constraints are not supported by HiGHS")
                }
                Section::End => {
                    let after = self.peek(0);
                    if after.kind != Kind::EndOfFile {
                        return self.error(
                            &after,
                            format!("expected end of file, found {}", self.describe(&after)),
                        );
                    }
                    return Ok(());
                }
                Section::None => unreachable!(),
            }
        }
    }

    /// `<name> := (<identifier> | <number>) ":"`
    fn parse_name(&mut self) -> Option<Vec<u8>> {
        let kind = self.peek(0).kind;
        if !matches!(kind, Kind::Identifier | Kind::Number) || self.peek(1).kind != Kind::Colon {
            return None;
        }
        let t = self.next();
        self.next(); // :
        Some(self.bytes(&t).to_vec())
    }

    /// `<number> := [<signs>] (<constant> | "inf" | "infinity")`
    fn parse_number(&mut self, label: &str) -> Result<f64, LpFileError> {
        let mut sign = 1.0f64;
        while is_sign(&self.peek(0)) {
            if self.next().kind == Kind::Minus {
                sign = -sign;
            }
        }
        let token = self.peek(0);
        if token.kind == Kind::Number {
            self.next();
            return Ok(sign * token.value);
        }
        if token.kind == Kind::Identifier
            && (self.eq_ci(&token, b"inf") || self.eq_ci(&token, b"infinity"))
        {
            self.next();
            return Ok(sign * INF);
        }
        self.error(
            &token,
            format!(
                "expected a number, found {} ({label})",
                self.describe(&token)
            ),
        )
    }

    fn parse_variable(&mut self, label: &str) -> Result<usize, LpFileError> {
        let var = self.peek(0);
        if var.kind != Kind::Identifier || self.at_keyword() {
            return self.error(
                &var,
                format!(
                    "expected a variable, found {} ({label})",
                    self.describe(&var)
                ),
            );
        }
        let t = self.next();
        Ok(self.get_column(&t))
    }

    /// The next token must be on a new line.
    fn check_new_line(&self, label: &str) -> Result<(), LpFileError> {
        let t = self.peek(0);
        if t.kind != Kind::EndOfFile && !t.line_start {
            return self.error(
                &t,
                format!("expected a new line, found {} ({label})", self.describe(&t)),
            );
        }
        Ok(())
    }

    fn parse_objective(&mut self) -> Result<(), LpFileError> {
        if let Some(name) = self.parse_name() {
            self.objective_name = name;
        }
        self.parse_expression(true)
    }

    /// `<expression> := [<signs>] <term> { <signs> <term> }`; in the objective it ends
    /// at the next section, in a constraint at the comparison.
    fn parse_expression(&mut self, objective: bool) -> Result<(), LpFileError> {
        let mut first = true;
        loop {
            let token = self.peek(0);
            if token.kind == Kind::EndOfFile
                || self.at_keyword()
                || (!objective && is_comparison(&token))
            {
                return Ok(());
            }
            let mut has_sign = false;
            let mut sign = 1.0f64;
            while is_sign(&self.peek(0)) {
                if self.next().kind == Kind::Minus {
                    sign = -sign;
                }
                has_sign = true;
            }
            if !first && !has_sign {
                return self.error(
                    &token,
                    format!("expected `+` or `-`, found {}", self.describe(&token)),
                );
            }
            self.parse_term(sign, objective)?;
            first = false;
        }
    }

    /// `<term> := <constant> [["*"] <identifier>] | <identifier> | <quadratic>`
    fn parse_term(&mut self, sign: f64, objective: bool) -> Result<(), LpFileError> {
        let token = self.peek(0);
        if token.kind == Kind::Number {
            self.next();
            let coef = sign * token.value;
            let next = self.peek(0);
            if next.kind == Kind::Times {
                self.next();
                let col = self.parse_variable("expected a variable after `*`")?;
                self.add_variable_term(col, coef, objective)?;
            } else if next.kind == Kind::Identifier && !self.at_keyword() {
                let var = self.next();
                let col = self.get_column(&var);
                self.add_variable_term(col, coef, objective)?;
            } else if next.kind == Kind::OpenBracket {
                return self.error(&next, "a coefficient cannot multiply `[ ]`");
            } else if objective {
                self.offset += coef;
            }
            // else: a constant on the left-hand side of a constraint is ignored
            // (a warning in HiGHS, for backwards compatibility).
        } else if token.kind == Kind::Identifier && !self.at_keyword() {
            let var = self.next();
            let col = self.get_column(&var);
            self.add_variable_term(col, sign, objective)?;
        } else if token.kind == Kind::OpenBracket {
            if !objective {
                return self.error(&token, "quadratic constraints are not supported by HiGHS");
            }
            self.parse_quadratic(sign)?;
        } else {
            return self.error(
                &token,
                format!("expected a term, found {}", self.describe(&token)),
            );
        }
        Ok(())
    }

    fn add_variable_term(
        &mut self,
        col: usize,
        coef: f64,
        objective: bool,
    ) -> Result<(), LpFileError> {
        if objective {
            self.col_cost[col] += coef;
        } else {
            // Sum repeated variables (a warning in HiGHS).
            match self.col_slot[col] {
                None => {
                    self.col_slot[col] = Some(self.row_entries.len());
                    self.row_entries.push(RowEntry { col, value: coef });
                }
                Some(slot) => self.row_entries[slot].value += coef,
            }
        }
        let next = self.peek(0);
        if matches!(next.kind, Kind::Times | Kind::Power) {
            return self.error(&next, "quadratic terms must be inside `[` and `]`");
        }
        Ok(())
    }

    /// `<quadratic> := "[" [[<signs>] <quad-term> { <signs> <quad-term> }] "]" ["/" "2"]`.
    /// The terms of 2 q(x) are added, and halved when "/ 2" follows.
    fn parse_quadratic(&mut self, sign: f64) -> Result<(), LpFileError> {
        let open = self.next();
        let first_entry = self.hessian_entries.len();
        let mut first = true;
        while self.peek(0).kind != Kind::CloseBracket {
            let token = self.peek(0);
            if token.kind == Kind::EndOfFile || self.at_keyword() {
                return self.error(
                    &token,
                    format!(
                        "expected `]`, found {} (the `[` on line {} is never closed)",
                        self.describe(&token),
                        open.line
                    ),
                );
            }
            let mut has_sign = false;
            let mut term_sign = sign;
            while is_sign(&self.peek(0)) {
                if self.next().kind == Kind::Minus {
                    term_sign = -term_sign;
                }
                has_sign = true;
            }
            if !first && !has_sign {
                return self.error(
                    &token,
                    format!("expected `+`, `-`, or `]`, found {}", self.describe(&token)),
                );
            }
            self.parse_quadratic_term(term_sign, 2.0)?;
            first = false;
        }
        self.next(); // ]
        if self.peek(0).kind == Kind::Divide {
            self.next();
            let two = self.peek(0);
            if two.kind != Kind::Number || two.value != 2.0 {
                return self.error(
                    &two,
                    format!("expected `2` after `/`, found {}", self.describe(&two)),
                );
            }
            self.next();
            for e in &mut self.hessian_entries[first_entry..] {
                e.value /= 2.0;
            }
        }
        Ok(())
    }

    /// `<quad-term> := [<constant> ["*"]] <identifier> ("^" "2" | "*" <identifier>)`
    fn parse_quadratic_term(&mut self, sign: f64, scale: f64) -> Result<(), LpFileError> {
        let mut coef = sign * scale;
        if self.peek(0).kind == Kind::Number {
            coef *= self.next().value;
            if self.peek(0).kind == Kind::Times {
                self.next();
            }
        }
        let col1 = self.parse_variable("expected a variable in this term")?;
        let op = self.peek(0);
        let mut col2 = col1;
        if op.kind == Kind::Power {
            self.next();
            let two = self.peek(0);
            if two.kind != Kind::Number || two.value != 2.0 {
                return self.error(
                    &two,
                    format!("expected `2` after `^`, found {}", self.describe(&two)),
                );
            }
            self.next();
        } else if op.kind == Kind::Times {
            self.next();
            col2 = self.parse_variable("expected a variable after `*`")?;
        } else {
            return self.error(
                &op,
                format!("expected `^` or `*`, found {}", self.describe(&op)),
            );
        }
        // addHessianEntry: coef * x[col1] * x[col2] added to x'Qx.
        if col1 == col2 {
            self.hessian_entries.push(HessianEntry {
                col: col1,
                row: col1,
                value: coef,
            });
        } else {
            self.hessian_entries.push(HessianEntry {
                col: col1,
                row: col2,
                value: coef / 2.0,
            });
            self.hessian_entries.push(HessianEntry {
                col: col2,
                row: col1,
                value: coef / 2.0,
            });
        }
        Ok(())
    }

    /// Constraints that are valid LP syntax but not supported by HiGHS.
    fn check_unsupported_constraint(&self) -> Result<(), LpFileError> {
        let t = self.peek(0);
        // <name> S1:: x1:1 x2:2
        if matches!(t.kind, Kind::Identifier | Kind::Number)
            && self.peek(1).kind == Kind::Colon
            && self.peek(2).kind == Kind::Colon
        {
            return self.error(&t, "SOS constraints are not supported by HiGHS");
        }
        // z = 1 -> x + y <= 1
        if t.kind == Kind::Identifier
            && self.peek(1).kind == Kind::Equal
            && self.peek(2).kind == Kind::Number
            && self.peek(3).kind == Kind::Implies
        {
            return self.error(&t, "indicator constraints are not supported by HiGHS");
        }
        Ok(())
    }

    /// `<constraint> := [<name>] <expression> <comparison> <number> <newline>`
    fn parse_constraint(&mut self) -> Result<(), LpFileError> {
        self.check_unsupported_constraint()?;
        let mut name = Vec::new();
        if let Some(n) = self.parse_name() {
            name = n;
            self.check_unsupported_constraint()?;
        }
        self.parse_expression(false)?;
        let comparison = self.peek(0);
        if !is_comparison(&comparison) {
            return self.error(
                &comparison,
                format!(
                    "expected a comparison, found {}",
                    self.describe(&comparison)
                ),
            );
        }
        self.next();
        let rhs = self.parse_number("expected the right-hand side")?;
        let after = self.peek(0);
        if after.kind == Kind::Identifier && !after.line_start {
            return self.error(
                &after,
                format!(
                    "expected a new line, found {} (variable on the right-hand side)",
                    self.describe(&after)
                ),
            );
        }
        self.check_new_line("expected the constraint to end before this")?;
        let lower = if comparison.kind != Kind::Less {
            rhs
        } else {
            -INF
        };
        let upper = if comparison.kind != Kind::Greater {
            rhs
        } else {
            INF
        };
        self.finish_row(name, lower, upper);
        Ok(())
    }

    fn finish_row(&mut self, mut name: Vec<u8>, lower: f64, upper: f64) {
        for e in self.row_entries.drain(..) {
            self.col_slot[e.col] = None;
            if e.value == 0.0 {
                continue;
            }
            self.ar_index.push(e.col);
            self.ar_value.push(e.value);
        }
        self.ar_start.push(self.ar_index.len());
        // An unnamed row gets a name, remembering if it could clash with one in the file.
        if name.starts_with(ROW_PREFIX) {
            self.prefix_ok = false;
        } else if name.is_empty() {
            name = format!("HiGHS_R{}", self.row_names.len()).into_bytes();
            self.used_prefix = true;
        }
        self.row_names.push(name);
        self.row_lower.push(lower);
        self.row_upper.push(upper);
    }

    /// `<bound> := <identifier> "free" | <identifier> <comparison> <number>
    ///           | <number> <comparison> <identifier> [<comparison> <number>]`,
    /// each followed by a new line.
    fn parse_bound(&mut self) -> Result<(), LpFileError> {
        let token = self.peek(0);
        let infinity = self.eq_ci(&token, b"inf") || self.eq_ci(&token, b"infinity");
        if token.kind == Kind::Identifier && !infinity {
            // x free | x <= 1
            let t = self.next();
            let col = self.get_column(&t);
            let op = self.peek(0);
            if op.kind == Kind::Identifier && self.eq_ci(&op, b"free") {
                self.next();
                self.col_lower[col] = -INF;
                self.col_upper[col] = INF;
            } else if is_comparison(&op) {
                self.next();
                let value = self.parse_number("expected the value of the bound")?;
                if op.kind != Kind::Less {
                    self.col_lower[col] = value;
                }
                if op.kind != Kind::Greater {
                    self.col_upper[col] = value;
                }
            } else {
                return self.error(
                    &op,
                    format!(
                        "expected `free` or a comparison, found {}",
                        self.describe(&op)
                    ),
                );
            }
        } else {
            // 1 <= x | 1 <= x <= 2
            if !is_sign(&token) && token.kind != Kind::Number && !infinity {
                return self.error(
                    &token,
                    format!("expected a bound, found {}", self.describe(&token)),
                );
            }
            let value = self.parse_number("expected a number")?;
            let op = self.peek(0);
            if !is_comparison(&op) {
                return self.error(
                    &op,
                    format!("expected a comparison, found {}", self.describe(&op)),
                );
            }
            self.next();
            let col = self.parse_variable("expected a variable")?;
            let op2 = self.peek(0);
            if is_comparison(&op2) {
                // 1 <= x <= 2 | 2 >= x >= 1
                if op.kind == Kind::Equal {
                    return self.error(&op, "a bound on both sides of a variable cannot use `=`");
                }
                if op2.kind != op.kind {
                    return self.error(
                        &op2,
                        "the comparisons in a bound on both sides of a variable must match",
                    );
                }
                self.next();
                let value2 = self.parse_number("expected the value of the bound")?;
                let less = op.kind == Kind::Less;
                self.col_lower[col] = if less { value } else { value2 };
                self.col_upper[col] = if less { value2 } else { value };
            } else {
                if op.kind != Kind::Greater {
                    self.col_lower[col] = value;
                }
                if op.kind != Kind::Less {
                    self.col_upper[col] = value;
                }
            }
        }
        self.check_new_line("each bound must be on its own line")
    }

    /// `{ <identifier> }` in the general, binary and semi-continuous sections.
    fn parse_variable_list(&mut self, section: Section) -> Result<(), LpFileError> {
        while !self.at_section_end() {
            let col = self.parse_variable("expected a variable")?;
            let ty = self.integrality[col];
            let semi = ty == SEMI_CONTINUOUS || ty == SEMI_INTEGER;
            let integer = ty == INTEGER || ty == SEMI_INTEGER;
            self.integrality[col] = if section == Section::Semi {
                if integer {
                    SEMI_INTEGER
                } else {
                    SEMI_CONTINUOUS
                }
            } else {
                // A binary column has upper bound 1 unless the bounds section gives one.
                if section == Section::Binary {
                    self.binary_cols.push(col);
                }
                if semi {
                    SEMI_INTEGER
                } else {
                    INTEGER
                }
            };
            self.has_integrality = true;
        }
        Ok(())
    }
}

/// `Parser::finish`'s square Hessian: entries sorted by (column, row), repeats summed
/// (in sorted order, starting from 0), zero sums dropped; `None` when empty. HiGHS's
/// `std::sort` is not stable; see the module documentation.
fn finish_hessian(mut entries: Vec<HessianEntry>, num_col: usize) -> Option<LpHessian> {
    entries.sort_by_key(|e| (e.col, e.row));
    let mut start: Vec<usize> = Vec::new();
    let mut index = Vec::new();
    let mut value = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let (col, row) = (entries[i].col, entries[i].row);
        let mut v = 0.0f64;
        while i < entries.len() && entries[i].col == col && entries[i].row == row {
            v += entries[i].value;
            i += 1;
        }
        if v == 0.0 {
            continue;
        }
        while start.len() <= col {
            start.push(index.len());
        }
        index.push(row);
        value.push(v);
    }
    if index.is_empty() {
        return None;
    }
    while start.len() <= num_col {
        start.push(index.len());
    }
    Some(LpHessian {
        dim: num_col,
        start,
        index,
        value,
    })
}

// ---------------------------------------------------------------------------
// passModel: assessLp, assessHessian.
// ---------------------------------------------------------------------------

/// `assessMatrix` (MU:36-330, HiGHS 1.16.0) on a column-wise matrix. An entry's
/// magnitude is OK unless `|v| >= large` (an error) or `|v| <= small` (dropped), so a
/// NaN entry is KEPT (1.15.1 dropped it). Without `sum_duplicates` the magnitude is
/// checked first and a repeated index among the kept entries is an error; with it,
/// repeats are summed into their first position and the sums checked afterwards.
fn assess_matrix(
    start: &mut [usize],
    index: &mut Vec<usize>,
    value: &mut Vec<f64>,
    small: f64,
    large: f64,
    sum_duplicates: bool,
) -> Result<(), LpFileError> {
    let num_vec = start.len() - 1;
    let mut new_nz = 0usize;
    let mut large_found = false;
    let mut ok_magnitude = |v: f64| {
        let a = v.abs();
        if a >= large {
            large_found = true;
            return false;
        }
        a > small || a.is_nan()
    };
    let mut seen: HashMap<usize, usize> = HashMap::new();
    for ix in 0..num_vec {
        let (from, to) = (start[ix], start[ix + 1]);
        start[ix] = new_nz;
        for el in from..to {
            let comp = index[el];
            if !sum_duplicates && !ok_magnitude(value[el]) {
                continue;
            }
            if let Some(&prev) = seen.get(&comp) {
                if !sum_duplicates {
                    return err("duplicate matrix index");
                }
                value[prev] += value[el];
                continue;
            }
            seen.insert(comp, new_nz);
            index[new_nz] = index[el];
            value[new_nz] = value[el];
            new_nz += 1;
        }
        if sum_duplicates {
            let (from, to) = (start[ix], new_nz);
            new_nz = start[ix];
            for el in from..to {
                if !ok_magnitude(value[el]) {
                    continue;
                }
                index[new_nz] = index[el];
                value[new_nz] = value[el];
                new_nz += 1;
            }
        }
        seen.clear();
    }
    start[num_vec] = new_nz;
    index.truncate(new_nz);
    value.truncate(new_nz);
    if large_found {
        return err("matrix entry too large");
    }
    Ok(())
}

/// `normaliseHessian` (HU:309-545) for a SQUARE Hessian (the only format the LP
/// reader builds), ported literally: the strict upper triangle is moved into a
/// row-wise `upper`, the lower part compacted in place, and the gather then writes
/// into the same arrays it reads (`from_hessian` aliases `hessian`, HU:387), which
/// is what fixes the final entry order.
fn normalise_square_hessian(
    dim: usize,
    start: &mut [usize],
    index: &mut Vec<usize>,
    value: &mut Vec<f64>,
) -> Result<(), LpFileError> {
    let mut upper_length = vec![0usize; dim];
    let (mut num_upper, mut num_lower) = (0usize, 0usize);
    for col in 0..dim {
        for &row in &index[start[col]..start[col + 1]] {
            if row < col {
                upper_length[row] += 1;
                num_upper += 1;
            } else if row > col {
                num_lower += 1;
            }
        }
    }
    if num_upper != num_lower {
        return err("Hessian is not symmetric");
    }
    let mut upper_start = vec![0usize];
    let mut upper_index: Vec<usize> = Vec::new();
    let mut upper_value: Vec<f64> = Vec::new();
    if num_upper > 0 {
        let mut nl = 0usize;
        for row in 0..dim {
            upper_start.push(upper_start[row] + upper_length[row]);
            upper_length[row] = upper_start[row];
        }
        upper_index.resize(num_upper, 0);
        upper_value.resize(num_upper, 0.0);
        for col in 0..dim {
            let from = start[col];
            start[col] = nl;
            for el in from..start[col + 1] {
                let row = index[el];
                if row < col {
                    upper_index[upper_length[row]] = col;
                    upper_value[upper_length[row]] = value[el];
                    upper_length[row] += 1;
                } else {
                    index[nl] = row;
                    value[nl] = value[el];
                    nl += 1;
                }
            }
        }
        start[dim] = nl;
    } else {
        upper_start.resize(dim + 1, 0);
    }
    let mut lower_on_below = vec![0.0f64; dim];
    let mut upper_off = vec![0.0f64; dim];
    let mut illegal_asymmetry = false;
    let mut n_el = 0usize;
    let truthy = |v: f64| v != 0.0; // C++ `if (double)`: NaN is true
                                    // A write past the end can only follow an asymmetric input, which the LP
                                    // reader's construction never produces; extend rather than index out of bounds.
    let put = |index: &mut Vec<usize>, value: &mut Vec<f64>, at: usize, r: usize, v: f64| {
        if at < index.len() {
            index[at] = r;
            value[at] = v;
        } else {
            index.push(r);
            value.push(v);
        }
    };
    for col in 0..dim {
        for el in start[col]..start[col + 1] {
            lower_on_below[index[el]] = value[el];
        }
        for el in upper_start[col]..upper_start[col + 1] {
            let row = upper_index[el];
            upper_off[row] = upper_value[el];
            let asym = (upper_off[row] - lower_on_below[row]).abs();
            if asym > 1e-10 {
                illegal_asymmetry = true;
            } else if truthy(asym) {
                let avg = (upper_off[row] + lower_on_below[row]) * 0.5;
                upper_off[row] = avg;
                lower_on_below[row] = avg;
            }
        }
        // Gather: `from_hessian` IS `hessian`, so these writes can overwrite entries
        // of this column not yet read (HU:459-492).
        let from = start[col];
        start[col] = n_el;
        if truthy(lower_on_below[col]) {
            put(index, value, n_el, col, lower_on_below[col]);
            n_el += 1;
            lower_on_below[col] = 0.0;
        }
        for el in from..start[col + 1] {
            let row = index[el];
            if truthy(lower_on_below[row]) {
                put(index, value, n_el, row, lower_on_below[row]);
                n_el += 1;
                lower_on_below[row] = 0.0;
            }
        }
        for &row in &upper_index[upper_start[col]..upper_start[col + 1]] {
            if truthy(lower_on_below[row]) {
                put(index, value, n_el, row, lower_on_below[row]);
                n_el += 1;
                lower_on_below[row] = 0.0;
            }
            upper_off[row] = 0.0;
        }
    }
    start[dim] = n_el;
    index.truncate(n_el);
    value.truncate(n_el);
    if illegal_asymmetry {
        return err("Hessian asymmetry beyond tolerance");
    }
    Ok(())
}

/// `completeHessianDiagonal` (HU:93-150): an explicit zero first in every column
/// whose first entry is not its diagonal.
fn complete_hessian_diagonal(h: &mut LpHessian) {
    let dim = h.dim;
    let nnz = h.start[dim];
    let missing = (0..dim)
        .filter(|&c| {
            let el = h.start[c];
            el >= nnz || h.index[el] != c
        })
        .count();
    if missing == 0 {
        return;
    }
    let new_nnz = nnz + missing;
    h.index.resize(new_nnz, 0);
    h.value.resize(new_nnz, 0.0);
    let mut to = new_nnz;
    let mut next_start = nnz;
    h.start[dim] = to;
    for col in (0..dim).rev() {
        let mut el = next_start;
        while el > h.start[col] + 1 {
            el -= 1;
            to -= 1;
            h.index[to] = h.index[el];
            h.value[to] = h.value[el];
        }
        let no_diag = if h.start[col] < next_start {
            let el = h.start[col];
            to -= 1;
            h.index[to] = h.index[el];
            h.value[to] = h.value[el];
            h.index[el] != col
        } else {
            true
        };
        if no_diag {
            to -= 1;
            h.index[to] = col;
            h.value[to] = 0.0;
        }
        next_start = h.start[col];
        h.start[col] = to;
    }
    debug_assert_eq!(to, 0);
}

/// `assessHessian` (HU:22-89) plus `passModel`'s clearing of an empty Hessian
/// (H:520-533). `None` when nothing survives.
fn assess_hessian(mut h: LpHessian) -> Result<Option<LpHessian>, LpFileError> {
    assess_matrix(&mut h.start, &mut h.index, &mut h.value, 0.0, INF, true)?;
    normalise_square_hessian(h.dim, &mut h.start, &mut h.index, &mut h.value)?;
    assess_matrix(&mut h.start, &mut h.index, &mut h.value, 1e-9, 1e15, false)?;
    if h.start[h.dim] == 0 {
        return Ok(None);
    }
    complete_hessian_diagonal(&mut h);
    Ok(Some(h))
}

fn name_string(s: &[u8]) -> String {
    String::from_utf8_lossy(s).into_owned()
}

/// Read an LP-format model from its bytes, as HiGHS 1.16.0's `readModel` does.
pub fn read_lp_bytes(text: &[u8]) -> Result<LpFileModel, LpFileError> {
    let toks = lex(text)?;
    let mut p = Parser::new(text, toks);
    p.parse()?;

    // Parser::finish (LpReader.cpp:1165-1215).
    for &col in &p.binary_cols {
        if p.col_upper[col] == INF {
            p.col_upper[col] = 1.0;
        }
    }
    let mut integrality = std::mem::take(&mut p.integrality);
    if !p.has_integrality {
        integrality.clear();
    }
    let row_names: Vec<String> = if p.used_prefix && !p.prefix_ok {
        Vec::new()
    } else {
        p.row_names.iter().map(|s| name_string(s)).collect()
    };
    let n = p.col_names.len();
    let m = p.row_lower.len();
    // ensureColwise: the transpose of the row-wise matrix, rows ascending in a column.
    let mut a_start = vec![0usize; n + 1];
    for &j in &p.ar_index {
        a_start[j + 1] += 1;
    }
    for j in 0..n {
        a_start[j + 1] += a_start[j];
    }
    let mut a_index = vec![0usize; p.ar_index.len()];
    let mut a_value = vec![0.0f64; p.ar_index.len()];
    let mut fill = a_start.clone();
    for i in 0..m {
        for k in p.ar_start[i]..p.ar_start[i + 1] {
            let j = p.ar_index[k];
            a_index[fill[j]] = i;
            a_value[fill[j]] = p.ar_value[k];
            fill[j] += 1;
        }
    }
    let hessian = finish_hessian(std::mem::take(&mut p.hessian_entries), n);

    let mut col_cost = p.col_cost;
    let mut col_lower = p.col_lower;
    let mut col_upper = p.col_upper;
    let mut row_lower = p.row_lower;
    let mut row_upper = p.row_upper;

    // passModel (H:476-487): no rows or no columns means an empty matrix.
    if n == 0 || m == 0 {
        a_start = vec![0; n + 1];
        a_index.clear();
        a_value.clear();
    }

    // assessLp (HighsLpUtils.cpp:34-101): costs, bounds, then the matrix.
    let big = 1e20;
    let fix_inf = |v: f64| {
        if v >= big {
            INF
        } else if v <= -big {
            -INF
        } else {
            v
        }
    };
    for c in col_cost.iter_mut() {
        *c = fix_inf(*c);
    }
    for (lo, hi) in [
        (&mut col_lower, &mut col_upper),
        (&mut row_lower, &mut row_upper),
    ] {
        for (l, u) in lo.iter_mut().zip(hi.iter_mut()) {
            *l = fix_inf(*l);
            *u = fix_inf(*u);
            // NaN fails both comparisons, so a NaN bound is an error here too.
            if l.is_nan() || *l >= big {
                return err("lower bound of +infinity (or NaN)");
            }
            if u.is_nan() || *u <= -big {
                return err("upper bound of -infinity (or NaN)");
            }
        }
    }
    if n > 0 {
        assess_matrix(&mut a_start, &mut a_index, &mut a_value, 1e-9, 1e15, false)?;
    }

    let hessian = match hessian {
        Some(h) => assess_hessian(h)?,
        None => None,
    };

    Ok(LpFileModel {
        num_row: m,
        num_col: n,
        a_start,
        a_index,
        a_value,
        col_cost,
        offset: p.offset,
        maximize: p.maximize,
        col_lower,
        col_upper,
        row_lower,
        row_upper,
        integrality,
        col_names: p.col_names.iter().map(|s| name_string(s)).collect(),
        row_names,
        objective_name: name_string(&p.objective_name),
        hessian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(s: &str) -> LpFileModel {
        read_lp_bytes(s.as_bytes()).unwrap_or_else(|e| panic!("{s:?}: {e}"))
    }

    fn rejects(s: &str) {
        assert!(
            read_lp_bytes(s.as_bytes()).is_err(),
            "{s:?} should be rejected"
        );
    }

    #[test]
    fn lexer_tokens() {
        let toks = lex(b"x1 <= 5. .5e1 2e3x 0x10 =< => == -> [ ] ^ / :").unwrap();
        let kinds: Vec<Kind> = toks.iter().map(|t| t.kind).collect();
        use Kind::*;
        assert_eq!(
            kinds,
            vec![
                Identifier,
                Less,
                Number,
                Number,
                Number,
                Identifier,
                Number,
                Identifier,
                Less,
                Greater,
                Equal,
                Implies,
                OpenBracket,
                CloseBracket,
                Power,
                Divide,
                Colon,
                EndOfFile
            ]
        );
        let values: Vec<f64> = toks
            .iter()
            .filter(|t| t.kind == Number)
            .map(|t| t.value)
            .collect();
        assert_eq!(values, vec![5.0, 5.0, 2000.0, 0.0]);
        // "1e" and "1e+" are the number 1 followed by an identifier/sign.
        let toks = lex(b"1e 1e+").unwrap();
        assert_eq!(toks[0].kind, Number);
        assert_eq!(toks[0].len, 1);
        assert_eq!(toks[1].kind, Identifier);
        assert!(lex(b"x\x01").is_err());
        assert!(lex(b"x\x7f").is_err());
        // Bytes >= 0x80 are identifier characters.
        assert_eq!(lex("é".as_bytes()).unwrap()[0].kind, Identifier);
        // Overflow and underflow as strtod.
        let toks = lex(b"1e400 1e-400").unwrap();
        assert_eq!(toks[0].value, INF);
        assert_eq!(toks[1].value, 0.0);
    }

    #[test]
    fn keywords_only_start_a_line() {
        // `st` mid-line is a variable; at the start of a line it starts the constraints.
        let m = read("min\n x + st\nst\n c: x + st >= 1\nend\n");
        assert_eq!(m.col_names, vec!["x", "st"]);
        assert_eq!(m.col_cost, vec![1.0, 1.0]);
        assert_eq!(m.num_row, 1);
        // A keyword followed by ':' is a name.
        let m = read("max\n bounds: x\nend");
        assert!(m.maximize);
        assert_eq!(m.objective_name, "bounds");
        // "subject to" must be on one line.
        rejects("min\n x\nsubject\nto\n x >= 1\nend");
        let m = read("min\n x\nsubject to\n x >= 1\nsemi - continuous\n x\nend");
        assert_eq!(m.integrality, vec![2]);
    }

    #[test]
    fn assembly_rules() {
        // Costs are summed; columns in order of first mention in the file.
        let m = read("min\n 2 x + 3 y - x + 4\nst\n c: z + x + x - 2 x >= 1\nend");
        assert_eq!(m.col_names, vec!["x", "y", "z"]);
        assert_eq!(m.col_cost, vec![1.0, 3.0, 0.0]);
        assert_eq!(m.offset, 4.0);
        // x + x - 2x sums to zero and is dropped.
        assert_eq!(m.a_start, vec![0, 0, 0, 1]);
        assert_eq!(m.row_names, vec!["c"]);
        // Sections in file order; binary upper bound 1 only when still infinite.
        let m = read("bounds\n b <= 5\nbinary\n b g\nmax\n g\nend");
        assert_eq!(m.col_names, vec!["b", "g"]);
        assert_eq!(m.col_upper, vec![5.0, 1.0]);
        assert_eq!(m.integrality, vec![1, 1]);
        // Automatic row names, cleared on a clash with an explicit HiGHS_R name.
        let m = read("min\n x\nst\n x >= 1\n x <= 2\nend");
        assert_eq!(m.row_names, vec!["HiGHS_R0", "HiGHS_R1"]);
        let m = read("min\n x\nst\n x >= 1\n HiGHS_R7: x <= 2\nend");
        assert!(m.row_names.is_empty());
        // inf is a variable in an expression and a number in a bound.
        let m = read("min\n x + inf\nbounds\n x <= inf\n -inf <= y <= 3\nend");
        assert_eq!(m.col_names, vec!["x", "inf", "y"]);
        assert_eq!(m.col_lower[2], -INF);
        // An empty file is an empty model.
        let m = read("\\ just a comment\n");
        assert_eq!((m.num_col, m.num_row), (0, 0));
        // A constant on a constraint's left-hand side is ignored.
        let m = read("min\n x\nst\n x + 3 >= 1\nend");
        assert_eq!(m.row_lower, vec![1.0]);
    }

    #[test]
    fn rejections() {
        rejects("garbage");
        rejects("min\n x\n");
        rejects("min\n x\nmin\n y\nend");
        rejects("min\n x y\nend");
        rejects("min\n x\nst\n x + y\nend");
        rejects("min\n x\nst\n x >= y\nend");
        rejects("min\n x\nst\n c: [ x^2 ] <= 1\nend");
        rejects("min\n x\nst\n z = 1 -> x >= 1\nend");
        rejects("min\n x\nst\n s1: S1:: x:1 y:2\nend");
        rejects("min\n x\nsos\n s1: S1:: x:1\nend");
        rejects("min\n [ x^3 ]\nend");
        rejects("min\n [ x^2 ]/3\nend");
        rejects("min\n 2 [ x^2 ]\nend");
        rejects("min\n x * y\nend");
        rejects("min\n x\nbounds\n x <= 1 y <= 2\nend");
        rejects("min\n x\nbounds\n 1 = x = 2\nend");
        rejects("min\n x\nbounds\n 1 <= x >= 0\nend");
        rejects("min\n x\nend\nx");
        rejects("min\n x\nbounds\n x >= inf\nend");
        rejects("min\n x\nst\n 1e15 x >= 1\nend");
    }

    #[test]
    fn hessian_entries_sorted_and_normalised() {
        // `[ ... ]/2` is x'Qx/2; terms are summed after sorting by (column, row).
        let m = read("min\n obj: x + [ z*x + y*x + x^2 ]/2\nst\n c: x + y + z >= 1\nend");
        assert_eq!(m.col_names, vec!["x", "z", "y"]);
        let h = m.hessian.expect("hessian");
        assert_eq!(h.dim, 3);
        // Lower triangle, column-wise, diagonals first, zero-filled.
        assert_eq!(h.start, vec![0, 3, 4, 5]);
        assert_eq!(h.index, vec![0, 1, 2, 1, 2]);
        assert_eq!(h.value, vec![1.0, 0.5, 0.5, 0.0, 0.0]);
        // Without "/ 2" the bracket is doubled; cancelling terms leave no Hessian.
        let m = read("min\n [ x^2 ]\nend");
        assert_eq!(m.hessian.unwrap().value, vec![2.0]);
        let m = read("min\n [ x^2 - x^2 ]\nend");
        assert!(m.hessian.is_none());
    }

    #[test]
    fn assess_matrix_keeps_nan_and_drops_small() {
        let mut start = vec![0, 3];
        let mut index = vec![0, 1, 0];
        let mut value = vec![f64::NAN, 1e-10, 2.0];
        // The small entry is dropped before duplicates are checked, but the NaN at
        // row 0 is kept, so row 0 repeats.
        assert!(assess_matrix(&mut start, &mut index, &mut value, 1e-9, 1e15, false).is_err());
        let mut start = vec![0, 2];
        let mut index = vec![0, 1];
        let mut value = vec![f64::NAN, 1e-10];
        assess_matrix(&mut start, &mut index, &mut value, 1e-9, 1e15, false).unwrap();
        assert_eq!(start, vec![0, 1]);
        assert!(value[0].is_nan());
        // Summing duplicates before the magnitude test.
        let mut start = vec![0, 3];
        let mut index = vec![1, 1, 0];
        let mut value = vec![1.0, -1.0, 3.0];
        assess_matrix(&mut start, &mut index, &mut value, 0.0, INF, true).unwrap();
        assert_eq!((start, index, value), (vec![0, 1], vec![0], vec![3.0]));
    }
}
