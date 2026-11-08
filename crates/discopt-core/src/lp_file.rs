//! CPLEX-LP-format (`.lp`) reader that reproduces HiGHS 1.15.1's `readModel` ->
//! `getLp` (and `getModel().hessian_`) exactly, quirks included (highs-parity P6:
//! the in-house replacement for HiGHS's reader in the benchmark loader, and the
//! reader users get for `.lp` models).
//!
//! "Exactly" is the contract, tested by an exact differential against highspy
//! (`python/tests/test_lp_reader.py`, `docs/dev/data/highs-parity/lp_reader/`).
//! HiGHS's reader is not a grammar but a staged pipeline, and a grammar-driven reader
//! disagrees with it, so this module ports the pipeline stage by stage. Sources are
//! `ref/HiGHS/highs/io/filereaderlp/reader.cpp` (R), `io/FilereaderLp.cpp` (F),
//! `filereaderlp/builder.hpp` (B), `lp_data/Highs.cpp` (H) and
//! `model/HighsHessianUtils.cpp` (HU):
//!
//! 1. **Raw lexer** (R:1224-1361, [`lex`]): lines from `getline`, one trailing `\r`
//!    dropped; only `' '`/`'\t'` separate; `\` comments to end of line anywhere; a
//!    `;` at the START of a token drops the rest of the line (inside a name it is a
//!    name character); NUL at a token start is an error; single-character tokens
//!    `[ ] < > = : + ^ / * -`; otherwise C `strtod` at the token start ([`c_strtod`]:
//!    leading `isspace` skipped, hex floats, `inf`/`infinity`/`nan(...)`, longest
//!    valid prefix, so `2e3x` is `2000` then `x`); otherwise a name up to the next
//!    character of `"\t\n\\:+<>^= /-*[]"` (R:1349).
//! 2. **Token classifier** (R:915-1175, `classify`), in R's order: an exact
//!    lowercase section keyword followed by `:` is a row name (R:919); `/* ... */`
//!    skipped TWO raw tokens at a time (so an odd number of tokens inside is an
//!    error, R:925-935); `STR - STR` -> `semi-continuous`; `STR STR` ->
//!    `subject to` / `such that`; a single keyword (case-insensitive) is recognised
//!    ANYWHERE; `S1 ::` / `S2 ::`; `STR :` or `number :` -> row name (literal text);
//!    `free`; at most two signs combine and a sign before a name pushes `+-1`;
//!    `-[` and `number [` are errors; `<` `=` merge into `<=` across whitespace.
//! 3. **Section splitter** (R:849-913, `split`): tokens before the first keyword are
//!    ignored; an empty section is not recorded; a section type may repeat only
//!    adjacently; the last non-empty section must be closed by a keyword; a non-empty
//!    `end` section is an error.
//! 4. **Section parsers in HiGHS's FIXED order** (R:837-847): objective (MIN wins over
//!    MAX), constraints, bounds, general, binary, semi-continuous, SOS. Columns are
//!    created on first mention in that order (B:15-22), default `[0, +inf]`.
//! 5. **Assembly** (F:40-269): cost ASSIGNED per term (last wins), offset the sum of
//!    constants, `HiGHS_R<i>` automatic row names (cleared when an explicit name
//!    starts with `HiGHS_R` and one was needed), a square Hessian (`x*y` coefficient
//!    halved into both columns), the column-wise matrix with duplicates summed into
//!    the first position and zero sums dropped.
//! 6. **`passModel`** (H:443-512): `assessLp` (costs/bounds at or past 1e20 become
//!    infinite, NaN or wrong-side-infinite bounds and `|a| >= 1e15` are errors,
//!    `|a| <= 1e-9` and NaN entries dropped) and `assessHessian` (HU:22-169):
//!    duplicates summed, `normaliseHessian` ported literally (HU:320-560: in the
//!    square case the gather overwrites its own input, which fixes the entry order),
//!    small entries dropped, missing diagonals filled with explicit zeros; a Hessian
//!    with no nonzeros is cleared.
//!
//! A file HiGHS rejects (`readModel` returns `kError`) is rejected with
//! [`LpFileError`]; a file HiGHS reads with `kWarning` (duplicate entries, explicit
//! zeros, crossed bounds, tiny entries) is read the same way. SOS sets with entries
//! are refused, as in HiGHS (F:31).

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

/// HiGHS's `lpassert` (def.hpp): a failed check aborts the read.
fn lpassert(cond: bool, what: &str) -> Result<(), LpFileError> {
    if cond {
        Ok(())
    } else {
        err(what.to_string())
    }
}

// ---------------------------------------------------------------------------
// Stage 1: C strtod and the raw lexer.
// ---------------------------------------------------------------------------

fn c_isspace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn starts_with_ci(s: &[u8], pat: &[u8]) -> bool {
    s.len() >= pat.len() && s[..pat.len()].eq_ignore_ascii_case(pat)
}

/// glibc `strtod` (C locale) on the start of `s`: `(value, bytes consumed)`, with
/// `0` consumed when no conversion is possible. Leading `isspace` characters are
/// skipped, then an optional sign, then `infinity`/`inf`, `nan` (optionally
/// `nan(chars)`), a hexadecimal float (`0x` + at least one hex digit, optional
/// `p` exponent) or a decimal float; the longest valid prefix is taken. Overflow is
/// `+-inf`, underflow keeps its sign (`-1e-400` is `-0.0`).
///
/// A hexadecimal mantissa is rounded once to nearest from up to 124 significant bits
/// (a sticky bit beyond); only a result in the subnormal range can round twice.
pub fn c_strtod(s: &[u8]) -> (f64, usize) {
    let mut i = 0;
    while i < s.len() && c_isspace(s[i]) {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let sign = if neg { -1.0 } else { 1.0 };
    let rest = &s[i..];
    if starts_with_ci(rest, b"infinity") {
        return (sign * INF, i + 8);
    }
    if starts_with_ci(rest, b"inf") {
        return (sign * INF, i + 3);
    }
    if starts_with_ci(rest, b"nan") {
        let mut j = 3;
        if rest.get(3) == Some(&b'(') {
            let mut k = 4;
            while k < rest.len() && (rest[k].is_ascii_alphanumeric() || rest[k] == b'_') {
                k += 1;
            }
            if rest.get(k) == Some(&b')') {
                j = k + 1;
            }
        }
        let nan = if neg { -f64::NAN } else { f64::NAN };
        return (nan, i + j);
    }
    if starts_with_ci(rest, b"0x") {
        if let Some((v, used)) = hex_float(&rest[2..]) {
            return (sign * v, i + 2 + used);
        }
        // "0x" without a hex digit: only the "0" converts (falls through).
    }
    // Decimal: digits [. digits] (at least one digit) [e [sign] digits].
    let mut j = 0;
    let mut mant = 0usize;
    while j < rest.len() && rest[j].is_ascii_digit() {
        j += 1;
        mant += 1;
    }
    if j < rest.len() && rest[j] == b'.' {
        j += 1;
        while j < rest.len() && rest[j].is_ascii_digit() {
            j += 1;
            mant += 1;
        }
    }
    if mant == 0 {
        return (0.0, 0);
    }
    let mut end = j;
    if j < rest.len() && (rest[j] == b'e' || rest[j] == b'E') {
        let mut k = j + 1;
        if k < rest.len() && (rest[k] == b'+' || rest[k] == b'-') {
            k += 1;
        }
        let d0 = k;
        while k < rest.len() && rest[k].is_ascii_digit() {
            k += 1;
        }
        if k > d0 {
            end = k;
        }
    }
    let txt = std::str::from_utf8(&rest[..end]).expect("ASCII digits");
    let v: f64 = txt.parse().expect("a valid decimal float literal");
    (sign * v, i + end)
}

/// The hexadecimal float after `0x`: `None` without a hex digit.
fn hex_float(b: &[u8]) -> Option<(f64, usize)> {
    let mut j = 0;
    let mut mant: u128 = 0;
    let mut exp2: i64 = 0;
    let mut sticky = false;
    let mut any = false;
    let cap: u128 = 1u128 << 120;
    while j < b.len() && b[j].is_ascii_hexdigit() {
        let d = (b[j] as char).to_digit(16).unwrap() as u128;
        if mant < cap {
            mant = mant * 16 + d;
        } else {
            exp2 += 4;
            sticky |= d != 0;
        }
        j += 1;
        any = true;
    }
    if j < b.len() && b[j] == b'.' {
        let mut k = j + 1;
        let mut frac_any = false;
        while k < b.len() && b[k].is_ascii_hexdigit() {
            let d = (b[k] as char).to_digit(16).unwrap() as u128;
            if mant < cap {
                mant = mant * 16 + d;
                exp2 -= 4;
            } else {
                sticky |= d != 0;
            }
            k += 1;
            frac_any = true;
        }
        if any || frac_any {
            j = k;
            any = true;
        }
    }
    if !any {
        return None;
    }
    if j < b.len() && (b[j] == b'p' || b[j] == b'P') {
        let mut k = j + 1;
        let mut eneg = false;
        if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
            eneg = b[k] == b'-';
            k += 1;
        }
        let d0 = k;
        let mut e: i64 = 0;
        while k < b.len() && b[k].is_ascii_digit() {
            e = e.saturating_mul(10).saturating_add((b[k] - b'0') as i64);
            e = e.min(1 << 40);
            k += 1;
        }
        if k > d0 {
            exp2 += if eneg { -e } else { e };
            j = k;
        }
    }
    if mant == 0 {
        return Some((0.0, j));
    }
    if sticky {
        mant |= 1;
    }
    let mut v = mant as f64;
    let mut e = exp2;
    while e > 0 {
        let step = e.min(1000);
        v *= 2f64.powi(step as i32);
        e -= step;
        if v.is_infinite() {
            break;
        }
    }
    while e < 0 {
        let step = (-e).min(1000);
        v *= 2f64.powi(-(step as i32));
        e += step;
        if v == 0.0 {
            break;
        }
    }
    Some((v, j))
}

#[derive(Debug, Clone, PartialEq)]
enum Raw {
    Str(Vec<u8>),
    /// A number (`strtod` value) and its literal text (a row name when `:` follows).
    Cons(f64, Vec<u8>),
    Less,
    Greater,
    Equal,
    Colon,
    Flend,
    BrkOp,
    BrkCl,
    Plus,
    Minus,
    Hat,
    Slash,
    Asterisk,
}

/// `Reader::readnexttoken` (R:1224-1361) over the whole file, ending in `Flend`.
/// Lexing is context-free, so lexing ahead is equivalent to HiGHS's lazy lexing.
fn lex(text: &[u8]) -> Result<Vec<Raw>, LpFileError> {
    const DELIMS: &[u8] = b"\t\n\\:+<>^= /-*[]";
    let mut out = Vec::new();
    for line in text.split(|&b| b == b'\n') {
        let line = match line.last() {
            Some(b'\r') => &line[..line.len() - 1],
            _ => line,
        };
        let mut pos = 0;
        while pos < line.len() {
            let c = line[pos];
            let single = match c {
                b'\\' | b';' => break,
                b'[' => Some(Raw::BrkOp),
                b']' => Some(Raw::BrkCl),
                b'<' => Some(Raw::Less),
                b'>' => Some(Raw::Greater),
                b'=' => Some(Raw::Equal),
                b':' => Some(Raw::Colon),
                b'+' => Some(Raw::Plus),
                b'^' => Some(Raw::Hat),
                b'/' => Some(Raw::Slash),
                b'*' => Some(Raw::Asterisk),
                b'-' => Some(Raw::Minus),
                b' ' | b'\t' => {
                    pos += 1;
                    continue;
                }
                0 => return err("NUL character in the file (R:1336)"),
                _ => None,
            };
            if let Some(t) = single {
                out.push(t);
                pos += 1;
                continue;
            }
            let (v, used) = c_strtod(&line[pos..]);
            if used > 0 {
                out.push(Raw::Cons(v, line[pos..pos + used].to_vec()));
                pos += used;
                continue;
            }
            let end = line[pos..]
                .iter()
                .position(|b| DELIMS.contains(b))
                .map_or(line.len(), |k| pos + k);
            out.push(Raw::Str(line[pos..end].to_vec()));
            pos = end;
        }
    }
    out.push(Raw::Flend);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Stage 2: token classifier.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kw {
    ObjMin,
    ObjMax,
    Con,
    Bounds,
    Gen,
    Bin,
    Semi,
    Sos,
    End,
}

impl Kw {
    fn slot(self) -> usize {
        self as usize
    }
}

/// `sectionkeywordmap` (R:154-179), exact (callers lowercase where R does).
fn section_keyword(s: &[u8]) -> Option<Kw> {
    Some(match s {
        b"minimize" | b"min" | b"minimum" => Kw::ObjMin,
        b"maximize" | b"max" | b"maximum" => Kw::ObjMax,
        b"subject to" | b"such that" | b"st" | b"s.t." => Kw::Con,
        b"bounds" | b"bound" => Kw::Bounds,
        b"binary" | b"binaries" | b"bin" => Kw::Bin,
        b"general" | b"generals" | b"gen" | b"integer" | b"integers" => Kw::Gen,
        b"semi-continuous" | b"semi" | b"semis" => Kw::Semi,
        b"sos" => Kw::Sos,
        b"end" => Kw::End,
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Leq,
    L,
    Eq,
    G,
    Geq,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    SecId(Kw),
    VarId(Vec<u8>),
    ConId(Vec<u8>),
    Const(f64),
    Free,
    BrkOp,
    BrkCl,
    Comp(Cmp),
    Slash,
    Asterisk,
    Hat,
    SosType(u8),
}

fn lower(s: &[u8]) -> Vec<u8> {
    s.to_ascii_lowercase()
}

/// `Reader::processtokens` (R:915-1175) over a 3-token window of raw tokens.
fn classify(mut raw: Vec<Raw>) -> Result<Vec<Tok>, LpFileError> {
    let last = raw.len() - 1; // the Flend
    let mut i = 0usize;
    let mut out = Vec::new();
    macro_rules! r {
        ($k:expr) => {
            &raw[(i + $k).min(last)]
        };
    }
    while *r!(0) != Raw::Flend {
        // R:917-922: an exact (case-sensitive) keyword followed by ':' is a name.
        if let Raw::Str(s) = r!(0) {
            if section_keyword(s).is_some() && *r!(1) == Raw::Colon {
                let s = s.clone();
                raw[i] = Raw::Cons(0.0, s);
            }
        }
        // R:924-935: "/*" skips raw tokens two at a time up to "*/".
        if *r!(0) == Raw::Slash && *r!(1) == Raw::Asterisk {
            loop {
                i += 2;
                if (*r!(0) == Raw::Asterisk && *r!(1) == Raw::Slash) || *r!(0) == Raw::Flend {
                    break;
                }
            }
            i += 2;
            continue;
        }
        let lc = match r!(0) {
            Raw::Str(s) => Some(lower(s)),
            _ => None,
        };
        // R:942-954: STR - STR, "semi-continuous".
        if let (Some(lc0), Raw::Minus, Raw::Str(s2)) = (&lc, r!(1), r!(2)) {
            let mut k = lc0.clone();
            k.push(b'-');
            k.extend(lower(s2));
            if let Some(kw) = section_keyword(&k) {
                out.push(Tok::SecId(kw));
                i += 3;
                continue;
            }
        }
        // R:956-967: STR STR, "subject to" / "such that".
        if let (Some(lc0), Raw::Str(s1)) = (&lc, r!(1)) {
            let mut k = lc0.clone();
            k.push(b' ');
            k.extend(lower(s1));
            if let Some(kw) = section_keyword(&k) {
                out.push(Tok::SecId(kw));
                i += 2;
                continue;
            }
        }
        // R:969-977: a single keyword, case-insensitive.
        if let Some(lc0) = &lc {
            if let Some(kw) = section_keyword(lc0) {
                out.push(Tok::SecId(kw));
                i += 1;
                continue;
            }
        }
        // R:979-990: "S1 ::" / "S2 ::".
        if let (Raw::Str(s), Raw::Colon, Raw::Colon) = (r!(0), r!(1), r!(2)) {
            lpassert(s.len() == 2, "bad SOS type before '::'")?;
            lpassert(s[0] == b'S' || s[0] == b's', "bad SOS type before '::'")?;
            lpassert(s[1] == b'1' || s[1] == b'2', "bad SOS type before '::'")?;
            out.push(Tok::SosType(if s[1] == b'1' { 1 } else { 2 }));
            i += 3;
            continue;
        }
        // R:992-1007: "name :" and "number :" (the number's literal text) are row names.
        match (r!(0), r!(1)) {
            (Raw::Str(s), Raw::Colon) | (Raw::Cons(_, s), Raw::Colon) => {
                out.push(Tok::ConId(s.clone()));
                i += 2;
                continue;
            }
            _ => {}
        }
        if let Raw::Str(s) = r!(0) {
            let lc0 = lc.as_deref().unwrap();
            if lc0 == b"free" {
                out.push(Tok::Free); // R:1010
            } else if lc0 == b"infinity" || lc0 == b"inf" {
                out.push(Tok::Const(INF)); // R:1017
            } else {
                out.push(Tok::VarId(s.clone())); // R:1025
            }
            i += 1;
            continue;
        }
        // R:1033-1076: signs.
        if matches!(r!(0), Raw::Plus | Raw::Minus) {
            let mut sign = if *r!(0) == Raw::Plus { 1.0 } else { -1.0 };
            i += 1;
            if matches!(r!(0), Raw::Plus | Raw::Minus) {
                sign *= if *r!(0) == Raw::Plus { 1.0 } else { -1.0 };
                i += 1;
            }
            match r!(0) {
                Raw::Cons(v, _) => {
                    out.push(Tok::Const(sign * v));
                    i += 1;
                    continue;
                }
                Raw::BrkOp if sign == 1.0 => {
                    out.push(Tok::BrkOp);
                    i += 1;
                    continue;
                }
                Raw::BrkOp => return err("'-[' (negated quadratic bracket, R:1058)"),
                Raw::Str(_) => {
                    // Push the sign; the name is classified on the next pass.
                    out.push(Tok::Const(sign));
                    continue;
                }
                _ => return err("sign followed by an operator (R:1075)"),
            }
        }
        if let Raw::Cons(v, _) = r!(0) {
            lpassert(*r!(1) != Raw::BrkOp, "number followed by '[' (R:1081)")?;
            out.push(Tok::Const(*v));
            i += 1;
            continue;
        }
        let (tok, used) = match (r!(0), r!(1)) {
            (Raw::BrkOp, _) => (Tok::BrkOp, 1),
            (Raw::BrkCl, _) => (Tok::BrkCl, 1),
            (Raw::Slash, _) => (Tok::Slash, 1),
            (Raw::Asterisk, _) => (Tok::Asterisk, 1),
            (Raw::Hat, _) => (Tok::Hat, 1),
            (Raw::Less, Raw::Equal) => (Tok::Comp(Cmp::Leq), 2),
            (Raw::Less, _) => (Tok::Comp(Cmp::L), 1),
            (Raw::Greater, Raw::Equal) => (Tok::Comp(Cmp::Geq), 2),
            (Raw::Greater, _) => (Tok::Comp(Cmp::G), 1),
            (Raw::Equal, _) => (Tok::Comp(Cmp::Eq), 1),
            _ => return err("unexpected ':' (R:1171)"),
        };
        out.push(tok);
        i += used;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Stage 3: section splitter.
// ---------------------------------------------------------------------------

/// `[begin, end)` token range per section keyword (`Reader::sectiontokens`).
type Sections = [Option<(usize, usize)>; 9];

/// `Reader::splittokens` (R:849-913), including its `debug_open_section` asserts.
fn split(toks: &[Tok]) -> Result<Sections, LpFileError> {
    let mut sec: Sections = [None; 9];
    let mut current: Option<Kw> = None;
    let mut open = false;
    let n = toks.len();
    for it in 0..n {
        let Tok::SecId(kw) = toks[it] else { continue };
        let new_type = current != Some(kw);
        if new_type {
            if let Some(c) = current {
                lpassert(open, "section bookkeeping (R:868)")?;
                sec[c.slot()].as_mut().unwrap().1 = it;
                open = false;
                current = None;
            }
        }
        let next = it + 1;
        let next_kw = match toks.get(next) {
            None => None,
            Some(Tok::SecId(k)) => Some(*k),
            Some(_) => {
                // Next section is non-empty.
                if new_type {
                    current = Some(kw);
                    lpassert(sec[kw.slot()].is_none(), "repeated section (R:896)")?;
                    lpassert(!open, "section bookkeeping (R:899)")?;
                    sec[kw.slot()] = Some((next, usize::MAX));
                    open = true;
                }
                lpassert(open != current.is_none(), "section bookkeeping (R:905)")?;
                continue;
            }
        };
        // End of tokens or an empty section. At the end of the tokens R reads the
        // keyword of the past-the-end element; any value but the open section's
        // closes it, which is what highspy does (`general x general` at EOF reads).
        if let Some(c) = current {
            if next_kw != Some(c) {
                lpassert(open, "section bookkeeping (R:884)")?;
                sec[c.slot()].as_mut().unwrap().1 = it;
                open = false;
            }
        }
        current = None;
        lpassert(!open, "empty section inside an open section (R:889)")?;
    }
    lpassert(
        current.is_none(),
        "last section not closed by a keyword (R:910)",
    )?;
    Ok(sec)
}

// ---------------------------------------------------------------------------
// Stage 4: section parsers (model.hpp, builder.hpp).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum VarType {
    Continuous,
    Binary,
    General,
    SemiContinuous,
    SemiInteger,
}

struct Var {
    ty: VarType,
    lb: f64,
    ub: f64,
    name: Vec<u8>,
}

#[derive(Default)]
struct Expr {
    lin: Vec<(usize, f64)>,
    quad: Vec<(usize, usize, f64)>,
    offset: f64,
    name: Vec<u8>,
}

struct Con {
    lb: f64,
    ub: f64,
    expr: Expr,
}

#[derive(Default)]
struct Builder {
    index: HashMap<Vec<u8>, usize>,
    vars: Vec<Var>,
}

impl Builder {
    /// `Builder::getvarbyname` (B:15-22): created on first mention, `[0, +inf]`.
    fn var(&mut self, name: &[u8]) -> usize {
        if let Some(&j) = self.index.get(name) {
            return j;
        }
        let j = self.vars.len();
        self.vars.push(Var {
            ty: VarType::Continuous,
            lb: 0.0,
            ub: INF,
            name: name.to_vec(),
        });
        self.index.insert(name.to_vec(), j);
        j
    }
}

fn is_const(t: Option<&Tok>) -> Option<f64> {
    match t {
        Some(Tok::Const(v)) => Some(*v),
        _ => None,
    }
}

fn is_var(t: Option<&Tok>) -> Option<&[u8]> {
    match t {
        Some(Tok::VarId(s)) => Some(s),
        _ => None,
    }
}

/// `Reader::parseexpression` (R:366-532) on `toks[*it..end]`.
fn parse_expression(
    b: &mut Builder,
    toks: &[Tok],
    it: &mut usize,
    end: usize,
    expr: &mut Expr,
    isobj: bool,
) -> Result<(), LpFileError> {
    // Tokens at or past `end` are invisible (HiGHS compares iterators with `end`).
    let at = |k: usize| if k < end { toks.get(k) } else { None };
    if let Some(Tok::ConId(s)) = at(*it) {
        expr.name = s.clone();
        *it += 1;
    }
    while *it < end {
        let cur = &toks[*it];
        // const var
        if let (Some(c), Some(name)) = (is_const(Some(cur)), is_var(at(*it + 1))) {
            let j = b.var(name);
            expr.lin.push((j, c));
            *it += 2;
            continue;
        }
        if let Tok::Const(c) = cur {
            expr.offset += *c;
            *it += 1;
            continue;
        }
        if let Tok::VarId(name) = cur {
            let j = b.var(name);
            expr.lin.push((j, 1.0));
            *it += 1;
            continue;
        }
        if *cur == Tok::BrkOp && *it + 1 < end {
            *it += 1;
            while *it < end && toks[*it] != Tok::BrkCl {
                let (t0, t1, t2, t3) = (at(*it), at(*it + 1), at(*it + 2), at(*it + 3));
                // const var ^ const
                if let (Some(c), Some(name), Some(Tok::Hat), Some(e)) =
                    (is_const(t0), is_var(t1), t2, is_const(t3))
                {
                    lpassert(e == 2.0, "exponent other than 2 (R:432)")?;
                    let j1 = b.var(name);
                    let j2 = b.var(name);
                    expr.quad.push((j1, j2, c));
                    *it += 4;
                    continue;
                }
                // var ^ const
                if let (Some(name), Some(Tok::Hat), Some(e)) = (is_var(t0), t1, is_const(t2)) {
                    lpassert(e == 2.0, "exponent other than 2 (R:453)")?;
                    let j1 = b.var(name);
                    let j2 = b.var(name);
                    expr.quad.push((j1, j2, 1.0));
                    *it = (*it + 3).min(end);
                    continue;
                }
                // const var * var
                if let (Some(c), Some(n1), Some(Tok::Asterisk), Some(n2)) =
                    (is_const(t0), is_var(t1), t2, is_var(t3))
                {
                    let j1 = b.var(n1);
                    let j2 = b.var(n2);
                    expr.quad.push((j1, j2, c));
                    *it += 4;
                    continue;
                }
                // var * var
                if let (Some(n1), Some(Tok::Asterisk), Some(n2)) = (is_var(t0), t1, is_var(t2)) {
                    let j1 = b.var(n1);
                    let j2 = b.var(n2);
                    expr.quad.push((j1, j2, 1.0));
                    *it = (*it + 3).min(end);
                    continue;
                }
                break;
            }
            if isobj {
                // R:505-518: "] / 2" must follow in the objective.
                lpassert(*it < end, "unclosed quadratic bracket")?;
                lpassert(at(*it + 2).is_some(), "quadratic bracket without '/ 2'")?;
                lpassert(toks[*it] == Tok::BrkCl, "expected ']' (R:513)")?;
                lpassert(
                    toks[*it + 1] == Tok::Slash,
                    "expected '/' after ']' (R:514)",
                )?;
                lpassert(
                    is_const(at(*it + 2)) == Some(2.0),
                    "quadratic bracket not divided by 2 (R:515-516)",
                )?;
                *it += 3;
            } else {
                lpassert(*it < end, "unclosed quadratic bracket (R:519)")?;
                lpassert(toks[*it] == Tok::BrkCl, "expected ']' (R:520)")?;
                *it += 1;
            }
            continue;
        }
        break;
    }
    Ok(())
}

struct Parsed {
    b: Builder,
    maximize: bool,
    obj: Expr,
    cons: Vec<Con>,
}

/// `Reader::processsections` (R:837-847): FIXED order, whatever the file order.
fn process_sections(toks: &[Tok], sec: &Sections) -> Result<Parsed, LpFileError> {
    let mut b = Builder::default();
    let range = |kw: Kw| sec[kw.slot()];
    // processendsec (R:833): an `end` section with content is an error. Checked
    // last by HiGHS; any error is an error, so order does not matter.
    lpassert(range(Kw::End).is_none(), "content after 'end' (R:834)")?;

    // processobjsec (R:534-553).
    let mut obj = Expr::default();
    let mut maximize = false;
    if let Some((s, e)) = range(Kw::ObjMin).or(range(Kw::ObjMax)) {
        maximize = range(Kw::ObjMin).is_none();
        let mut it = s;
        parse_expression(&mut b, toks, &mut it, e, &mut obj, true)?;
        lpassert(it == e, "unparsed tokens in the objective (R:541)")?;
    }

    // processconsec (R:555-592).
    let mut cons = Vec::new();
    if let Some((mut it, e)) = range(Kw::Con) {
        while it != e {
            let mut con = Con {
                lb: -INF,
                ub: INF,
                expr: Expr::default(),
            };
            parse_expression(&mut b, toks, &mut it, e, &mut con.expr, false)?;
            lpassert(it != e, "constraint without a comparison (R:566)")?;
            let Tok::Comp(dir) = toks[it] else {
                return err("constraint without a comparison (R:567)");
            };
            it += 1;
            lpassert(it != e, "constraint without a right-hand side (R:572)")?;
            let Some(v) = is_const(Some(&toks[it])) else {
                return err("right-hand side is not a constant (R:573)");
            };
            match dir {
                Cmp::Eq => {
                    con.lb = v;
                    con.ub = v;
                }
                Cmp::Leq => con.ub = v,
                Cmp::Geq => con.lb = v,
                _ => return err("'<' or '>' in a constraint (R:585)"),
            }
            cons.push(con);
            it += 1;
        }
    }

    // processboundssec (R:594-712).
    if let Some((mut it, e)) = range(Kw::Bounds) {
        let at = |k: usize| if k < e { toks.get(k) } else { None };
        while it != e {
            if let (Some(Tok::VarId(name)), Some(Tok::Free)) = (at(it), at(it + 1)) {
                let j = b.var(name);
                b.vars[j].lb = -INF;
                b.vars[j].ub = INF;
                it += 2;
                continue;
            }
            let (t0, t1, t2, t3, t4) = (at(it), at(it + 1), at(it + 2), at(it + 3), at(it + 4));
            // CONST COMP VAR COMP CONST
            if let (Some(lo), Some(Tok::Comp(d1)), Some(name), Some(Tok::Comp(d3)), Some(hi)) =
                (is_const(t0), t1, is_var(t2), t3, is_const(t4))
            {
                lpassert(*d1 == Cmp::Leq, "double bound must use '<=' (R:651)")?;
                lpassert(*d3 == Cmp::Leq, "double bound must use '<=' (R:652)")?;
                let j = b.var(name);
                b.vars[j].lb = lo;
                b.vars[j].ub = hi;
                it += 5;
                continue;
            }
            // CONST COMP VAR
            if let (Some(v), Some(Tok::Comp(d)), Some(name)) = (is_const(t0), t1, is_var(t2)) {
                let j = b.var(name);
                match d {
                    Cmp::Leq => b.vars[j].lb = v,
                    Cmp::Geq => b.vars[j].ub = v,
                    Cmp::Eq => {
                        b.vars[j].lb = v;
                        b.vars[j].ub = v;
                    }
                    _ => return err("'<' or '>' in a bound (R:677)"),
                }
                it += 3;
                continue;
            }
            // VAR COMP CONST
            if let (Some(name), Some(Tok::Comp(d)), Some(v)) = (is_var(t0), t1, is_const(t2)) {
                let j = b.var(name);
                match d {
                    Cmp::Leq => b.vars[j].ub = v,
                    Cmp::Geq => b.vars[j].lb = v,
                    Cmp::Eq => {
                        b.vars[j].lb = v;
                        b.vars[j].ub = v;
                    }
                    _ => return err("'<' or '>' in a bound (R:701)"),
                }
                it += 3;
                continue;
            }
            return err("unrecognised bound (R:710)");
        }
    }

    // processgensec, processbinsec, processsemisec (R:714-782).
    for kw in [Kw::Gen, Kw::Bin, Kw::Semi] {
        let Some((s, e)) = range(kw) else { continue };
        for t in &toks[s..e] {
            if let Tok::SecId(k) = t {
                lpassert(*k == kw, "foreign keyword in a type section (R:723)")?;
                continue;
            }
            let Tok::VarId(name) = t else {
                return err("non-variable in a type section (R:727)");
            };
            let j = b.var(name);
            let v = &mut b.vars[j];
            match kw {
                Kw::Gen => {
                    v.ty = if v.ty == VarType::SemiContinuous {
                        VarType::SemiInteger
                    } else {
                        VarType::General
                    }
                }
                Kw::Bin => {
                    v.ty = VarType::Binary;
                    if v.ub == INF {
                        v.ub = 1.0; // R:732: only an infinite upper bound becomes 1
                    }
                }
                _ => {
                    v.ty = if v.ty == VarType::General {
                        VarType::SemiInteger
                    } else {
                        VarType::SemiContinuous
                    }
                }
            }
        }
    }

    // processsossec (R:784-831).
    let mut n_sos = 0usize;
    if let Some((mut it, e)) = range(Kw::Sos) {
        while it != e {
            lpassert(
                matches!(toks[it], Tok::ConId(_)),
                "SOS without a name (R:797)",
            )?;
            it += 1;
            lpassert(it != e, "SOS without a type (R:802)")?;
            lpassert(
                matches!(toks[it], Tok::SosType(_)),
                "SOS without a type (R:803)",
            )?;
            it += 1;
            while it != e {
                let Tok::ConId(name) = &toks[it] else { break };
                if let Some(Tok::Const(_)) = toks.get(it + 1).filter(|_| it + 1 < e) {
                    b.var(name);
                    it += 2;
                    continue;
                }
                break;
            }
            n_sos += 1;
        }
    }
    // F:31-35: any SOS set (even an empty one) makes `m.soss` non-empty.
    if n_sos > 0 {
        return err("SOS not supported by HiGHS (F:32)");
    }
    Ok(Parsed {
        b,
        maximize,
        obj,
        cons,
    })
}

// ---------------------------------------------------------------------------
// Stages 5-6: FilereaderLp assembly, assessLp, assessHessian.
// ---------------------------------------------------------------------------

/// `assessMatrix` (HighsMatrixUtils.cpp:50-279) on a column-wise matrix: duplicates
/// summed into their first position (when `sum_duplicates`) or an error, entries
/// with `|v| >= large` an error, entries with `!(|v| > small)` (NaN included)
/// dropped.
fn assess_matrix(
    start: &mut [usize],
    index: &mut Vec<usize>,
    value: &mut Vec<f64>,
    small: f64,
    large: f64,
    sum_duplicates: bool,
) -> Result<(), LpFileError> {
    let num_vec = start.len() - 1;
    let mut num_new = 0usize;
    let mut large_found = false;
    let mut seen: HashMap<usize, usize> = HashMap::new();
    for ix in 0..num_vec {
        let (from, to) = (start[ix], start[ix + 1]);
        start[ix] = num_new;
        for el in from..to {
            let comp = index[el];
            if let Some(&prev) = seen.get(&comp) {
                if !sum_duplicates {
                    return err("duplicate matrix index");
                }
                value[prev] += value[el];
                continue;
            }
            index[num_new] = index[el];
            value[num_new] = value[el];
            seen.insert(comp, num_new);
            num_new += 1;
        }
        let (from, to) = (start[ix], num_new);
        num_new = start[ix];
        for el in from..to {
            let a = value[el].abs();
            if a >= large {
                large_found = true;
            }
            if a > small {
                index[num_new] = index[el];
                value[num_new] = value[el];
                num_new += 1;
            }
        }
        seen.clear();
    }
    start[num_vec] = num_new;
    index.truncate(num_new);
    value.truncate(num_new);
    if large_found {
        return err("matrix entry too large");
    }
    Ok(())
}

/// `normaliseHessian` (HU:320-532) for a SQUARE Hessian (the only format the LP
/// reader builds), ported literally: the strict upper triangle is moved into a
/// row-wise `upper`, the lower part compacted in place, and the gather then writes
/// into the same arrays it reads (`from_hessian` aliases `hessian`, HU:396), which
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
        return err("Hessian is not symmetric (HU:345)");
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
        // of this column not yet read (HU:468-501).
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
        return err("Hessian asymmetry beyond tolerance (HU:525)");
    }
    Ok(())
}

/// `completeHessianDiagonal` (HU:110-169): an explicit zero first in every column
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

/// `assessHessian` (HU:22-90) plus `passModel`'s clearing of an empty Hessian
/// (H:496-505). `None` when nothing survives.
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

/// Read an LP-format model from its bytes, as HiGHS 1.15.1's `readModel` does.
pub fn read_lp_bytes(text: &[u8]) -> Result<LpFileModel, LpFileError> {
    let toks = classify(lex(text)?)?;
    let sec = split(&toks)?;
    let p = process_sections(&toks, &sec)?;
    let truthy = |v: f64| v != 0.0;
    let vars = &p.b.vars;
    let n = vars.len();
    let m = p.cons.len();

    // F:40-64: bounds, names, integrality.
    let mut col_lower: Vec<f64> = vars.iter().map(|v| v.lb).collect();
    let mut col_upper: Vec<f64> = vars.iter().map(|v| v.ub).collect();
    let col_names: Vec<String> = vars.iter().map(|v| name_string(&v.name)).collect();
    let mut integrality: Vec<u8> = vars
        .iter()
        .map(|v| match v.ty {
            VarType::Binary | VarType::General => 1,
            VarType::SemiContinuous => 2,
            VarType::SemiInteger => 3,
            VarType::Continuous => 0,
        })
        .collect();
    if integrality.iter().all(|&t| t == 0) {
        integrality.clear();
    }

    // F:66-73: offset as written, cost assigned per term (the last term wins).
    let offset = p.obj.offset;
    let mut col_cost = vec![0.0f64; n];
    for &(j, c) in &p.obj.lin {
        col_cost[j] = c;
    }

    // F:75-118: the square Hessian.
    let mut qmat: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(j1, j2, c) in &p.obj.quad {
        if j1 != j2 {
            qmat[j1].push((j2, c / 2.0));
            qmat[j2].push((j1, c / 2.0));
        } else {
            qmat[j1].push((j2, c));
        }
    }
    let qnnz = qmat.iter().flatten().filter(|e| truthy(e.1)).count();
    let hessian = if qnnz > 0 {
        let mut h = LpHessian {
            dim: n,
            start: Vec::with_capacity(n + 1),
            index: Vec::new(),
            value: Vec::new(),
        };
        for col in &qmat {
            h.start.push(h.index.len());
            for &(r, v) in col {
                if truthy(v) {
                    h.index.push(r);
                    h.value.push(v);
                }
            }
        }
        h.start.push(h.index.len());
        Some(h)
    } else {
        None
    };

    // F:120-145: rows; a quadratic term in a row is an error.
    let mut col_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut row_names: Vec<Vec<u8>> = Vec::with_capacity(m);
    let mut row_lower = Vec::with_capacity(m);
    let mut row_upper = Vec::with_capacity(m);
    for (i, con) in p.cons.iter().enumerate() {
        row_names.push(con.expr.name.clone());
        for &(j, c) in &con.expr.lin {
            col_rows[j].push((i, c));
        }
        row_lower.push(con.lb);
        row_upper.push(con.ub);
        if !con.expr.quad.is_empty() {
            return err("Quadratic constraints not supported by HiGHS (F:140)");
        }
    }

    // F:147-169: automatic row names and the HiGHS_R clash rule.
    let mut prefix_ok = true;
    let mut used_prefix = false;
    for (i, name) in row_names.iter_mut().enumerate() {
        // strncmp(name, "HiGHS_R", 7) on a C string: stops at the first NUL.
        let c_name = name.split(|&b| b == 0).next().unwrap_or(&[]);
        if c_name.starts_with(b"HiGHS_R") {
            prefix_ok = false;
        } else if name.is_empty() {
            *name = format!("HiGHS_R{i}").into_bytes();
            used_prefix = true;
        }
    }
    let row_names: Vec<String> = if used_prefix && !prefix_ok {
        Vec::new()
    } else {
        row_names.iter().map(|s| name_string(s)).collect()
    };

    // F:171-262: the column-wise matrix, duplicates summed into the first position.
    let mut a_start = Vec::with_capacity(n + 1);
    let mut a_index = Vec::new();
    let mut a_value = Vec::new();
    let mut column = vec![0.0f64; m];
    a_start.push(0);
    for entries in &col_rows {
        for &(i, v) in entries.iter().filter(|e| truthy(e.1)) {
            column[i] += v;
        }
        for &(i, v) in entries.iter().filter(|e| truthy(e.1)) {
            let _ = v;
            if truthy(column[i]) {
                a_index.push(i);
                a_value.push(column[i]);
            }
            column[i] = 0.0;
        }
        a_start.push(a_index.len());
    }

    // passModel (H:461-475): no rows or no columns means an empty matrix.
    if n == 0 || m == 0 {
        a_start = vec![0; n + 1];
        a_index.clear();
        a_value.clear();
    }

    // assessLp (HighsLpUtils.cpp:33-105): costs, bounds, then the matrix.
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
        offset,
        maximize: p.maximize,
        col_lower,
        col_upper,
        row_lower,
        row_upper,
        integrality,
        col_names,
        row_names,
        objective_name: name_string(&p.obj.name),
        hessian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `c_strtod` is glibc `strtod`: longest valid prefix, hex floats, inf/nan,
    /// leading isspace, signed underflow. Every case is counted.
    #[test]
    fn strtod_matches_glibc_prefix_semantics() {
        let cases: &[(&[u8], f64, usize)] = &[
            (b"2e3x", 2000.0, 3),
            (b"1d2", 1.0, 1),
            (b"1e", 1.0, 1),
            (b"1e+", 1.0, 1),
            (b".5", 0.5, 2),
            (b"5.", 5.0, 2),
            (b"0x10", 16.0, 4),
            (b"0x1.8p1", 3.0, 7),
            (b"0x", 0.0, 1),
            (b"0xg", 0.0, 1),
            (b"0x.8", 0.5, 4),
            (b"infeasible", INF, 3),
            (b"Infinity2", INF, 8),
            (b"1e400", INF, 5),
            (b"\x0b5 y", 5.0, 2),
            (b"\x0b-5", -5.0, 3),
            (b"x", 0.0, 0),
            (b".", 0.0, 0),
            (b"", 0.0, 0),
        ];
        let mut checked = 0;
        for &(s, want, used) in cases {
            assert_eq!(
                c_strtod(s),
                (want, used),
                "{:?}",
                String::from_utf8_lossy(s)
            );
            checked += 1;
        }
        let (v, used) = c_strtod(b"nancy");
        assert!(v.is_nan() && used == 3);
        let (v, used) = c_strtod(b"nan(12)x");
        assert!(v.is_nan() && used == 7);
        let (v, used) = c_strtod(b"nan(12");
        assert!(v.is_nan() && used == 3);
        let (v, _) = c_strtod(b"-1e-400");
        assert!(v == 0.0 && v.is_sign_negative());
        assert_eq!(checked, cases.len());
    }

    /// The lexer splits names at HiGHS's delimiter set only and keeps a number's
    /// literal text.
    #[test]
    fn lexer_tokens() {
        let toks = lex(b"a/b 2e3x;c\n1: x;y \\ comment\r\n").unwrap();
        assert_eq!(
            toks,
            vec![
                Raw::Str(b"a".to_vec()),
                Raw::Slash,
                Raw::Str(b"b".to_vec()),
                Raw::Cons(2000.0, b"2e3".to_vec()),
                Raw::Str(b"x;c".to_vec()),
                Raw::Cons(1.0, b"1".to_vec()),
                Raw::Colon,
                Raw::Str(b"x;y".to_vec()),
                Raw::Flend,
            ]
        );
        assert!(lex(b"x\0").is_ok()); // NUL inside a name is a name character
        assert!(lex(b"x \0").is_err()); // ... at a token start an error
    }

    /// normaliseHessian's self-overwriting gather: for `[z*x + y*x + x^2]` the
    /// x column is "diagonal, below-diagonal survivors, recovered upper entries".
    #[test]
    fn hessian_order_follows_the_aliased_gather() {
        let m =
            read_lp_bytes(b"min\n obj: x + [ z*x + y*x + x^2 ]/2\nst\n c: x + y + z >= 2\nend\n")
                .unwrap();
        assert_eq!(m.col_names, vec!["x", "z", "y"]);
        let h = m.hessian.unwrap();
        assert_eq!(h.dim, 3);
        // Column x (0): diagonal 1, then z (1) and y (2) in the order they survive.
        assert_eq!(h.start, vec![0, 3, 4, 5]);
        assert_eq!(&h.index[..3], &[0, 2, 1]);
        assert_eq!(&h.value[..3], &[1.0, 0.5, 0.5]);
        // Columns z and y have no diagonal: explicit zeros.
        assert_eq!(&h.index[3..], &[1, 2]);
        assert_eq!(&h.value[3..], &[0.0, 0.0]);
    }

    /// End to end: cost last-wins, LHS constant dropped, auto row names, duplicate
    /// entries summed, bin on a bounded column keeps the bound.
    #[test]
    fn assembly_rules() {
        let m = read_lp_bytes(
            b"max\n obj: 2 x + 3 y - x + 5\nst\n c1: x + y - 3 >= 2\n x + 2 y + x <= 4\n\
              bounds\n y <= 5\nbin\n y\nend\n",
        )
        .unwrap();
        assert!(m.maximize);
        assert_eq!(m.col_cost, vec![-1.0, 3.0]);
        assert_eq!(m.offset, 5.0);
        assert_eq!(m.row_lower, vec![2.0, -INF]);
        assert_eq!(m.row_names, vec!["c1", "HiGHS_R1"]);
        assert_eq!(m.a_start, vec![0, 2, 4]);
        assert_eq!(m.a_value, vec![1.0, 2.0, 1.0, 2.0]);
        assert_eq!((m.col_upper[1], m.integrality.clone()), (5.0, vec![0, 1]));
        assert!(read_lp_bytes(b"min\n x\nst\n c: x >= 1\n").is_err()); // unclosed
        assert!(read_lp_bytes(b"min\n x /* a */\nst\n c: x >= 1\nend\n").is_err());
        assert!(read_lp_bytes(b"min\n x /* a b */\nst\n c: x >= 1\nend\n").is_ok());
    }
}
