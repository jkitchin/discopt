//! Free-format MPS reader that reproduces HiGHS 1.15.1's `readModel` -> `getLp`
//! exactly (highs-parity P6: the in-house replacement for HiGHS's reader in
//! `discopt.mipcc` and the benchmark loader).
//!
//! "Exactly" is the contract, and it is tested by an exact differential against
//! highspy (`python/tests/test_mps_reader.py`, `docs/dev/data/highs-parity/
//! mps_reader/`). So HiGHS's quirks are reproduced on purpose, each with its
//! source line (`ref/HiGHS/highs/io/HMpsFF.cpp`, `lp_data/HighsLpUtils.cpp`):
//!
//! * lines split on `\n`; a line is skipped if blank or if column 0 is `*`; data
//!   lines are trimmed of `" \t\n\v\f\r"`; tokens are whitespace-separated;
//! * section keywords are case-insensitive; NAME / OBJSENSE / QSECTION / QCMATRIX /
//!   CSECTION count with trailing text, every other keyword only as the sole token
//!   (`checkFirstWord`, 399-489); a single-character first token is never one;
//! * ROWS: the first N row is the objective, later N rows are dropped (their
//!   COLUMNS entries discarded; an RHS on one still sets the offset, 1060-1099);
//! * COLUMNS: a new column whenever the name differs from the previous line's, so a
//!   name that reappears later is a SECOND column; zero values are dropped; the
//!   first value of a (column, row) pair and of a cost wins; entries keep file
//!   order; `'MARKER'` / `'INTORG'` / `'INTEND'` are case-sensitive (740-1042);
//! * an INTORG column is flagged binary and forced to [0, 1] at the end unless a
//!   BOUNDS entry other than BV clears the flag (877-886, 327-333, 1450-1539);
//! * RHS: set name optional (SIF form when the first token is a row name), objective
//!   RHS is `offset = -value`, first value per row wins (1044-1258);
//! * RANGES: always a set name; L or E with R < 0 sets `lo = up - |R|`, G or E with
//!   R > 0 sets `up = lo + |R|`, applied to the row's CURRENT bounds (1541-1733);
//! * BOUNDS: case-sensitive types; the column is the second token if it names a
//!   column, else the third; an unknown column is APPENDED; a duplicate lower or
//!   upper setter is ignored as a whole; UP < 0 does NOT move the lower bound;
//! * numbers are C `atof` after the first `D` (else `d`) becomes `E`: the longest
//!   valid prefix, hex floats, `inf` and `nan` included, garbage is 0 (2088-2112);
//! * `assessLp`: costs and bounds at or past 1e20 become infinite; a NaN bound, an
//!   infinite-from-the-wrong-side bound or a matrix entry with |a| >= 1e15 is an
//!   error; matrix entries with |a| <= 1e-9 (and NaN) are dropped.
//!
//! The legacy fixed-format reader HiGHS falls back to (`HMPSIO.cpp`) is NOT
//! emulated: a file that triggers the fallback is refused with
//! [`MpsError::FixedFormat`], so it can never be read differently than HiGHS reads it.

use std::collections::HashMap;

/// A model read from an MPS file, as HiGHS's `getLp` returns it.
#[derive(Debug, Clone, PartialEq)]
pub struct MpsModel {
    /// Rows (the objective and dropped N rows excluded).
    pub num_row: usize,
    /// Columns, in creation order (COLUMNS, then BOUNDS / Hessian sections).
    pub num_col: usize,
    /// Column-wise matrix (CSC): `a_start` has `num_col + 1` entries.
    pub a_start: Vec<usize>,
    /// Row index of each matrix entry.
    pub a_index: Vec<usize>,
    /// Value of each matrix entry.
    pub a_value: Vec<f64>,
    /// Objective coefficients.
    pub col_cost: Vec<f64>,
    /// Objective constant (minus the objective row's RHS).
    pub offset: f64,
    /// OBJSENSE MAX.
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
    /// Empty when two columns (rows) share a name (HiGHS clears the vector).
    pub col_names: Vec<String>,
    /// Row names (see `col_names`).
    pub row_names: Vec<String>,
    /// A nonzero objective Hessian was read (QUADOBJ / QMATRIX / QSECTION naming
    /// the objective). Linear callers refuse such a model.
    pub has_hessian: bool,
}

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq)]
pub enum MpsError {
    /// HiGHS would refuse the file too (`readModel` returns `kError`).
    Parse(String),
    /// HiGHS would fall back to its fixed-format reader; not emulated here.
    FixedFormat(String),
}

impl std::fmt::Display for MpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MpsError::Parse(m) => write!(f, "MPS parse error: {m}"),
            MpsError::FixedFormat(m) => write!(
                f,
                "fixed-format MPS (HiGHS would switch to its legacy fixed reader, \
                 which discopt does not emulate): {m}"
            ),
        }
    }
}

const INF: f64 = f64::INFINITY;
const WS: &[char] = &[' ', '\t', '\n', '\u{0B}', '\u{0C}', '\r'];

fn is_ws(c: char) -> bool {
    WS.contains(&c)
}

/// C `atof` on `word` after HiGHS's `D`/`d` -> `E` substitution: the longest valid
/// floating-point prefix (decimal, hexadecimal, `inf`/`infinity`, `nan`), 0 if none.
pub fn mps_value(word: &str) -> f64 {
    let mut w = word.to_string();
    if let Some(i) = w.find('D') {
        w.replace_range(i..i + 1, "E");
    } else if let Some(i) = w.find('d') {
        w.replace_range(i..i + 1, "E");
    }
    strtod_prefix(&w)
}

fn strtod_prefix(s: &str) -> f64 {
    let b = s.trim_start_matches(WS).as_bytes();
    let mut i = 0;
    let mut neg = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        neg = b[i] == b'-';
        i += 1;
    }
    let rest = &b[i..];
    let lower: Vec<u8> = rest
        .iter()
        .take(8)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let sign = if neg { -1.0 } else { 1.0 };
    if lower.starts_with(b"inf") {
        return sign * INF;
    }
    if lower.starts_with(b"nan") {
        return f64::NAN;
    }
    if lower.starts_with(b"0x") {
        return sign * hex_prefix(&rest[2..]);
    }
    // Decimal: digits [. digits] [e [sign] digits], longest valid prefix.
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
        return 0.0;
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
    let txt = std::str::from_utf8(&rest[..end]).unwrap_or("0");
    let v: f64 = txt.parse().unwrap_or(0.0);
    sign * v
}

/// The value of a hexadecimal floating-point prefix (after `0x`): hex digits,
/// optional `.` and fraction digits, optional `p[sign]digits` binary exponent.
fn hex_prefix(b: &[u8]) -> f64 {
    let mut j = 0;
    let mut v = 0.0f64;
    let mut any = false;
    while j < b.len() && (b[j] as char).is_ascii_hexdigit() {
        v = v * 16.0 + (b[j] as char).to_digit(16).unwrap() as f64;
        j += 1;
        any = true;
    }
    if j < b.len() && b[j] == b'.' {
        j += 1;
        let mut scale = 1.0 / 16.0;
        while j < b.len() && (b[j] as char).is_ascii_hexdigit() {
            v += (b[j] as char).to_digit(16).unwrap() as f64 * scale;
            scale /= 16.0;
            j += 1;
            any = true;
        }
    }
    if !any {
        return 0.0;
    }
    if j < b.len() && (b[j] == b'p' || b[j] == b'P') {
        let mut k = j + 1;
        let mut eneg = false;
        if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
            eneg = b[k] == b'-';
            k += 1;
        }
        let d0 = k;
        let mut e: i32 = 0;
        while k < b.len() && b[k].is_ascii_digit() {
            e = e.saturating_mul(10).saturating_add((b[k] - b'0') as i32);
            k += 1;
        }
        if k > d0 {
            v *= 2f64.powi(if eneg { -e } else { e });
        }
    }
    v
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Key {
    None,
    Name,
    Objsense,
    Max,
    Min,
    Rows,
    Cols,
    Rhs,
    Bounds,
    Ranges,
    Qsection,
    Qmatrix,
    Quadobj,
    Qcmatrix,
    Csection,
    Unsupported,
    Sets,
    End,
}

/// HiGHS's `checkFirstWord` on a trimmed, non-empty line: the section key (if
/// any), the first word, and the rest of the line after it.
fn check_first_word(line: &str) -> (Key, &str, &str) {
    let mut chars = line.char_indices();
    let (_, c0) = chars.next().expect("non-empty line");
    let second = chars.next();
    if second.is_none() || second.is_some_and(|(_, c)| is_ws(c)) {
        let w = &line[..c0.len_utf8()];
        return (Key::None, w, &line[c0.len_utf8()..]);
    }
    let end = line.find(is_ws).unwrap_or(line.len());
    let word = &line[..end];
    let rest = &line[end..];
    let up = word.to_ascii_uppercase();
    let key = match up.as_str() {
        "NAME" => Key::Name,
        "OBJSENSE" => Key::Objsense,
        _ if up.starts_with("MAX") => Key::Max,
        _ if up.starts_with("MIN") => Key::Min,
        "ROWS" => Key::Rows,
        "COLUMNS" => Key::Cols,
        "RHS" => Key::Rhs,
        "BOUNDS" => Key::Bounds,
        "RANGES" => Key::Ranges,
        "QSECTION" => Key::Qsection,
        "QMATRIX" => Key::Qmatrix,
        "QUADOBJ" => Key::Quadobj,
        "QCMATRIX" => Key::Qcmatrix,
        "CSECTION" => Key::Csection,
        "DELAYEDROWS" | "MODELCUTS" | "USERCUTS" | "INDICATORS" | "GENCONS" | "PWLOBJ"
        | "PWLNAM" | "PWLCON" => Key::Unsupported,
        "SETS" | "SOS" => Key::Sets,
        "ENDATA" => Key::End,
        _ => return (Key::None, word, rest),
    };
    if matches!(
        key,
        Key::Name | Key::Objsense | Key::Qcmatrix | Key::Qsection | Key::Csection
    ) {
        return (key, word, rest);
    }
    if rest.trim_matches(WS).is_empty() {
        (key, word, rest)
    } else {
        (Key::None, word, rest)
    }
}

/// Whitespace tokens of `s`.
fn tokens(s: &str) -> Vec<&str> {
    s.split(is_ws).filter(|t| !t.is_empty()).collect()
}

#[derive(Clone, Copy, PartialEq)]
enum RowKind {
    G,
    E,
    L,
}

#[derive(Clone, Copy, PartialEq)]
enum RowRef {
    Row(usize),
    Obj,
    DroppedN,
}

#[derive(Default)]
struct Parser {
    mps_name: String,
    maximize: bool,
    row_map: HashMap<String, RowRef>,
    row_kind: Vec<RowKind>,
    row_lower: Vec<f64>,
    row_upper: Vec<f64>,
    row_names: Vec<String>,
    dup_row: bool,
    col_map: HashMap<String, usize>,
    col_names: Vec<String>,
    dup_col: bool,
    col_lower: Vec<f64>,
    col_upper: Vec<f64>,
    col_integrality: Vec<u8>,
    col_binary: Vec<bool>,
    col_cost: Vec<f64>,
    // COLUMNS entries per column, in file order: (row, value).
    col_entries: Vec<Vec<(usize, f64)>>,
    offset: f64,
    has_obj_entry: bool,
    has_hessian: bool,
    /// The argument of the last QSECTION / QCMATRIX header (HiGHS's
    /// `section_args`, set inside `checkFirstWord`).
    section_arg: String,
}

/// [`check_first_word`], recording a QSECTION / QCMATRIX header's argument as
/// HiGHS's `checkFirstWord` does.
fn cfw<'a>(p: &mut Parser, line: &'a str) -> (Key, &'a str, &'a str) {
    let r = check_first_word(line);
    if matches!(r.0, Key::Qsection | Key::Qcmatrix | Key::Csection) {
        p.section_arg = tokens(r.2)
            .first()
            .map(|s| s.to_string())
            .unwrap_or_default();
    }
    r
}

impl Parser {
    fn new_col(&mut self, name: &str, integer: bool, binary: bool) -> usize {
        let j = self.col_names.len();
        if self.col_map.contains_key(name) {
            self.dup_col = true;
        } else {
            self.col_map.insert(name.to_string(), j);
        }
        self.col_names.push(name.to_string());
        self.col_lower.push(0.0);
        self.col_upper.push(INF);
        self.col_integrality.push(u8::from(integer));
        self.col_binary.push(binary);
        self.col_cost.push(0.0);
        self.col_entries.push(Vec::new());
        j
    }

    fn col_or_new(&mut self, name: &str) -> usize {
        match self.col_map.get(name) {
            Some(&j) => j,
            None => self.new_col(name, false, false),
        }
    }
}

/// Read an MPS model from its text (decompress `.gz` before calling).
pub fn read_mps_str(text: &str) -> Result<MpsModel, MpsError> {
    let mut p = Parser::default();
    let mut lines = text.split('\n');
    let next_data = |lines: &mut std::str::Split<'_, char>| -> Option<String> {
        for raw in lines.by_ref() {
            if raw.is_empty() || raw.starts_with('*') {
                continue;
            }
            let t = raw.trim_matches(WS);
            if t.is_empty() {
                continue;
            }
            return Some(t.to_string());
        }
        None
    };
    let err = |m: &str| Err(MpsError::Parse(m.to_string()));
    let mut key = Key::None;
    // `parseDefault`: one line, outside any section.
    loop {
        match key {
            Key::End => break,
            Key::Unsupported => return err("unsupported section"),
            Key::Csection => return err("cones (CSECTION) not supported"),
            Key::None | Key::Name | Key::Max | Key::Min => {
                let Some(line) = next_data(&mut lines) else {
                    return err("missing ENDATA");
                };
                let (k, _w, rest) = cfw(&mut p, &line);
                match k {
                    Key::Name => {
                        if let Some(t) = tokens(rest).first() {
                            p.mps_name = t.to_string();
                        }
                        key = Key::None;
                    }
                    Key::Objsense => {
                        if let Some(t) = tokens(rest).first() {
                            match t.to_ascii_uppercase().as_str() {
                                "MAX" => p.maximize = true,
                                "MIN" => p.maximize = false,
                                _ => {}
                            }
                        }
                        key = Key::Objsense;
                    }
                    other => key = other,
                }
            }
            Key::Objsense => {
                key = loop {
                    let Some(line) = next_data(&mut lines) else {
                        return err("missing ENDATA");
                    };
                    let (k, _, _) = cfw(&mut p, &line);
                    match k {
                        Key::Max => p.maximize = true,
                        Key::Min => p.maximize = false,
                        Key::None => {}
                        other => break other,
                    }
                };
            }
            Key::Rows => {
                let mut hasobj = false;
                key = loop {
                    let Some(line) = next_data(&mut lines) else {
                        return err("missing ENDATA");
                    };
                    let (k, _, _) = cfw(&mut p, &line);
                    if k != Key::None {
                        if !hasobj {
                            p.row_map
                                .entry("artificial_empty_objective".to_string())
                                .or_insert(RowRef::Obj);
                        }
                        break k;
                    }
                    let c0 = line.chars().next().unwrap();
                    let after = &line[c0.len_utf8()..];
                    let toks = tokens(after);
                    let name = toks.first().copied().unwrap_or("");
                    if toks.len() > 1 {
                        let t = after.trim_matches(WS);
                        return if t.len() > 8 {
                            err("ROWS name longer than 8 with spaces")
                        } else {
                            Err(MpsError::FixedFormat(format!("ROWS line {line:?}")))
                        };
                    }
                    let kind = match c0 {
                        'G' => Some(RowKind::G),
                        'E' => Some(RowKind::E),
                        'L' => Some(RowKind::L),
                        'N' => None,
                        _ => return err(&format!("unidentified ROWS entry {line:?}")),
                    };
                    match kind {
                        None => {
                            if !hasobj {
                                hasobj = true;
                                if p.row_map.contains_key(name) {
                                    p.dup_row = true;
                                } else {
                                    p.row_map.insert(name.to_string(), RowRef::Obj);
                                }
                            } else {
                                p.row_map
                                    .entry(name.to_string())
                                    .or_insert(RowRef::DroppedN);
                            }
                        }
                        Some(kd) => {
                            let i = p.row_kind.len();
                            p.row_kind.push(kd);
                            let (lo, hi) = match kd {
                                RowKind::G => (0.0, INF),
                                RowKind::E => (0.0, 0.0),
                                RowKind::L => (-INF, 0.0),
                            };
                            p.row_lower.push(lo);
                            p.row_upper.push(hi);
                            p.row_names.push(name.to_string());
                            if p.row_map.contains_key(name) {
                                p.dup_row = true;
                            } else {
                                p.row_map.insert(name.to_string(), RowRef::Row(i));
                            }
                        }
                    }
                };
            }
            Key::Cols => {
                let mut integral = false;
                let mut colname: Option<String> = None;
                let mut cur: usize = usize::MAX;
                let mut seen: Vec<bool> = vec![false; p.row_kind.len()];
                let mut cost_set = false;
                let mut touched: Vec<usize> = Vec::new();
                key = loop {
                    let Some(line) = next_data(&mut lines) else {
                        return err("missing ENDATA");
                    };
                    let (k, first, rest) = cfw(&mut p, &line);
                    if k != Key::None {
                        break k;
                    }
                    let t = tokens(rest);
                    let marker = t.first().copied().unwrap_or("");
                    if marker == "'MARKER'" {
                        let m = t.get(1).copied().unwrap_or("");
                        if (integral && m != "'INTEND'") || (!integral && m != "'INTORG'") {
                            return err("integrality marker error in COLUMNS");
                        }
                        integral = !integral;
                        continue;
                    }
                    // Offset (in the trimmed line) where the second token ends.
                    let first_end = first.len();
                    let end_marker = if marker.is_empty() {
                        line.len()
                    } else {
                        let ms = first_end + rest.find(marker).unwrap_or(0);
                        ms + marker.len()
                    };
                    if end_marker < 9 && !p.row_map.contains_key(marker) {
                        let head: String = line.chars().take(10).collect();
                        let head = head.trim_matches(WS);
                        return if head.len() > 8 {
                            err("COLUMNS name longer than 8 with spaces")
                        } else {
                            Err(MpsError::FixedFormat(format!("COLUMNS line {line:?}")))
                        };
                    }
                    if colname.as_deref() != Some(first) {
                        for &i in &touched {
                            seen[i] = false;
                        }
                        touched.clear();
                        cost_set = false;
                        cur = p.new_col(first, integral, integral);
                        colname = Some(first.to_string());
                    }
                    let Some(w1) = t.get(1).copied() else {
                        return err("no coefficient given in COLUMNS");
                    };
                    let mut pairs = vec![(marker, w1)];
                    if t.len() > 2 {
                        pairs.push((t[2], t.get(3).copied().unwrap_or("")));
                    }
                    for (rname, vtxt) in pairs {
                        let Some(&r) = p.row_map.get(rname) else {
                            continue;
                        };
                        let v = mps_value(vtxt);
                        if v == 0.0 {
                            continue;
                        }
                        match r {
                            RowRef::Row(i) => {
                                if !seen[i] {
                                    seen[i] = true;
                                    touched.push(i);
                                    p.col_entries[cur].push((i, v));
                                }
                            }
                            RowRef::Obj => {
                                if !cost_set {
                                    cost_set = true;
                                    p.col_cost[cur] = v;
                                }
                            }
                            RowRef::DroppedN => {}
                        }
                    }
                };
            }
            Key::Rhs => {
                let mut has_entry = vec![false; p.row_kind.len()];
                key = loop {
                    let Some(line) = next_data(&mut lines) else {
                        return err("missing ENDATA");
                    };
                    let (k, _, _) = cfw(&mut p, &line);
                    if k != Key::None && k != Key::Rhs {
                        break k;
                    }
                    let t = tokens(&line);
                    let mut s = if p.row_map.contains_key(t[0]) { 0 } else { 1 };
                    let mut marker = t.get(s).copied().unwrap_or("");
                    let Some(mut word) = t.get(s + 1).copied() else {
                        return err("no bound given in RHS");
                    };
                    if !p.row_map.contains_key(marker) && marker == p.mps_name {
                        s += 1;
                        marker = word;
                        let Some(w) = t.get(s + 1).copied() else {
                            return err("no bound given for SIF row in RHS");
                        };
                        word = w;
                    }
                    let mut pairs = vec![(marker, word)];
                    if t.len() > s + 2 {
                        pairs.push((t[s + 2], t.get(s + 3).copied().unwrap_or("")));
                    }
                    for (rname, vtxt) in pairs {
                        let Some(&r) = p.row_map.get(rname) else {
                            continue;
                        };
                        let v = mps_value(vtxt);
                        match r {
                            RowRef::Row(i) => {
                                if has_entry[i] {
                                    continue;
                                }
                                match p.row_kind[i] {
                                    RowKind::E => {
                                        p.row_upper[i] = v;
                                        p.row_lower[i] = v;
                                    }
                                    RowKind::L => p.row_upper[i] = v,
                                    RowKind::G => p.row_lower[i] = v,
                                }
                                has_entry[i] = true;
                            }
                            RowRef::Obj | RowRef::DroppedN => {
                                if !p.has_obj_entry {
                                    p.offset = -v;
                                    p.has_obj_entry = true;
                                }
                            }
                        }
                    }
                };
            }
            Key::Ranges => {
                let mut has_entry = vec![false; p.row_kind.len()];
                key = loop {
                    let Some(line) = next_data(&mut lines) else {
                        return err("missing ENDATA");
                    };
                    let (k, _, rest) = cfw(&mut p, &line);
                    if k != Key::None {
                        break k;
                    }
                    let t = tokens(rest);
                    if t.len() > 4 {
                        return err("more than two entries on a RANGES line");
                    }
                    let mut idx = 0;
                    while idx < t.len() {
                        let rname = t[idx];
                        let Some(vtxt) = t.get(idx + 1).copied() else {
                            return err("no range given in RANGES");
                        };
                        idx += 2;
                        let Some(&RowRef::Row(i)) = p.row_map.get(rname) else {
                            continue;
                        };
                        if has_entry[i] {
                            continue;
                        }
                        let v = mps_value(vtxt);
                        let kd = p.row_kind[i];
                        if (kd == RowKind::E && v < 0.0) || kd == RowKind::L {
                            p.row_lower[i] = p.row_upper[i] - v.abs();
                        } else if (kd == RowKind::E && v > 0.0) || kd == RowKind::G {
                            p.row_upper[i] = p.row_lower[i] + v.abs();
                        }
                        has_entry[i] = true;
                    }
                };
            }
            Key::Bounds => {
                let mut has_lower = vec![false; p.col_names.len()];
                let mut has_upper = vec![false; p.col_names.len()];
                key = loop {
                    let Some(line) = next_data(&mut lines) else {
                        return err("missing ENDATA");
                    };
                    let (k, first, rest) = cfw(&mut p, &line);
                    if k != Key::None {
                        break k;
                    }
                    let (is_lb, is_ub, integral, semi, default) = match first {
                        "UP" => (false, true, false, false, false),
                        "LO" => (true, false, false, false, false),
                        "FX" => (true, true, false, false, false),
                        "MI" => (true, false, false, false, true),
                        "PL" => (false, true, false, false, true),
                        "BV" => (true, true, true, false, true),
                        "LI" => (true, false, true, false, false),
                        "UI" => (false, true, true, false, false),
                        "FR" => (true, true, false, false, true),
                        "SI" => (false, true, true, true, false),
                        "SC" => (false, true, false, true, false),
                        other => return err(&format!("BOUNDS entry of type {other:?}")),
                    };
                    let t = tokens(rest);
                    let b0 = t.first().copied().unwrap_or("");
                    let (cname, vi) = if p.col_map.contains_key(b0) {
                        (b0, 1)
                    } else {
                        (t.get(1).copied().unwrap_or(""), 2)
                    };
                    let j = if let Some(&j) = p.col_map.get(cname) {
                        j
                    } else {
                        let j = p.col_or_new(cname);
                        has_lower.push(false);
                        has_upper.push(false);
                        j
                    };
                    if (is_lb && has_lower[j]) || (is_ub && has_upper[j]) {
                        continue;
                    }
                    if default {
                        if integral {
                            p.col_integrality[j] = 1;
                            p.col_binary[j] = true;
                            p.col_upper[j] = 1.0;
                        } else {
                            p.col_binary[j] = false;
                            if is_lb {
                                p.col_lower[j] = -INF;
                            }
                            if is_ub {
                                p.col_upper[j] = INF;
                            }
                        }
                        if is_lb {
                            has_lower[j] = true;
                        }
                        if is_ub {
                            has_upper[j] = true;
                        }
                        continue;
                    }
                    let Some(vtxt) = t.get(vi).copied() else {
                        return err("no bound value given in BOUNDS");
                    };
                    let v = mps_value(vtxt);
                    if integral {
                        p.col_integrality[j] = if semi { 3 } else { 1 };
                    } else if semi {
                        p.col_integrality[j] = 2;
                    }
                    if is_lb {
                        p.col_lower[j] = v;
                        has_lower[j] = true;
                    }
                    if is_ub {
                        p.col_upper[j] = v;
                        has_upper[j] = true;
                    }
                    p.col_binary[j] = false;
                };
            }
            Key::Qmatrix | Key::Quadobj | Key::Qsection | Key::Qcmatrix => {
                // Hessian sections: created columns and a nonzero flag are what the
                // linear callers need (they refuse a quadratic objective).
                let objective = if matches!(key, Key::Qsection | Key::Qcmatrix) {
                    // The row this section belongs to: HiGHS keeps it only for the
                    // objective; a real row is an error, anything else is skipped.
                    match p.row_map.get(p.section_arg.as_str()) {
                        Some(RowRef::Obj) => true,
                        Some(RowRef::Row(_)) => {
                            return err("quadratic constraints (QCMATRIX) not supported");
                        }
                        _ => false,
                    }
                } else {
                    true
                };
                key = loop {
                    let Some(line) = next_data(&mut lines) else {
                        return err("missing ENDATA");
                    };
                    let (k, first, rest) = cfw(&mut p, &line);
                    if k != Key::None {
                        break k;
                    }
                    if !objective {
                        continue;
                    }
                    let t = tokens(rest);
                    let c2 = t.first().copied().unwrap_or("");
                    let v = mps_value(t.get(1).copied().unwrap_or(""));
                    p.col_or_new(first);
                    p.col_or_new(c2);
                    if v != 0.0 {
                        p.has_hessian = true;
                    }
                };
            }
            Key::Sets => {
                // SOS/SETS with entries is an error; an empty section is skipped.
                let Some(line) = next_data(&mut lines) else {
                    return err("missing ENDATA");
                };
                let (k, _, _) = cfw(&mut p, &line);
                if k == Key::None {
                    return err("SOS constraints not supported");
                }
                key = k;
            }
        }
    }
    finish(p)
}

fn finish(mut p: Parser) -> Result<MpsModel, MpsError> {
    for j in 0..p.col_names.len() {
        if p.col_binary[j] {
            p.col_lower[j] = 0.0;
            p.col_upper[j] = 1.0;
        }
    }
    let n = p.col_names.len();
    let m = p.row_kind.len();
    // assessLp: costs and bounds, then the matrix.
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
    for c in p.col_cost.iter_mut() {
        *c = fix_inf(*c);
    }
    for (lo, hi) in [
        (&mut p.col_lower, &mut p.col_upper),
        (&mut p.row_lower, &mut p.row_upper),
    ] {
        for (l, u) in lo.iter_mut().zip(hi.iter_mut()) {
            if l.is_nan() || u.is_nan() {
                return Err(MpsError::Parse("NaN bound".into()));
            }
            *l = fix_inf(*l);
            *u = fix_inf(*u);
            if *l >= big {
                return Err(MpsError::Parse("lower bound of +infinity".into()));
            }
            if *u <= -big {
                return Err(MpsError::Parse("upper bound of -infinity".into()));
            }
        }
    }
    let mut a_start = Vec::with_capacity(n + 1);
    let mut a_index = Vec::new();
    let mut a_value = Vec::new();
    a_start.push(0);
    for col in &p.col_entries {
        for &(i, v) in col {
            if v.abs() >= 1e15 {
                return Err(MpsError::Parse(format!("matrix entry {v} too large")));
            }
            if v.is_nan() || v.abs() <= 1e-9 {
                continue;
            }
            a_index.push(i);
            a_value.push(v);
        }
        a_start.push(a_index.len());
    }
    let integrality = if p.col_integrality.iter().any(|&t| t != 0) {
        p.col_integrality
    } else {
        Vec::new()
    };
    Ok(MpsModel {
        num_row: m,
        num_col: n,
        a_start,
        a_index,
        a_value,
        col_cost: p.col_cost,
        offset: p.offset,
        maximize: p.maximize,
        col_lower: p.col_lower,
        col_upper: p.col_upper,
        row_lower: p.row_lower,
        row_upper: p.row_upper,
        integrality,
        col_names: if p.dup_col { Vec::new() } else { p.col_names },
        row_names: if p.dup_row { Vec::new() } else { p.row_names },
        has_hessian: p.has_hessian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mps_value` is C `atof` after HiGHS's `D` -> `E` substitution: longest valid
    /// prefix, hex floats, inf/nan, garbage 0. Every case is counted.
    #[test]
    fn mps_value_matches_atof_semantics() {
        let cases: &[(&str, f64)] = &[
            ("1.5D+2", 150.0),
            ("2d1", 20.0),
            ("0x10", 16.0),
            ("0x1.8p1", 3.0),
            ("1.0e", 1.0),
            ("-3.25e-1x", -0.325),
            ("+.5", 0.5),
            ("abc", 0.0),
            ("1e400", f64::INFINITY),
            ("-Infinity", f64::NEG_INFINITY),
            ("", 0.0),
        ];
        let mut checked = 0;
        for &(s, want) in cases {
            assert_eq!(mps_value(s), want, "{s:?}");
            checked += 1;
        }
        assert!(mps_value("nan").is_nan());
        assert_eq!(checked, cases.len());
    }

    /// The INTORG default and objective offset, end to end.
    #[test]
    fn intorg_column_defaults_to_binary_and_offset_is_minus_rhs() {
        let m = read_mps_str(
            "NAME t\nROWS\n N obj\n L c\nCOLUMNS\n M 'MARKER' 'INTORG'\n x obj 1 c 1\n \
             M 'MARKER' 'INTEND'\nRHS\n r obj 3 c 4\nENDATA\n",
        )
        .unwrap();
        assert_eq!((m.col_lower[0], m.col_upper[0]), (0.0, 1.0));
        assert_eq!(m.integrality, vec![1]);
        assert_eq!(m.offset, -3.0);
        assert_eq!(m.row_upper, vec![4.0]);
        assert!(matches!(
            read_mps_str("NAME t\nROWS\n N obj\n"),
            Err(MpsError::Parse(_))
        ));
    }
}
