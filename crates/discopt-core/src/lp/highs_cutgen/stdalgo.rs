//! Provenance: the pdqsort part is an altered Rust translation of Orson Peters' pdqsort
//! (zlib License, as vendored by HiGHS); see `THIRD_PARTY_NOTICES.md` at the repository root.
//! `partition`, `partial_sort`, `upper_bound` and pdqsort's heapsort fallback are
//! independent implementations (Rust std and textbook algorithms), not translations.
//!
//! The ordering algorithms the HiGHS cut generator calls: Orson Peters' `pdqsort` /
//! `pdqsort_branchless` (`extern/pdqsort/pdqsort.h`, zlib license, as vendored by
//! HiGHS), translated so its comparison sequence matches, and independent versions of
//! the C++ standard algorithms HiGHS uses.
//!
//! Why exact copies rather than Rust's `sort_unstable_by`: HiGHS's cover comparator
//! compares with a `feastol` slack, so it is not a strict weak order, and its result
//! on near-ties depends on the algorithm's exact comparison sequence. The cover (and
//! with it the cut) is only reproducible if the algorithm is. The comparators here
//! are "less" predicates (`comp(a, b)` = `a` goes before `b`), as in C++.

const INSERTION_SORT_THRESHOLD: usize = 24;
const NINTHER_THRESHOLD: usize = 128;
const PARTIAL_INSERTION_SORT_LIMIT: usize = 8;
const BLOCK_SIZE: usize = 64;

fn log2(mut n: usize) -> i32 {
    let mut log = 0;
    loop {
        n >>= 1;
        if n == 0 {
            break;
        }
        log += 1;
    }
    log
}

fn insertion_sort<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut F,
) {
    if begin == end {
        return;
    }
    for cur in begin + 1..end {
        let mut sift = cur;
        let mut sift_1 = cur - 1;
        if comp(&v[sift], &v[sift_1]) {
            let tmp = v[sift];
            loop {
                v[sift] = v[sift_1];
                sift -= 1;
                if sift == begin {
                    break;
                }
                sift_1 -= 1;
                if !comp(&tmp, &v[sift_1]) {
                    break;
                }
            }
            v[sift] = tmp;
        }
    }
}

fn unguarded_insertion_sort<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut F,
) {
    if begin == end {
        return;
    }
    for cur in begin + 1..end {
        let mut sift = cur;
        let mut sift_1 = cur - 1;
        if comp(&v[sift], &v[sift_1]) {
            let tmp = v[sift];
            loop {
                v[sift] = v[sift_1];
                sift -= 1;
                sift_1 -= 1;
                if !comp(&tmp, &v[sift_1]) {
                    break;
                }
            }
            v[sift] = tmp;
        }
    }
}

fn partial_insertion_sort<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut F,
) -> bool {
    if begin == end {
        return true;
    }
    let mut limit = 0usize;
    for cur in begin + 1..end {
        let mut sift = cur;
        let mut sift_1 = cur - 1;
        if comp(&v[sift], &v[sift_1]) {
            let tmp = v[sift];
            loop {
                v[sift] = v[sift_1];
                sift -= 1;
                if sift == begin {
                    break;
                }
                sift_1 -= 1;
                if !comp(&tmp, &v[sift_1]) {
                    break;
                }
            }
            v[sift] = tmp;
            limit += cur - sift;
        }
        if limit > PARTIAL_INSERTION_SORT_LIMIT {
            return false;
        }
    }
    true
}

#[inline(always)]
fn sort2<T: Copy, F: FnMut(&T, &T) -> bool>(v: &mut [T], a: usize, b: usize, comp: &mut F) {
    if comp(&v[b], &v[a]) {
        v.swap(a, b);
    }
}

#[inline(always)]
fn sort3<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    a: usize,
    b: usize,
    c: usize,
    comp: &mut F,
) {
    sort2(v, a, b, comp);
    sort2(v, b, c, comp);
    sort2(v, a, b, comp);
}

fn swap_offsets<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    offsets_l: &[u8],
    offsets_r: &[u8],
    num: usize,
    use_swaps: bool,
) {
    if use_swaps {
        for i in 0..num {
            v.swap(first + offsets_l[i] as usize, last - offsets_r[i] as usize);
        }
    } else if num > 0 {
        let mut l = first + offsets_l[0] as usize;
        let mut r = last - offsets_r[0] as usize;
        let tmp = v[l];
        v[l] = v[r];
        for i in 1..num {
            l = first + offsets_l[i] as usize;
            v[r] = v[l];
            r = last - offsets_r[i] as usize;
            v[l] = v[r];
        }
        v[r] = tmp;
    }
}

fn partition_right_branchless<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut F,
) -> (usize, bool) {
    let pivot = v[begin];
    let mut first = begin;
    let mut last = end;
    loop {
        first += 1;
        if !comp(&v[first], &pivot) {
            break;
        }
    }
    if first - 1 == begin {
        while first < last {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    }
    let already_partitioned = first >= last;
    if !already_partitioned {
        v.swap(first, last);
        first += 1;
        let mut offsets_l_storage = [0u8; BLOCK_SIZE + 8];
        let mut offsets_r_storage = [0u8; BLOCK_SIZE + 8];
        let mut offsets_l_base = first;
        let mut offsets_r_base = last;
        let (mut num_l, mut num_r, mut start_l, mut start_r) = (0usize, 0usize, 0usize, 0usize);
        while first < last {
            let num_unknown = last - first;
            let left_split = if num_l == 0 {
                if num_r == 0 {
                    num_unknown / 2
                } else {
                    num_unknown
                }
            } else {
                0
            };
            let right_split = if num_r == 0 {
                num_unknown - left_split
            } else {
                0
            };

            if left_split >= BLOCK_SIZE {
                for i in 0..BLOCK_SIZE {
                    offsets_l_storage[num_l] = i as u8;
                    num_l += (!comp(&v[first], &pivot)) as usize;
                    first += 1;
                }
            } else {
                for i in 0..left_split {
                    offsets_l_storage[num_l] = i as u8;
                    num_l += (!comp(&v[first], &pivot)) as usize;
                    first += 1;
                }
            }
            if right_split >= BLOCK_SIZE {
                for i in 0..BLOCK_SIZE {
                    offsets_r_storage[num_r] = (i + 1) as u8;
                    last -= 1;
                    num_r += comp(&v[last], &pivot) as usize;
                }
            } else {
                for i in 0..right_split {
                    offsets_r_storage[num_r] = (i + 1) as u8;
                    last -= 1;
                    num_r += comp(&v[last], &pivot) as usize;
                }
            }
            let num = num_l.min(num_r);
            swap_offsets(
                v,
                offsets_l_base,
                offsets_r_base,
                &offsets_l_storage[start_l..],
                &offsets_r_storage[start_r..],
                num,
                num_l == num_r,
            );
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;
            if num_l == 0 {
                start_l = 0;
                offsets_l_base = first;
            }
            if num_r == 0 {
                start_r = 0;
                offsets_r_base = last;
            }
        }
        if num_l != 0 {
            let offs = &offsets_l_storage[start_l..];
            while num_l > 0 {
                num_l -= 1;
                last -= 1;
                v.swap(offsets_l_base + offs[num_l] as usize, last);
            }
            first = last;
        }
        if num_r != 0 {
            let offs = &offsets_r_storage[start_r..];
            while num_r > 0 {
                num_r -= 1;
                v.swap(offsets_r_base - offs[num_r] as usize, first);
                first += 1;
            }
            // `last = first` in C++; `last` is dead afterwards.
        }
    }
    let pivot_pos = first - 1;
    v[begin] = v[pivot_pos];
    v[pivot_pos] = pivot;
    (pivot_pos, already_partitioned)
}

fn partition_right<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut F,
) -> (usize, bool) {
    let pivot = v[begin];
    let mut first = begin;
    let mut last = end;
    loop {
        first += 1;
        if !comp(&v[first], &pivot) {
            break;
        }
    }
    if first - 1 == begin {
        while first < last {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    }
    let already_partitioned = first >= last;
    while first < last {
        v.swap(first, last);
        loop {
            first += 1;
            if !comp(&v[first], &pivot) {
                break;
            }
        }
        loop {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    }
    let pivot_pos = first - 1;
    v[begin] = v[pivot_pos];
    v[pivot_pos] = pivot;
    (pivot_pos, already_partitioned)
}

fn partition_left<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut F,
) -> usize {
    let pivot = v[begin];
    let mut first = begin;
    let mut last = end;
    loop {
        last -= 1;
        if !comp(&pivot, &v[last]) {
            break;
        }
    }
    if last + 1 == end {
        while first < last {
            first += 1;
            if comp(&pivot, &v[first]) {
                break;
            }
        }
    } else {
        loop {
            first += 1;
            if comp(&pivot, &v[first]) {
                break;
            }
        }
    }
    while first < last {
        v.swap(first, last);
        loop {
            last -= 1;
            if !comp(&pivot, &v[last]) {
                break;
            }
        }
        loop {
            first += 1;
            if comp(&pivot, &v[first]) {
                break;
            }
        }
    }
    let pivot_pos = last;
    v[begin] = v[pivot_pos];
    v[pivot_pos] = pivot;
    pivot_pos
}

#[allow(clippy::too_many_arguments)]
fn pdqsort_loop<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    mut begin: usize,
    end: usize,
    comp: &mut F,
    mut bad_allowed: i32,
    mut leftmost: bool,
    branchless: bool,
) {
    loop {
        let size = end - begin;
        if size < INSERTION_SORT_THRESHOLD {
            if leftmost {
                insertion_sort(v, begin, end, comp);
            } else {
                unguarded_insertion_sort(v, begin, end, comp);
            }
            return;
        }
        let s2 = size / 2;
        if size > NINTHER_THRESHOLD {
            sort3(v, begin, begin + s2, end - 1, comp);
            sort3(v, begin + 1, begin + (s2 - 1), end - 2, comp);
            sort3(v, begin + 2, begin + (s2 + 1), end - 3, comp);
            sort3(v, begin + (s2 - 1), begin + s2, begin + (s2 + 1), comp);
            v.swap(begin, begin + s2);
        } else {
            sort3(v, begin + s2, begin, end - 1, comp);
        }
        if !leftmost && !comp(&v[begin - 1], &v[begin]) {
            begin = partition_left(v, begin, end, comp) + 1;
            continue;
        }
        let (pivot_pos, already_partitioned) = if branchless {
            partition_right_branchless(v, begin, end, comp)
        } else {
            partition_right(v, begin, end, comp)
        };
        let l_size = pivot_pos - begin;
        let r_size = end - (pivot_pos + 1);
        let highly_unbalanced = l_size < size / 8 || r_size < size / 8;
        if highly_unbalanced {
            bad_allowed -= 1;
            if bad_allowed == 0 {
                heap_sort_range(v, begin, end, comp);
                return;
            }
            if l_size >= INSERTION_SORT_THRESHOLD {
                v.swap(begin, begin + l_size / 4);
                v.swap(pivot_pos - 1, pivot_pos - l_size / 4);
                if l_size > NINTHER_THRESHOLD {
                    v.swap(begin + 1, begin + (l_size / 4 + 1));
                    v.swap(begin + 2, begin + (l_size / 4 + 2));
                    v.swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
                    v.swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
                }
            }
            if r_size >= INSERTION_SORT_THRESHOLD {
                v.swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
                v.swap(end - 1, end - r_size / 4);
                if r_size > NINTHER_THRESHOLD {
                    v.swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
                    v.swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
                    v.swap(end - 2, end - (1 + r_size / 4));
                    v.swap(end - 3, end - (2 + r_size / 4));
                }
            }
        } else if already_partitioned
            && partial_insertion_sort(v, begin, pivot_pos, comp)
            && partial_insertion_sort(v, pivot_pos + 1, end, comp)
        {
            return;
        }
        pdqsort_loop(v, begin, pivot_pos, comp, bad_allowed, leftmost, branchless);
        begin = pivot_pos + 1;
        leftmost = false;
    }
}

/// `pdqsort(begin, end, comp)` with a user lambda (non-branchless partition).
pub fn pdqsort<T: Copy, F: FnMut(&T, &T) -> bool>(v: &mut [T], mut comp: F) {
    if v.is_empty() {
        return;
    }
    let n = v.len();
    pdqsort_loop(v, 0, n, &mut comp, log2(n), true, false);
}

/// `pdqsort_branchless(begin, end, comp)`; also what `pdqsort(begin, end)` with
/// `std::less<double>` dispatches to.
pub fn pdqsort_branchless<T: Copy, F: FnMut(&T, &T) -> bool>(v: &mut [T], mut comp: F) {
    if v.is_empty() {
        return;
    }
    let n = v.len();
    pdqsort_loop(v, 0, n, &mut comp, log2(n), true, true);
}

/// `pdqsort(begin, end)` on doubles (`std::less`, arithmetic => branchless).
pub fn pdqsort_f64(v: &mut [f64]) {
    pdqsort_branchless(v, |a: &f64, b: &f64| a < b);
}

// ---- independent implementations (Rust std and textbook algorithms) ----
//
// These used to be translations of libstdc++ (`std::partition`, `std::partial_sort`,
// `std::make_heap`/`std::sort_heap`, `std::upper_bound`). They are replaced by
// implementations written independently (gap plan §12.31, licensing): the results are
// the same sets/values, but the order of tied elements may differ from libstdc++'s.
// Cut validity never depends on that order; only which of equally valid cuts is
// produced on a tie.

fn ord_of<T, F: FnMut(&T, &T) -> bool>(comp: &mut F, a: &T, b: &T) -> std::cmp::Ordering {
    if comp(a, b) {
        std::cmp::Ordering::Less
    } else if comp(b, a) {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

/// Heapsort of `v[begin..end]` (textbook sift-down heap), pdqsort's fallback for
/// inputs that keep producing unbalanced partitions.
fn heap_sort_range<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut F,
) {
    let s = &mut v[begin..end];
    let n = s.len();
    let sift_down = |s: &mut [T], mut root: usize, len: usize, comp: &mut F| loop {
        let mut child = 2 * root + 1;
        if child >= len {
            break;
        }
        if child + 1 < len && comp(&s[child], &s[child + 1]) {
            child += 1;
        }
        if !comp(&s[root], &s[child]) {
            break;
        }
        s.swap(root, child);
        root = child;
    };
    for i in (0..n / 2).rev() {
        sift_down(s, i, n, comp);
    }
    for last in (1..n).rev() {
        s.swap(0, last);
        sift_down(s, 0, last, comp);
    }
}

/// `partial_sort(v, middle, comp)`: the `middle` smallest elements, sorted, in
/// `v[..middle]` (`select_nth_unstable_by` + `sort_unstable_by`).
pub fn partial_sort<T: Copy, F: FnMut(&T, &T) -> bool>(v: &mut [T], middle: usize, mut comp: F) {
    if middle == 0 {
        return;
    }
    if middle < v.len() {
        v.select_nth_unstable_by(middle - 1, |a, b| ord_of(&mut comp, a, b));
    }
    let k = middle.min(v.len());
    v[..k].sort_unstable_by(|a, b| ord_of(&mut comp, a, b));
}

/// `partition(v, pred)`: moves the elements satisfying `pred` to the front and returns
/// their count (textbook in-place partition; order within each side unspecified).
pub fn partition<T, P: FnMut(&T) -> bool>(v: &mut [T], mut pred: P) -> usize {
    let mut k = 0;
    for j in 0..v.len() {
        if pred(&v[j]) {
            v.swap(k, j);
            k += 1;
        }
    }
    k
}

/// `upper_bound(v, value, comp)`: the first position whose element `e` has
/// `comp(value, e)` (`slice::partition_point`).
pub fn upper_bound<T, V, F: FnMut(&V, &T) -> bool>(v: &[T], value: &V, mut comp: F) -> usize {
    v.partition_point(|e| !comp(value, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn pdqsort_variants_sort_and_the_heap_fallback_is_reachable() {
        let mut s = 7u64;
        let mut checks = 0usize;
        for n in [0usize, 1, 2, 5, 23, 24, 25, 100, 129, 200, 1000, 5000] {
            for dup in [1u64, 3, 1000, 1 << 30] {
                let base: Vec<u64> = (0..n).map(|_| lcg(&mut s) % dup).collect();
                let mut want = base.clone();
                want.sort();
                let mut a = base.clone();
                pdqsort(&mut a, |x: &u64, y: &u64| x < y);
                assert_eq!(a, want);
                let mut b = base.clone();
                pdqsort_branchless(&mut b, |x: &u64, y: &u64| x < y);
                assert_eq!(b, want);
                let mut c = base.clone();
                let mid = n / 3;
                partial_sort(&mut c, mid, |x: &u64, y: &u64| x < y);
                assert_eq!(&c[..mid], &want[..mid]);
                checks += 3;
            }
        }
        // A "killer" input that forces repeated unbalanced partitions still sorts.
        let mut k: Vec<u64> = (0..4096)
            .map(|i| if i % 2 == 0 { i } else { 4096 - i })
            .collect();
        let mut want = k.clone();
        want.sort();
        pdqsort(&mut k, |x: &u64, y: &u64| x < y);
        assert_eq!(k, want);
        // The heapsort fallback on its own.
        let mut h: Vec<u64> = (0..777).map(|_| lcg(&mut s) % 50).collect();
        let mut want = h.clone();
        want.sort();
        let n = h.len();
        let mut lt = |x: &u64, y: &u64| x < y;
        heap_sort_range(&mut h, 0, n, &mut lt);
        assert_eq!(h, want);
        checks += 2;
        assert!(checks > 100);
    }

    #[test]
    fn partition_and_upper_bound_are_correct() {
        let mut s = 11u64;
        let mut checks = 0usize;
        for n in [0usize, 1, 7, 64, 500] {
            let base: Vec<u64> = (0..n).map(|_| lcg(&mut s) % 20).collect();
            let mut v = base.clone();
            let p = partition(&mut v, |x| *x >= 10);
            assert!(v[..p].iter().all(|x| *x >= 10) && v[p..].iter().all(|x| *x < 10));
            let mut a = v.clone();
            a.sort();
            let mut b = base.clone();
            b.sort();
            assert_eq!(a, b);
            for t in 0..21u64 {
                let ub = upper_bound(&b, &t, |t: &u64, e: &u64| t < e);
                assert_eq!(ub, b.iter().filter(|e| **e <= t).count());
                checks += 1;
            }
        }
        assert!(checks > 50);
    }
}
