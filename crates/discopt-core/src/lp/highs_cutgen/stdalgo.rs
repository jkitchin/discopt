//! Exact ports of the ordering algorithms the HiGHS cut generator calls: Orson
//! Peters' `pdqsort` / `pdqsort_branchless` (`extern/pdqsort/pdqsort.h`, zlib
//! license, as vendored by HiGHS) and the libstdc++ `std::partition`,
//! `std::partial_sort`, `std::make_heap`/`std::sort_heap` and `std::upper_bound`.
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
                make_heap(v, begin, end, comp);
                sort_heap(v, begin, end, comp);
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

// ---- libstdc++ heap algorithms (bits/stl_heap.h) ----

fn push_heap_impl<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    first: usize,
    mut hole: usize,
    top: usize,
    value: T,
    comp: &mut F,
) {
    let mut parent = if hole == 0 { 0 } else { (hole - 1) / 2 };
    while hole > top && comp(&v[first + parent], &value) {
        v[first + hole] = v[first + parent];
        hole = parent;
        parent = if hole == 0 { 0 } else { (hole - 1) / 2 };
    }
    v[first + hole] = value;
}

fn adjust_heap<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    first: usize,
    mut hole: usize,
    len: usize,
    value: T,
    comp: &mut F,
) {
    let top = hole;
    let mut second = hole;
    // `(len - 1) / 2` with signed ptrdiff in C++; len >= 1 here.
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if comp(&v[first + second], &v[first + second - 1]) {
            second -= 1;
        }
        v[first + hole] = v[first + second];
        hole = second;
    }
    if (len & 1) == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[first + hole] = v[first + second - 1];
        hole = second - 1;
    }
    push_heap_impl(v, first, hole, top, value, comp);
}

pub(crate) fn make_heap<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: &mut F,
) {
    let len = last - first;
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent];
        adjust_heap(v, first, parent, len, value, comp);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn pop_heap_to<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    first: usize,
    last: usize,
    result: usize,
    comp: &mut F,
) {
    let value = v[result];
    v[result] = v[first];
    adjust_heap(v, first, 0, last - first, value, comp);
}

pub(crate) fn sort_heap<T: Copy, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    comp: &mut F,
) {
    while last - first > 1 {
        last -= 1;
        pop_heap_to(v, first, last, last, comp);
    }
}

/// `std::partial_sort(first, middle, last, comp)` (libstdc++: heap select + sort heap).
pub fn partial_sort<T: Copy, F: FnMut(&T, &T) -> bool>(v: &mut [T], middle: usize, mut comp: F) {
    let last = v.len();
    if middle == 0 {
        return;
    }
    make_heap(v, 0, middle, &mut comp);
    for i in middle..last {
        if comp(&v[i], &v[0]) {
            pop_heap_to(v, 0, middle, i, &mut comp);
        }
    }
    sort_heap(v, 0, middle, &mut comp);
}

/// `std::partition(first, last, pred)` (libstdc++ bidirectional version); returns
/// the partition point.
pub fn partition<T, P: FnMut(&T) -> bool>(v: &mut [T], mut pred: P) -> usize {
    let mut first = 0usize;
    let mut last = v.len();
    loop {
        loop {
            if first == last {
                return first;
            } else if pred(&v[first]) {
                first += 1;
            } else {
                break;
            }
        }
        last -= 1;
        loop {
            if first == last {
                return first;
            } else if !pred(&v[last]) {
                last -= 1;
            } else {
                break;
            }
        }
        v.swap(first, last);
        first += 1;
    }
}

/// `std::upper_bound(first, last, value, comp)` with `comp(value, elem)`.
pub fn upper_bound<T, V, F: FnMut(&V, &T) -> bool>(v: &[T], value: &V, mut comp: F) -> usize {
    let mut first = 0usize;
    let mut len = v.len();
    while len > 0 {
        let half = len >> 1;
        let middle = first + half;
        if comp(value, &v[middle]) {
            len = half;
        } else {
            first = middle + 1;
            len = len - half - 1;
        }
    }
    first
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
        // The heap algorithms on their own.
        let mut h: Vec<u64> = (0..777).map(|_| lcg(&mut s) % 50).collect();
        let mut want = h.clone();
        want.sort();
        let n = h.len();
        let mut lt = |x: &u64, y: &u64| x < y;
        make_heap(&mut h, 0, n, &mut lt);
        sort_heap(&mut h, 0, n, &mut lt);
        assert_eq!(h, want);
        checks += 2;
        assert!(checks > 100);
    }

    #[test]
    fn partition_matches_libstdcxx_swap_order() {
        // libstdc++ swaps the first misplaced element from the left with the last
        // misplaced one from the right; the order of each side is thus fixed.
        let mut v = vec![1, 8, 3, 9, 5, 6, 7, 2];
        let p = partition(&mut v, |x| *x >= 5);
        assert_eq!(p, 5);
        assert_eq!(v, vec![7, 8, 6, 9, 5, 3, 1, 2]);
    }
}
