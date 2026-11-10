//! Faithful port of `util/HighsRandom.h` and the parts of `util/HighsHash.h` it and
//! the cut generator use (`pair_hash`, `hash(std::pair<int,int>)`, `log2i`).
//! HiGHS is built with 32-bit `HighsInt` (`HIGHSINT64` undefined), which this
//! port assumes. Bit-exactness is checked by the cut-generation replay tests,
//! which compare the generator state before and after each cut against HiGHS.

const C: [u64; 64] = [
    0xc8497d2a400d9551,
    0x80c8963be3e4c2f3,
    0x042d8680e260ae5b,
    0x8a183895eeac1536,
    0xa94e9c75f80ad6de,
    0x7e92251dec62835e,
    0x07294165cb671455,
    0x89b0f6212b0a4292,
    0x31900011b96bf554,
    0xa44540f8eee2094f,
    0xce7ffd372e4c64fc,
    0x51c9d471bfe6a10f,
    0x758c2a674483826f,
    0xf91a20abe63f8b02,
    0xc2a069024a1fcc6f,
    0xd5bb18b70c5dbd59,
    0xd510adac6d1ae289,
    0x571d069b23050a79,
    0x60873b8872933e06,
    0x780481cc19670350,
    0x7a48551760216885,
    0xb5d68b918231e6ca,
    0xa7e5571699aa5274,
    0x7b6d309b2cfdcf01,
    0x04e77c3d474daeff,
    0x4dbf099fd7247031,
    0x5d70dca901130beb,
    0x9f8b5f0df4182499,
    0x293a74c9686092da,
    0xd09bdab6840f52b3,
    0xc05d47f3ab302263,
    0x6b79e62b884b65d6,
    0xa581106fc980c34d,
    0xf081b7145ea2293e,
    0xfb27243dd7c3f5ad,
    0x5211bf8860ea667f,
    0x9455e65cb2385e7f,
    0x0dfaf6731b449b33,
    0x4ec98b3c6f5e68c7,
    0x007bfd4a42ae936b,
    0x65c93061f8674518,
    0x640816f17127c5d1,
    0x6dd4bab17b7c3a74,
    0x34d9268c256fa1ba,
    0x0b4d0c6b5b50d7f4,
    0x30aa965bc9fadaff,
    0xc0ac1d0c2771404d,
    0xc5e64509abb76ef2,
    0xd606b11990624a36,
    0x0d3f05d242ce2fb7,
    0x469a803cb276fe32,
    0xa4a44d177a3e23f4,
    0xb9d9a120dcc1ca03,
    0x2e15af8165234a2e,
    0x10609ba2720573d4,
    0xaa4191b60368d1d5,
    0x333dd2300bc57762,
    0xdf6ec48f79fb402f,
    0x5ed20fcef1b734fa,
    0x4c94924ec8be21ee,
    0x5abe6ad9d131e631,
    0xbe10136a522e602d,
    0x53671115c340e779,
    0x9f392fe43e2144da,
];

/// `HighsHashHelpers::pair_hash<k>(a, b)`.
#[inline(always)]
pub fn pair_hash(k: usize, a: u32, b: u32) -> u64 {
    (a as u64)
        .wrapping_add(C[2 * k])
        .wrapping_mul((b as u64).wrapping_add(C[2 * k + 1]))
}

/// `HighsHashHelpers::hash(std::make_pair(HighsInt a, HighsInt b))` (8-byte case).
#[inline(always)]
pub fn hash_pair_i32(a: i32, b: i32) -> u64 {
    let (b0, b1) = (a as u32, b as u32);
    pair_hash(1, b0, b1) ^ (pair_hash(0, b0, b1) >> 32)
}

#[inline(always)]
fn log2i_u32(n: u32) -> i32 {
    31 - n.leading_zeros() as i32
}

/// `HighsRandom`: xorshift state with hashed outputs.
#[derive(Clone, Debug)]
pub struct HighsRandom {
    state: u64,
}

impl HighsRandom {
    /// `HighsRandom(seed)`.
    pub fn new(seed: u32) -> Self {
        let mut r = HighsRandom { state: 0 };
        r.initialise(seed);
        r
    }
    /// Resume from a recorded state (the replay tests use HiGHS's dumped state).
    pub fn from_state(state: u64) -> Self {
        HighsRandom { state }
    }
    /// The raw generator state.
    pub fn state(&self) -> u64 {
        self.state
    }
    /// `initialise(seed)`.
    pub fn initialise(&mut self, seed: u32) {
        self.state = seed as u64;
        loop {
            self.state = pair_hash(0, self.state as u32, (self.state >> 32) as u32);
            self.state ^= pair_hash(1, (self.state >> 32) as u32, seed) >> 32;
            if self.state != 0 {
                break;
            }
        }
    }
    /// `advance()`: one xorshift step.
    #[inline(always)]
    pub fn advance(&mut self) {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
    }
    #[inline(always)]
    fn lohi(&self) -> (u32, u32) {
        (self.state as u32, (self.state >> 32) as u32)
    }
    fn draw_uniform32(&mut self, sup: u32, nbits: i32) -> u32 {
        // Output functions 0..=7 and 9..=31: HiGHS skips 8, reproduced here.
        const OUTS: [usize; 31] = [
            0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31,
        ];
        loop {
            self.advance();
            let (lo, hi) = self.lohi();
            for &n in OUTS.iter() {
                let val = (pair_hash(n, lo, hi) >> (64 - nbits)) as u32;
                if val < sup {
                    return val;
                }
            }
        }
    }
    /// A random integer in `[0, 2^31)`.
    pub fn integer(&mut self) -> i32 {
        self.advance();
        let (lo, hi) = self.lohi();
        (pair_hash(0, lo, hi) >> 33) as i32
    }
    /// A random integer in `[0, sup)`.
    pub fn integer_below(&mut self, sup: i32) -> i32 {
        if sup <= 1 {
            return 0;
        }
        let nbits = log2i_u32((sup - 1) as u32) + 1;
        self.draw_uniform32(sup as u32, nbits) as i32
    }
    /// A random fraction in the open interval (0, 1).
    pub fn fraction(&mut self) -> f64 {
        self.advance();
        let (lo, hi) = self.lohi();
        let output = (pair_hash(0, lo, hi) >> (64 - 52)) ^ (pair_hash(1, lo, hi) >> (64 - 26));
        (1 + output) as f64 * 2.2204460492503125e-16
    }
    /// A random fraction in the closed interval [0, 1].
    pub fn closed_fraction(&mut self) -> f64 {
        self.advance();
        let (lo, hi) = self.lohi();
        let output = (pair_hash(0, lo, hi) >> (64 - 53)) ^ (pair_hash(1, lo, hi) >> 32);
        output as f64 * 1.1102230246251566e-16
    }
    /// `real(a, b)`: uniform in [a, b].
    pub fn real(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.closed_fraction()
    }
    /// `bit()`.
    pub fn bit(&mut self) -> bool {
        self.advance();
        let (lo, hi) = self.lohi();
        (pair_hash(0, lo, hi) >> 63) != 0
    }
    /// `shuffle(data, N)` (Fisher-Yates from the back).
    pub fn shuffle<T>(&mut self, data: &mut [T]) {
        let n = data.len();
        let mut i = n;
        while i > 1 {
            let pos = self.integer_below(i as i32) as usize;
            data.swap(pos, i - 1);
            i -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values printed by a C++ program built against HiGHS's own
    /// `HighsRandom.h`/`HighsHash.h` (clang -O2, x86-64, 32-bit `HighsInt`).
    /// The cut replay never exercises the cover tiebreak hash on the recorded
    /// population, so this test is what pins it.
    #[test]
    fn hash_and_generator_match_highs_reference_values() {
        let h: [(i32, i32, u64); 8] = [
            (0, 0, 2135356889229817085),
            (0, 42, 14779231693550946974),
            (0, -1, 18146541805284781345),
            (1, 2147483647, 1494448546770062214),
            (7, 42, 10647614550169158049),
            (12345, -1, 5032312969141746148),
            (-3, 0, 7927948579550475679),
            (-3, 2147483647, 6559646670366787762),
        ];
        for (a, b, want) in h {
            assert_eq!(hash_pair_i32(a, b), want, "hash({a},{b})");
        }
        #[allow(clippy::type_complexity)]
        let r: [(u32, u64, [i32; 3], i32, i32, f64, f64, bool); 4] = [
            (
                0,
                11452595370619992822,
                [949769484, 1049131993, 1409581922],
                1,
                991493,
                0.74103233832769788,
                0.065204513679423926,
                false,
            ),
            (
                1,
                2285683514435278526,
                [744992014, 1023482941, 761245146],
                2,
                961252,
                0.9402844842002005,
                0.63993387496963261,
                true,
            ),
            (
                12345,
                16700606225564332918,
                [1442184564, 69784783, 1213288113],
                6,
                58745,
                0.34648949869863216,
                0.04753717556242143,
                true,
            ),
            (
                4000000000,
                8020933233738222706,
                [1175169038, 962136917, 1606426238],
                1,
                322115,
                0.25466231749948037,
                0.46909334574619943,
                true,
            ),
        ];
        let mut checks = 0;
        for (seed, state, ints, i10, ibig, fr, cfr, bit) in r {
            let mut g = HighsRandom::new(seed);
            assert_eq!(g.state(), state, "seed {seed}");
            for want in ints {
                assert_eq!(g.integer(), want);
            }
            assert_eq!(g.integer_below(10), i10);
            assert_eq!(g.integer_below(1000003), ibig);
            assert_eq!(g.fraction(), fr);
            assert_eq!(g.closed_fraction(), cfr);
            assert_eq!(g.bit(), bit);
            checks += 9;
        }
        assert_eq!(checks, 36);
    }
}
