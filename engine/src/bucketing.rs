//! Card-abstraction bucket tables, loaded at solve/play time (Phase 3c, B2).
//!
//! The Phase-3b pipeline (`src/parciver/bucketing.py`) clustered every
//! suit-isomorphic postflop situation into buckets and persisted one `uint16`
//! per dense hand-isomorphism index to
//! `data/abstraction/{flop,turn,river}_buckets.npy`. This module loads those
//! tables into memory and reproduces, in Rust, the exact Python lookup
//!
//! ```text
//! bucket_of(street, cards) = table[canonical_index(street, cards)]
//! ```
//!
//! so the NLHE game ([`crate::nlhe`]) can key its information sets on buckets —
//! the card abstraction that makes the tree solvable. Preflop is NOT bucketed:
//! it is the 169 exact suit-isomorphic classes
//! ([`crate::abstraction::preflop_index`]).
//!
//! The parity that matters is enforced by a gate: Rust `bucket_of` must equal the
//! Python reference `parciver.bucketing.bucket_of` for every sampled situation
//! (see `tests/test_nlhe.py` and `phase_audit.py::check_nlhe_bucket_key`). Both
//! sides call the SAME `canonical_index` (this crate's [`HandIndexer`]), so the
//! gate really certifies the `.npy` reader below loaded the table values
//! faithfully and the street mapping is right.

use std::path::Path;

use crate::abstraction::preflop_index;
use crate::cards::Card;
use crate::isomorphism::HandIndexer;

/// Postflop street file stems, in street-id order (0=flop, 1=turn, 2=river) —
/// matching the Python indexer and the persisted table names.
const STREET_NAMES: [&str; 3] = ["flop", "turn", "river"];

/// Cards-per-round profile of each street's hand-isomorphism indexer: the hole
/// is one round, the whole board another (equity is a card-*set* function, so the
/// order within the board is strategically irrelevant). Mirrors
/// `lib.rs::street_indexers` and the Python driver exactly.
const CARDS_PER_ROUND: [[u8; 2]; 3] = [[2, 3], [2, 4], [2, 5]];

/// The three postflop bucket tables plus the hand-isomorphism indexers that
/// address them. Constructed once and shared (e.g. behind an `Arc`) by the game.
pub struct Abstraction {
    indexers: [HandIndexer; 3],
    tables: [Vec<u16>; 3],
}

impl Abstraction {
    /// Load the flop/turn/river bucket tables from `dir` (the `data/abstraction`
    /// directory). Each table is validated to be a 1-D little-endian `uint16`
    /// array whose length equals its street's canonical-index size, so a wrong or
    /// stale table fails loudly at construction rather than mis-keying infosets.
    pub fn load(dir: &Path) -> Result<Abstraction, String> {
        let indexers = [
            HandIndexer::new(&CARDS_PER_ROUND[0]),
            HandIndexer::new(&CARDS_PER_ROUND[1]),
            HandIndexer::new(&CARDS_PER_ROUND[2]),
        ];
        let mut tables: Vec<Vec<u16>> = Vec::with_capacity(3);
        for (s, name) in STREET_NAMES.iter().enumerate() {
            let path = dir.join(format!("{name}_buckets.npy"));
            let table = read_npy_u16_1d(&path)?;
            let expect = indexers[s].size() as usize;
            if table.len() != expect {
                return Err(format!(
                    "{name} bucket table has {} entries, expected {expect} \
                     (canonical index size); table is stale or for a different street",
                    table.len()
                ));
            }
            tables.push(table);
        }
        let tables: [Vec<u16>; 3] = tables
            .try_into()
            .map_err(|_| "internal: expected exactly 3 bucket tables".to_string())?;
        Ok(Abstraction { indexers, tables })
    }

    /// The abstraction's identity: the [`crate::mccfr::Fnv64`] of the three
    /// bucket tables, each as its length (`u64`) then its values (`u16`), little-
    /// endian. A checkpoint records it so a solve resumes only on the tables it
    /// was trained on. It depends on the tables, not on the files that held them.
    pub fn fingerprint(&self) -> u64 {
        let mut h = crate::mccfr::Fnv64::new();
        for table in &self.tables {
            h.update(&(table.len() as u64).to_le_bytes());
            for v in table {
                h.update(&v.to_le_bytes());
            }
        }
        h.value()
    }

    /// Bucket id of a `(hole, board)` situation.
    ///
    /// * empty board -> preflop class `0..169` ([`preflop_index`]);
    /// * a 3/4/5-card board -> the postflop bucket
    ///   `table[canonical_index(street, hole ++ board)]`, exactly the Python
    ///   `parciver.bucketing.bucket_of` semantics.
    ///
    /// Panics on an unsupported board length; the NLHE tree only ever queries
    /// decision nodes, whose boards hold 0/3/4/5 cards.
    pub fn bucket_of(&self, hole: [Card; 2], board: &[Card]) -> u32 {
        match board.len() {
            0 => preflop_index(hole[0], hole[1]) as u32,
            3 => self.postflop_bucket(0, hole, board) as u32,
            4 => self.postflop_bucket(1, hole, board) as u32,
            5 => self.postflop_bucket(2, hole, board) as u32,
            n => panic!("bucket_of: unsupported board length {n} (want 0/3/4/5)"),
        }
    }

    /// Postflop bucket via `table[index(hole ++ board)]`, where `street` is
    /// 0=flop / 1=turn / 2=river. The index is order-invariant within the hole and
    /// within the board (it is set-based per round), so the caller need not sort.
    fn postflop_bucket(&self, street: usize, hole: [Card; 2], board: &[Card]) -> u16 {
        // Stack buffer (hole + at most 5 board cards): this runs at every
        // decision node of the blueprint game, so it must not allocate.
        let mut cards = [0u8; 7];
        cards[0] = hole[0].index();
        cards[1] = hole[1].index();
        for (slot, c) in cards[2..].iter_mut().zip(board) {
            *slot = c.index();
        }
        let idx = self.indexers[street].index(&cards[..2 + board.len()]) as usize;
        self.tables[street][idx]
    }
}

/// Minimal reader for a 1-D little-endian `uint16` NumPy `.npy` array (format
/// v1/v2/v3). Returns the array as a `Vec<u16>`. Rejects anything that is not
/// exactly a C-ordered 1-D `'<u2'` array — the shape the bucket tables always
/// have — so a corrupt or unexpected file fails loudly instead of being misread.
fn read_npy_u16_1d(path: &Path) -> Result<Vec<u16>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {path:?}: {e}"))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{path:?}: not a .npy file (bad magic)"));
    }
    let major = bytes[6];
    // v1 stores the header length in 2 bytes; v2/v3 in 4 bytes.
    let (header_start, header_len) = match major {
        1 => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(format!("{path:?}: truncated v{major} header"));
            }
            let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (12usize, hl)
        }
        _ => return Err(format!("{path:?}: unsupported .npy major version {major}")),
    };
    let data_start = header_start + header_len;
    if bytes.len() < data_start {
        return Err(format!("{path:?}: truncated header"));
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .map_err(|_| format!("{path:?}: header is not valid UTF-8"))?;
    if !header.contains("'<u2'") {
        return Err(format!(
            "{path:?}: expected dtype '<u2', header was: {header}"
        ));
    }
    if !header.contains("'fortran_order': False") {
        return Err(format!(
            "{path:?}: expected C-order (fortran_order False), header was: {header}"
        ));
    }
    let n = parse_npy_shape_1d(header)?;
    let expected_bytes = data_start + n * 2;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{path:?}: data is {} bytes, expected {} for shape ({n},)",
            bytes.len(),
            expected_bytes
        ));
    }
    // Little-endian u16 decode; `chunks_exact(2)` vectorizes well in release.
    let out: Vec<u16> = bytes[data_start..]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(out)
}

/// Parse the single dimension out of a `.npy` header's `'shape': (N,)`. Errors on
/// a missing, malformed, or non-1-D shape.
fn parse_npy_shape_1d(header: &str) -> Result<usize, String> {
    let key = "'shape':";
    let at = header.find(key).ok_or("no 'shape' in .npy header")?;
    let rest = &header[at + key.len()..];
    let lp = rest.find('(').ok_or("malformed shape (no '(')")?;
    let rp = rest[lp..].find(')').ok_or("malformed shape (no ')')")? + lp;
    let inner = &rest[lp + 1..rp];
    let dims: Vec<usize> = inner
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .map_err(|_| format!("bad shape dim {t:?}"))
        })
        .collect::<Result<_, _>>()?;
    if dims.len() != 1 {
        return Err(format!("expected a 1-D shape, got {dims:?}"));
    }
    Ok(dims[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid v1.0 `.npy` byte image for a 1-D `uint16` array. The header
    /// dict is terminated with `\n`; NumPy also pads the header to a 64-byte
    /// boundary, which our reader does not require, so we skip that padding here
    /// (the reader keys off the header-length field, not alignment).
    fn npy_u16_image(vals: &[u16]) -> Vec<u8> {
        let mut header = format!(
            "{{'descr': '<u2', 'fortran_order': False, 'shape': ({},), }}\n",
            vals.len()
        )
        .into_bytes();
        // Header length must fit in the v1 u16 field.
        assert!(header.len() <= u16::MAX as usize);
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1); // major
        out.push(0); // minor
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.append(&mut header);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn npy_reader_round_trips_a_small_u16_array() {
        let vals = [0u16, 7, 65_535, 1234, 9999, 42];
        let path =
            std::env::temp_dir().join(format!("parciver_npy_test_{}.npy", std::process::id()));
        std::fs::write(&path, npy_u16_image(&vals)).unwrap();
        let got = read_npy_u16_1d(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(got.unwrap(), vals);
    }

    #[test]
    fn fingerprint_hashes_each_tables_length_then_values() {
        let fingerprint = |tables: [Vec<u16>; 3]| {
            let indexers = CARDS_PER_ROUND.map(|c| HandIndexer::new(&c));
            Abstraction { indexers, tables }.fingerprint()
        };
        let base = fingerprint([vec![1, 2], vec![3], vec![]]);
        let mut h = crate::mccfr::Fnv64::new();
        for bytes in [&2u64.to_le_bytes()[..], &[1, 0, 2, 0], &1u64.to_le_bytes(), &[3, 0], &0u64.to_le_bytes()] {
            h.update(bytes);
        }
        assert_eq!(base, h.value());
        assert_ne!(base, fingerprint([vec![1, 4], vec![3], vec![]]), "a value");
        assert_ne!(base, fingerprint([vec![1], vec![2, 3], vec![]]), "a table boundary");
    }

    #[test]
    fn npy_reader_rejects_bad_magic_and_wrong_dtype() {
        let dir = std::env::temp_dir();
        // Bad magic.
        let p1 = dir.join(format!("parciver_npy_badmagic_{}.npy", std::process::id()));
        std::fs::write(&p1, b"not a numpy file at all").unwrap();
        assert!(read_npy_u16_1d(&p1).is_err());
        let _ = std::fs::remove_file(&p1);

        // Correct magic but a float64 dtype must be rejected (not '<u2').
        let p2 = dir.join(format!("parciver_npy_f8_{}.npy", std::process::id()));
        let mut header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }\n".to_vec();
        let mut img = Vec::new();
        img.extend_from_slice(b"\x93NUMPY");
        img.push(1);
        img.push(0);
        img.extend_from_slice(&(header.len() as u16).to_le_bytes());
        img.append(&mut header);
        img.extend_from_slice(&0f64.to_le_bytes());
        std::fs::write(&p2, &img).unwrap();
        assert!(read_npy_u16_1d(&p2).is_err());
        let _ = std::fs::remove_file(&p2);
    }
}
