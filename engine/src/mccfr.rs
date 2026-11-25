// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! External-sampling Monte-Carlo CFR (ES-MCCFR) — the sampled counterpart to the
//! full-traversal [`crate::cfr`] solver, and the hot loop that scales to the
//! abstracted hold'em tree where a full traversal is intractable.
//!
//! On each iteration one player is the *traverser*: the tree is explored over ALL
//! of the traverser's actions, but only a SINGLE sampled action at each opponent
//! node and a single sampled outcome at each chance node (Lanctot, Waugh,
//! Zinkevich & Bowling, NIPS 2009). Sampling the opponent and chance in
//! proportion to their own reach is exactly what lets the regret update drop the
//! explicit counterfactual-reach weight that full CFR carries — the sampling
//! supplies it in expectation. So:
//!   * regret is accumulated at the TRAVERSER's infosets (increment `v[a] - v`,
//!     no reach factor), and
//!   * the average strategy is accumulated at the OPPONENT's infosets, where
//!     visits already occur in proportion to reach.
//! This is the update structure of OpenSpiel's (validated) external-sampling
//! MCCFR; the per-iteration weighting below is the only extension.
//!
//! Correctness is certified the same way as the rest of the engine and is NOT
//! asserted by construction here: the Python best-response oracle scores
//! `average_strategy()` and must watch exploitability fall to equilibrium on Kuhn
//! and Leduc, independent of this file.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};

use crate::game::Game;

/// Upper bound on the number of legal actions at any node, across every game this
/// solver runs (Kuhn 2, Leduc 3, NLHE fold/check-call/two sized bets/all-in). The
/// history token asserts `< 10`; a size-10 stack array therefore holds any node's
/// strategy/regret snapshot with room to spare, so the hot loop needs no per-visit
/// heap allocation for them (only the two persistent per-infoset `Vec`s allocate,
/// once each, on first visit).
const MAX_ACTIONS: usize = 10;

/// Stack capacity for a responder node's saved infoset key (short ASCII: street
/// tag, card token, betting history). The recursion clobbers the shared scratch
/// buffer, so the key is copied here before descending and reused for the regret
/// write-back instead of being rebuilt. A longer key falls back to the rebuild, so
/// the bytes used for the write-back are identical either way.
const KEY_STACK: usize = 128;

/// Sampling-robust CFR weighting. Unlike the tabular [`crate::cfr::Variant`] —
/// which rescales every node once per iteration, sound only when a full traversal
/// touches every node — these weights are applied at UPDATE time to the touched
/// node alone, so they stay correct when sampling visits only a subset.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Plain external-sampling MCCFR: unit regret increments, uniform averaging.
    /// The provably-convergent baseline and primary correctness anchor.
    Vanilla,
    /// Regret-matching-plus (negative regret floored to zero) with linear
    /// averaging — CFR+ under sampling; the robust practical workhorse.
    Plus,
    /// Linear CFR (Brown & Sandholm 2019; the Pluribus choice): iteration `t`
    /// weights both the regret increment and the strategy increment by `t`, so
    /// later near-equilibrium iterates dominate without a separate discount pass.
    Linear,
}

impl Variant {
    /// Weight on this iteration's regret increment. Only Linear CFR scales it.
    fn regret_increment_weight(self, t: u64) -> f64 {
        match self {
            Variant::Linear => t as f64,
            _ => 1.0,
        }
    }

    /// Weight on this iteration's contribution to the average strategy. Vanilla
    /// averages uniformly; Plus/Linear weight by `t` (linear averaging).
    fn strategy_weight(self, t: u64) -> f64 {
        match self {
            Variant::Vanilla => 1.0,
            Variant::Plus | Variant::Linear => t as f64,
        }
    }

    /// `sum_{t=1..t_max} strategy_weight(t)`: the total averaging weight after
    /// `t_max` iterations (normalizes an infoset's accumulated strategy weight into
    /// a reach-probability estimate).
    fn strategy_weight_sum(self, t_max: u64) -> f64 {
        let t = t_max as f64;
        match self {
            Variant::Vanilla => t,
            Variant::Plus | Variant::Linear => t * (t + 1.0) / 2.0,
        }
    }

    /// Whether to floor cumulative regret at zero after each update (RM+).
    fn regret_matching_plus(self) -> bool {
        matches!(self, Variant::Plus)
    }
}

/// Resolve a variant by its Python-side name, or `None` if unknown.
pub fn variant_by_name(name: &str) -> Option<Variant> {
    match name {
        "mccfr" | "vanilla" => Some(Variant::Vanilla),
        "mccfr+" | "plus" => Some(Variant::Plus),
        "linear" | "linear-mccfr" => Some(Variant::Linear),
        _ => None,
    }
}

impl Variant {
    /// Stable one-byte tag for the resumable checkpoint header.
    fn tag(self) -> u8 {
        match self {
            Variant::Vanilla => 0,
            Variant::Plus => 1,
            Variant::Linear => 2,
        }
    }

    /// The Python-side name [`variant_by_name`] resolves to this variant.
    pub fn name(self) -> &'static str {
        match self {
            Variant::Vanilla => "mccfr",
            Variant::Plus => "mccfr+",
            Variant::Linear => "linear",
        }
    }

    fn from_tag(tag: u8) -> Option<Variant> {
        match tag {
            0 => Some(Variant::Vanilla),
            1 => Some(Variant::Plus),
            2 => Some(Variant::Linear),
            _ => None,
        }
    }
}

/// Per-infoset accumulators. No `player` field is needed: regret is only written
/// when the node's owner is the traverser, the average strategy only when it is
/// the opponent, and the recursion already knows which case it is in.
///
/// One heap block per infoset holds its key AND its values, and the map slot is
/// this one pointer: `[key_len: u16 | n: u16 | key bytes | pad to 8 | regret_sum
/// (n f64) | strategy_sum (n f64)]`. A lookup reads the slot, then one block whose
/// key and values are adjacent, instead of a 32-byte slot and two separate
/// allocations (the key and the values), one cache miss fewer per decision.
/// The node is the map's KEY (`Hash`, `Eq` and `Borrow<[u8]>` all go through the
/// key bytes, hashing exactly as the `Box<[u8]>` key it replaces), so the map's
/// layout and iteration order, and with them the checkpoint bytes, are unchanged.
#[repr(transparent)]
struct Node {
    ptr: NonNull<u8>,
}

// SAFETY: `Node` uniquely owns its block, exactly like a `Box`.
unsafe impl Send for Node {}
unsafe impl Sync for Node {}

impl Node {
    const HEADER: usize = 4;

    #[inline]
    fn values_offset(key_len: usize) -> usize {
        (Self::HEADER + key_len + 7) & !7
    }

    fn layout(key_len: usize, n: usize) -> Layout {
        Layout::from_size_align(Self::values_offset(key_len) + 16 * n, 8).expect("node layout")
    }

    /// A node for `key` with `n` actions and every accumulator 0.0.
    fn new(key: &[u8], n: usize) -> Self {
        assert!(key.len() <= u16::MAX as usize && n <= u16::MAX as usize, "node key or action count too large");
        let layout = Self::layout(key.len(), n);
        // SAFETY: the layout is non-zero-sized (the header alone is 4 bytes); the
        // header and key are written in bounds, and zeroed bytes are 0.0 f64s.
        unsafe {
            let raw = alloc_zeroed(layout);
            let Some(ptr) = NonNull::new(raw) else { handle_alloc_error(layout) };
            raw.cast::<u16>().write(key.len() as u16);
            raw.add(2).cast::<u16>().write(n as u16);
            std::ptr::copy_nonoverlapping(key.as_ptr(), raw.add(Self::HEADER), key.len());
            Node { ptr }
        }
    }

    /// A node for `key` holding `values` = `[regret_sum | strategy_sum]`.
    fn with_values(key: &[u8], values: &[f64]) -> Self {
        let n = values.len() / 2;
        let mut node = Node::new(key, n);
        node.values_mut().copy_from_slice(values);
        node
    }

    #[inline]
    fn key_len(&self) -> usize {
        // SAFETY: the header is in bounds and 8-aligned.
        unsafe { self.ptr.as_ptr().cast::<u16>().read() as usize }
    }

    #[inline]
    fn n(&self) -> usize {
        // SAFETY: as `key_len`.
        unsafe { self.ptr.as_ptr().add(2).cast::<u16>().read() as usize }
    }

    #[inline]
    fn key(&self) -> &[u8] {
        // SAFETY: `key_len` initialized bytes follow the header.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(Self::HEADER), self.key_len()) }
    }

    #[inline]
    fn values_ptr(&self) -> *mut f64 {
        // SAFETY: the values start in bounds, 8-aligned.
        unsafe { self.ptr.as_ptr().add(Self::values_offset(self.key_len())).cast::<f64>() }
    }

    /// `[regret_sum (n) | strategy_sum (n)]`.
    #[inline]
    fn values(&self) -> &[f64] {
        // SAFETY: 2n initialized f64s; shared access through `&self`.
        unsafe { std::slice::from_raw_parts(self.values_ptr(), 2 * self.n()) }
    }

    #[inline]
    fn values_mut(&mut self) -> &mut [f64] {
        // SAFETY: 2n initialized f64s; exclusive access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.values_ptr(), 2 * self.n()) }
    }

    #[inline]
    fn regret_sum(&self) -> &[f64] {
        &self.values()[..self.n()]
    }

    #[inline]
    fn strategy_sum(&self) -> &[f64] {
        &self.values()[self.n()..]
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // SAFETY: allocated in `Node::new` with this same layout.
        unsafe { dealloc(self.ptr.as_ptr(), Self::layout(self.key_len(), self.n())) }
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Node {}

impl Borrow<[u8]> for Node {
    fn borrow(&self) -> &[u8] {
        self.key()
    }
}

/// Write access to one node of a locked shard, for as long as the shard is
/// borrowed mutably (so it is exclusive). The block never moves when the map
/// grows: only the slot holding its pointer does.
struct NodeMut<'a> {
    ptr: NonNull<u8>,
    _shard: PhantomData<&'a mut Shard>,
}

impl NodeMut<'_> {
    #[inline]
    fn node(&self) -> &Node {
        // SAFETY: `Node` is `repr(transparent)` over one `NonNull`, and this
        // pointer came from a live node in the borrowed shard.
        unsafe { &*(&self.ptr as *const NonNull<u8>).cast::<Node>() }
    }

    #[inline]
    fn regret_sum(&self) -> &[f64] {
        self.node().regret_sum()
    }

    #[inline]
    fn split_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        let n = self.node().n();
        let p = self.node().values_ptr();
        // SAFETY: 2n initialized f64s, exclusively ours while the shard is borrowed.
        unsafe { std::slice::from_raw_parts_mut(p, 2 * n) }.split_at_mut(n)
    }

    #[inline]
    fn regret_sum_mut(&mut self) -> &mut [f64] {
        self.split_mut().0
    }

    #[inline]
    fn strategy_sum_mut(&mut self) -> &mut [f64] {
        self.split_mut().1
    }
}

/// One lock-stripe of the [`NodeTable`]: the nodes keyed by their own key bytes.
#[derive(Default)]
struct Shard {
    map: FxHashMap<Node, ()>,
}

impl Shard {
    #[inline]
    fn get(&self, key: &[u8]) -> Option<&Node> {
        self.map.get_key_value(key).map(|(node, _)| node)
    }

    #[inline]
    fn get_mut(&mut self, key: &[u8]) -> Option<NodeMut<'_>> {
        let ptr = self.get(key)?.ptr;
        Some(NodeMut { ptr, _shard: PhantomData })
    }

    /// The node for `key`, inserted with `n` zeroed actions on a first visit (one
    /// lookup on a hit; a miss then inserts, as `get_mut` + `insert` did).
    #[inline]
    fn get_or_insert(&mut self, key: &[u8], n: usize) -> NodeMut<'_> {
        let ptr = match self.get(key) {
            Some(node) => node.ptr,
            None => {
                let node = Node::new(key, n);
                let ptr = node.ptr;
                self.map.insert(node, ());
                ptr
            }
        };
        NodeMut { ptr, _shard: PhantomData }
    }

    /// Insert a node; false (and the new node dropped) if its key is present.
    fn insert(&mut self, node: Node) -> bool {
        self.map.insert(node, ()).is_none()
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn iter(&self) -> std::collections::hash_map::Keys<'_, Node, ()> {
        self.map.keys()
    }
}

/// Deterministic SplitMix64 — the same generator the equity sampler uses, so the
/// whole engine shares one well-understood, reproducible stream.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` from the top 53 bits (the exact f64 mantissa width).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Positive-regret matching into a caller-provided slice `out` (length == regret's);
/// uniform when no action has positive regret. Writing into a stack buffer instead
/// of returning a `Vec` keeps the hot loop allocation-free. Byte-for-byte the
/// tabular solver's rule, so both engines define the current strategy identically.
fn regret_matching_into(regret_sum: &[f64], out: &mut [f64]) {
    let n = regret_sum.len();
    let mut total = 0.0;
    for i in 0..n {
        let p = if regret_sum[i] > 0.0 {
            regret_sum[i]
        } else {
            0.0
        };
        out[i] = p;
        total += p;
    }
    if total > 0.0 {
        for x in out.iter_mut().take(n) {
            *x /= total;
        }
    } else {
        let u = 1.0 / n as f64;
        for x in out.iter_mut().take(n) {
            *x = u;
        }
    }
}

/// Sample an index from a weight vector that sums to ~1 with one uniform draw. The
/// final index is returned if a floating-point shortfall leaves `r` past the end.
fn sample_index(weights: &[f64], rng: &mut Rng) -> usize {
    let r = rng.next_f64();
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if r < acc {
            return i;
        }
    }
    weights.len() - 1
}

/// Magic + version tag at the head of a resumable checkpoint (see
/// [`MCCFRSolver::save_state`]). Bump the trailing digit on any format change.
/// Version 2 adds a tag (the game's bet grid) right after the magic. Version 3
/// adds the abstraction's fingerprint after the tag and a trailer: the [`Fnv64`]
/// of every byte before it. Versions 1 and 2 are still read (without either check).
const CKPT_MAGIC: &[u8; 8] = b"PCVRCKP1";
const CKPT_MAGIC_TAGGED: &[u8; 8] = b"PCVRCKP2";
const CKPT_MAGIC_V3: &[u8; 8] = b"PCVRCKP3";

/// FNV-1a, 64-bit: a checkpoint's checksum and the abstraction's fingerprint.
/// Fixed by its definition (unlike `std`'s hashers), so a file written by one
/// build is checked by any other.
#[derive(Clone, Copy)]
pub(crate) struct Fnv64(u64);

impl Fnv64 {
    pub(crate) const fn new() -> Self {
        Fnv64(0xcbf2_9ce4_8422_2325)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub(crate) fn value(self) -> u64 {
        self.0
    }
}

/// A reader or writer that hashes every byte passing through it while `hash`
/// is set (version 3 only: the older versions carry no checksum).
struct Hashing<T> {
    inner: T,
    hash: Option<Fnv64>,
}

impl<T> Hashing<T> {
    fn new(inner: T, hashing: bool) -> Self {
        Hashing { inner, hash: hashing.then(Fnv64::new) }
    }
}

impl<W: Write> Write for Hashing<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        if let Some(hash) = &mut self.hash {
            hash.update(&buf[..n]);
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl<R: Read> Read for Hashing<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if let Some(hash) = &mut self.hash {
            hash.update(&buf[..n]);
        }
        Ok(n)
    }
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Magic at the head of a TAGGED average-strategy blueprint (the tag names the
/// bet grid it was trained on). Untagged blueprints start directly with the row
/// count, which can never equal this value (the loader bounds the count by the
/// file size).
const BP_MAGIC_TAGGED: &[u8; 8] = b"PCVRBP02";

fn write_tag(w: &mut impl Write, tag: &str) -> std::io::Result<()> {
    let b = tag.as_bytes();
    if b.len() > u8::MAX as usize || !tag.is_ascii() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "file tag must be <= 255 ASCII bytes"));
    }
    w.write_all(&[b.len() as u8])?;
    w.write_all(b)
}

fn read_tag(r: &mut impl Read) -> std::io::Result<String> {
    let mut n = [0u8; 1];
    r.read_exact(&mut n)?;
    let mut b = vec![0u8; n[0] as usize];
    r.read_exact(&mut b)?;
    String::from_utf8(b)
        .ok()
        .filter(|t| t.is_ascii())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "file tag is not ASCII"))
}

/// Number of lock-striped shards in the regret table. A power of two, so the shard
/// index is a mask of the key hash. Chosen well above the hardware thread count
/// (16) so concurrent traversals — which touch mostly-disjoint infosets — rarely
/// contend on the same shard's lock; empty shards cost almost nothing.
const SHARDS: usize = 1024;

/// The regret / average-strategy table, lock-striped into [`SHARDS`] independent
/// maps so many external-sampling traversals update it CONCURRENTLY with low
/// contention. Two design choices matter:
///   * **Interned byte keys.** Keys are the infoset-key bytes stored once per
///     infoset, inside the node's own block (see [`Node`]). The hot loop builds the key into a reused scratch
///     buffer and looks the node up by slice, so a repeat visit neither allocates a
///     `String` nor clones the key — the dominant per-node cost the Phase-2 note
///     flagged, and (crucially) it removes the per-visit allocator traffic that
///     would otherwise serialize threads on the global allocator lock.
///   * **Additive updates under a per-shard `Mutex`.** Regret/strategy increments
///     commute, so applying them in any thread order yields the same sums up to
///     floating-point associativity; a traversal holds a shard lock only to read a
///     snapshot or add increments, never across the recursive descent.
struct NodeTable {
    shards: Box<[Mutex<Shard>]>,
}

impl NodeTable {
    fn new() -> Self {
        let shards = (0..SHARDS)
            .map(|_| Mutex::new(Shard::default()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        NodeTable { shards }
    }

    #[inline]
    fn shard_index(key: &[u8]) -> usize {
        let mut h = FxHasher::default();
        h.write(key);
        (h.finish() as usize) & (SHARDS - 1)
    }

    #[inline]
    fn shard(&self, key: &[u8]) -> &Mutex<Shard> {
        &self.shards[Self::shard_index(key)]
    }

    /// Total interned infosets (locks each shard once; off the hot loop).
    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap().len()).sum()
    }

    /// Visit every `(key, node)`, one shard at a time. Iteration ORDER differs from
    /// a single map, which only perturbs whole-table float reductions in their last
    /// bits (no gate depends on that); every per-node value is order-independent.
    fn for_each<F: FnMut(&[u8], &Node)>(&self, mut f: F) {
        for s in self.shards.iter() {
            for node in s.lock().unwrap().iter() {
                f(node.key(), node);
            }
        }
    }

    /// `f` over each shard's `(key, node)` pairs, shards in parallel on the current rayon
    /// pool, results in SHARD order. Reducing that Vec sequentially gives the same answer
    /// on any thread count (unlike a parallel float reduce, whose split order varies).
    fn map_shards<T: Send, F: Fn(&mut dyn Iterator<Item = (&[u8], &Node)>) -> T + Sync>(
        &self,
        f: F,
    ) -> Vec<T> {
        self.shards
            .par_iter()
            .map(|s| {
                f(&mut s.lock().unwrap().iter().map(|n| (n.key(), n)))
            })
            .collect()
    }

    /// Like [`NodeTable::for_each`] but the visitor may fail (for streaming a node
    /// at a time to disk); the first error short-circuits.
    fn try_for_each<F: FnMut(&[u8], &Node) -> std::io::Result<()>>(
        &self,
        mut f: F,
    ) -> std::io::Result<()> {
        for s in self.shards.iter() {
            for node in s.lock().unwrap().iter() {
                f(node.key(), node)?;
            }
        }
        Ok(())
    }
}

/// Well-spread per-iteration RNG seed for the parallel path: a function of the
/// immutable base `seed` and the GLOBAL iteration index `g`, so (a) each concurrent
/// traversal draws an independent stream and (b) the seed schedule is identical
/// however the run is chunked — a chunked parallel solve samples the same per-
/// iteration streams as one long call (the parallel analog of the serial path's
/// persisted continuous stream).
#[inline]
fn mix_seed(seed: u64, g: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(g.wrapping_mul(0xD1B5_4A32_D192_ED03))
}

pub struct MCCFRSolver {
    variant: Variant,
    table: NodeTable,
    /// Immutable base seed; the parallel path derives each iteration's stream from
    /// it via [`mix_seed`], and it is stamped into checkpoints so a resumed solve
    /// continues the same schedule.
    seed: u64,
    /// The SERIAL sampling stream's live state, PERSISTED across `run` calls so a
    /// single-thread solve driven in chunks draws one continuous stream — identical
    /// to a single `run` of the same total. Initialized to the seed, so a single
    /// serial `run` starts from `seed` and every existing single-call gate is
    /// byte-for-byte unchanged. (Unused by the parallel path, which reseeds per
    /// iteration; carried through checkpoints for completeness.)
    rng_state: u64,
    t: u64,
    /// Worker threads for [`MCCFRSolver::run`]: `1` (default) selects the exact
    /// serial code path — one continuous RNG, deterministic, byte-identical to the
    /// pre-parallel engine — used by every existing correctness gate; `> 1` fans
    /// traversals over a private rayon pool.
    num_threads: usize,
    /// Lazily built rayon pool sized to `num_threads`, reused across `run` calls and
    /// rebuilt only if the thread count changes. `None` while serial.
    pool: Option<rayon::ThreadPool>,
}

/// Fixed-opponent action source for a sampled response solve. Implementations
/// receive the real game state as well as its infoset key, which lets the later
/// S3.2 public model condition on observable board/history while still denying it
/// private cards. The output is normalized by the solver before sampling.
pub trait FixedPolicy<G: Game>: Sync {
    fn action_probabilities(&self, game: &G, state: &G::State, infoset_key: &[u8], out: &mut [f64]);
}

/// A conventional infoset-keyed fixed policy. Missing or malformed rows fall
/// back to uniform, matching every existing engine evaluator's policy convention.
impl<G: Game> FixedPolicy<G> for HashMap<String, Vec<f64>> {
    fn action_probabilities(
        &self,
        _game: &G,
        _state: &G::State,
        infoset_key: &[u8],
        out: &mut [f64],
    ) {
        let row = std::str::from_utf8(infoset_key)
            .ok()
            .and_then(|key| self.get(key));
        let valid = row.is_some_and(|p| {
            p.len() == out.len()
                && p.iter().all(|x| x.is_finite() && *x >= 0.0)
                && p.iter().sum::<f64>() > 0.0
        });
        if valid {
            let row = row.unwrap();
            let total: f64 = row.iter().sum();
            for (dst, src) in out.iter_mut().zip(row) {
                *dst = *src / total;
            }
        } else {
            out.fill(1.0 / out.len() as f64);
        }
    }
}

/// Compact saved-blueprint rows use byte keys and f32 probabilities. The solver
/// reads them directly without expanding a 13M-infoset artifact into Python/f64.
impl<G: Game> FixedPolicy<G> for FxHashMap<Box<[u8]>, Vec<f32>> {
    fn action_probabilities(
        &self,
        _game: &G,
        _state: &G::State,
        infoset_key: &[u8],
        out: &mut [f64],
    ) {
        let row = self.get(infoset_key);
        let valid = row.is_some_and(|p| {
            p.len() == out.len()
                && p.iter().all(|x| x.is_finite() && *x >= 0.0)
                && p.iter().map(|x| *x as f64).sum::<f64>() > 0.0
        });
        if valid {
            let row = row.unwrap();
            let total: f64 = row.iter().map(|x| *x as f64).sum();
            for (dst, src) in out.iter_mut().zip(row) {
                *dst = *src as f64 / total;
            }
        } else {
            out.fill(1.0 / out.len() as f64);
        }
    }
}

/// External-sampling CFR against one immutable opponent. Only `responder` learns;
/// chance and opponent actions are sampled, while every responder action is
/// explored. The responder's average strategy is accumulated with its own reach,
/// because there is no alternating opponent traversal to provide that reach by
/// sampling as in ordinary two-sided ES-MCCFR.
pub struct FixedOpponentMCCFR {
    inner: MCCFRSolver,
    responder: usize,
}

impl FixedOpponentMCCFR {
    pub fn new(variant: Variant, seed: u64, responder: usize) -> Self {
        assert!(
            responder < 2,
            "fixed-opponent solver currently supports HU only"
        );
        Self {
            inner: MCCFRSolver::new(variant, seed),
            responder,
        }
    }

    pub fn set_threads(&mut self, n: usize) {
        self.inner.set_threads(n);
    }

    #[allow(dead_code)]
    pub fn iterations(&self) -> u64 {
        self.inner.iterations()
    }

    pub fn num_infosets(&self) -> usize {
        self.inner.num_infosets()
    }

    fn traverse<G: Game, P: FixedPolicy<G>>(
        table: &NodeTable,
        variant: Variant,
        game: &G,
        fixed: &P,
        state: &G::State,
        responder: usize,
        responder_reach: f64,
        t: u64,
        rng: &mut Rng,
        scratch: &mut Vec<u8>,
    ) -> f64 {
        if game.is_terminal(state) {
            return game.terminal_utility_for(state, responder);
        }
        if game.is_chance(state) {
            let child = game.sample_chance_with(state, |probabilities| {
                let r = rng.next_f64();
                let mut acc = 0.0;
                let mut idx = probabilities.len() - 1;
                for (i, probability) in probabilities.iter().enumerate() {
                    acc += *probability;
                    if r < acc {
                        idx = i;
                        break;
                    }
                }
                idx
            });
            return Self::traverse(
                table,
                variant,
                game,
                fixed,
                &child,
                responder,
                responder_reach,
                t,
                rng,
                scratch,
            );
        }

        let player = game.current_player(state);
        let n = game.num_actions(state);
        debug_assert!(n <= MAX_ACTIONS, "num_actions {n} exceeds MAX_ACTIONS");
        scratch.clear();
        game.infoset_key_into(state, scratch);

        if player != responder {
            let mut probabilities = [0.0f64; MAX_ACTIONS];
            fixed.action_probabilities(game, state, scratch, &mut probabilities[..n]);
            let total: f64 = probabilities[..n].iter().sum();
            if !total.is_finite()
                || total <= 0.0
                || probabilities[..n]
                    .iter()
                    .any(|p| !p.is_finite() || *p < 0.0)
            {
                probabilities[..n].fill(1.0 / n as f64);
            } else {
                for p in &mut probabilities[..n] {
                    *p /= total;
                }
            }
            let action = sample_index(&probabilities[..n], rng);
            let child = game.next_state(state, action);
            return Self::traverse(
                table,
                variant,
                game,
                fixed,
                &child,
                responder,
                responder_reach,
                t,
                rng,
                scratch,
            );
        }

        // One shard lock and one map lookup (a first visit inserts the node): the
        // current strategy from its regrets, then the responder policy averaged
        // with its realization reach — done before recursion, while `scratch`
        // still holds this infoset key.
        let mut strategy = [0.0f64; MAX_ACTIONS];
        {
            let mut map = table.shard(scratch).lock().unwrap();
            let mut node = map.get_or_insert(scratch, n);
            debug_assert_eq!(node.regret_sum().len(), n);
            regret_matching_into(node.regret_sum(), &mut strategy[..n]);
            let sw = variant.strategy_weight(t) * responder_reach;
            for (s, &p) in node.strategy_sum_mut().iter_mut().zip(&strategy[..n]) {
                *s += sw * p;
            }
        }

        // The recursion below clobbers `scratch`; keep this node's key on the stack
        // so the regret write-back needn't rebuild it.
        let key_len = scratch.len();
        let mut saved_key = [0u8; KEY_STACK];
        let saved = key_len <= KEY_STACK;
        if saved {
            saved_key[..key_len].copy_from_slice(scratch);
        }

        let mut action_values = [0.0f64; MAX_ACTIONS];
        let mut node_util = 0.0;
        for a in 0..n {
            let child = game.next_state(state, a);
            let value = Self::traverse(
                table,
                variant,
                game,
                fixed,
                &child,
                responder,
                responder_reach * strategy[a],
                t,
                rng,
                scratch,
            );
            action_values[a] = value;
            node_util += strategy[a] * value;
        }

        let rw = variant.regret_increment_weight(t);
        let plus = variant.regret_matching_plus();
        let key: &[u8] = if saved {
            &saved_key[..key_len]
        } else {
            scratch.clear();
            game.infoset_key_into(state, scratch);
            scratch
        };
        let mut map = table.shard(key).lock().unwrap();
        let mut node = map.get_mut(key).unwrap();
        let regret_sum = node.regret_sum_mut();
        for a in 0..n {
            regret_sum[a] += rw * (action_values[a] - node_util);
            if plus && regret_sum[a] < 0.0 {
                regret_sum[a] = 0.0;
            }
        }
        node_util
    }

    pub fn run<G: Game + Sync, P: FixedPolicy<G>>(&mut self, game: &G, fixed: &P, iterations: u64) {
        if iterations == 0 {
            return;
        }
        if self.inner.num_threads <= 1 {
            let root = game.initial_state();
            let mut rng = Rng::new(self.inner.rng_state);
            let mut scratch = Vec::<u8>::with_capacity(64);
            for _ in 0..iterations {
                self.inner.t += 1;
                Self::traverse(
                    &self.inner.table,
                    self.inner.variant,
                    game,
                    fixed,
                    &root,
                    self.responder,
                    1.0,
                    self.inner.t,
                    &mut rng,
                    &mut scratch,
                );
            }
            self.inner.rng_state = rng.state;
        } else {
            self.inner.ensure_pool();
            let variant = self.inner.variant;
            let seed = self.inner.seed;
            let base_t = self.inner.t;
            let responder = self.responder;
            {
                let pool = self.inner.pool.as_ref().unwrap();
                let table = &self.inner.table;
                pool.install(|| {
                    (0..iterations).into_par_iter().for_each_init(
                        || Vec::<u8>::with_capacity(64),
                        |scratch, i| {
                            let t = base_t + i + 1;
                            let mut rng = Rng::new(mix_seed(seed, t));
                            let root = game.initial_state();
                            Self::traverse(
                                table, variant, game, fixed, &root, responder, 1.0, t, &mut rng,
                                scratch,
                            );
                        },
                    );
                });
            }
            self.inner.t += iterations;
        }
    }

    pub fn average_strategy(&self) -> HashMap<String, Vec<f64>> {
        self.inner.average_strategy()
    }

    /// The average strategy restricted to infosets whose estimated reach
    /// probability is at least `min_reach`. A responder infoset is visited with the
    /// opponent's-and-chance's reach probability and each visit adds
    /// `strategy_weight(t) * responder_reach`, so its accumulated weight divided by
    /// `strategy_weight_sum(T)` estimates its full reach probability under
    /// (response, opponent, chance). Omitted (rarely reached, hence noisily
    /// averaged) infosets let the caller fall back to another policy. With
    /// `min_reach <= 0` every infoset is kept: exactly [`Self::average_strategy`].
    pub fn average_strategy_min_reach(&self, min_reach: f64) -> HashMap<Vec<u8>, Vec<f64>> {
        let solver = &self.inner;
        let total_weight = solver.variant.strategy_weight_sum(solver.t);
        let mut out = HashMap::new();
        solver.table.for_each(|kb, node| {
            let total: f64 = node.strategy_sum().iter().sum();
            if min_reach > 0.0 && !(total / total_weight >= min_reach) {
                return;
            }
            let probs = if total > 0.0 {
                node.strategy_sum().iter().map(|x| x / total).collect()
            } else {
                let n = node.strategy_sum().len();
                vec![1.0 / n as f64; n]
            };
            out.insert(kb.to_vec(), probs);
        });
        out
    }

}

impl MCCFRSolver {
    pub fn new(variant: Variant, seed: u64) -> Self {
        MCCFRSolver {
            variant,
            table: NodeTable::new(),
            seed,
            rng_state: seed,
            t: 0,
            num_threads: 1,
            pool: None,
        }
    }

    /// Set the worker-thread count for subsequent `run` calls. `1` keeps the exact
    /// deterministic serial path (the correctness gates); `> 1` parallelizes. Values
    /// are clamped to at least 1.
    pub fn set_threads(&mut self, n: usize) {
        self.num_threads = n.max(1);
    }

    /// The worker-thread count in effect.
    #[allow(dead_code)]
    pub fn threads(&self) -> usize {
        self.num_threads
    }

    /// Infoset count reached so far; a coverage cross-check for the tests.
    #[allow(dead_code)]
    pub fn num_infosets(&self) -> usize {
        self.table.len()
    }

    /// Total iterations run so far (across all `run` calls).
    #[allow(dead_code)]
    pub fn iterations(&self) -> u64 {
        self.t
    }

    /// The CFR weighting this solver runs (a resumed one: its checkpoint's).
    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// A variant-normalized average-regret convergence proxy: the mean over every
    /// infoset action of `|cumulative regret|`, divided by the accumulated regret
    /// weight (`t` for unit-weighted variants, `t(t+1)/2` for Linear's `t`-weighted
    /// increments). This is the sampled analog of average regret `R_T/T`, which the
    /// theory drives to zero — so it TRENDS DOWN as the solve converges, which is
    /// what the dashboard's convergence panel plots when exact exploitability is
    /// intractable (the full blueprint game). Computed off the hot loop, at the
    /// caller's checkpoint cadence.
    /// Scans the table on the solver's threads (it runs after every telemetry chunk, and
    /// a single-thread scan of tens of millions of infosets left the other threads idle
    /// for longer than the chunk itself took).
    #[allow(dead_code)]
    pub fn avg_regret(&self) -> f64 {
        if self.t == 0 {
            return 0.0;
        }
        let parts = self.scan(|it| {
            let (mut sum, mut cnt) = (0.0f64, 0usize);
            for (_key, node) in it {
                for &r in node.regret_sum() {
                    sum += r.abs();
                    cnt += 1;
                }
            }
            (sum, cnt)
        });
        let (sum, cnt) = parts.iter().fold((0.0, 0usize), |(s, c), &(ps, pc)| (s + ps, c + pc));
        if cnt == 0 {
            return 0.0;
        }
        let raw = sum / cnt as f64;
        let weight_sum = match self.variant {
            Variant::Linear => (self.t as f64) * (self.t as f64 + 1.0) / 2.0,
            _ => self.t as f64,
        };
        raw / weight_sum
    }

    /// Infoset counts by betting street (`[P, F, T, R]`), from the key prefix — the
    /// coverage-by-street breakdown the dashboard shows. O(infosets), off the hot
    /// loop.
    #[allow(dead_code)]
    pub fn infosets_by_street(&self) -> [u64; 4] {
        let parts = self.scan(|it| {
            let mut c = [0u64; 4];
            for (key, _node) in it {
                match key.first() {
                    Some(b'P') => c[0] += 1,
                    Some(b'F') => c[1] += 1,
                    Some(b'T') => c[2] += 1,
                    Some(b'R') => c[3] += 1,
                    _ => {}
                }
            }
            c
        });
        parts.iter().fold([0u64; 4], |a, p| [a[0] + p[0], a[1] + p[1], a[2] + p[2], a[3] + p[3]])
    }

    /// [`NodeTable::map_shards`] on this solver's thread pool (serial when it has one
    /// thread), so a table scan uses the threads the solve does.
    fn scan<T: Send, F: Fn(&mut dyn Iterator<Item = (&[u8], &Node)>) -> T + Sync + Send>(&self, f: F) -> Vec<T> {
        match &self.pool {
            Some(pool) if self.num_threads > 1 => pool.install(|| self.table.map_shards(f)),
            _ => self
                .table
                .shards
                .iter()
                .map(|s| f(&mut s.lock().unwrap().iter().map(|n| (n.key(), n))))
                .collect(),
        }
    }

    /// The time-average strategy for ONE infoset key, or `None` if unvisited —
    /// without materializing the whole (millions-of-entries) strategy map. Used to
    /// read the 169 preflop opening nodes for the live range grid each checkpoint.
    #[allow(dead_code)]
    pub fn average_strategy_for(&self, key: &str) -> Option<Vec<f64>> {
        let kb = key.as_bytes();
        self.table.shard(kb).lock().unwrap().get(kb).map(|node| {
            let total: f64 = node.strategy_sum().iter().sum();
            if total > 0.0 {
                node.strategy_sum().iter().map(|x| x / total).collect()
            } else {
                let n = node.strategy_sum().len();
                vec![1.0 / n as f64; n]
            }
        })
    }

    /// Persist the average strategy to a compact binary blueprint file: a `u64`
    /// count, then per infoset `[u16 key-len][key bytes][u8 n][n × f32 avg-prob]`.
    /// This is the trainer's durable output for a long run (and a resumable
    /// artifact); the caller writes it periodically and at the end. O(infosets).
    #[allow(dead_code)]
    pub fn save_average_strategy(&self, path: &Path) -> std::io::Result<()> {
        self.save_average_strategy_tagged(path, None)
    }

    /// [`MCCFRSolver::save_average_strategy`] with `tag` (the bet grid) in a
    /// header, read back by [`read_strategy_tag`]; `None` writes the untagged
    /// original format.
    pub fn save_average_strategy_tagged(&self, path: &Path, tag: Option<&str>) -> std::io::Result<()> {
        crate::atomic_file::save(path, true, |w| {
            if let Some(tag) = tag {
                w.write_all(BP_MAGIC_TAGGED)?;
                write_tag(w, tag)?;
            }
            w.write_all(&(self.table.len() as u64).to_le_bytes())?;
            self.table.try_for_each(|kb, node| {
                w.write_all(&(kb.len() as u16).to_le_bytes())?;
                w.write_all(kb)?;
                let n = node.strategy_sum().len();
                w.write_all(&[n as u8])?;
                let total: f64 = node.strategy_sum().iter().sum();
                for &s in node.strategy_sum() {
                    let p = if total > 0.0 {
                        (s / total) as f32
                    } else {
                        1.0 / n as f32
                    };
                    w.write_all(&p.to_le_bytes())?;
                }
                Ok(())
            })?;
            Ok(())
        })
    }

    /// Persist the FULL solver state to a resumable checkpoint (version 1; see
    /// [`MCCFRSolver::save_state_tagged`] for version 3's additions): the header
    /// (`magic | version-tagged variant | seed | rng_state | t | count`) then, per
    /// infoset, `[u16 key-len][key bytes][u8 n][n × f64 regret_sum][n × f64
    /// strategy_sum]`. Unlike [`MCCFRSolver::save_average_strategy`] (which writes
    /// only the normalized average, enough to PLAY but not to CONTINUE), this keeps
    /// full f64 accumulators so a reload resumes the solve exactly — the artifact
    /// that lets a multi-day 100bb solve survive a restart.
    #[allow(dead_code)]
    pub fn save_state(&self, path: &Path) -> std::io::Result<()> {
        self.save_state_tagged(path, None)
    }

    /// [`MCCFRSolver::save_state`] as version 3: `header` is the tag (the bet
    /// grid) and the abstraction's fingerprint
    /// ([`crate::bucketing::Abstraction::fingerprint`]), read back by
    /// [`MCCFRSolver::load_state_tagged`]; `None` writes version 1.
    pub fn save_state_tagged(&self, path: &Path, header: Option<(&str, u64)>) -> std::io::Result<()> {
        crate::atomic_file::save(path, true, |w| {
            let w = &mut Hashing::new(w, header.is_some());
            match header {
                Some((tag, abstraction)) => {
                    w.write_all(CKPT_MAGIC_V3)?;
                    write_tag(w, tag)?;
                    w.write_all(&abstraction.to_le_bytes())?;
                }
                None => w.write_all(CKPT_MAGIC)?,
            }
            w.write_all(&[self.variant.tag()])?;
            w.write_all(&self.seed.to_le_bytes())?;
            w.write_all(&self.rng_state.to_le_bytes())?;
            w.write_all(&self.t.to_le_bytes())?;
            w.write_all(&(self.table.len() as u64).to_le_bytes())?;
            self.table.try_for_each(|kb, node| {
                w.write_all(&(kb.len() as u16).to_le_bytes())?;
                w.write_all(kb)?;
                let n = node.regret_sum().len();
                w.write_all(&[n as u8])?;
                for &r in node.regret_sum() {
                    w.write_all(&r.to_le_bytes())?;
                }
                for &s in node.strategy_sum() {
                    w.write_all(&s.to_le_bytes())?;
                }
                Ok(())
            })?;
            if let Some(sum) = w.hash {
                w.inner.write_all(&sum.value().to_le_bytes())?;
            }
            Ok(())
        })
    }

    /// Reconstruct a solver from a [`MCCFRSolver::save_state`] checkpoint. The
    /// resumed solver is serial (`num_threads = 1`); the caller re-applies its
    /// desired thread count with [`MCCFRSolver::set_threads`].
    #[allow(dead_code)]
    pub fn load_state(path: &Path) -> std::io::Result<Self> {
        Self::load_state_tagged(path, None).map(|(s, _)| s)
    }

    /// [`MCCFRSolver::load_state`] plus the checkpoint's tag ("" for version 1).
    /// With `abstraction` (the fingerprint of the tables the solve will run
    /// on), refuses a version-3 checkpoint written over other tables; versions
    /// 1 and 2 record none. Refuses a corrupt file: an impossible infoset count,
    /// an empty or non-ASCII key, a node with no actions or more than
    /// `MAX_ACTIONS`, a non-finite accumulator, a negative strategy sum, a
    /// negative regret where the variant floors them at 0 (mccfr+), a duplicate
    /// key, a checksum that does not match (version 3), trailing bytes.
    pub fn load_state_tagged(path: &Path, abstraction: Option<u64>) -> std::io::Result<(Self, String)> {
        use std::io::{Error, ErrorKind};
        let f = std::fs::File::open(path)?;
        let file_len = f.metadata()?.len();
        let mut r = Hashing::new(std::io::BufReader::new(f), true);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        let version = match &magic {
            CKPT_MAGIC => 1,
            CKPT_MAGIC_TAGGED => 2,
            CKPT_MAGIC_V3 => 3,
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "not a parciver MCCFR checkpoint",
                ))
            }
        };
        if version < 3 {
            r.hash = None;
        }
        let tag = if version >= 2 { read_tag(&mut r)? } else { String::new() };
        if version == 3 {
            let written = read_u64(&mut r)?;
            if let Some(want) = abstraction.filter(|&want| want != written) {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "checkpoint was trained on a different abstraction \
                         (fingerprint {written:016x}; these tables: {want:016x}), or the file is corrupt"
                    ),
                ));
            }
        }
        let mut b1 = [0u8; 1];
        r.read_exact(&mut b1)?;
        let variant = Variant::from_tag(b1[0]).ok_or_else(|| {
            Error::new(ErrorKind::InvalidData, "unknown variant tag in checkpoint")
        })?;
        let seed = read_u64(&mut r)?;
        let rng_state = read_u64(&mut r)?;
        let t = read_u64(&mut r)?;
        let count = read_u64(&mut r)?;
        // Magic, the tag (its length byte and bytes), the fingerprint, the
        // variant byte, four u64s; then the checksum after the rows.
        let header = 8 + if version >= 2 { 1 + tag.len() as u64 } else { 0 } + if version == 3 { 8 } else { 0 } + 1 + 32;
        let trailer = if version == 3 { 8 } else { 0 };
        // Every row holds at least a 2-byte key length, one key byte, the action
        // count and one action's regret and strategy sums: 20 bytes.
        let max_rows = file_len.saturating_sub(header + trailer) / 20;
        if count > max_rows {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("checkpoint declares {count} infosets but {file_len} bytes can hold at most {max_rows}"),
            ));
        }
        let invalid = |what: String| Error::new(ErrorKind::InvalidData, what);

        let table = NodeTable::new();
        for _ in 0..count {
            let mut lb = [0u8; 2];
            r.read_exact(&mut lb)?;
            let klen = u16::from_le_bytes(lb) as usize;
            if klen == 0 {
                return Err(invalid("checkpoint contains an empty infoset key".into()));
            }
            let mut kb = vec![0u8; klen];
            r.read_exact(&mut kb)?;
            if !kb.is_ascii() {
                return Err(invalid("checkpoint infoset key is not ASCII".into()));
            }
            let mut nb = [0u8; 1];
            r.read_exact(&mut nb)?;
            let n = nb[0] as usize;
            if n == 0 {
                return Err(invalid(format!("checkpoint has a zero-action node at {}", String::from_utf8_lossy(&kb))));
            }
            // The solver's per-node buffers hold MAX_ACTIONS: a larger row would
            // panic at its first visit, hours into a resumed run.
            if n > MAX_ACTIONS {
                return Err(invalid(format!(
                    "checkpoint has a node with {n} actions (at most {MAX_ACTIONS}) at {}",
                    String::from_utf8_lossy(&kb)
                )));
            }
            // File order is n regrets then n strategy sums — the in-memory layout.
            let mut values = vec![0.0f64; 2 * n];
            for x in values.iter_mut() {
                let mut fb = [0u8; 8];
                r.read_exact(&mut fb)?;
                *x = f64::from_le_bytes(fb);
            }
            // Regret sums are any finite reals (mccfr+ floors them at 0 after
            // every update); a strategy sum adds non-negative weights times
            // probabilities.
            let what = if values.iter().any(|x| !x.is_finite()) {
                Some("a non-finite accumulator")
            } else if values[n..].iter().any(|s| *s < 0.0) {
                Some("a negative strategy sum")
            } else if variant.regret_matching_plus() && values[..n].iter().any(|r| *r < 0.0) {
                Some("a negative regret (mccfr+ floors them at 0)")
            } else {
                None
            };
            if let Some(what) = what {
                return Err(invalid(format!("checkpoint has {what} at {}", String::from_utf8_lossy(&kb))));
            }
            if !table.shard(&kb).lock().unwrap().insert(Node::with_values(&kb, &values)) {
                return Err(invalid("checkpoint contains a duplicate infoset key".into()));
            }
        }
        if let Some(sum) = r.hash {
            if read_u64(&mut r.inner)? != sum.value() {
                return Err(invalid("checkpoint checksum does not match its contents (the file is corrupt)".into()));
            }
        }
        let mut trailing = [0u8; 1];
        if r.inner.read(&mut trailing)? != 0 {
            return Err(invalid("checkpoint contains trailing bytes".into()));
        }
        Ok((
            MCCFRSolver {
                variant,
                table,
                seed,
                rng_state,
                t,
                num_threads: 1,
                pool: None,
            },
            tag,
        ))
    }

    /// One external-sampling traversal for `traverser`. Returns the sampled value
    /// of `state` TO `traverser`; utilities are player-0-view, so player 1 negates.
    ///
    /// An associated function over a SHARED `&NodeTable` (interior mutability via
    /// the per-shard locks), not `&mut self` — this is what lets many traversals
    /// run concurrently. Reads and writes to a node each take the shard lock
    /// briefly and NEVER hold it across the recursive descent, so a child that maps
    /// to the same shard cannot deadlock. `scratch` is a per-thread byte buffer the
    /// key is rebuilt into (no per-visit `String`); the strategy / snapshot /
    /// action-value buffers are stack arrays (no per-visit heap traffic).
    fn traverse<G: Game>(
        table: &NodeTable,
        variant: Variant,
        game: &G,
        state: &G::State,
        traverser: usize,
        t: u64,
        rng: &mut Rng,
        scratch: &mut Vec<u8>,
    ) -> f64 {
        if game.is_terminal(state) {
            let u = game.terminal_utility(state);
            return if traverser == 0 { u } else { -u };
        }
        if game.is_chance(state) {
            // Sample one outcome in proportion to its probability, building only the
            // chosen child (`sample_chance_with`). One rng draw, exactly as the
            // serial path's `sample_index` over the outcome probabilities.
            let child = game.sample_chance_with(state, |probabilities| {
                let r = rng.next_f64();
                let mut acc = 0.0;
                let mut idx = probabilities.len() - 1;
                for (i, p) in probabilities.iter().enumerate() {
                    acc += *p;
                    if r < acc {
                        idx = i;
                        break;
                    }
                }
                idx
            });
            return Self::traverse(
                table,
                variant,
                game,
                &child,
                traverser,
                t,
                rng,
                scratch,
            );
        }

        let player = game.current_player(state);
        // One legal-action pass gives both the action count and every child built below.
        game.with_children(state, |n, child_of| {
            debug_assert!(n <= MAX_ACTIONS, "num_actions {n} exceeds MAX_ACTIONS");

            // Build the infoset key into the reused scratch buffer (no allocation on a
            // repeat visit), then, under ONE shard lock and one map lookup (a first
            // visit inserts the node): take the current strategy from its regrets and,
            // at an opponent node, accumulate the average strategy — visits already
            // occur in proportion to reach. Released before recursing.
            scratch.clear();
            game.infoset_key_into(state, scratch);
            let mut strategy = [0.0f64; MAX_ACTIONS];
            {
                let mut map = table.shard(scratch).lock().unwrap();
                let mut node = map.get_or_insert(scratch, n);
                debug_assert_eq!(node.regret_sum().len(), n);
                regret_matching_into(node.regret_sum(), &mut strategy[..n]);
                if player != traverser {
                    let sw = variant.strategy_weight(t);
                    for (s, &p) in node.strategy_sum_mut().iter_mut().zip(&strategy[..n]) {
                        *s += sw * p;
                    }
                }
            }

            if player == traverser {
                // Explore every action; accumulate regret from the sampled subtrees.
                // The recursion clobbers `scratch`; keep this node's key on the stack
                // so the write-back needn't rebuild it.
                let key_len = scratch.len();
                let mut saved_key = [0u8; KEY_STACK];
                let saved = key_len <= KEY_STACK;
                if saved {
                    saved_key[..key_len].copy_from_slice(scratch);
                }
                let mut action_values = [0.0f64; MAX_ACTIONS];
                let mut node_util = 0.0;
                for a in 0..n {
                    let child = child_of(a);
                    let v = Self::traverse(table, variant, game, &child, traverser, t, rng, scratch);
                    action_values[a] = v;
                    node_util += strategy[a] * v;
                }
                let rw = variant.regret_increment_weight(t);
                let plus = variant.regret_matching_plus();
                let key: &[u8] = if saved {
                    &saved_key[..key_len]
                } else {
                    scratch.clear();
                    game.infoset_key_into(state, scratch);
                    scratch
                };
                let mut map = table.shard(key).lock().unwrap();
                let mut node = map.get_mut(key).unwrap();
                let regret_sum = node.regret_sum_mut();
                for a in 0..n {
                    regret_sum[a] += rw * (action_values[a] - node_util);
                    if plus && regret_sum[a] < 0.0 {
                        regret_sum[a] = 0.0;
                    }
                }
                node_util
            } else {
                // Opponent (average strategy already accumulated above): sample a single
                // action to descend.
                let a = sample_index(&strategy[..n], rng);
                let child = child_of(a);
                Self::traverse(table, variant, game, &child, traverser, t, rng, scratch)
            }
        })
    }

    /// Run `iterations` full iterations; each is one external-sampling traversal per
    /// player (the alternating traverser is intrinsic to external sampling). Uses
    /// [`MCCFRSolver::set_threads`]: `1` is the exact deterministic serial loop;
    /// `> 1` fans iterations over the private rayon pool.
    pub fn run<G: Game + Sync>(&mut self, game: &G, iterations: u64) {
        if iterations == 0 {
            return;
        }
        if self.num_threads <= 1 {
            self.run_serial(game, iterations);
        } else {
            self.run_parallel(game, iterations);
        }
    }

    /// The single-thread path: one continuous SplitMix64 stream resumed from
    /// `rng_state`, iterations in order — byte-for-byte the pre-parallel engine
    /// (only the node MAP type changed, which does not touch any per-node value), so
    /// every existing serial gate (Kuhn value, Leduc/small-game convergence,
    /// seed-determinism) is unchanged.
    fn run_serial<G: Game>(&mut self, game: &G, iterations: u64) {
        let variant = self.variant;
        let root = game.initial_state();
        let mut rng = Rng::new(self.rng_state);
        let mut scratch: Vec<u8> = Vec::with_capacity(64);
        let mut t = self.t;
        for _ in 0..iterations {
            t += 1;
            Self::traverse(
                &self.table,
                variant,
                game,
                &root,
                0,
                t,
                &mut rng,
                &mut scratch,
            );
            Self::traverse(
                &self.table,
                variant,
                game,
                &root,
                1,
                t,
                &mut rng,
                &mut scratch,
            );
        }
        self.t = t;
        self.rng_state = rng.state;
    }

    /// The parallel path: fan the batch's iterations over `num_threads` rayon
    /// workers, each traversal drawing an independent stream seeded from the global
    /// iteration index ([`mix_seed`]). Increments land in the shared lock-striped
    /// table and commute, so this converges to the same equilibrium as the serial
    /// path (certified, not assumed, by the same Python best-response oracle); it is
    /// not bit-reproducible because concurrent float additions reorder.
    fn run_parallel<G: Game + Sync>(&mut self, game: &G, iterations: u64) {
        self.ensure_pool();
        let variant = self.variant;
        let seed = self.seed;
        let base_t = self.t;
        {
            let pool = self.pool.as_ref().unwrap();
            let table = &self.table;
            pool.install(|| {
                (0..iterations).into_par_iter().for_each_init(
                    || Vec::<u8>::with_capacity(64),
                    |scratch, i| {
                        let g = base_t + i + 1;
                        let mut rng = Rng::new(mix_seed(seed, g));
                        let root = game.initial_state();
                        Self::traverse(table, variant, game, &root, 0, g, &mut rng, scratch);
                        Self::traverse(table, variant, game, &root, 1, g, &mut rng, scratch);
                    },
                );
            });
        }
        self.t += iterations;
    }

    /// Build the rayon pool sized to `num_threads`, reusing an existing one of the
    /// right size. A private pool (not the global one) keeps the thread count under
    /// this solver's control — needed to measure speedup at 1/2/4/8/16 threads.
    fn ensure_pool(&mut self) {
        let n = self.num_threads;
        let ok = self
            .pool
            .as_ref()
            .map(|p| p.current_num_threads() == n)
            .unwrap_or(false);
        if !ok {
            self.pool = Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("mccfr-{i}"))
                    .build()
                    .expect("failed to build rayon thread pool for MCCFR"),
            );
        }
    }

    /// The converged object: the time-AVERAGE strategy keyed by infoset — never
    /// the current strategy (the MCCFR guarantee, like CFR's, is on the average).
    pub fn average_strategy(&self) -> HashMap<String, Vec<f64>> {
        let mut out = HashMap::with_capacity(self.table.len());
        self.table.for_each(|kb, node| {
            let total: f64 = node.strategy_sum().iter().sum();
            let probs = if total > 0.0 {
                node.strategy_sum().iter().map(|x| x / total).collect()
            } else {
                let n = node.strategy_sum().len();
                vec![1.0 / n as f64; n]
            };
            // Keys are ASCII by construction; this is the exact inverse of the
            // interned byte key.
            let key = String::from_utf8(kb.to_vec()).expect("infoset key is valid UTF-8");
            out.insert(key, probs);
        });
        out
    }
}

/// Load a saved average-strategy blueprint (written by
/// [`MCCFRSolver::save_average_strategy`]) into a map from infoset-key BYTES to the
/// per-action average probabilities. Read-only and solver-independent — the
/// blueprint analysis (`crate::analysis`) queries this by the same key bytes the
/// game produces, so it needs no live solver and never touches a running one.
pub(crate) fn load_average_strategy(
    path: &Path,
) -> std::io::Result<FxHashMap<Box<[u8]>, Vec<f32>>> {
    let f = std::fs::File::open(path)?;
    let file_len = f.metadata()?.len();
    let mut r = std::io::BufReader::new(f);
    let mut u64b = [0u8; 8];
    r.read_exact(&mut u64b)?;
    let mut header = 8;
    if &u64b == BP_MAGIC_TAGGED {
        header += 1 + read_tag(&mut r)?.len() as u64 + 8;
        r.read_exact(&mut u64b)?;
    }
    let count = u64::from_le_bytes(u64b);
    // Every row consumes at least 2 key-length bytes, one non-empty key byte,
    // one action-count byte and one f32. Reject an impossible header before it
    // can request a corrupt-artifact-sized allocation.
    let max_rows = file_len.saturating_sub(header) / 8;
    if count > max_rows {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "blueprint declares {count} rows but {file_len} bytes can hold at most {max_rows}"
            ),
        ));
    }
    let mut out: FxHashMap<Box<[u8]>, Vec<f32>> =
        FxHashMap::with_capacity_and_hasher(count as usize, Default::default());
    for _ in 0..count {
        let mut lb = [0u8; 2];
        r.read_exact(&mut lb)?;
        let klen = u16::from_le_bytes(lb) as usize;
        if klen == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "blueprint contains an empty infoset key",
            ));
        }
        let mut kb = vec![0u8; klen];
        r.read_exact(&mut kb)?;
        if !kb.is_ascii() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "blueprint infoset key is not ASCII",
            ));
        }
        let mut nb = [0u8; 1];
        r.read_exact(&mut nb)?;
        let n = nb[0] as usize;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "blueprint contains a zero-action strategy row",
            ));
        }
        let mut probs = vec![0.0f32; n];
        for x in probs.iter_mut() {
            let mut fb = [0u8; 4];
            r.read_exact(&mut fb)?;
            *x = f32::from_le_bytes(fb);
        }
        let total: f32 = probs.iter().sum();
        if probs.iter().any(|p| !p.is_finite() || *p < 0.0)
            || !total.is_finite()
            || (total - 1.0).abs() > 1e-3
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "blueprint has an invalid probability row at {}",
                    String::from_utf8_lossy(&kb)
                ),
            ));
        }
        if out.insert(kb.into_boxed_slice(), probs).is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "blueprint contains a duplicate infoset key",
            ));
        }
    }
    let mut trailing = [0u8; 1];
    if r.read(&mut trailing)? != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "blueprint contains trailing bytes",
        ));
    }
    Ok(out)
}

/// The tag of a blueprint written by [`MCCFRSolver::save_average_strategy_tagged`]
/// ("" for an untagged one).
pub(crate) fn read_strategy_tag(path: &Path) -> std::io::Result<String> {
    let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic == BP_MAGIC_TAGGED {
        read_tag(&mut r)
    } else {
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exploit::{best_response_value, on_policy_value};
    use crate::game::count_infosets;
    use crate::kuhn::{Kuhn, GAME_VALUE};
    use std::collections::HashMap;

    /// On-policy value of an average strategy via a small exact tree walk (the
    /// same pattern `kuhn.rs` uses to check the tabular solver).
    fn kuhn_value(strat: &HashMap<String, Vec<f64>>) -> f64 {
        use crate::kuhn::KuhnState;
        fn walk(game: &Kuhn, s: &KuhnState, strat: &HashMap<String, Vec<f64>>) -> f64 {
            if game.is_terminal(s) {
                return game.terminal_utility(s);
            }
            if game.is_chance(s) {
                return game
                    .chance_outcomes(s)
                    .into_iter()
                    .map(|(c, p)| p * walk(game, &c, strat))
                    .sum();
            }
            let probs = &strat[&game.infoset_key(s)];
            (0..game.num_actions(s))
                .map(|i| probs[i] * walk(game, &game.next_state(s, i), strat))
                .sum()
        }
        walk(&Kuhn::new(), &Kuhn::new().initial_state(), strat)
    }

    #[test]
    fn linear_mccfr_approaches_kuhn_value() {
        // External sampling is higher-variance than full CFR, so this is a looser
        // gate than the tabular test; the Python oracle supplies the strict
        // exploitability check. Value near -1/18 is necessary for equilibrium.
        let mut solver = MCCFRSolver::new(Variant::Linear, 1);
        solver.run(&Kuhn::new(), 300_000);
        let v = kuhn_value(&solver.average_strategy());
        assert!((v - GAME_VALUE).abs() < 8e-3, "value {v} not near -1/18");
    }

    #[test]
    fn is_deterministic_for_a_fixed_seed() {
        let run = || {
            let mut s = MCCFRSolver::new(Variant::Plus, 7);
            s.run(&Kuhn::new(), 20_000);
            s.average_strategy()
        };
        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len());
        for (k, va) in &a {
            let vb = &b[k];
            assert_eq!(va, vb, "strategy for {k} differs run-to-run");
        }
    }

    #[test]
    fn covers_every_kuhn_infoset() {
        // With positive-probability sampling everywhere, enough iterations visit
        // the full infoset partition (the structural gate is count_infosets).
        let mut s = MCCFRSolver::new(Variant::Vanilla, 3);
        s.run(&Kuhn::new(), 20_000);
        assert_eq!(s.num_infosets(), count_infosets(&Kuhn::new()));
    }

    #[test]
    fn parallel_run_converges_to_kuhn_value() {
        // The multi-threaded path reaches the same equilibrium value as the serial
        // path (the whole point of parallelizing). Not bit-identical — concurrent
        // float additions reorder — so this checks convergence, not reproducibility;
        // the Python oracle supplies the strict exploitability gate. Uses more
        // threads than cores would ever matter for a tiny game; correctness is what
        // is under test, not speed.
        let mut solver = MCCFRSolver::new(Variant::Linear, 1);
        solver.set_threads(4);
        solver.run(&Kuhn::new(), 400_000);
        assert!(solver.num_infosets() == count_infosets(&Kuhn::new()));
        let v = kuhn_value(&solver.average_strategy());
        assert!(
            (v - GAME_VALUE).abs() < 1e-2,
            "parallel value {v} not near -1/18"
        );
    }

    #[test]
    fn checkpoint_roundtrip_preserves_state() {
        // A full-state checkpoint reloads to a byte-identical solver: same iteration
        // count and same average strategy (f64 accumulators are stored losslessly),
        // so a multi-day solve can stop and resume without drift.
        let mut s = MCCFRSolver::new(Variant::Linear, 42);
        s.run(&Kuhn::new(), 5_000);
        let before = s.average_strategy();
        let iters = s.iterations();

        let path = std::env::temp_dir().join(format!("parciver_ckpt_{}.bin", std::process::id()));
        s.save_state(&path).expect("save_state");
        let loaded = MCCFRSolver::load_state(&path).expect("load_state");

        assert_eq!(loaded.iterations(), iters);
        assert_eq!(loaded.num_infosets(), s.num_infosets());
        let after = loaded.average_strategy();
        assert_eq!(before.len(), after.len());
        for (k, vb) in &before {
            assert_eq!(vb, &after[k], "strategy for {k} changed across checkpoint");
        }

        // And a resumed solver keeps solving: more iterations advance the counter
        // (past where the checkpoint stopped) and stay near the Kuhn value.
        let mut resumed = MCCFRSolver::load_state(&path).expect("load_state again");
        std::fs::remove_file(&path).ok();
        resumed.run(&Kuhn::new(), 100_000);
        assert_eq!(resumed.iterations(), iters + 100_000);
        let v = kuhn_value(&resumed.average_strategy());
        assert!(
            (v - GAME_VALUE).abs() < 1e-2,
            "resumed value {v} not near -1/18"
        );
    }

    #[test]
    fn checkpoint_loader_refuses_corrupt_files_and_keeps_the_variant() {
        // A valid checkpoint loads with its own variant; each single corruption
        // of it is refused as InvalidData, naming what is wrong.
        for v in [Variant::Vanilla, Variant::Plus, Variant::Linear] {
            assert!(variant_by_name(v.name()) == Some(v));
        }
        let mut s = MCCFRSolver::new(Variant::Plus, 11);
        s.run(&Kuhn::new(), 2_000);
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let path = dir.join(format!("pcvr_ck_corrupt_src_{pid}.bin"));
        s.save_state(&path).unwrap();
        let good = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(path.with_extension("bin.previous")).ok();
        let loaded = MCCFRSolver::load_state(&path_with(&dir, pid, "good", &good)).unwrap();
        assert!(loaded.variant() == Variant::Plus && loaded.num_infosets() == s.num_infosets());

        // Untagged header: magic 8, variant 1, seed/rng/t/count 4 x 8 -> rows at 41.
        let (count_at, row) = (33, 41);
        let klen = u16::from_le_bytes([good[row], good[row + 1]]) as usize;
        let n_at = row + 2 + klen;
        let n = good[n_at] as usize;
        let row_end = n_at + 1 + 16 * n;
        let regret0 = n_at + 1;
        let strategy0 = regret0 + 8 * n;
        let set_count = |b: &mut Vec<u8>, c: u64| b[count_at..count_at + 8].copy_from_slice(&c.to_le_bytes());
        let count = u64::from_le_bytes(good[count_at..count_at + 8].try_into().unwrap());

        let mut cases: Vec<(&str, Vec<u8>, &str)> = Vec::new();
        let mut b = good.clone();
        b.push(0xA5);
        cases.push(("trailing", b, "trailing bytes"));
        let mut b = good.clone();
        set_count(&mut b, u64::MAX);
        cases.push(("count", b, "can hold at most"));
        let mut b = good.clone();
        b.extend_from_slice(&good[row..row_end]);
        set_count(&mut b, count + 1);
        cases.push(("duplicate", b, "duplicate infoset key"));
        let mut b = good.clone();
        b[regret0..regret0 + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        cases.push(("nan", b, "non-finite accumulator"));
        let mut b = good.clone();
        b[strategy0..strategy0 + 8].copy_from_slice(&(-1.0f64).to_le_bytes());
        cases.push(("negative", b, "negative strategy sum"));
        // The checkpoint is mccfr+ (Linear's negative regrets load: the round trip test).
        let mut b = good.clone();
        b[regret0..regret0 + 8].copy_from_slice(&(-1.0f64).to_le_bytes());
        cases.push(("plus_regret", b, "negative regret"));
        let mut b = good.clone();
        b[n_at] = 0;
        cases.push(("zero_actions", b, "zero-action node"));
        let mut b = good.clone();
        b[n_at] = (MAX_ACTIONS + 1) as u8;
        cases.push(("too_many_actions", b, "actions (at most"));
        let mut b = good.clone();
        b[row..row + 2].copy_from_slice(&0u16.to_le_bytes());
        cases.push(("empty_key", b, "empty infoset key"));
        let mut b = good.clone();
        b[row + 2] = 0xC3;
        cases.push(("non_ascii", b, "not ASCII"));
        for (name, bytes, want) in cases {
            let err = match MCCFRSolver::load_state(&path_with(&dir, pid, name, &bytes)) {
                Ok(_) => panic!("{name}: a corrupt checkpoint loaded"),
                Err(e) => e,
            };
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{name}: {err}");
            assert!(err.to_string().contains(want), "{name}: {err}");
        }
        // The same negative regret loads where the variant does not floor regrets.
        for v in [Variant::Vanilla, Variant::Linear] {
            let mut b = good.clone();
            b[regret0..regret0 + 8].copy_from_slice(&(-1.0f64).to_le_bytes());
            b[8] = v.tag();
            let p = path_with(&dir, pid, &format!("regret_{}", v.tag()), &b);
            let loaded = MCCFRSolver::load_state(&p).unwrap_or_else(|e| panic!("{}: {e}", v.name()));
            assert!(loaded.variant() == v && loaded.num_infosets() == s.num_infosets(), "{}", v.name());
            std::fs::remove_file(&p).ok();
        }
        for name in ["good", "trailing", "count", "duplicate", "nan", "negative", "plus_regret", "zero_actions", "too_many_actions", "empty_key", "non_ascii"] {
            std::fs::remove_file(dir.join(format!("pcvr_ck_corrupt_{name}_{pid}.bin"))).ok();
        }

        fn path_with(dir: &Path, pid: u32, name: &str, bytes: &[u8]) -> std::path::PathBuf {
            let p = dir.join(format!("pcvr_ck_corrupt_{name}_{pid}.bin"));
            std::fs::write(&p, bytes).unwrap();
            p
        }
    }

    /// The loader's checks on real checkpoints (none may be refused):
    /// `PARCIVER_CKPTS="a.bin;b.bin" cargo test --release --no-default-features --lib
    /// -- --ignored --nocapture real_checkpoints_load`.
    #[test]
    #[ignore]
    fn real_checkpoints_load() {
        let paths = std::env::var("PARCIVER_CKPTS").expect("set PARCIVER_CKPTS to ';'-separated checkpoint paths");
        for p in paths.split(';').filter(|p| !p.is_empty()) {
            let (s, tag) = MCCFRSolver::load_state_tagged(Path::new(p), None).unwrap_or_else(|e| panic!("{p}: {e}"));
            println!(
                "{p}: tag {tag:?}, {}, {} iterations, {} infosets",
                s.variant().name(),
                s.iterations(),
                s.num_infosets()
            );
        }
    }

    #[test]
    fn average_strategy_loader_validates_structure_and_eof() {
        let mut solver = MCCFRSolver::new(Variant::Linear, 9);
        solver.run(&Kuhn::new(), 2_000);
        let path = std::env::temp_dir().join(format!(
            "parciver_average_strategy_{}.bin",
            std::process::id()
        ));
        solver
            .save_average_strategy(&path)
            .expect("save average strategy");
        let loaded = load_average_strategy(&path).expect("valid strategy must load");
        assert_eq!(loaded.len(), solver.num_infosets());

        use std::io::Write as _;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0xA5])
            .unwrap();
        let err = load_average_strategy(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("trailing bytes"));
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(path.with_extension("bin.previous")).ok();

        let bad = std::env::temp_dir().join(format!(
            "parciver_average_strategy_bad_{}.bin",
            std::process::id()
        ));
        std::fs::write(&bad, u64::MAX.to_le_bytes()).unwrap();
        let err = load_average_strategy(&bad).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(bad).ok();
    }

    #[test]
    fn tagged_blueprint_and_checkpoint_round_trip_and_untagged_read_as_empty() {
        let mut solver = MCCFRSolver::new(Variant::Linear, 5);
        solver.run(&Kuhn::new(), 500);
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (bp, bp_tagged) = (dir.join(format!("pcvr_bp_{pid}.bin")), dir.join(format!("pcvr_bp_t_{pid}.bin")));
        solver.save_average_strategy(&bp).unwrap();
        solver.save_average_strategy_tagged(&bp_tagged, Some("200bb")).unwrap();
        assert_eq!(read_strategy_tag(&bp).unwrap(), "");
        assert_eq!(read_strategy_tag(&bp_tagged).unwrap(), "200bb");
        let (a, b) = (load_average_strategy(&bp).unwrap(), load_average_strategy(&bp_tagged).unwrap());
        assert_eq!(a, b);
        assert_eq!(a.len(), solver.num_infosets());
        let (ck, ck_tagged) = (dir.join(format!("pcvr_ck_{pid}.bin")), dir.join(format!("pcvr_ck_t_{pid}.bin")));
        solver.save_state(&ck).unwrap();
        solver.save_state_tagged(&ck_tagged, Some(("200bb", 7))).unwrap();
        let (s1, t1) = MCCFRSolver::load_state_tagged(&ck, None).unwrap();
        let (s2, t2) = MCCFRSolver::load_state_tagged(&ck_tagged, None).unwrap();
        assert_eq!((t1.as_str(), t2.as_str()), ("", "200bb"));
        assert_eq!((s1.iterations(), s1.num_infosets()), (s2.iterations(), s2.num_infosets()));
        assert_eq!(s2.iterations(), solver.iterations());
        assert!(solver.save_average_strategy_tagged(&bp_tagged, Some("é")).is_err());
        for p in [&bp, &bp_tagged, &ck, &ck_tagged] {
            std::fs::remove_file(p).ok();
            std::fs::remove_file(p.with_extension("bin.previous")).ok();
        }
    }

    #[test]
    fn fnv64_matches_the_published_test_vectors() {
        let fnv = |bytes: &[u8]| {
            let mut h = Fnv64::new();
            h.update(bytes);
            h.value()
        };
        assert_eq!(fnv(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn version_3_checkpoint_checks_its_abstraction_and_checksum_and_older_versions_load() {
        // Version 3 is version 1's body under a header naming the grid and the
        // abstraction, then the FNV-1a of every byte before it. A corruption
        // that keeps every value finite and signed right loads from version 1
        // (the gap version 3 closes) and is refused by version 3's checksum.
        let mut s = MCCFRSolver::new(Variant::Plus, 3);
        s.run(&Kuhn::new(), 2_000);
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let file = |name: &str| dir.join(format!("pcvr_ck_v3_{name}_{pid}.bin"));
        s.save_state(&file("v1")).unwrap();
        s.save_state_tagged(&file("v3"), Some(("200bb", 0xABCD))).unwrap();
        let (v1, v3) = (std::fs::read(file("v1")).unwrap(), std::fs::read(file("v3")).unwrap());
        for name in ["v1", "v3"] {
            std::fs::remove_file(file(name)).ok();
            std::fs::remove_file(file(name).with_extension("bin.previous")).ok();
        }
        let mut want = CKPT_MAGIC_V3.to_vec();
        want.push(5);
        want.extend_from_slice(b"200bb");
        want.extend_from_slice(&0xABCDu64.to_le_bytes());
        want.extend_from_slice(&v1[8..]);
        let mut h = Fnv64::new();
        h.update(&want);
        want.extend_from_slice(&h.value().to_le_bytes());
        assert!(v3 == want, "version 3 layout");

        let load = |name: &str, bytes: &[u8], abstraction: Option<u64>| {
            let p = file(name);
            std::fs::write(&p, bytes).unwrap();
            let out = MCCFRSolver::load_state_tagged(&p, abstraction);
            std::fs::remove_file(&p).ok();
            out
        };
        let same = |name: &str, out: std::io::Result<(MCCFRSolver, String)>, tag: &str| {
            let (loaded, got) = out.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(got, tag, "{name}");
            assert!(loaded.variant() == Variant::Plus && loaded.iterations() == s.iterations(), "{name}");
            assert!(loaded.average_strategy() == s.average_strategy(), "{name}");
        };
        same("checked", load("checked", &v3, Some(0xABCD)), "200bb");
        same("unchecked", load("unchecked", &v3, None), "200bb");
        let err = load("other", &v3, Some(0xABCE)).err().expect("a checkpoint of other tables loaded");
        assert!(err.kind() == std::io::ErrorKind::InvalidData && err.to_string().contains("different abstraction"), "{err}");
        // Versions 1 and 2 record no abstraction: they load whatever is asked.
        let mut v2 = CKPT_MAGIC_TAGGED.to_vec();
        v2.push(5);
        v2.extend_from_slice(b"200bb");
        v2.extend_from_slice(&v1[8..]);
        same("v2", load("v2", &v2, Some(0xABCE)), "200bb");
        same("v1", load("v1", &v1, Some(0xABCE)), "");

        // The lowest mantissa byte of the first row's first strategy sum.
        let row = 8 + 1 + 5 + 8 + 1 + 32;
        let klen = u16::from_le_bytes([v3[row], v3[row + 1]]) as usize;
        let n_at = row + 2 + klen;
        let strategy0 = n_at + 1 + 8 * v3[n_at] as usize;
        let flip = |bytes: &[u8], at: usize| {
            let mut b = bytes.to_vec();
            b[at] ^= 1;
            b
        };
        assert!(load("v1_flip", &flip(&v1, strategy0 - (row - 41)), None).is_ok(), "version 1 cannot see it");
        let cases = [
            ("value", flip(&v3, strategy0), "checksum does not match"),
            ("fingerprint", flip(&v3, 14), "checksum does not match"),
            ("trailer", flip(&v3, v3.len() - 1), "checksum does not match"),
            ("trailing", [&v3[..], &[0xA5]].concat(), "trailing bytes"),
        ];
        for (name, bytes, want) in cases {
            let err = load(name, &bytes, None).err().unwrap_or_else(|| panic!("{name}: a corrupt checkpoint loaded"));
            assert!(err.kind() == std::io::ErrorKind::InvalidData && err.to_string().contains(want), "{name}: {err}");
        }
        let err = load("truncated", &v3[..v3.len() - 8], None).err().expect("a checkpoint without its checksum loaded");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof, "{err}");
    }

    #[test]
    fn fixed_opponent_response_is_bounded_by_exact_br() {
        let game = Kuhn::new();
        let mut seed_solver = MCCFRSolver::new(Variant::Vanilla, 4);
        seed_solver.run(&game, 20_000);
        let uniform: HashMap<String, Vec<f64>> = seed_solver
            .average_strategy()
            .into_iter()
            .map(|(key, row)| {
                let n = row.len();
                (key, vec![1.0 / n as f64; n])
            })
            .collect();
        let none = |_: &_| None::<String>;
        let baseline = on_policy_value(&game, &uniform, 0, none);
        let exact = best_response_value(&game, &uniform, 0, &none);

        let mut response = FixedOpponentMCCFR::new(Variant::Linear, 17, 0);
        response.run(&game, &uniform, 200_000);
        assert_eq!(response.iterations(), 200_000);
        assert!(response.num_infosets() > 0);
        let mut profile = uniform.clone();
        profile.extend(response.average_strategy());
        let realized = on_policy_value(&game, &profile, 0, none);
        assert!(
            realized <= exact + 1e-12,
            "sampled response {realized} exceeded BR {exact}"
        );
        let captured = (realized - baseline) / (exact - baseline);
        assert!(
            captured >= 0.95,
            "sampled response captured only {captured:.3} of BR gain"
        );
    }

    #[test]
    fn min_reach_gate_keeps_exactly_the_reached_infosets() {
        let game = Kuhn::new();
        let mut seed_solver = MCCFRSolver::new(Variant::Vanilla, 4);
        seed_solver.run(&game, 20_000);
        let uniform: HashMap<String, Vec<f64>> = seed_solver
            .average_strategy()
            .into_iter()
            .map(|(key, row)| {
                let n = row.len();
                (key, vec![1.0 / n as f64; n])
            })
            .collect();
        let mut response = FixedOpponentMCCFR::new(Variant::Linear, 17, 0);
        response.run(&game, &uniform, 200_000);

        // tau <= 0 is exactly the ungated average strategy.
        let all: HashMap<Vec<u8>, Vec<f64>> = response
            .average_strategy()
            .into_iter()
            .map(|(key, row)| (key.into_bytes(), row))
            .collect();
        assert_eq!(response.average_strategy_min_reach(0.0), all);
        assert_eq!(all.len(), response.num_infosets());

        // Player 0's first decision is reached with the deal's 1/3; every later
        // player-0 infoset needs a check and a uniform opponent bet (<= 1/6). So
        // tau = 0.3 keeps exactly the three opening infosets, each with its
        // ungated strategy.
        let root = game.initial_state();
        let opening: Vec<Vec<u8>> = game
            .chance_outcomes(&root)
            .into_iter()
            .flat_map(|(first, _)| game.chance_outcomes(&first))
            .map(|(dealt, _)| game.infoset_key(&dealt).into_bytes())
            .collect();
        let gated = response.average_strategy_min_reach(0.3);
        assert_eq!(gated.len(), 3, "kept {:?}", gated.keys().collect::<Vec<_>>());
        for key in &opening {
            assert_eq!(gated.get(key), all.get(key));
        }
        assert!(response.average_strategy_min_reach(0.34).is_empty());
    }

    /// FNV-1a over every node (key bytes, then regret and strategy sums as raw
    /// bits), in key order: equal iff the two tables are bit-identical.
    fn table_checksum(solver: &MCCFRSolver) -> u64 {
        let mut rows: Vec<(Vec<u8>, Vec<u64>)> = Vec::new();
        solver.table.for_each(|k, node| rows.push((k.to_vec(), node.values().iter().map(|v| v.to_bits()).collect())));
        rows.sort();
        let mut h = 0xcbf29ce484222325u64;
        let mut eat = |x: u64| h = (h ^ x).wrapping_mul(0x100000001b3);
        for (k, vals) in &rows {
            k.iter().for_each(|&b| eat(b as u64));
            vals.iter().for_each(|&v| eat(v));
        }
        h
    }

    /// Blueprint-training throughput on the real bucketed 200bb game, serial (with
    /// a checksum of the whole table: the serial path is bit-reproducible, so an
    /// optimization must leave it unchanged) and at 16 threads (PARCIVER_MCCFR_THREADS
    /// overrides; 0 skips). Needs the baked tables (PARCIVER_ABSTRACTION, else
    /// data/abstraction):
    ///   cargo test --release --no-default-features --lib -- --ignored --nocapture print_mccfr_speed
    #[test]
    #[ignore]
    fn print_mccfr_speed() {
        use crate::nlhe::{Nlhe, NlheConfig};
        use std::path::PathBuf;
        use std::sync::Arc;
        use std::time::Instant;
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let dir = std::env::var("PARCIVER_ABSTRACTION").map(PathBuf::from).unwrap_or_else(|_| root.join("data/abstraction"));
        if !dir.join("river_buckets.npy").exists() {
            eprintln!("abstraction absent at {dir:?}; skipping");
            return;
        }
        let abs = Arc::new(crate::bucketing::Abstraction::load(&dir).unwrap());
        let game = Nlhe::with_abstraction(NlheConfig::deep_200bb(), abs);
        let mut serial = MCCFRSolver::new(Variant::Linear, 11);
        serial.run(&game, 2_000);
        let t0 = Instant::now();
        serial.run(&game, 20_000);
        let dt = t0.elapsed().as_secs_f64();
        println!("serial: {:.0} it/s, {} infosets, checksum {:016x}", 20_000.0 / dt, serial.num_infosets(), table_checksum(&serial));
        // PARCIVER_MCCFR_THREADS=0 skips the parallel run (e.g. on a busy machine).
        let threads = std::env::var("PARCIVER_MCCFR_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        if threads == 0 {
            return;
        }
        let mut par = MCCFRSolver::new(Variant::Linear, 11);
        par.set_threads(threads);
        par.run(&game, 20_000);
        let t0 = Instant::now();
        par.run(&game, 200_000);
        let dt = t0.elapsed().as_secs_f64();
        println!("{threads} threads: {:.0} it/s, {} infosets", 200_000.0 / dt, par.num_infosets());
    }
}
