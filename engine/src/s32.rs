// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! S3.2 policy and public-observation primitives.
//!
//! This module is deliberately below the protocol/gate layer. It provides only:
//! blueprint-derived behavioral leak transforms; immutable policy composition;
//! public-only action counts; and the locked exact/backoff Dirichlet model. It
//! knows no FIT/HELD labels, thresholds, alpha schedule, or verdict.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::game::Game;
use crate::mccfr::FixedPolicy;
use crate::nlhe::{ActLabel, Nlhe, NlheState};

const MAX_ACTIONS: usize = 10;

type CompactStrategy = FxHashMap<Box<[u8]>, Vec<f32>>;
type Counts = FxHashMap<String, Vec<u64>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeakFamily {
    Nit,
    Station,
    Maniac,
}

impl LeakFamily {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "nit" => Some(Self::Nit),
            "station" => Some(Self::Station),
            "maniac" => Some(Self::Maniac),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Policy(Arc<PolicyKind>);

enum PolicyKind {
    Tabular(Arc<CompactStrategy>),
    Uniform,
    Leak {
        base: Policy,
        family: LeakFamily,
        strength: f64,
    },
    Blend {
        base: Policy,
        target: Policy,
        alpha: f64,
    },
    PublicModel(Arc<PublicModel>),
    HierarchicalPublicModel(Arc<HierarchicalPublicModel>),
    FiveLevelPublicModel(Arc<FiveLevelPublicModel>),
}

impl Policy {
    pub(crate) fn tabular(strategy: Arc<CompactStrategy>) -> Self {
        Self(Arc::new(PolicyKind::Tabular(strategy)))
    }

    pub(crate) fn uniform() -> Self {
        Self(Arc::new(PolicyKind::Uniform))
    }

    pub(crate) fn leak(&self, family: LeakFamily, strength: f64) -> Self {
        Self(Arc::new(PolicyKind::Leak {
            base: self.clone(),
            family,
            strength,
        }))
    }

    pub(crate) fn blend(&self, target: &Policy, alpha: f64) -> Self {
        Self(Arc::new(PolicyKind::Blend {
            base: self.clone(),
            target: target.clone(),
            alpha,
        }))
    }

    pub(crate) fn public_model(
        train: &Observation,
        reference: &Observation,
        min_count: u64,
        prior_strength: f64,
    ) -> Self {
        Self(Arc::new(PolicyKind::PublicModel(Arc::new(
            PublicModel::fit(train, reference, min_count, prior_strength),
        ))))
    }

    pub(crate) fn score(&self, validation: &Observation) -> Option<ModelScore> {
        match self.0.as_ref() {
            PolicyKind::PublicModel(model) => Some(model.score(validation)),
            _ => None,
        }
    }

    pub(crate) fn hierarchical_public_model(
        train: &Observation,
        reference: &Observation,
        exact_strength: f64,
        backoff_strength: f64,
        context_strength: f64,
    ) -> Self {
        Self(Arc::new(PolicyKind::HierarchicalPublicModel(Arc::new(
            HierarchicalPublicModel::fit(
                train,
                reference,
                exact_strength,
                backoff_strength,
                context_strength,
            ),
        ))))
    }

    pub(crate) fn score_hierarchical(
        &self,
        validation: &Observation,
    ) -> Option<HierarchicalModelScore> {
        match self.0.as_ref() {
            PolicyKind::HierarchicalPublicModel(model) => Some(model.score(validation)),
            _ => None,
        }
    }

    pub(crate) fn five_level_public_model(
        train: &Observation,
        reference: &Observation,
        exact_strength: f64,
        backoff_strength: f64,
        context_strength: f64,
        position_strength: f64,
        global_strength: f64,
    ) -> Self {
        Self(Arc::new(PolicyKind::FiveLevelPublicModel(Arc::new(
            FiveLevelPublicModel::fit(
                train,
                reference,
                exact_strength,
                backoff_strength,
                context_strength,
                position_strength,
                global_strength,
            ),
        ))))
    }

    pub(crate) fn score_five_level(&self, validation: &Observation) -> Option<FiveLevelModelScore> {
        match self.0.as_ref() {
            PolicyKind::FiveLevelPublicModel(model) => Some(model.score(validation)),
            _ => None,
        }
    }

    /// Fill one row and report whether this policy had a row. Sparse response
    /// policies return `false`, allowing a blend to retain its base exactly.
    pub(crate) fn try_fill(&self, game: &Nlhe, state: &NlheState, key: &[u8], out: &mut [f64]) -> bool {
        match self.0.as_ref() {
            PolicyKind::Tabular(strategy) => fill_tabular(strategy, key, out),
            PolicyKind::Uniform => {
                out.fill(1.0 / out.len() as f64);
                true
            }
            PolicyKind::Leak {
                base,
                family,
                strength,
            } => {
                let found = base.try_fill(game, state, key, out);
                if found {
                    apply_leak(out, &game.labeled_actions(state), *family, *strength);
                }
                found
            }
            PolicyKind::Blend {
                base,
                target,
                alpha,
            } => {
                let mut target_row = [0.0f64; MAX_ACTIONS];
                let base_found = base.try_fill(game, state, key, out);
                if !base_found {
                    out.fill(1.0 / out.len() as f64);
                }
                if !target.try_fill(game, state, key, &mut target_row[..out.len()]) {
                    return base_found;
                }
                for (a, b) in out.iter_mut().zip(target_row) {
                    *a = (1.0 - alpha) * *a + alpha * b;
                }
                normalize(out);
                true
            }
            PolicyKind::PublicModel(model) => {
                model.fill(game, state, out);
                true
            }
            PolicyKind::HierarchicalPublicModel(model) => {
                model.fill(game, state, out);
                true
            }
            PolicyKind::FiveLevelPublicModel(model) => {
                model.fill(game, state, out);
                true
            }
        }
    }
}

impl FixedPolicy<Nlhe> for Policy {
    fn action_probabilities(
        &self,
        game: &Nlhe,
        state: &NlheState,
        infoset_key: &[u8],
        out: &mut [f64],
    ) {
        if !self.try_fill(game, state, infoset_key, out) {
            out.fill(1.0 / out.len() as f64);
        }
    }
}

fn fill_tabular(strategy: &CompactStrategy, key: &[u8], out: &mut [f64]) -> bool {
    match strategy.get(key) {
        Some(row) if row.len() == out.len() => {
            for (dst, src) in out.iter_mut().zip(row) {
                *dst = *src as f64;
            }
            normalize(out);
            true
        }
        _ => false,
    }
}

fn normalize(row: &mut [f64]) {
    let total: f64 = row.iter().sum();
    if total.is_finite() && total > 0.0 && row.iter().all(|p| p.is_finite() && *p >= 0.0) {
        for p in row {
            *p /= total;
        }
    } else {
        row.fill(1.0 / row.len() as f64);
    }
}

fn transfer_to(
    row: &mut [f64],
    target: usize,
    sources: impl Iterator<Item = usize>,
    strength: f64,
) {
    let mut moved = 0.0;
    for index in sources {
        if index != target {
            let amount = strength * row[index];
            row[index] -= amount;
            moved += amount;
        }
    }
    row[target] += moved;
}

fn apply_leak(row: &mut [f64], labels: &[ActLabel], family: LeakFamily, strength: f64) {
    match family {
        LeakFamily::Nit => {
            if let Some(target) = labels
                .iter()
                .position(|label| matches!(label, ActLabel::Fold))
            {
                transfer_to(row, target, 0..row.len(), strength);
            }
        }
        LeakFamily::Station => {
            if labels.iter().any(|label| matches!(label, ActLabel::Fold)) {
                let target = labels
                    .iter()
                    .position(|label| matches!(label, ActLabel::Call))
                    .or_else(|| {
                        labels
                            .iter()
                            .position(|label| matches!(label, ActLabel::AllIn))
                    });
                if let Some(target) = target {
                    let sources: Vec<usize> = labels
                        .iter()
                        .enumerate()
                        .filter_map(|(index, label)| {
                            matches!(label, ActLabel::Fold | ActLabel::Bet(_) | ActLabel::AllIn)
                                .then_some(index)
                        })
                        .collect();
                    transfer_to(row, target, sources.into_iter(), strength);
                }
            } else if let Some(target) = labels
                .iter()
                .position(|label| matches!(label, ActLabel::Check))
            {
                let sources: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter_map(|(index, label)| {
                        matches!(label, ActLabel::Bet(_) | ActLabel::AllIn).then_some(index)
                    })
                    .collect();
                transfer_to(row, target, sources.into_iter(), strength);
            }
        }
        LeakFamily::Maniac => {
            let target = labels
                .iter()
                .enumerate()
                .rev()
                .find_map(|(index, label)| matches!(label, ActLabel::Bet(_)).then_some(index))
                .or_else(|| {
                    labels
                        .iter()
                        .position(|label| matches!(label, ActLabel::AllIn))
                });
            if let Some(target) = target {
                let sources: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter_map(|(index, label)| {
                        matches!(label, ActLabel::Fold | ActLabel::Check | ActLabel::Call)
                            .then_some(index)
                    })
                    .collect();
                transfer_to(row, target, sources.into_iter(), strength);
            }
        }
    }
    normalize(row);
}

fn action_shape(labels: &[ActLabel]) -> String {
    labels
        .iter()
        .map(|label| match label {
            ActLabel::Fold => 'f',
            ActLabel::Check => 'x',
            ActLabel::Call => 'c',
            ActLabel::Bet(_) => 'r',
            ActLabel::AllIn => 'A',
        })
        .collect()
}

#[derive(Clone, Default)]
pub(crate) struct Observation {
    exact: Counts,
    backoff: Counts,
    context: Counts,
    position: Counts,
    global: Counts,
    exact_meta: FxHashMap<String, (String, String, String)>,
    backoff_context: FxHashMap<String, String>,
    context_shape: FxHashMap<String, String>,
    context_position: FxHashMap<String, String>,
    position_shape: FxHashMap<String, String>,
    backoff_shape: FxHashMap<String, String>,
    decisions: u64,
}

impl Observation {
    fn add(
        &mut self,
        exact_key: String,
        backoff_key: String,
        context_key: String,
        position_key: String,
        shape: String,
        action: usize,
        actions: usize,
    ) {
        fn increment(table: &mut Counts, key: String, action: usize, actions: usize) {
            let row = table.entry(key).or_insert_with(|| vec![0; actions]);
            assert_eq!(row.len(), actions, "public key changed legal-action count");
            row[action] += 1;
        }
        // Metadata is derived deterministically from the keys and is first-wins,
        // so only a key's first sighting stores it. Guarding on `contains_key`
        // keeps the common already-seen path allocation-free: the `entry` API
        // would clone the key string on every decision even when present, and
        // opponent decisions dominate the observation-stage cost.
        if !self.exact_meta.contains_key(&exact_key) {
            self.exact_meta.insert(
                exact_key.clone(),
                (backoff_key.clone(), context_key.clone(), shape.clone()),
            );
        }
        if !self.backoff_context.contains_key(&backoff_key) {
            self.backoff_context
                .insert(backoff_key.clone(), context_key.clone());
        }
        if !self.context_shape.contains_key(&context_key) {
            self.context_shape.insert(context_key.clone(), shape.clone());
        }
        if !self.context_position.contains_key(&context_key) {
            self.context_position
                .insert(context_key.clone(), position_key.clone());
        }
        if !self.position_shape.contains_key(&position_key) {
            self.position_shape.insert(position_key.clone(), shape.clone());
        }
        if !self.backoff_shape.contains_key(&backoff_key) {
            self.backoff_shape.insert(backoff_key.clone(), shape.clone());
        }
        increment(&mut self.exact, exact_key, action, actions);
        increment(&mut self.backoff, backoff_key, action, actions);
        increment(&mut self.context, context_key, action, actions);
        increment(&mut self.position, position_key, action, actions);
        increment(&mut self.global, shape, action, actions);
        self.decisions += 1;
    }

    /// Fold one block into this accumulator in place. Counts are summed and
    /// metadata is first-wins, so this is associative and order-independent for
    /// disjoint-seed blocks (identical result to a `merged` fold).
    fn merge_from(&mut self, other: &Self) -> Result<(), String> {
        fn merge_counts(target: &mut Counts, source: &Counts) -> Result<(), String> {
            for (key, values) in source {
                let row = target
                    .entry(key.clone())
                    .or_insert_with(|| vec![0; values.len()]);
                if row.len() != values.len() {
                    return Err(format!("public row action-count mismatch at {key}"));
                }
                for (dst, src) in row.iter_mut().zip(values) {
                    *dst = dst
                        .checked_add(*src)
                        .ok_or_else(|| format!("public observation count overflow at {key}"))?;
                }
            }
            Ok(())
        }
        fn merge_meta<T: Clone + PartialEq + std::fmt::Debug>(
            target: &mut FxHashMap<String, T>,
            source: &FxHashMap<String, T>,
        ) -> Result<(), String> {
            for (key, value) in source {
                if let Some(existing) = target.get(key) {
                    if existing != value {
                        return Err(format!("public observation metadata mismatch at {key}"));
                    }
                } else {
                    target.insert(key.clone(), value.clone());
                }
            }
            Ok(())
        }

        merge_counts(&mut self.exact, &other.exact)?;
        merge_counts(&mut self.backoff, &other.backoff)?;
        merge_counts(&mut self.context, &other.context)?;
        merge_counts(&mut self.position, &other.position)?;
        merge_counts(&mut self.global, &other.global)?;
        merge_meta(&mut self.exact_meta, &other.exact_meta)?;
        merge_meta(&mut self.backoff_context, &other.backoff_context)?;
        merge_meta(&mut self.context_shape, &other.context_shape)?;
        merge_meta(&mut self.context_position, &other.context_position)?;
        merge_meta(&mut self.position_shape, &other.position_shape)?;
        merge_meta(&mut self.backoff_shape, &other.backoff_shape)?;
        self.decisions = self
            .decisions
            .checked_add(other.decisions)
            .ok_or_else(|| "public observation decision-count overflow".to_string())?;
        Ok(())
    }

    pub(crate) fn merged(&self, other: &Self) -> Result<Self, String> {
        let mut out = self.clone();
        out.merge_from(other)?;
        Ok(out)
    }

    /// Merge many independently sampled blocks with a single accumulator clone
    /// instead of re-cloning the growing accumulator once per block. The result
    /// is identical to folding `merged` left-to-right over the same blocks.
    pub(crate) fn merge_all(blocks: &[&Self]) -> Result<Self, String> {
        let (first, rest) = blocks
            .split_first()
            .ok_or_else(|| "merge_all needs at least one observation block".to_string())?;
        let mut out = (*first).clone();
        for other in rest {
            out.merge_from(other)?;
        }
        Ok(out)
    }

    pub(crate) fn decisions(&self) -> u64 {
        self.decisions
    }

    pub(crate) fn exact_keys(&self) -> impl Iterator<Item = &str> {
        self.exact.keys().map(String::as_str)
    }

    pub(crate) fn deterministic_fingerprint(&self) -> u64 {
        fn hash_counts<H: Hasher>(table: &Counts, state: &mut H) {
            let mut keys: Vec<_> = table.keys().collect();
            keys.sort_unstable();
            for key in keys {
                key.hash(state);
                table[key].hash(state);
            }
        }
        fn hash_meta<T: Hash, H: Hasher>(table: &FxHashMap<String, T>, state: &mut H) {
            let mut keys: Vec<_> = table.keys().collect();
            keys.sort_unstable();
            for key in keys {
                key.hash(state);
                table[key].hash(state);
            }
        }
        let mut state = std::collections::hash_map::DefaultHasher::new();
        self.decisions.hash(&mut state);
        hash_counts(&self.exact, &mut state);
        hash_counts(&self.backoff, &mut state);
        hash_counts(&self.context, &mut state);
        hash_counts(&self.global, &mut state);
        hash_meta(&self.exact_meta, &mut state);
        hash_meta(&self.backoff_context, &mut state);
        hash_meta(&self.context_shape, &mut state);
        hash_meta(&self.backoff_shape, &mut state);
        state.finish()
    }

    /// S3.4 fingerprint including the additive position-shape table.  The older
    /// fingerprint above deliberately remains byte-stable for S3.3 parity.
    pub(crate) fn support_fingerprint(&self) -> u64 {
        fn hash_counts<H: Hasher>(table: &Counts, state: &mut H) {
            let mut keys: Vec<_> = table.keys().collect();
            keys.sort_unstable();
            for key in keys {
                key.hash(state);
                table[key].hash(state);
            }
        }
        fn hash_meta<T: Hash, H: Hasher>(table: &FxHashMap<String, T>, state: &mut H) {
            let mut keys: Vec<_> = table.keys().collect();
            keys.sort_unstable();
            for key in keys {
                key.hash(state);
                table[key].hash(state);
            }
        }
        let mut state = std::collections::hash_map::DefaultHasher::new();
        self.deterministic_fingerprint().hash(&mut state);
        hash_counts(&self.position, &mut state);
        hash_meta(&self.context_position, &mut state);
        hash_meta(&self.position_shape, &mut state);
        state.finish()
    }

    pub(crate) fn count_table_bytes(&self, five_level: bool) -> usize {
        let value_count = self.exact.values().map(Vec::len).sum::<usize>()
            + self.backoff.values().map(Vec::len).sum::<usize>()
            + self.context.values().map(Vec::len).sum::<usize>()
            + self.global.values().map(Vec::len).sum::<usize>()
            + if five_level {
                self.position.values().map(Vec::len).sum::<usize>()
            } else {
                0
            };
        value_count * std::mem::size_of::<u64>()
    }

    pub(crate) fn public_keys_valid(&self) -> bool {
        self.exact_meta
            .iter()
            .all(|(exact, (backoff, context, shape))| {
                let position = self.context_position.get(context);
                let street = exact.chars().next();
                matches!(street, Some('P' | 'F' | 'T' | 'R'))
                    && exact.contains("|B[")
                    && (exact.ends_with("|P0") || exact.ends_with("|P1"))
                    && (backoff.starts_with("P|")
                        || backoff.starts_with("F|")
                        || backoff.starts_with("T|")
                        || backoff.starts_with("R|"))
                    && (backoff.ends_with("|P0") || backoff.ends_with("|P1"))
                    && (context.starts_with("P|A")
                        || context.starts_with("F|A")
                        || context.starts_with("T|A")
                        || context.starts_with("R|A"))
                    && (context.ends_with("|P0") || context.ends_with("|P1"))
                    && position.is_some_and(|value| {
                        (value.starts_with('A'))
                            && (value.ends_with("|P0") || value.ends_with("|P1"))
                            && self.position_shape.get(value) == Some(shape)
                    })
                    && !exact.contains("|H")
                    && !backoff.contains("|H")
                    && !context.contains("|H")
                    && position.is_some_and(|value| !value.contains("|H"))
            })
    }
}

#[derive(Clone)]
struct PublicModel {
    exact: FxHashMap<String, Vec<f64>>,
    backoff: FxHashMap<String, Vec<f64>>,
    reference_exact: FxHashMap<String, Vec<f64>>,
    reference_backoff: FxHashMap<String, Vec<f64>>,
    reference_global: FxHashMap<String, Vec<f64>>,
    train_exact_count: FxHashMap<String, u64>,
    train_backoff_count: FxHashMap<String, u64>,
    min_count: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct ModelScore {
    pub coverage: f64,
    pub model_log_loss: f64,
    pub reference_log_loss: f64,
    pub deviation_l1: f64,
    pub decisions: u64,
}

fn normalized_counts(row: &[u64]) -> Vec<f64> {
    let total: u64 = row.iter().sum();
    if total == 0 {
        vec![1.0 / row.len() as f64; row.len()]
    } else {
        row.iter()
            .map(|value| *value as f64 / total as f64)
            .collect()
    }
}

impl PublicModel {
    fn reference_row(
        reference: &Observation,
        exact_key: &str,
        backoff_key: &str,
        shape: &str,
        actions: usize,
    ) -> Vec<f64> {
        reference
            .exact
            .get(exact_key)
            .or_else(|| reference.backoff.get(backoff_key))
            .or_else(|| reference.global.get(shape))
            .map(|row| normalized_counts(row))
            .unwrap_or_else(|| vec![1.0 / actions as f64; actions])
    }

    fn smoothed(row: &[u64], prior: &[f64], strength: f64) -> Vec<f64> {
        let total: u64 = row.iter().sum();
        let denom = total as f64 + strength;
        row.iter()
            .zip(prior)
            .map(|(value, base)| (*value as f64 + strength * base) / denom)
            .collect()
    }

    fn fit(
        train: &Observation,
        reference: &Observation,
        min_count: u64,
        prior_strength: f64,
    ) -> Self {
        let reference_exact = reference
            .exact
            .iter()
            .map(|(key, row)| (key.clone(), normalized_counts(row)))
            .collect();
        let reference_backoff = reference
            .backoff
            .iter()
            .map(|(key, row)| (key.clone(), normalized_counts(row)))
            .collect();
        let reference_global = reference
            .global
            .iter()
            .map(|(key, row)| (key.clone(), normalized_counts(row)))
            .collect();
        let train_exact_count = train
            .exact
            .iter()
            .map(|(key, row)| (key.clone(), row.iter().sum()))
            .collect();
        let train_backoff_count = train
            .backoff
            .iter()
            .map(|(key, row)| (key.clone(), row.iter().sum()))
            .collect();

        let mut exact = FxHashMap::default();
        for (key, row) in &train.exact {
            if row.iter().sum::<u64>() < min_count {
                continue;
            }
            let (backoff, _context, shape) = &train.exact_meta[key];
            let prior = Self::reference_row(reference, key, backoff, shape, row.len());
            exact.insert(key.clone(), Self::smoothed(row, &prior, prior_strength));
        }
        let mut backoff = FxHashMap::default();
        for (key, row) in &train.backoff {
            if row.iter().sum::<u64>() < min_count {
                continue;
            }
            let shape = train
                .backoff_shape
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let prior = reference
                .backoff
                .get(key)
                .or_else(|| reference.global.get(shape))
                .map(|values| normalized_counts(values))
                .unwrap_or_else(|| vec![1.0 / row.len() as f64; row.len()]);
            backoff.insert(key.clone(), Self::smoothed(row, &prior, prior_strength));
        }
        Self {
            exact,
            backoff,
            reference_exact,
            reference_backoff,
            reference_global,
            train_exact_count,
            train_backoff_count,
            min_count,
        }
    }

    fn rows<'a>(
        &'a self,
        exact_key: &str,
        backoff_key: &str,
        shape: &str,
    ) -> (Option<&'a [f64]>, Option<&'a [f64]>) {
        let model = self
            .exact
            .get(exact_key)
            .or_else(|| self.backoff.get(backoff_key))
            .or_else(|| self.reference_exact.get(exact_key))
            .or_else(|| self.reference_backoff.get(backoff_key))
            .or_else(|| self.reference_global.get(shape))
            .map(Vec::as_slice);
        let reference = self
            .reference_exact
            .get(exact_key)
            .or_else(|| self.reference_backoff.get(backoff_key))
            .or_else(|| self.reference_global.get(shape))
            .map(Vec::as_slice);
        (model, reference)
    }

    fn fill(&self, game: &Nlhe, state: &NlheState, out: &mut [f64]) {
        let (exact, backoff) = game.public_model_keys(state);
        let shape = action_shape(&game.labeled_actions(state));
        if let Some(row) = self.rows(&exact, &backoff, &shape).0 {
            if row.len() == out.len() {
                out.copy_from_slice(row);
                return;
            }
        }
        out.fill(1.0 / out.len() as f64);
    }

    fn score(&self, validation: &Observation) -> ModelScore {
        let mut supported = 0u64;
        let mut model_loss = 0.0;
        let mut reference_loss = 0.0;
        let mut deviation_l1 = 0.0;
        for (exact_key, observed) in &validation.exact {
            let (backoff, _context, shape) = &validation.exact_meta[exact_key];
            let n: u64 = observed.iter().sum();
            let exact_count = self.train_exact_count.get(exact_key).copied().unwrap_or(0);
            let backoff_count = self.train_backoff_count.get(backoff).copied().unwrap_or(0);
            if exact_count >= self.min_count || backoff_count >= self.min_count {
                supported += n;
            }
            let actions = observed.len();
            let uniform = vec![1.0 / actions as f64; actions];
            let (model, reference) = self.rows(exact_key, backoff, shape);
            let model = model.unwrap_or(&uniform);
            let reference = reference.unwrap_or(&uniform);
            deviation_l1 += n as f64
                * model
                    .iter()
                    .zip(reference)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
            for ((times, probability), baseline) in observed.iter().zip(model).zip(reference) {
                model_loss -= *times as f64 * probability.max(1e-15).ln();
                reference_loss -= *times as f64 * baseline.max(1e-15).ln();
            }
        }
        let decisions = validation.decisions.max(1);
        ModelScore {
            coverage: supported as f64 / decisions as f64,
            model_log_loss: model_loss / decisions as f64,
            reference_log_loss: reference_loss / decisions as f64,
            deviation_l1: deviation_l1 / decisions as f64,
            decisions: validation.decisions,
        }
    }
}

/// S3.3's fixed, continuously interpolated public model.  This lives beside the
/// frozen S3.2 `PublicModel`; no S3.2 threshold or fallback behavior is changed.
#[derive(Clone)]
struct HierarchicalPublicModel {
    exact: FxHashMap<String, Vec<f64>>,
    backoff: FxHashMap<String, Vec<f64>>,
    context: FxHashMap<String, Vec<f64>>,
    reference_context: FxHashMap<String, Vec<f64>>,
    reference_global: FxHashMap<String, Vec<f64>>,
    train_exact_count: FxHashMap<String, u64>,
    train_backoff_count: FxHashMap<String, u64>,
    train_context_count: FxHashMap<String, u64>,
    legacy: PublicModel,
    exact_strength: f64,
    backoff_strength: f64,
    context_strength: f64,
}

#[derive(Clone)]
pub(crate) struct HierarchicalModelScore {
    pub decisions: u64,
    pub effective_weight_sum: f64,
    pub zero_weight_decisions: u64,
    pub legacy_supported_decisions: u64,
    pub model_nll_sum: f64,
    pub reference_nll_sum: f64,
    pub legacy_nll_sum: f64,
    pub sparse_decisions: u64,
    pub sparse_model_nll_sum: f64,
    pub sparse_reference_nll_sum: f64,
    pub model_brier_sum: f64,
    pub reference_brier_sum: f64,
    pub deviation_l1_sum: f64,
    pub ece_count: Vec<u64>,
    pub ece_confidence_sum: Vec<f64>,
    pub ece_accuracy_sum: Vec<u64>,
}

impl HierarchicalPublicModel {
    fn reference_context_row(
        reference: &Observation,
        context: &str,
        shape: &str,
        actions: usize,
    ) -> Vec<f64> {
        reference
            .context
            .get(context)
            .or_else(|| reference.global.get(shape))
            .map(|row| normalized_counts(row))
            .unwrap_or_else(|| vec![1.0 / actions as f64; actions])
    }

    fn fit(
        train: &Observation,
        reference: &Observation,
        exact_strength: f64,
        backoff_strength: f64,
        context_strength: f64,
    ) -> Self {
        let reference_context = reference
            .context
            .iter()
            .map(|(key, row)| (key.clone(), normalized_counts(row)))
            .collect();
        let reference_global = reference
            .global
            .iter()
            .map(|(key, row)| (key.clone(), normalized_counts(row)))
            .collect();

        let mut context = FxHashMap::default();
        for (key, row) in &train.context {
            let shape = train
                .context_shape
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let prior = Self::reference_context_row(reference, key, shape, row.len());
            context.insert(
                key.clone(),
                PublicModel::smoothed(row, &prior, context_strength),
            );
        }

        let mut backoff = FxHashMap::default();
        for (key, row) in &train.backoff {
            let context_key = train
                .backoff_context
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let shape = train
                .backoff_shape
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let parent = context.get(context_key).cloned().unwrap_or_else(|| {
                Self::reference_context_row(reference, context_key, shape, row.len())
            });
            backoff.insert(
                key.clone(),
                PublicModel::smoothed(row, &parent, backoff_strength),
            );
        }

        let mut exact = FxHashMap::default();
        for (key, row) in &train.exact {
            let (backoff_key, context_key, shape) = &train.exact_meta[key];
            let parent = backoff
                .get(backoff_key)
                .or_else(|| context.get(context_key))
                .cloned()
                .unwrap_or_else(|| {
                    Self::reference_context_row(reference, context_key, shape, row.len())
                });
            exact.insert(
                key.clone(),
                PublicModel::smoothed(row, &parent, exact_strength),
            );
        }

        let train_exact_count = train
            .exact
            .iter()
            .map(|(key, row)| (key.clone(), row.iter().sum()))
            .collect();
        let train_backoff_count = train
            .backoff
            .iter()
            .map(|(key, row)| (key.clone(), row.iter().sum()))
            .collect();
        let train_context_count = train
            .context
            .iter()
            .map(|(key, row)| (key.clone(), row.iter().sum()))
            .collect();

        Self {
            exact,
            backoff,
            context,
            reference_context,
            reference_global,
            train_exact_count,
            train_backoff_count,
            train_context_count,
            legacy: PublicModel::fit(train, reference, 200, 20.0),
            exact_strength,
            backoff_strength,
            context_strength,
        }
    }

    fn rows<'a>(
        &'a self,
        exact: &str,
        backoff: &str,
        context: &str,
        shape: &str,
    ) -> (Option<&'a [f64]>, Option<&'a [f64]>) {
        let model = self
            .exact
            .get(exact)
            .or_else(|| self.backoff.get(backoff))
            .or_else(|| self.context.get(context))
            .or_else(|| self.reference_context.get(context))
            .or_else(|| self.reference_global.get(shape))
            .map(Vec::as_slice);
        let reference = self
            .reference_context
            .get(context)
            .or_else(|| self.reference_global.get(shape))
            .map(Vec::as_slice);
        (model, reference)
    }

    fn fill(&self, game: &Nlhe, state: &NlheState, out: &mut [f64]) {
        let (exact, backoff, context) = game.public_hierarchy_keys(state);
        let shape = action_shape(&game.labeled_actions(state));
        if let Some(row) = self.rows(&exact, &backoff, &context, &shape).0 {
            if row.len() == out.len() {
                out.copy_from_slice(row);
                return;
            }
        }
        out.fill(1.0 / out.len() as f64);
    }

    fn score(&self, validation: &Observation) -> HierarchicalModelScore {
        let mut score = HierarchicalModelScore {
            decisions: validation.decisions,
            effective_weight_sum: 0.0,
            zero_weight_decisions: 0,
            legacy_supported_decisions: 0,
            model_nll_sum: 0.0,
            reference_nll_sum: 0.0,
            legacy_nll_sum: 0.0,
            sparse_decisions: 0,
            sparse_model_nll_sum: 0.0,
            sparse_reference_nll_sum: 0.0,
            model_brier_sum: 0.0,
            reference_brier_sum: 0.0,
            deviation_l1_sum: 0.0,
            ece_count: vec![0; 10],
            ece_confidence_sum: vec![0.0; 10],
            ece_accuracy_sum: vec![0; 10],
        };

        for (exact_key, observed) in &validation.exact {
            let (backoff_key, context_key, shape) = &validation.exact_meta[exact_key];
            let n: u64 = observed.iter().sum();
            let actions = observed.len();
            let uniform = vec![1.0 / actions as f64; actions];
            let (model, reference) = self.rows(exact_key, backoff_key, context_key, shape);
            let model = model.unwrap_or(&uniform);
            let reference = reference.unwrap_or(&uniform);
            let legacy = self
                .legacy
                .rows(exact_key, backoff_key, shape)
                .0
                .unwrap_or(&uniform);

            let exact_count = self.train_exact_count.get(exact_key).copied().unwrap_or(0);
            let backoff_count = self
                .train_backoff_count
                .get(backoff_key)
                .copied()
                .unwrap_or(0);
            let context_count = self
                .train_context_count
                .get(context_key)
                .copied()
                .unwrap_or(0);
            let anchor_weight = self.exact_strength / (exact_count as f64 + self.exact_strength)
                * self.backoff_strength
                / (backoff_count as f64 + self.backoff_strength)
                * self.context_strength
                / (context_count as f64 + self.context_strength);
            let opponent_weight = 1.0 - anchor_weight;
            score.effective_weight_sum += n as f64 * opponent_weight;
            if exact_count == 0 && backoff_count == 0 && context_count == 0 {
                score.zero_weight_decisions += n;
            }
            let sparse = exact_count < 200 && backoff_count < 200;
            if !sparse {
                score.legacy_supported_decisions += n;
            } else {
                score.sparse_decisions += n;
            }

            score.deviation_l1_sum += n as f64
                * model
                    .iter()
                    .zip(reference)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();

            let top = model
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(index, _)| index)
                .unwrap_or(0);
            let confidence = model[top].clamp(0.0, 1.0);
            let bin = if confidence == 1.0 {
                9
            } else {
                (confidence * 10.0).floor() as usize
            };
            score.ece_count[bin] += n;
            score.ece_confidence_sum[bin] += n as f64 * confidence;
            score.ece_accuracy_sum[bin] += observed[top];

            for (action, times) in observed.iter().copied().enumerate() {
                let times_f = times as f64;
                let model_nll = -model[action].max(1e-15).ln() * times_f;
                let reference_nll = -reference[action].max(1e-15).ln() * times_f;
                score.model_nll_sum += model_nll;
                score.reference_nll_sum += reference_nll;
                score.legacy_nll_sum += -legacy[action].max(1e-15).ln() * times_f;
                if sparse {
                    score.sparse_model_nll_sum += model_nll;
                    score.sparse_reference_nll_sum += reference_nll;
                }
                score.model_brier_sum += times_f
                    * model
                        .iter()
                        .enumerate()
                        .map(|(index, p)| {
                            let expected = usize::from(index == action) as f64;
                            (p - expected).powi(2)
                        })
                        .sum::<f64>();
                score.reference_brier_sum += times_f
                    * reference
                        .iter()
                        .enumerate()
                        .map(|(index, p)| {
                            let expected = usize::from(index == action) as f64;
                            (p - expected).powi(2)
                        })
                        .sum::<f64>();
            }
        }
        score
    }
}

/// S3.4's locked five-level public model.  The frozen S3.3 model is embedded as
/// an exact shadow comparator; its fit and scoring path above is not modified.
#[derive(Clone)]
struct FiveLevelPublicModel {
    exact: FxHashMap<String, Vec<f64>>,
    backoff: FxHashMap<String, Vec<f64>>,
    context: FxHashMap<String, Vec<f64>>,
    position: FxHashMap<String, Vec<f64>>,
    global: FxHashMap<String, Vec<f64>>,
    reference_global: FxHashMap<String, Vec<f64>>,
    train_exact_count: FxHashMap<String, u64>,
    train_backoff_count: FxHashMap<String, u64>,
    train_context_count: FxHashMap<String, u64>,
    train_position_count: FxHashMap<String, u64>,
    train_global_count: FxHashMap<String, u64>,
    shadow: HierarchicalPublicModel,
    exact_strength: f64,
    backoff_strength: f64,
    context_strength: f64,
    position_strength: f64,
    global_strength: f64,
}

#[derive(Clone)]
pub(crate) struct FiveLevelModelScore {
    pub decisions: u64,
    pub effective_weight_sum: f64,
    pub zero_weight_decisions: u64,
    pub three_level_zero_decisions: u64,
    pub recovered_three_level_zero_decisions: u64,
    pub legacy_supported_decisions: u64,
    pub model_nll_sum: f64,
    pub reference_nll_sum: f64,
    pub shadow_nll_sum: f64,
    pub legacy_nll_sum: f64,
    pub sparse_decisions: u64,
    pub sparse_model_nll_sum: f64,
    pub sparse_reference_nll_sum: f64,
    pub model_brier_sum: f64,
    pub reference_brier_sum: f64,
    pub deviation_l1_sum: f64,
    pub ece_count: Vec<u64>,
    pub ece_confidence_sum: Vec<f64>,
    pub ece_accuracy_sum: Vec<u64>,
}

impl FiveLevelPublicModel {
    #[allow(clippy::too_many_arguments)]
    fn fit(
        train: &Observation,
        reference: &Observation,
        exact_strength: f64,
        backoff_strength: f64,
        context_strength: f64,
        position_strength: f64,
        global_strength: f64,
    ) -> Self {
        let shadow = HierarchicalPublicModel::fit(
            train,
            reference,
            exact_strength,
            backoff_strength,
            context_strength,
        );
        let reference_global: FxHashMap<_, _> = reference
            .global
            .iter()
            .map(|(key, row)| (key.clone(), normalized_counts(row)))
            .collect();

        let mut global = FxHashMap::default();
        for (key, row) in &train.global {
            let prior = reference_global
                .get(key)
                .cloned()
                .unwrap_or_else(|| vec![1.0 / row.len() as f64; row.len()]);
            global.insert(
                key.clone(),
                PublicModel::smoothed(row, &prior, global_strength),
            );
        }

        let mut position = FxHashMap::default();
        for (key, row) in &train.position {
            let shape = train
                .position_shape
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let parent = global
                .get(shape)
                .or_else(|| reference_global.get(shape))
                .cloned()
                .unwrap_or_else(|| vec![1.0 / row.len() as f64; row.len()]);
            position.insert(
                key.clone(),
                PublicModel::smoothed(row, &parent, position_strength),
            );
        }

        let mut context = FxHashMap::default();
        for (key, row) in &train.context {
            let position_key = train
                .context_position
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let shape = train
                .context_shape
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let parent = position
                .get(position_key)
                .or_else(|| global.get(shape))
                .or_else(|| reference_global.get(shape))
                .cloned()
                .unwrap_or_else(|| vec![1.0 / row.len() as f64; row.len()]);
            context.insert(
                key.clone(),
                PublicModel::smoothed(row, &parent, context_strength),
            );
        }

        let mut backoff = FxHashMap::default();
        for (key, row) in &train.backoff {
            let context_key = train
                .backoff_context
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let position_key = train
                .context_position
                .get(context_key)
                .map(String::as_str)
                .unwrap_or("");
            let shape = train
                .backoff_shape
                .get(key)
                .map(String::as_str)
                .unwrap_or("");
            let parent = context
                .get(context_key)
                .or_else(|| position.get(position_key))
                .or_else(|| global.get(shape))
                .or_else(|| reference_global.get(shape))
                .cloned()
                .unwrap_or_else(|| vec![1.0 / row.len() as f64; row.len()]);
            backoff.insert(
                key.clone(),
                PublicModel::smoothed(row, &parent, backoff_strength),
            );
        }

        let mut exact = FxHashMap::default();
        for (key, row) in &train.exact {
            let (backoff_key, context_key, shape) = &train.exact_meta[key];
            let position_key = train
                .context_position
                .get(context_key)
                .map(String::as_str)
                .unwrap_or("");
            let parent = backoff
                .get(backoff_key)
                .or_else(|| context.get(context_key))
                .or_else(|| position.get(position_key))
                .or_else(|| global.get(shape))
                .or_else(|| reference_global.get(shape))
                .cloned()
                .unwrap_or_else(|| vec![1.0 / row.len() as f64; row.len()]);
            exact.insert(
                key.clone(),
                PublicModel::smoothed(row, &parent, exact_strength),
            );
        }

        let totals = |table: &Counts| {
            table
                .iter()
                .map(|(key, row)| (key.clone(), row.iter().sum()))
                .collect()
        };
        Self {
            exact,
            backoff,
            context,
            position,
            global,
            reference_global,
            train_exact_count: totals(&train.exact),
            train_backoff_count: totals(&train.backoff),
            train_context_count: totals(&train.context),
            train_position_count: totals(&train.position),
            train_global_count: totals(&train.global),
            shadow,
            exact_strength,
            backoff_strength,
            context_strength,
            position_strength,
            global_strength,
        }
    }

    fn rows<'a>(
        &'a self,
        exact: &str,
        backoff: &str,
        context: &str,
        position: &str,
        shape: &str,
    ) -> (Option<&'a [f64]>, Option<&'a [f64]>, Option<&'a [f64]>) {
        let model = self
            .exact
            .get(exact)
            .or_else(|| self.backoff.get(backoff))
            .or_else(|| self.context.get(context))
            .or_else(|| self.position.get(position))
            .or_else(|| self.global.get(shape))
            .or_else(|| self.reference_global.get(shape))
            .map(Vec::as_slice);
        let (shadow, reference) = self.shadow.rows(exact, backoff, context, shape);
        (model, reference, shadow)
    }

    fn fill(&self, game: &Nlhe, state: &NlheState, out: &mut [f64]) {
        let (exact, backoff, context) = game.public_hierarchy_keys(state);
        let position = context
            .split_once('|')
            .map(|(_, suffix)| suffix)
            .unwrap_or("");
        let shape = action_shape(&game.labeled_actions(state));
        if let Some(row) = self.rows(&exact, &backoff, &context, position, &shape).0 {
            if row.len() == out.len() {
                out.copy_from_slice(row);
                return;
            }
        }
        out.fill(1.0 / out.len() as f64);
    }

    fn score(&self, validation: &Observation) -> FiveLevelModelScore {
        let mut score = FiveLevelModelScore {
            decisions: validation.decisions,
            effective_weight_sum: 0.0,
            zero_weight_decisions: 0,
            three_level_zero_decisions: 0,
            recovered_three_level_zero_decisions: 0,
            legacy_supported_decisions: 0,
            model_nll_sum: 0.0,
            reference_nll_sum: 0.0,
            shadow_nll_sum: 0.0,
            legacy_nll_sum: 0.0,
            sparse_decisions: 0,
            sparse_model_nll_sum: 0.0,
            sparse_reference_nll_sum: 0.0,
            model_brier_sum: 0.0,
            reference_brier_sum: 0.0,
            deviation_l1_sum: 0.0,
            ece_count: vec![0; 10],
            ece_confidence_sum: vec![0.0; 10],
            ece_accuracy_sum: vec![0; 10],
        };

        for (exact_key, observed) in &validation.exact {
            let (backoff_key, context_key, shape) = &validation.exact_meta[exact_key];
            let position_key = validation
                .context_position
                .get(context_key)
                .map(String::as_str)
                .unwrap_or("");
            let n: u64 = observed.iter().sum();
            let actions = observed.len();
            let uniform = vec![1.0 / actions as f64; actions];
            let (model, reference, shadow) =
                self.rows(exact_key, backoff_key, context_key, position_key, shape);
            let model = model.unwrap_or(&uniform);
            let reference = reference.unwrap_or(&uniform);
            let shadow = shadow.unwrap_or(&uniform);
            let legacy = self
                .shadow
                .legacy
                .rows(exact_key, backoff_key, shape)
                .0
                .unwrap_or(&uniform);

            let exact_count = self.train_exact_count.get(exact_key).copied().unwrap_or(0);
            let backoff_count = self
                .train_backoff_count
                .get(backoff_key)
                .copied()
                .unwrap_or(0);
            let context_count = self
                .train_context_count
                .get(context_key)
                .copied()
                .unwrap_or(0);
            let position_count = self
                .train_position_count
                .get(position_key)
                .copied()
                .unwrap_or(0);
            let global_count = self.train_global_count.get(shape).copied().unwrap_or(0);
            let anchor_weight = self.exact_strength / (exact_count as f64 + self.exact_strength)
                * self.backoff_strength
                / (backoff_count as f64 + self.backoff_strength)
                * self.context_strength
                / (context_count as f64 + self.context_strength)
                * self.position_strength
                / (position_count as f64 + self.position_strength)
                * self.global_strength
                / (global_count as f64 + self.global_strength);
            score.effective_weight_sum += n as f64 * (1.0 - anchor_weight);

            let three_level_zero = exact_count == 0 && backoff_count == 0 && context_count == 0;
            let five_level_zero = three_level_zero && position_count == 0 && global_count == 0;
            if three_level_zero {
                score.three_level_zero_decisions += n;
                if !five_level_zero {
                    score.recovered_three_level_zero_decisions += n;
                }
            }
            if five_level_zero {
                score.zero_weight_decisions += n;
            }

            let sparse = exact_count < 200 && backoff_count < 200;
            if sparse {
                score.sparse_decisions += n;
            } else {
                score.legacy_supported_decisions += n;
            }
            score.deviation_l1_sum += n as f64
                * model
                    .iter()
                    .zip(reference)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();

            let top = model
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(index, _)| index)
                .unwrap_or(0);
            let confidence = model[top].clamp(0.0, 1.0);
            let bin = if confidence == 1.0 {
                9
            } else {
                (confidence * 10.0).floor() as usize
            };
            score.ece_count[bin] += n;
            score.ece_confidence_sum[bin] += n as f64 * confidence;
            score.ece_accuracy_sum[bin] += observed[top];

            for (action, times) in observed.iter().copied().enumerate() {
                let times_f = times as f64;
                let model_nll = -model[action].max(1e-15).ln() * times_f;
                let reference_nll = -reference[action].max(1e-15).ln() * times_f;
                score.model_nll_sum += model_nll;
                score.reference_nll_sum += reference_nll;
                score.shadow_nll_sum += -shadow[action].max(1e-15).ln() * times_f;
                score.legacy_nll_sum += -legacy[action].max(1e-15).ln() * times_f;
                if sparse {
                    score.sparse_model_nll_sum += model_nll;
                    score.sparse_reference_nll_sum += reference_nll;
                }
                score.model_brier_sum += times_f
                    * model
                        .iter()
                        .enumerate()
                        .map(|(index, p)| {
                            let expected = usize::from(index == action) as f64;
                            (p - expected).powi(2)
                        })
                        .sum::<f64>();
                score.reference_brier_sum += times_f
                    * reference
                        .iter()
                        .enumerate()
                        .map(|(index, p)| {
                            let expected = usize::from(index == action) as f64;
                            (p - expected).powi(2)
                        })
                        .sum::<f64>();
            }
        }
        score
    }
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let value = value ^ (value >> 31);
        (value >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

fn sample(probabilities: &[f64], rng: &mut Rng) -> usize {
    let draw = rng.next_f64();
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

pub(crate) fn sample_observation(
    game: &Nlhe,
    hero: &Policy,
    opponent: &Policy,
    hands: u64,
    seed: u64,
) -> Observation {
    sample_observation_split(game, hero, opponent, hands, 0, seed).0
}

/// Independently seeded fixed observation blocks.  Indexed parallel iteration
/// preserves block order; each block owns its RNG, so worker scheduling cannot
/// alter counts or downstream statistics.
pub(crate) fn sample_observation_blocks(
    game: &Nlhe,
    hero: &Policy,
    opponent: &Policy,
    hands_per_block: u64,
    seeds: &[u64],
    threads: usize,
) -> Result<Vec<Observation>, String> {
    if hands_per_block == 0 || seeds.is_empty() {
        return Err("observation blocks require positive hands and at least one seed".to_string());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|error| format!("failed to build observation thread pool: {error}"))?;
    Ok(pool.install(|| {
        seeds
            .par_iter()
            .map(|seed| sample_observation(game, hero, opponent, hands_per_block, *seed))
            .collect()
    }))
}

pub(crate) fn sample_observation_split(
    game: &Nlhe,
    hero: &Policy,
    opponent: &Policy,
    fit_hands: u64,
    validation_hands: u64,
    seed: u64,
) -> (Observation, Observation) {
    let mut fit = Observation::default();
    let mut validation = Observation::default();
    let mut rng = Rng::new(seed ^ 0x5333_325F_4F42_5345); // "S32_OBSE"-ish
    let mut key = Vec::<u8>::with_capacity(48);
    for hand in 0..fit_hands + validation_hands {
        let hero_seat = (hand & 1) as usize;
        let opponent_seat = 1 - hero_seat;
        let mut state = game.initial_state();
        loop {
            if game.is_terminal(&state) {
                break;
            }
            if game.is_chance(&state) {
                state = game.sample_chance_with(&state, |outcome_probabilities| {
                    let mut probabilities = outcome_probabilities.to_vec();
                    normalize(&mut probabilities);
                    sample(&probabilities, &mut rng)
                });
                continue;
            }
            let actions = game.num_actions(&state);
            key.clear();
            game.infoset_key_into(&state, &mut key);
            let mut probabilities = [0.0f64; MAX_ACTIONS];
            let policy = if game.current_player(&state) == hero_seat {
                hero
            } else {
                opponent
            };
            if !policy.try_fill(game, &state, &key, &mut probabilities[..actions]) {
                probabilities[..actions].fill(1.0 / actions as f64);
            }
            let action = sample(&probabilities[..actions], &mut rng);
            if game.current_player(&state) == opponent_seat {
                let (exact, backoff, context) = game.public_hierarchy_keys(&state);
                let shape = action_shape(&game.labeled_actions(&state));
                let position = context
                    .split_once('|')
                    .map(|(_, suffix)| suffix.to_string())
                    .expect("public context contains street separator");
                let destination = if hand < fit_hands {
                    &mut fit
                } else {
                    &mut validation
                };
                destination.add(exact, backoff, context, position, shape, action, actions);
            }
            state = game.next_state(&state, action);
        }
    }
    (fit, validation)
}

#[derive(Clone)]
pub(crate) struct PairedMatchStats {
    pub hands: u64,
    pub block_size: u64,
    pub candidate_bb_per_100: f64,
    pub anchor_bb_per_100: f64,
    pub gain_bb_per_100: f64,
    pub difference_stddev_bb_per_hand: f64,
    pub block_gains_bb_per_100: Vec<f64>,
}

fn hand_seed(seed: u64, hand: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(hand.wrapping_mul(0xD1B5_4A32_D192_ED03))
        ^ 0x5333_325F_5041_4952 // "S32_PAIR"
}

fn play_hand(game: &Nlhe, hero: &Policy, opponent: &Policy, hero_seat: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut key = Vec::<u8>::with_capacity(48);
    let mut state = game.initial_state();
    let utility = loop {
        if game.is_terminal(&state) {
            break game.terminal_utility(&state);
        }
        if game.is_chance(&state) {
            state = game.sample_chance_with(&state, |outcome_probabilities| {
                let mut probabilities = outcome_probabilities.to_vec();
                normalize(&mut probabilities);
                sample(&probabilities, &mut rng)
            });
            continue;
        }
        let actions = game.num_actions(&state);
        key.clear();
        game.infoset_key_into(&state, &mut key);
        let policy = if game.current_player(&state) == hero_seat {
            hero
        } else {
            opponent
        };
        let mut probabilities = [0.0f64; MAX_ACTIONS];
        if !policy.try_fill(game, &state, &key, &mut probabilities[..actions]) {
            probabilities[..actions].fill(1.0 / actions as f64);
        }
        let action = sample(&probabilities[..actions], &mut rng);
        state = game.next_state(&state, action);
    };
    let hero_chips = if hero_seat == 0 { utility } else { -utility };
    hero_chips / game.big_blind() as f64
}

#[derive(Clone, Copy, Default)]
struct PairedBlockSums {
    candidate: f64,
    anchor: f64,
    difference_sq: f64,
}

fn paired_block(
    game: &Nlhe,
    candidate: &Policy,
    anchor: &Policy,
    opponent: &Policy,
    seed: u64,
    start_hand: u64,
    block_size: u64,
) -> PairedBlockSums {
    let mut sums = PairedBlockSums::default();
    for hand in start_hand..start_hand + block_size {
        let hero_seat = (hand & 1) as usize;
        let stream = hand_seed(seed, hand);
        let candidate_result = play_hand(game, candidate, opponent, hero_seat, stream);
        let anchor_result = play_hand(game, anchor, opponent, hero_seat, stream);
        let difference = candidate_result - anchor_result;
        sums.candidate += candidate_result;
        sums.anchor += anchor_result;
        sums.difference_sq += difference * difference;
    }
    sums
}

/// Fixed-block paired evaluation for S3.3.  Every worker count evaluates the
/// same globally indexed hands and the ordered reduction below is identical.
pub(crate) fn paired_evaluate_blocks(
    game: &Nlhe,
    candidate: &Policy,
    anchor: &Policy,
    opponent: &Policy,
    hands: u64,
    seed: u64,
    block_size: u64,
    threads: usize,
) -> Result<PairedMatchStats, String> {
    if hands == 0 || block_size == 0 || hands % block_size != 0 {
        return Err(
            "hands and block_size must be positive, with hands divisible by block_size".to_string(),
        );
    }
    let blocks = hands / block_size;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|error| format!("failed to build paired-evaluation thread pool: {error}"))?;
    let rows: Vec<PairedBlockSums> = pool.install(|| {
        (0..blocks)
            .into_par_iter()
            .map(|block| {
                paired_block(
                    game,
                    candidate,
                    anchor,
                    opponent,
                    seed,
                    block * block_size,
                    block_size,
                )
            })
            .collect()
    });
    let mut candidate_sum = 0.0;
    let mut anchor_sum = 0.0;
    let mut difference_sum_sq = 0.0;
    let mut block_gains_bb_per_100 = Vec::with_capacity(rows.len());
    for row in rows {
        candidate_sum += row.candidate;
        anchor_sum += row.anchor;
        difference_sum_sq += row.difference_sq;
        block_gains_bb_per_100.push((row.candidate - row.anchor) / block_size as f64 * 100.0);
    }
    let n = hands as f64;
    let difference_sum = candidate_sum - anchor_sum;
    let variance = if hands > 1 {
        ((difference_sum_sq - difference_sum * difference_sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };
    Ok(PairedMatchStats {
        hands,
        block_size,
        candidate_bb_per_100: candidate_sum / n * 100.0,
        anchor_bb_per_100: anchor_sum / n * 100.0,
        gain_bb_per_100: difference_sum / n * 100.0,
        difference_stddev_bb_per_hand: variance.sqrt(),
        block_gains_bb_per_100,
    })
}

/// Common-random-number paired comparison. The two worlds receive the same
/// per-hand random stream, so deals and action draws remain paired until their
/// trajectories diverge. Blocks are fixed and consecutive as required by §43.5.
pub(crate) fn paired_evaluate(
    game: &Nlhe,
    candidate: &Policy,
    anchor: &Policy,
    opponent: &Policy,
    hands: u64,
    seed: u64,
    block_size: u64,
) -> Result<PairedMatchStats, String> {
    if hands == 0 || block_size == 0 || hands % block_size != 0 {
        return Err(
            "hands and block_size must be positive, with hands divisible by block_size".to_string(),
        );
    }
    let blocks = (hands / block_size) as usize;
    let mut block_differences = vec![0.0f64; blocks];
    let mut candidate_sum = 0.0;
    let mut anchor_sum = 0.0;
    let mut difference_sum_sq = 0.0;
    for hand in 0..hands {
        let hero_seat = (hand & 1) as usize;
        let stream = hand_seed(seed, hand);
        let candidate_result = play_hand(game, candidate, opponent, hero_seat, stream);
        let anchor_result = play_hand(game, anchor, opponent, hero_seat, stream);
        let difference = candidate_result - anchor_result;
        candidate_sum += candidate_result;
        anchor_sum += anchor_result;
        difference_sum_sq += difference * difference;
        block_differences[(hand / block_size) as usize] += difference;
    }
    let n = hands as f64;
    let difference_sum = candidate_sum - anchor_sum;
    let difference_mean = difference_sum / n;
    let variance = if hands > 1 {
        ((difference_sum_sq - difference_sum * difference_sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };
    Ok(PairedMatchStats {
        hands,
        block_size,
        candidate_bb_per_100: candidate_sum / n * 100.0,
        anchor_bb_per_100: anchor_sum / n * 100.0,
        gain_bb_per_100: difference_mean * 100.0,
        difference_stddev_bb_per_hand: variance.sqrt(),
        block_gains_bb_per_100: block_differences
            .into_iter()
            .map(|sum| sum / block_size as f64 * 100.0)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mccfr::{MCCFRSolver, Variant};
    use crate::nlhe::NlheConfig;

    fn small_anchor() -> (Nlhe, Policy) {
        let game = Nlhe::new(NlheConfig::small());
        let mut solver = MCCFRSolver::new(Variant::Linear, 5);
        solver.run(&game, 50_000);
        let strategy = solver
            .average_strategy()
            .into_iter()
            .map(|(key, row)| {
                (
                    key.into_bytes().into_boxed_slice(),
                    row.into_iter().map(|p| p as f32).collect(),
                )
            })
            .collect();
        (game, Policy::tabular(Arc::new(strategy)))
    }

    #[test]
    fn leak_rows_remain_probabilities() {
        let (game, anchor) = small_anchor();
        let policies = [
            anchor.leak(LeakFamily::Nit, 0.6),
            anchor.leak(LeakFamily::Station, 0.6),
            anchor.leak(LeakFamily::Maniac, 0.6),
        ];
        let mut rng = Rng::new(3);
        for policy in policies {
            let mut state = game.initial_state();
            for _ in 0..100 {
                if game.is_terminal(&state) {
                    state = game.initial_state();
                    continue;
                }
                if game.is_chance(&state) {
                    state = game.sample_chance_with(&state, |probabilities| {
                        (rng.next_f64() * probabilities.len() as f64) as usize
                    });
                    continue;
                }
                let n = game.num_actions(&state);
                let mut key = Vec::new();
                game.infoset_key_into(&state, &mut key);
                let mut row = [0.0; MAX_ACTIONS];
                if !policy.try_fill(&game, &state, &key, &mut row[..n]) {
                    row[..n].fill(1.0 / n as f64);
                }
                assert!((row[..n].iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(row[..n].iter().all(|p| p.is_finite() && *p >= 0.0));
                state = game.next_state(&state, sample(&row[..n], &mut rng));
            }
        }
    }

    #[test]
    fn public_model_uses_supported_backoff_and_improves_log_loss() {
        let (game, anchor) = small_anchor();
        let station = anchor.leak(LeakFamily::Station, 0.6);
        let train = sample_observation(&game, &anchor, &station, 20_000, 11);
        let validation = sample_observation(&game, &anchor, &station, 5_000, 13);
        let reference = sample_observation(&game, &anchor, &anchor, 20_000, 17);
        let model = Policy::public_model(&train, &reference, 20, 20.0);
        let score = model.score(&validation).unwrap();
        assert!(score.decisions > 0);
        assert!(score.coverage >= 0.95, "coverage {}", score.coverage);
        assert!(score.model_log_loss < score.reference_log_loss);
        assert!(train.public_keys_valid());
        assert!(train.exact_keys().all(|key| !key.contains("|H")));
    }

    #[test]
    fn hierarchical_model_is_public_normalized_and_predictive() {
        let (game, anchor) = small_anchor();
        let station = anchor.leak(LeakFamily::Station, 0.6);
        let train = sample_observation(&game, &anchor, &station, 20_000, 101);
        let validation = sample_observation(&game, &anchor, &station, 5_000, 103);
        let reference = sample_observation(&game, &anchor, &anchor, 20_000, 107);
        let model = Policy::hierarchical_public_model(&train, &reference, 200.0, 50.0, 20.0);
        let score = model.score_hierarchical(&validation).unwrap();
        assert!(train.public_keys_valid());
        assert_eq!(score.decisions, validation.decisions());
        assert!(score.effective_weight_sum / score.decisions as f64 >= 0.99);
        assert_eq!(score.zero_weight_decisions, 0);
        assert!(score.model_nll_sum < score.reference_nll_sum);
        assert!(score.model_brier_sum < score.reference_brier_sum);
        assert_eq!(score.ece_count.iter().sum::<u64>(), score.decisions);
        assert_eq!(score.ece_accuracy_sum.len(), 10);
    }

    fn synthetic_observation(
        exact: &str,
        backoff: &str,
        context: &str,
        position: &str,
        shape: &str,
        exact_counts: Vec<u64>,
        backoff_counts: Vec<u64>,
        context_counts: Vec<u64>,
        position_counts: Vec<u64>,
        global_counts: Vec<u64>,
    ) -> Observation {
        let decisions = exact_counts.iter().sum();
        Observation {
            exact: [(exact.to_string(), exact_counts)].into_iter().collect(),
            backoff: [(backoff.to_string(), backoff_counts)]
                .into_iter()
                .collect(),
            context: [(context.to_string(), context_counts)]
                .into_iter()
                .collect(),
            position: [(position.to_string(), position_counts)]
                .into_iter()
                .collect(),
            global: [(shape.to_string(), global_counts)].into_iter().collect(),
            exact_meta: [(
                exact.to_string(),
                (backoff.to_string(), context.to_string(), shape.to_string()),
            )]
            .into_iter()
            .collect(),
            backoff_context: [(backoff.to_string(), context.to_string())]
                .into_iter()
                .collect(),
            context_shape: [(context.to_string(), shape.to_string())]
                .into_iter()
                .collect(),
            context_position: [(context.to_string(), position.to_string())]
                .into_iter()
                .collect(),
            position_shape: [(position.to_string(), shape.to_string())]
                .into_iter()
                .collect(),
            backoff_shape: [(backoff.to_string(), shape.to_string())]
                .into_iter()
                .collect(),
            decisions,
        }
    }

    #[test]
    fn five_level_equations_recover_three_level_zero_and_preserve_shadow() {
        let train = synthetic_observation(
            "F|B[1, 2, 3]|h|P0",
            "F|h|P0",
            "F|Axc|P0",
            "Axc|P0",
            "xc",
            vec![2, 0],
            vec![4, 0],
            vec![6, 0],
            vec![8, 0],
            vec![10, 0],
        );
        let reference = synthetic_observation(
            "F|B[1, 2, 3]|h|P0",
            "F|h|P0",
            "F|Axc|P0",
            "Axc|P0",
            "xc",
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
        );
        let model = FiveLevelPublicModel::fit(&train, &reference, 200.0, 50.0, 20.0, 10.0, 5.0);

        let q_global = [(10.0 + 5.0 * 0.5) / 15.0, (5.0 * 0.5) / 15.0];
        let q_position = [
            (8.0 + 10.0 * q_global[0]) / 18.0,
            (10.0 * q_global[1]) / 18.0,
        ];
        let q_context = [
            (6.0 + 20.0 * q_position[0]) / 26.0,
            (20.0 * q_position[1]) / 26.0,
        ];
        let q_backoff = [
            (4.0 + 50.0 * q_context[0]) / 54.0,
            (50.0 * q_context[1]) / 54.0,
        ];
        let q_exact = [
            (2.0 + 200.0 * q_backoff[0]) / 202.0,
            (200.0 * q_backoff[1]) / 202.0,
        ];
        for (actual, expected) in model.global["xc"].iter().zip(q_global) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in model.position["Axc|P0"].iter().zip(q_position) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in model.context["F|Axc|P0"].iter().zip(q_context) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in model.backoff["F|h|P0"].iter().zip(q_backoff) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in model.exact["F|B[1, 2, 3]|h|P0"].iter().zip(q_exact) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for table in [
            &model.global,
            &model.position,
            &model.context,
            &model.backoff,
            &model.exact,
        ] {
            assert!(table.values().all(|row| {
                row.iter().all(|p| p.is_finite() && *p >= 0.0)
                    && (row.iter().sum::<f64>() - 1.0).abs() < 1e-12
            }));
        }

        let validation = synthetic_observation(
            "R|B[4, 5, 6, 7, 8]|new|P0",
            "R|new|P0",
            "R|Axc|P0",
            "Axc|P0",
            "xc",
            vec![0, 1],
            vec![0, 1],
            vec![0, 1],
            vec![0, 1],
            vec![0, 1],
        );
        let score = model.score(&validation);
        let shadow_score = model.shadow.score(&validation);
        assert_eq!(score.decisions, 1);
        assert_eq!(score.three_level_zero_decisions, 1);
        assert_eq!(score.recovered_three_level_zero_decisions, 1);
        assert_eq!(score.zero_weight_decisions, 0);
        assert_eq!(
            score.shadow_nll_sum.to_bits(),
            shadow_score.model_nll_sum.to_bits()
        );
        assert!(score.model_nll_sum.is_finite());
    }

    #[test]
    fn five_level_all_zero_remains_zero_support_with_finite_fallback() {
        let train = synthetic_observation(
            "F|B[1, 2, 3]|h|P0",
            "F|h|P0",
            "F|Axc|P0",
            "Axc|P0",
            "xc",
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
        );
        let reference = train.clone();
        let model = FiveLevelPublicModel::fit(&train, &reference, 200.0, 50.0, 20.0, 10.0, 5.0);
        let validation = synthetic_observation(
            "T|B[9, 10, 11, 12]|z|P1",
            "T|z|P1",
            "T|AfrA|P1",
            "AfrA|P1",
            "frA",
            vec![1, 0, 0],
            vec![1, 0, 0],
            vec![1, 0, 0],
            vec![1, 0, 0],
            vec![1, 0, 0],
        );
        let score = model.score(&validation);
        assert_eq!(score.zero_weight_decisions, 1);
        assert_eq!(score.recovered_three_level_zero_decisions, 0);
        assert_eq!(score.effective_weight_sum, 0.0);
        assert!(score.model_nll_sum.is_finite());
        assert_eq!(score.ece_count.iter().sum::<u64>(), 1);
    }

    #[test]
    fn fixed_blocks_are_worker_count_invariant() {
        let (game, anchor) = small_anchor();
        let opponent = anchor.leak(LeakFamily::Nit, 0.35);
        let seeds = [701, 703, 709, 719];
        let serial = sample_observation_blocks(&game, &anchor, &opponent, 250, &seeds, 1).unwrap();
        let parallel =
            sample_observation_blocks(&game, &anchor, &opponent, 250, &seeds, 4).unwrap();
        let serial_merged = serial
            .iter()
            .skip(1)
            .try_fold(serial[0].clone(), |acc, row| acc.merged(row))
            .unwrap();
        let parallel_merged = parallel
            .iter()
            .skip(1)
            .try_fold(parallel[0].clone(), |acc, row| acc.merged(row))
            .unwrap();
        assert_eq!(serial_merged.decisions, parallel_merged.decisions);
        assert_eq!(serial_merged.exact, parallel_merged.exact);
        assert_eq!(serial_merged.backoff, parallel_merged.backoff);
        assert_eq!(serial_merged.context, parallel_merged.context);
        assert_eq!(serial_merged.position, parallel_merged.position);
        assert_eq!(serial_merged.global, parallel_merged.global);
        assert_eq!(
            serial_merged.support_fingerprint(),
            parallel_merged.support_fingerprint()
        );

        let one = paired_evaluate_blocks(&game, &anchor, &anchor, &opponent, 4_000, 727, 1_000, 1)
            .unwrap();
        let four = paired_evaluate_blocks(&game, &anchor, &anchor, &opponent, 4_000, 727, 1_000, 4)
            .unwrap();
        assert_eq!(
            one.candidate_bb_per_100.to_bits(),
            four.candidate_bb_per_100.to_bits()
        );
        assert_eq!(
            one.anchor_bb_per_100.to_bits(),
            four.anchor_bb_per_100.to_bits()
        );
        assert_eq!(
            one.gain_bb_per_100.to_bits(),
            four.gain_bb_per_100.to_bits()
        );
        assert_eq!(one.block_gains_bb_per_100, four.block_gains_bb_per_100);
    }

    #[test]
    fn sparse_response_blend_falls_back_to_anchor_and_pairing_is_exact() {
        let (game, anchor) = small_anchor();
        let empty = Policy::tabular(Arc::new(FxHashMap::default()));
        let blended = anchor.blend(&empty, 1.0);
        let opponent = anchor.leak(LeakFamily::Nit, 0.6);
        let stats =
            paired_evaluate(&game, &blended, &anchor, &opponent, 10_000, 29, 1_000).unwrap();
        assert_eq!(stats.hands, 10_000);
        assert_eq!(stats.gain_bb_per_100, 0.0);
        assert_eq!(stats.difference_stddev_bb_per_hand, 0.0);
        assert!(stats.block_gains_bb_per_100.iter().all(|gain| *gain == 0.0));
    }
}
