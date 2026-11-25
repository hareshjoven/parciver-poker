// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! Compact card representation shared by the hand evaluator.
//!
//! A card is a 0..52 index; internally `rank` is 0..13 (0=deuce … 12=ace) and
//! `suit` is 0..4. Rank 0..12 (ace high = 12) is exactly what the evaluator's
//! tiebreak encoding wants, and the ace-low wheel is special-cased there.

/// A single card. `rank` in 0..13 (0=2 … 8=T, 9=J, 10=Q, 11=K, 12=A), `suit` 0..4.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Card {
    pub rank: u8,
    pub suit: u8,
}

impl Card {
    /// Card from its 0..52 index (`rank * 4 + suit`).
    #[inline]
    pub fn from_index(index: u8) -> Card {
        Card {
            rank: index >> 2,
            suit: index & 3,
        }
    }

    /// The 0..52 index of this card.
    #[inline]
    pub fn index(self) -> u8 {
        (self.rank << 2) | self.suit
    }
}

/// Parse a two-char card like "As", "Td", "2c" into its 0..52 index.
/// Ranks `23456789TJQKA` (case-insensitive), suits `cdhs`.
pub fn parse_card(s: &str) -> Result<u8, String> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("card must be 2 chars, got {s:?}"));
    }
    let rank = match bytes[0].to_ascii_uppercase() {
        b'2' => 0,
        b'3' => 1,
        b'4' => 2,
        b'5' => 3,
        b'6' => 4,
        b'7' => 5,
        b'8' => 6,
        b'9' => 7,
        b'T' => 8,
        b'J' => 9,
        b'Q' => 10,
        b'K' => 11,
        b'A' => 12,
        _ => return Err(format!("bad rank in {s:?}")),
    };
    let suit = match bytes[1].to_ascii_lowercase() {
        b'c' => 0,
        b'd' => 1,
        b'h' => 2,
        b's' => 3,
        _ => return Err(format!("bad suit in {s:?}")),
    };
    Ok(Card { rank, suit }.index())
}
