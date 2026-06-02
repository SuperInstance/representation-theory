//! Character theory: character tables, orthogonality relations.

use nalgebra::DVector;
use num_complex::Complex64;
use serde::{Serialize, Deserialize};
use crate::group::FiniteGroup;
use crate::representation::Representation;

/// Character of a representation: χ(g) = Tr(ρ(g)).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Character {
    /// Values indexed by group element index.
    pub values: Vec<Complex64>,
}

impl Character {
    /// Compute character of a representation.
    pub fn from_representation(rep: &Representation) -> Self {
        Self {
            values: rep.matrices.iter().map(|m| m.trace()).collect(),
        }
    }

    /// Get character value at element index i.
    pub fn value(&self, i: usize) -> Complex64 {
        self.values[i]
    }

    /// Inner product of two characters: ⟨χ₁, χ₂⟩ = (1/|G|) Σ χ₁(g) conj(χ₂(g)).
    pub fn inner_product(&self, other: &Character, group: &FiniteGroup) -> Complex64 {
        let n = group.order() as f64;
        let sum: Complex64 = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b.conj())
            .sum();
        sum / n
    }

    /// Degree of the character (= χ(e) = dimension of representation).
    pub fn degree(&self) -> f64 {
        self.values[0].re // should be real and positive for element 0 being identity... 
    }

    /// Check if this character is irreducible: ⟨χ, χ⟩ = 1.
    pub fn is_irreducible(&self, group: &FiniteGroup) -> bool {
        let ip = self.inner_product(self, group);
        (ip.re - 1.0).abs() < 1e-10 && ip.im.abs() < 1e-10
    }
}

/// Character table of a finite group.
/// Rows = irreducible characters, Columns = conjugacy classes (rep'd by first element).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharacterTable {
    /// Number of conjugacy classes (= number of irreducible representations).
    pub num_classes: usize,
    /// Class representatives (element indices).
    pub class_representatives: Vec<usize>,
    /// Class sizes.
    pub class_sizes: Vec<usize>,
    /// Character values: table[i][j] = χ_i(class_j).
    pub table: Vec<Vec<Complex64>>,
    /// Group order.
    pub group_order: usize,
}

impl CharacterTable {
    /// Build character table from irreducible characters and conjugacy classes.
    pub fn new(
        group: &FiniteGroup,
        irreducible_chars: Vec<Character>,
    ) -> Self {
        let classes = group.conjugacy_classes();
        let class_sizes: Vec<usize> = classes.iter().map(|c| c.len()).collect();
        let class_representatives: Vec<usize> = classes.iter().map(|c| c[0]).collect();

        let table: Vec<Vec<Complex64>> = irreducible_chars
            .iter()
            .map(|chi| class_representatives.iter().map(|&rep| chi.value(rep)).collect())
            .collect();

        Self {
            num_classes: classes.len(),
            class_representatives,
            class_sizes,
            table,
            group_order: group.order(),
        }
    }

    /// Verify first orthogonality relation: Σ_g χ_i(g) conj(χ_j(g)) = |G| δ_{ij}.
    pub fn verify_first_orthogonality(&self) -> bool {
        let g = self.group_order as f64;
        for i in 0..self.table.len() {
            for j in 0..self.table.len() {
                let mut sum = Complex64::new(0.0, 0.0);
                // Sum over conjugacy classes, weighted by class size
                for k in 0..self.num_classes {
                    let size = self.class_sizes[k] as f64;
                    sum += size * self.table[i][k] * self.table[j][k].conj();
                }
                let expected = if i == j { g } else { 0.0 };
                if (sum.re - expected).abs() > 1e-8 || sum.im.abs() > 1e-8 {
                    return false;
                }
            }
        }
        true
    }

    /// Verify second orthogonality relation: Σ_χ χ(g_i) conj(χ(g_j)) = |G|/|C(g_i)| δ_{ij}.
    pub fn verify_second_orthogonality(&self) -> bool {
        for i in 0..self.num_classes {
            for j in 0..self.num_classes {
                let mut sum = Complex64::new(0.0, 0.0);
                for chi_row in &self.table {
                    sum += chi_row[i] * chi_row[j].conj();
                }
                let expected = if i == j {
                    self.group_order as f64 / self.class_sizes[i] as f64
                } else {
                    0.0
                };
                if (sum.re - expected).abs() > 1e-8 || sum.im.abs() > 1e-8 {
                    return false;
                }
            }
        }
        true
    }

    /// Decompose a character into irreducible components.
    /// Returns multiplicities: n_i = ⟨χ, χ_i⟩.
    pub fn decompose_character(&self, chi: &Character) -> Vec<f64> {
        self.table
            .iter()
            .map(|irr_row| {
                let mut inner = Complex64::new(0.0, 0.0);
                for (k, &class_rep) in self.class_representatives.iter().enumerate() {
                    let size = self.class_sizes[k] as f64;
                    inner += size * chi.value(class_rep) * irr_row[k].conj();
                }
                (inner / self.group_order as f64).re
            })
            .collect()
    }
}
