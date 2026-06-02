//! Induced representations and Frobenius reciprocity.

use nalgebra::DMatrix;
use num_complex::Complex64;
use crate::group::FiniteGroup;
use crate::representation::Representation;
use crate::character::Character;

/// A subgroup specification: maps subgroup element indices to group element indices.
pub struct SubgroupEmbedding {
    /// Maps subgroup element index → group element index.
    pub embedding: Vec<usize>,
}

impl SubgroupEmbedding {
    /// Create from a mapping.
    pub fn new(embedding: Vec<usize>) -> Self {
        Self { embedding }
    }

    /// Order of the subgroup.
    pub fn order(&self) -> usize {
        self.embedding.len()
    }
}

/// Compute the induced character Ind_H^G(χ).
/// Uses the embedding to identify which group elements are in the subgroup.
pub fn induced_character_embedded(
    group: &FiniteGroup,
    chi_h: &Character,
    embedding: &SubgroupEmbedding,
) -> Character {
    let h_order = embedding.order() as f64;
    let mut values = Vec::new();

    for g in 0..group.order() {
        let mut sum = Complex64::new(0.0, 0.0);
        for x in 0..group.order() {
            let x_inv = group.inverse(x);
            let xgx_inv = group.multiply(group.multiply(x_inv, g), x);
            // Check if xgx_inv is in the subgroup
            if let Some(h_idx) = embedding.embedding.iter().position(|&e| e == xgx_inv) {
                sum += chi_h.value(h_idx);
            }
        }
        values.push(sum / h_order);
    }

    Character { values }
}

/// Restrict a character from G to H using an embedding.
pub fn restrict_character_embedded(
    chi_g: &Character,
    embedding: &SubgroupEmbedding,
) -> Character {
    let values: Vec<Complex64> = embedding
        .embedding
        .iter()
        .map(|&g_idx| chi_g.value(g_idx))
        .collect();
    Character { values }
}

/// Verify Frobenius reciprocity: ⟨Ind_H^G(χ), ψ⟩_G = ⟨χ, Res_H^G(ψ)⟩_H
pub fn verify_frobenius_reciprocity_embedded(
    group: &FiniteGroup,
    subgroup: &FiniteGroup,
    chi_h: &Character,
    psi_g: &Character,
    embedding: &SubgroupEmbedding,
) -> bool {
    let ind_chi = induced_character_embedded(group, chi_h, embedding);
    let lhs = ind_chi.inner_product(psi_g, group);

    let res_psi = restrict_character_embedded(psi_g, embedding);
    let rhs = chi_h.inner_product(&res_psi, subgroup);

    (lhs - rhs).norm() < 1e-8
}

/// Compute the induced representation Ind_H^G(ρ).
pub fn induced_representation_embedded(
    group: &FiniteGroup,
    _subgroup: &FiniteGroup,
    rho_h: &Representation,
    embedding: &SubgroupEmbedding,
    coset_reps: &[usize],
) -> Representation {
    let index = coset_reps.len();
    let dim_h = rho_h.dimension;
    let dim_induced = index * dim_h;

    let mut matrices = Vec::new();

    for g in 0..group.order() {
        let mut mat = DMatrix::zeros(dim_induced, dim_induced);

        for (i, &ti) in coset_reps.iter().enumerate() {
            let gti = group.multiply(g, ti);

            for (j, &tj) in coset_reps.iter().enumerate() {
                let tj_inv = group.inverse(tj);
                let candidate_h = group.multiply(tj_inv, gti);

                // Check if candidate_h is in the subgroup
                if let Some(hi) = embedding.embedding.iter().position(|&e| e == candidate_h) {
                    let rho_block = &rho_h.matrices[hi];
                    for r in 0..dim_h {
                        for c in 0..dim_h {
                            mat[(j * dim_h + r, i * dim_h + c)] = rho_block[(r, c)];
                        }
                    }
                    break;
                }
            }
        }
        matrices.push(mat);
    }

    Representation::new(matrices)
}

// Keep old API for backward compatibility (uses label matching, works when subgroup labels match group labels)
pub fn induced_character(
    subgroup: &FiniteGroup,
    group: &FiniteGroup,
    chi_h: &Character,
    coset_reps: &[usize],
) -> Character {
    let h_order = subgroup.order() as f64;
    let mut values = Vec::new();

    for g in 0..group.order() {
        let mut sum = Complex64::new(0.0, 0.0);
        for x in 0..group.order() {
            let x_inv = group.inverse(x);
            let xgx_inv = group.multiply(group.multiply(x_inv, g), x);
            let label = &group.elements[xgx_inv];
            if let Some(h_idx) = subgroup.elements.iter().position(|e| e == label) {
                sum += chi_h.value(h_idx);
            }
        }
        values.push(sum / h_order);
    }

    Character { values }
}

pub fn restrict_character(
    subgroup: &FiniteGroup,
    group: &FiniteGroup,
    chi_g: &Character,
) -> Character {
    let values: Vec<Complex64> = subgroup
        .elements
        .iter()
        .map(|label| {
            let idx = group.elements.iter().position(|e| e == label).unwrap();
            chi_g.value(idx)
        })
        .collect();
    Character { values }
}

pub fn verify_frobenius_reciprocity(
    subgroup: &FiniteGroup,
    group: &FiniteGroup,
    chi_h: &Character,
    psi_g: &Character,
    coset_reps: &[usize],
) -> bool {
    let ind_chi = induced_character(subgroup, group, chi_h, coset_reps);
    let lhs = ind_chi.inner_product(psi_g, group);

    let res_psi = restrict_character(subgroup, group, psi_g);
    let rhs = chi_h.inner_product(&res_psi, subgroup);

    (lhs - rhs).norm() < 1e-8
}
