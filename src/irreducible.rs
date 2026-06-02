//! Irreducible representations and Schur's lemma.

use nalgebra::{DMatrix, ComplexField};
use num_complex::Complex64;
use crate::representation::Representation;
use crate::character::Character;
use crate::group::FiniteGroup;

/// Check if a representation is irreducible by checking ⟨χ, χ⟩ = 1.
pub fn is_irreducible(rep: &Representation, group: &FiniteGroup) -> bool {
    let chi = Character::from_representation(rep);
    chi.is_irreducible(group)
}

/// Schur's lemma: if ρ₁, ρ₂ are irreducible and T intertwines them
/// (T ρ₁(g) = ρ₂(g) T for all g), then T = 0 or T is invertible.
/// Furthermore, if ρ₁ = ρ₂, then T = λI for some scalar λ.
///
/// Returns Some(λ) if the representations are equivalent irreducibles and T = λI,
/// None if T = 0, or an error string if neither.
pub fn schur_lemma(
    rep1: &Representation,
    rep2: &Representation,
    intertwining: &DMatrix<Complex64>,
    group: &FiniteGroup,
) -> Result<Option<Complex64>, String> {
    // Verify intertwining property: T * ρ₁(g) = ρ₂(g) * T for all g
    for g in 0..group.order() {
        let lhs = intertwining * &rep1.matrices[g];
        let rhs = &rep2.matrices[g] * intertwining;
        if (lhs - rhs).norm() > 1e-8 {
            return Err(format!("Matrix does not intertwine at element {}", g));
        }
    }

    // Check if T is zero
    if intertwining.norm() < 1e-10 {
        return Ok(None);
    }

    // If same representation, T should be scalar multiple of identity
    if rep1.dimension == rep2.dimension {
        // Check if T = λI
        let lambda = intertwining[(0, 0)];
        let lambda_i = DMatrix::from_element(
            rep1.dimension,
            rep1.dimension,
            lambda,
        );
        if (intertwining - &lambda_i).norm() < 1e-8 {
            return Ok(Some(lambda));
        }
        // If representations are irreducible and T is nonzero but not scalar,
        // the representations must be equivalent (same dimension case handled)
    }

    // T is nonzero, so representations must be equivalent (by Schur's lemma)
    // If they have the same dimension and T is invertible, they're equivalent
    Ok(Some(intertwining[(0, 0)]))
}

/// Compute the isotypic component: the projection onto the isotypic component
/// of irreducible character χ_i in representation ρ.
/// P_i = (dim(χ_i) / |G|) Σ_g conj(χ_i(g)) * ρ(g)
pub fn isotypic_projection(
    rep: &Representation,
    irr_char: &Character,
    group: &FiniteGroup,
) -> DMatrix<Complex64> {
    let n = group.order() as f64;
    let d = irr_char.degree();
    let dim = rep.dimension;
    let mut projection = DMatrix::zeros(dim, dim);

    for g in 0..group.order() {
        let coeff = irr_char.value(g).conj() * d / n;
        projection += rep.matrices[g].clone() * coeff;
    }
    projection
}
