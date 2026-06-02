//! Group representations: homomorphisms G → GL(V).

use nalgebra::{DMatrix, DVector, ComplexField};
use num_complex::Complex64;
use serde::{Serialize, Deserialize};
use crate::group::FiniteGroup;

/// A finite-dimensional (complex) representation of a finite group.
///
/// Maps each group element to a matrix ρ(g) ∈ GL(n, C).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Representation {
    /// Dimension of the representation.
    pub dimension: usize,
    /// Matrices indexed by group element index: rho[element_index] = matrix.
    pub matrices: Vec<DMatrix<Complex64>>,
}

impl Representation {
    /// Create a representation from its matrices.
    pub fn new(matrices: Vec<DMatrix<Complex64>>) -> Self {
        let dimension = matrices.first().map(|m| m.nrows()).unwrap_or(0);
        for m in &matrices {
            assert_eq!(m.nrows(), dimension);
            assert_eq!(m.ncols(), dimension);
        }
        Self { dimension, matrices }
    }

    /// The trivial representation of dimension 1 (all elements map to [1]).
    pub fn trivial(group_order: usize) -> Self {
        let matrices = (0..group_order)
            .map(|_| DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0)))
            .collect();
        Self::new(matrices)
    }

    /// The sign representation of S_3: even permutations → [1], odd → [-1].
    pub fn sign_s3() -> Self {
        // S3 elements: e, (12), (13), (23), (123), (132)
        // Sign: +1, -1, -1, -1, +1, +1
        let signs = [1.0, -1.0, -1.0, -1.0, 1.0, 1.0];
        let matrices: Vec<_> = signs
            .iter()
            .map(|&s| DMatrix::from_element(1, 1, Complex64::new(s, 0.0)))
            .collect();
        Self::new(matrices)
    }

    /// The standard (2-dimensional) representation of S_3.
    pub fn standard_s3() -> Self {
        // S3 acts on R^2 by permuting coordinates of (x,y,z) with x+y+z=0
        // Using basis: e1 = (1,-1,0)/sqrt(2), e2 = (1,1,-2)/sqrt(6)
        // e=0, (12)=1, (13)=2, (23)=3, (123)=4, (132)=5
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let half = Complex64::new(0.5, 0.0);
        let neg_half = Complex64::new(-0.5, 0.0);
        let sqrt3_4 = Complex64::new(0.75_f64.sqrt(), 0.0);
        let neg_sqrt3_4 = Complex64::new(-0.75_f64.sqrt(), 0.0);

        let matrices = vec![
            // e: identity
            DMatrix::from_row_slice(2, 2, &[one, zero, zero, one]),
            // (12): swap x,y → [[-1,0],[0,1]] ... let me use standard matrices
            DMatrix::from_row_slice(2, 2, &[
                neg_half, sqrt3_4,
                sqrt3_4, half,
            ]),
            // (13): swap x,z
            DMatrix::from_row_slice(2, 2, &[
                neg_half, neg_sqrt3_4,
                neg_sqrt3_4, half,
            ]),
            // (23): swap y,z
            DMatrix::from_row_slice(2, 2, &[
                one, zero,
                zero, Complex64::new(-1.0, 0.0),
            ]),
            // (123): (x,y,z)→(z,x,y)
            DMatrix::from_row_slice(2, 2, &[
                neg_half, sqrt3_4,
                neg_sqrt3_4, neg_half,
            ]),
            // (132): (x,y,z)→(y,z,x)
            DMatrix::from_row_slice(2, 2, &[
                neg_half, neg_sqrt3_4,
                sqrt3_4, neg_half,
            ]),
        ];
        Self::new(matrices)
    }

    /// Direct sum of two representations.
    pub fn direct_sum(&self, other: &Representation) -> Representation {
        assert_eq!(self.matrices.len(), other.matrices.len());
        let matrices: Vec<_> = self
            .matrices
            .iter()
            .zip(&other.matrices)
            .map(|(a, b)| {
                let n = a.nrows() + b.nrows();
                let mut m = DMatrix::zeros(n, n);
                m.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
                m.view_mut((a.nrows(), a.ncols()), (b.nrows(), b.ncols())).copy_from(b);
                m
            })
            .collect();
        Representation::new(matrices)
    }

    /// Verify this is a valid representation of the given group.
    pub fn verify_homomorphism(&self, group: &FiniteGroup) -> bool {
        let e = group.identity();
        // Check identity maps to identity matrix
        let id_matrix = DMatrix::identity(self.dimension, self.dimension);
        if (self.matrices[e].clone() - &id_matrix).norm() > 1e-10 {
            return false;
        }
        // Check ρ(g*h) = ρ(g) * ρ(h)
        for g in 0..group.order() {
            for h in 0..group.order() {
                let gh = group.multiply(g, h);
                let prod = &self.matrices[g] * &self.matrices[h];
                if (prod - &self.matrices[gh]).norm() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Get matrix for element at index.
    pub fn matrix(&self, i: usize) -> &DMatrix<Complex64> {
        &self.matrices[i]
    }

    /// Compute trace of ρ(g) for element g.
    pub fn trace(&self, i: usize) -> Complex64 {
        self.matrices[i].trace()
    }
}
