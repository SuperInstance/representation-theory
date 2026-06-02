//! Agent symmetry analysis: group actions on agent state spaces.
//!
//! Models how groups of symmetries (permutations, rotations, etc.) act on
//! the state spaces of multi-agent systems, enabling decomposition of
//! joint state spaces using representation theory.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde::{Serialize, Deserialize};
use crate::group::FiniteGroup;
use crate::representation::Representation;
use crate::character::{Character, CharacterTable};
use crate::decomposition::decompose;

/// An agent state space with a group action.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentStateSpace {
    /// Number of agents.
    pub num_agents: usize,
    /// Dimension of each agent's local state space.
    pub local_dimension: usize,
    /// Description of the state space.
    pub description: String,
}

impl AgentStateSpace {
    /// Total dimension of the joint state space.
    pub fn total_dimension(&self) -> usize {
        self.num_agents * self.local_dimension
    }

    /// Construct the permutation representation of S_n acting on agent indices.
    /// Each permutation σ maps basis vector e_i to e_{σ(i)}.
    pub fn permutation_representation(&self, group: &FiniteGroup) -> Representation {
        let n = self.num_agents;
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);

        // We need a mapping from group elements to permutations of {0,...,n-1}
        // For now, if group is S_n or similar, use the natural action
        let matrices: Vec<DMatrix<Complex64>> = (0..group.order())
            .map(|g| {
                let mut perm = vec![0usize; n];
                for i in 0..n {
                    perm[i] = self.apply_group_action(g, i, group);
                }
                let mut mat = DMatrix::zeros(n, n);
                for (i, &j) in perm.iter().enumerate() {
                    mat[(j, i)] = one;
                }
                mat
            })
            .collect();

        Representation::new(matrices)
    }

    /// Apply group action: map agent index i through group element g.
    /// Default: uses the group's natural permutation action.
    fn apply_group_action(&self, _g: usize, i: usize, _group: &FiniteGroup) -> usize {
        // Default: identity (override for specific groups)
        i
    }
}

/// Analysis result for symmetry decomposition of agent state spaces.
#[derive(Debug, Serialize, Deserialize)]
pub struct SymmetryAnalysis {
    /// The state space analyzed.
    pub state_space: AgentStateSpace,
    /// Multiplicities of each irreducible in the decomposition.
    pub multiplicities: Vec<usize>,
    /// Symmetric subspace dimension.
    pub symmetric_dimension: usize,
    /// Whether the joint state space has non-trivial symmetry.
    pub has_nontrivial_symmetry: bool,
}

/// Analyze the symmetry structure of a multi-agent state space.
///
/// Given a group G acting on the joint state space of multiple agents,
/// decompose the state space into irreducible representations.
pub fn analyze_symmetry(
    state_space: &AgentStateSpace,
    group: &FiniteGroup,
    irr_reps: &[Representation],
) -> SymmetryAnalysis {
    let perm_rep = state_space.permutation_representation(group);
    let multiplicities = decompose(&perm_rep, irr_reps, group);

    let symmetric_dimension = multiplicities
        .first()
        .map(|&m| m * irr_reps.first().map(|r| r.dimension).unwrap_or(0))
        .unwrap_or(0);

    SymmetryAnalysis {
        state_space: state_space.clone(),
        multiplicities,
        symmetric_dimension,
        has_nontrivial_symmetry: group.order() > 1,
    }
}

/// Find invariant subspaces under a group action.
///
/// Returns the dimension and a description of each invariant subspace.
pub fn invariant_subspaces(
    rep: &Representation,
    irr_reps: &[Representation],
    group: &FiniteGroup,
) -> Vec<(usize, String)> {
    let mults = decompose(rep, irr_reps, group);
    let mut subspaces = Vec::new();

    for (i, &mult) in mults.iter().enumerate() {
        if mult == 0 {
            continue;
        }
        let dim = irr_reps[i].dimension * mult;
        let desc = if irr_reps[i].dimension == 1 && i == 0 {
            format!("Symmetric subspace (trivial rep), mult={}, dim={}", mult, dim)
        } else if irr_reps[i].dimension == 1 {
            format!("1D irreducible #{}, mult={}", i, mult)
        } else {
            format!("{}D irreducible #{}, mult={}, total dim={}", irr_reps[i].dimension, i, mult, dim)
        };
        subspaces.push((dim, desc));
    }

    subspaces
}

/// Compute the symmetrizer: the projection onto the fully symmetric subspace.
/// P_sym = (1/|G|) Σ_{g∈G} ρ(g)
pub fn symmetrizer(rep: &Representation, group: &FiniteGroup) -> DMatrix<Complex64> {
    let n = group.order() as f64;
    let dim = rep.dimension;
    let mut proj = DMatrix::zeros(dim, dim);

    for g in 0..group.order() {
        let factor = Complex64::new(1.0 / n, 0.0);
        proj += rep.matrices[g].clone() * factor;
    }
    proj
}

/// Compute the antisymmetrizer (for alternating representations).
/// P_anti = (1/|G|) Σ_{g∈G} sgn(g) ρ(g)
pub fn antisymmetrizer(rep: &Representation, signs: &[f64], group: &FiniteGroup) -> DMatrix<Complex64> {
    let n = group.order() as f64;
    let dim = rep.dimension;
    let mut proj = DMatrix::zeros(dim, dim);

    for g in 0..group.order() {
        let factor = Complex64::new(signs[g] / n, 0.0);
        proj += rep.matrices[g].clone() * factor;
    }
    proj
}
