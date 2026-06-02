# representation-theory

Representation theory in Rust. Groups acting on vector spaces.

A library for computing group representations, character tables, irreducible decompositions, induced representations, Young tableaux, Clebsch-Gordan coefficients, and symmetry analysis.

## What This Does

- **Group algebra** — define finite groups via Cayley tables (with built-in generators for ℤ/nℤ, S₃, and the Klein four-group)
- **Representations** — construct complex matrix representations ρ: G → GL(n, ℂ) and verify homomorphism properties
- **Character theory** — compute characters χ(g) = Tr(ρ(g)), build character tables, and verify both orthogonality relations
- **Irreducible decomposition** — decompose any representation into irreducibles using inner products of characters (Maschke's theorem)
- **Induced representations** — compute induced characters and representations Indᴴᴳ(χ), verify Frobenius reciprocity
- **Tensor products** — Kronecker products of representations, Clebsch-Gordan coefficients for SU(2), and symmetric powers
- **Young tableaux** — partitions, hook-length formula, standard tableaux, and Murnaghan-Nakayama rule for Sₙ characters
- **Symmetry analysis** — decompose multi-particle state spaces under group actions, find invariant subspaces, compute symmetrizers/antisymmetrizers

## Install

```toml
[dependencies]
representation-theory = "0.1.0"
```

Requires Rust 2021 edition. Depends on `nalgebra`, `num-complex`, and `serde`.

## Quick Start

```rust
use representation_theory::*;

// Create the symmetric group S₃
let s3 = group::FiniteGroup::s3();

// Build the three irreducible representations
let trivial = Representation::trivial(s3.order());
let sign = Representation::sign_s3();
let standard = Representation::standard_s3();

// Compute characters and build character table
let chi_triv = Character::from_representation(&trivial);
let chi_sign = Character::from_representation(&sign);
let chi_std = Character::from_representation(&standard);
let table = CharacterTable::new(&s3, vec![chi_triv, chi_sign, chi_std]);
assert!(table.verify_first_orthogonality());

// Decompose a tensor product into irreducibles
let tensor = tensor::tensor_product(&standard, &standard);
let irr_reps = vec![trivial, sign, standard];
let mults = decomposition::decompose(&tensor, &irr_reps, &s3);
// → [1, 1, 1]  (trivial ⊕ sign ⊕ standard)

// Young tableaux: dimension of the S₆ irrep for partition [3,2,1]
let diagram = young::YoungDiagram::new(vec![3, 2, 1]);
assert_eq!(diagram.dimension(), 16);
```

## License

MIT OR Apache-2.0
