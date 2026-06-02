//! # representation-theory
//!
//! Representation theory of finite groups: how groups act on vector spaces.
//!
//! Covers: group representations, character theory, irreducible representations,
//! Maschke's theorem, induced representations, tensor products, Young tableaux,
//! and symmetry analysis.

pub mod group;
pub mod representation;
pub mod character;
pub mod irreducible;
pub mod decomposition;
pub mod induced;
pub mod tensor;
pub mod young;
pub mod agent_symmetry;

pub use group::*;
pub use representation::*;
pub use character::*;
pub use irreducible::*;
pub use decomposition::*;
pub use induced::*;
pub use tensor::*;
pub use young::*;
pub use agent_symmetry::*;
