//! Core group types and operations.

use nalgebra::{DMatrix, ComplexField};
use num_complex::Complex64;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;

/// A finite group represented by its Cayley table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FiniteGroup {
    /// Ordered list of group element labels.
    pub elements: Vec<String>,
    /// Cayley table: table[i][j] = index of (element_i * element_j).
    pub table: Vec<Vec<usize>>,
}

impl FiniteGroup {
    /// Create a group from elements and a Cayley table.
    pub fn new(elements: Vec<String>, table: Vec<Vec<usize>>) -> Self {
        assert_eq!(elements.len(), table.len());
        Self { elements, table }
    }

    /// Order of the group.
    pub fn order(&self) -> usize {
        self.elements.len()
    }

    /// Multiply two elements by index.
    pub fn multiply(&self, i: usize, j: usize) -> usize {
        self.table[i][j]
    }

    /// Identity element index.
    pub fn identity(&self) -> usize {
        // Find e such that e*g = g for all g
        for candidate in 0..self.order() {
            let is_identity = (0..self.order())
                .all(|j| self.table[candidate][j] == j && self.table[j][candidate] == j);
            if is_identity {
                return candidate;
            }
        }
        panic!("No identity element found");
    }

    /// Inverse of element at index i.
    pub fn inverse(&self, i: usize) -> usize {
        let e = self.identity();
        for j in 0..self.order() {
            if self.table[i][j] == e && self.table[j][i] == e {
                return j;
            }
        }
        panic!("No inverse found for element {}", i);
    }

    /// Conjugacy classes: partition elements into conjugacy classes.
    pub fn conjugacy_classes(&self) -> Vec<Vec<usize>> {
        let n = self.order();
        let mut visited = vec![false; n];
        let mut classes = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }
            let mut cls = Vec::new();
            for j in 0..n {
                // Check if j is conjugate to i: exists g such that g*i*g^{-1} = j
                if self.is_conjugate(i, j) {
                    cls.push(j);
                    visited[j] = true;
                }
            }
            classes.push(cls);
        }
        classes
    }

    /// Check if elements i and j are conjugate.
    fn is_conjugate(&self, i: usize, j: usize) -> bool {
        let e = self.identity();
        for g in 0..self.order() {
            let g_inv = self.inverse(g);
            // g * i * g^{-1}
            let conjugated = self.table[self.table[g][i]][g_inv];
            if conjugated == j {
                return true;
            }
        }
        false
    }

    /// Generate the cyclic group Z/nZ.
    pub fn cyclic(n: usize) -> Self {
        let elements: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
        let table: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).map(|j| (i + j) % n).collect())
            .collect();
        Self::new(elements, table)
    }

    /// Generate the symmetric group S_3.
    pub fn s3() -> Self {
        // Elements: e, (12), (13), (23), (123), (132)
        // Using indices: 0=e, 1=(12), 2=(13), 3=(23), 4=(123), 5=(132)
        let elements = vec![
            "e".into(), "(12)".into(), "(13)".into(), "(23)".into(),
            "(123)".into(), "(132)".into(),
        ];
        // Composition table for S3
        let table = vec![
            vec![0, 1, 2, 3, 4, 5], // e * x
            vec![1, 0, 4, 5, 2, 3], // (12) * x
            vec![2, 5, 0, 4, 3, 1], // (13) * x
            vec![3, 4, 5, 0, 1, 2], // (23) * x
            vec![4, 3, 1, 2, 5, 0], // (123) * x
            vec![5, 2, 3, 1, 0, 4], // (132) * x
        ];
        Self::new(elements, table)
    }

    /// Generate the Klein four-group Z/2Z × Z/2Z.
    pub fn klein_four() -> Self {
        let elements = vec!["e".into(), "a".into(), "b".into(), "ab".into()];
        // e=0, a=1, b=2, ab=3
        let table = vec![
            vec![0, 1, 2, 3],
            vec![1, 0, 3, 2],
            vec![2, 3, 0, 1],
            vec![3, 2, 1, 0],
        ];
        Self::new(elements, table)
    }
}

/// Trait for groups that can provide element matrices for representations.
pub trait GroupOps {
    /// Get the group order.
    fn order(&self) -> usize;
    /// Multiply two group elements by index.
    fn multiply(&self, i: usize, j: usize) -> usize;
    /// Get the identity element index.
    fn identity(&self) -> usize;
    /// Get the inverse of element i.
    fn inverse(&self, i: usize) -> usize;
    /// Get conjugacy classes.
    fn conjugacy_classes(&self) -> Vec<Vec<usize>>;
}

impl GroupOps for FiniteGroup {
    fn order(&self) -> usize { self.order() }
    fn multiply(&self, i: usize, j: usize) -> usize { self.multiply(i, j) }
    fn identity(&self) -> usize { self.identity() }
    fn inverse(&self, i: usize) -> usize { self.inverse(i) }
    fn conjugacy_classes(&self) -> Vec<Vec<usize>> { self.conjugacy_classes() }
}
