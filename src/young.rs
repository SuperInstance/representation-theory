//! Young tableaux and symmetric group representations.

use serde::{Serialize, Deserialize};

/// A Young diagram (partition) represented as non-increasing row lengths.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct YoungDiagram {
    pub rows: Vec<usize>,
}

impl YoungDiagram {
    pub fn new(rows: Vec<usize>) -> Self {
        for i in 1..rows.len() {
            assert!(rows[i] <= rows[i - 1], "Partition must be non-increasing");
        }
        Self { rows }
    }

    pub fn n(&self) -> usize { self.rows.iter().sum() }
    pub fn num_rows(&self) -> usize { self.rows.len() }

    pub fn conjugate(&self) -> YoungDiagram {
        if self.rows.is_empty() { return YoungDiagram::new(vec![]); }
        let max_cols = self.rows[0];
        let conj_rows: Vec<usize> = (1..=max_cols)
            .map(|col| self.rows.iter().filter(|&&r| r >= col).count())
            .collect();
        YoungDiagram::new(conj_rows)
    }

    pub fn hook_length(&self, row: usize, col: usize) -> usize {
        let right = self.rows[row] - col - 1;
        let below = self.rows.iter().skip(row + 1).filter(|&&r| r > col).count();
        1 + right + below
    }

    pub fn hook_product(&self) -> usize {
        let mut prod = 1usize;
        for (i, &row_len) in self.rows.iter().enumerate() {
            for j in 0..row_len { prod *= self.hook_length(i, j); }
        }
        prod
    }

    pub fn dimension(&self) -> usize {
        factorial(self.n()) / self.hook_product()
    }

    pub fn partitions(n: usize) -> Vec<YoungDiagram> {
        if n == 0 { return vec![YoungDiagram::new(vec![])]; }
        let mut result = Vec::new();
        let mut partition = vec![0usize; n];
        generate_partitions(n, n, 0, &mut partition, &mut result);
        result
    }
}

fn generate_partitions(n: usize, max: usize, idx: usize, buf: &mut Vec<usize>, out: &mut Vec<YoungDiagram>) {
    if n == 0 {
        out.push(YoungDiagram::new(buf[..idx].to_vec()));
        return;
    }
    let mut i = max.min(n);
    while i >= 1 {
        buf[idx] = i;
        generate_partitions(n - i, i, idx + 1, buf, out);
        if i == 1 { break; }
        i -= 1;
    }
}

fn factorial(n: usize) -> usize { (1..=n).product() }

/// A Young tableau: filling of a Young diagram with numbers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct YoungTableau {
    pub diagram: YoungDiagram,
    pub entries: Vec<Vec<usize>>,
}

impl YoungTableau {
    pub fn new(diagram: YoungDiagram, entries: Vec<Vec<usize>>) -> Self {
        assert_eq!(entries.len(), diagram.rows.len());
        for (i, row) in entries.iter().enumerate() { assert_eq!(row.len(), diagram.rows[i]); }
        Self { diagram, entries }
    }

    /// Check if this is a standard Young tableau (strictly increasing in rows and columns, entries 1..n).
    pub fn is_standard(&self) -> bool {
        let n = self.diagram.n();
        let mut seen = vec![false; n + 1];
        for row in &self.entries {
            for &val in row {
                if val == 0 || val > n || seen[val] { return false; }
                seen[val] = true;
            }
        }
        for i in 1..=n { if !seen[i] { return false; } }
        for row in &self.entries {
            for j in 1..row.len() { if row[j] <= row[j - 1] { return false; } }
        }
        let max_cols = self.diagram.rows.first().copied().unwrap_or(0);
        for col in 0..max_cols {
            for i in 1..self.diagram.rows.len() {
                if col < self.entries[i].len() && col < self.entries[i - 1].len() {
                    if self.entries[i][col] <= self.entries[i - 1][col] { return false; }
                }
            }
        }
        true
    }

    /// Generate all standard Young tableaux of a given shape.
    pub fn standard_tableaux(diagram: &YoungDiagram) -> Vec<YoungTableau> {
        let n = diagram.n();
        let mut result = Vec::new();
        let mut entries: Vec<Vec<usize>> = diagram.rows.iter().map(|&len| vec![0; len]).collect();
        fill_standard(diagram, &mut entries, n, 1, &mut result);
        result
    }
}

fn fill_standard(d: &YoungDiagram, e: &mut Vec<Vec<usize>>, n: usize, num: usize, out: &mut Vec<YoungTableau>) {
    if num > n {
        out.push(YoungTableau::new(d.clone(), e.clone()));
        return;
    }
    for i in 0..d.rows.len() {
        for j in 0..d.rows[i] {
            if e[i][j] != 0 { continue; }
            let rok = j == 0 || e[i][j - 1] != 0;
            let cok = i == 0 || (j < e[i - 1].len() && e[i - 1][j] != 0);
            if rok && cok {
                e[i][j] = num;
                fill_standard(d, e, n, num + 1, out);
                e[i][j] = 0;
            }
        }
    }
}

/// Compute the character of the irreducible S_n representation via the Murnaghan-Nakayama rule.
pub fn symmetric_group_character(partition: &[usize], cycle_type: &[usize]) -> i64 {
    let pn: usize = partition.iter().sum();
    let cn: usize = cycle_type.iter().sum();
    if pn != cn { return 0; }
    if cycle_type.is_empty() {
        return if partition.is_empty() { 1 } else { 0 };
    }
    mn_rule(partition, cycle_type)
}

fn mn_rule(partition: &[usize], cycle_type: &[usize]) -> i64 {
    if cycle_type.is_empty() {
        return if partition.is_empty() { 1 } else { 0 };
    }
    if partition.is_empty() { return 0; }
    let r = cycle_type[0];
    let remaining = &cycle_type[1..];
    let mut result: i64 = 0;
    for (mu, leg) in find_border_strips(partition, r) {
        let sign = if leg % 2 == 0 { 1i64 } else { -1i64 };
        result += sign * mn_rule(&mu, remaining);
    }
    result
}

/// Find all border strips of given size in the partition.
fn find_border_strips(lambda: &[usize], size: usize) -> Vec<(Vec<usize>, usize)> {
    let n = lambda.len();
    let mut results = Vec::new();

    for top_row in 0..n {
        for bottom_row in 0..=top_row {
            let mut mu = lambda.to_vec();
            let mut total = 0usize;
            let leg = top_row - bottom_row;
            let mut valid = true;

            let top_shelf = if top_row + 1 < n { lambda[top_row + 1] } else { 0 };
            let top_removable = lambda[top_row].saturating_sub(top_shelf);

            if top_row == bottom_row {
                if lambda[top_row] < size { continue; }
                let new_width = lambda[top_row] - size;
                if new_width < top_shelf { continue; }
                mu[top_row] = new_width;
                total = size;
            } else {
                if top_removable == 0 { continue; }
                mu[top_row] = top_shelf;
                total += top_removable;

                for row in (bottom_row + 1..top_row).rev() {
                    let shelf_below = mu[row + 1];
                    let removable = lambda[row].saturating_sub(shelf_below);
                    if removable == 0 { valid = false; break; }
                    mu[row] = shelf_below;
                    total += removable;
                }
                if !valid { continue; }

                if total >= size { continue; }
                let needed = size - total;
                if needed > lambda[bottom_row] { continue; }
                let new_width = lambda[bottom_row] - needed;
                let shelf_below = if bottom_row + 1 < n { mu[bottom_row + 1] } else { 0 };
                if new_width < shelf_below { continue; }
                mu[bottom_row] = new_width;
                total += needed;
            }

            if total != size { continue; }

            while mu.last() == Some(&0) { mu.pop(); }
            let mut is_valid = true;
            for i in 1..mu.len() { if mu[i] > mu[i - 1] { is_valid = false; break; } }
            if !is_valid { continue; }

            // Connectivity
            let mut connected = true;
            for i in bottom_row..top_row {
                let s_i = if i < mu.len() { mu[i] } else { 0 };
                let s_i1 = if i + 1 < mu.len() { mu[i + 1] } else { 0 };
                if !(s_i < lambda[i + 1] && s_i1 < lambda[i]) { connected = false; break; }
            }
            if !connected { continue; }

            // No 2×2 blocks
            let mut has_2x2 = false;
            for i in bottom_row..top_row {
                let r_i = lambda[i] - if i < mu.len() { mu[i] } else { 0 };
                let r_i1 = lambda[i + 1] - if i + 1 < mu.len() { mu[i + 1] } else { 0 };
                if r_i >= 2 && r_i1 >= 2 {
                    let s_i = if i < mu.len() { mu[i] } else { 0 };
                    let s_i1 = if i + 1 < mu.len() { mu[i + 1] } else { 0 };
                    if s_i < lambda[i + 1] && s_i1 < lambda[i] { has_2x2 = true; break; }
                }
            }
            if has_2x2 { continue; }

            results.push((mu, leg));
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_length_formula() {
        let d = YoungDiagram::new(vec![3, 2, 1]);
        assert_eq!(d.hook_product(), 45);
        assert_eq!(d.dimension(), 16);
    }

    #[test]
    fn test_mn_s3() {
        // Trivial rep [3]
        assert_eq!(symmetric_group_character(&[3], &[3]), 1);
        assert_eq!(symmetric_group_character(&[3], &[2, 1]), 1);
        assert_eq!(symmetric_group_character(&[3], &[1, 1, 1]), 1);
        // Sign rep [1,1,1]
        assert_eq!(symmetric_group_character(&[1, 1, 1], &[3]), 1);
        assert_eq!(symmetric_group_character(&[1, 1, 1], &[2, 1]), -1);
        assert_eq!(symmetric_group_character(&[1, 1, 1], &[1, 1, 1]), 1);
        // Standard rep [2,1]
        assert_eq!(symmetric_group_character(&[2, 1], &[1, 1, 1]), 2);
        assert_eq!(symmetric_group_character(&[2, 1], &[2, 1]), 0);
        assert_eq!(symmetric_group_character(&[2, 1], &[3]), -1);
    }
}
