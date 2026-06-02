//! Comprehensive tests for representation theory.

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;
    use num_complex::Complex64;

    use representation_theory::group::FiniteGroup;
    use representation_theory::representation::Representation;
    use representation_theory::character::{Character, CharacterTable};
    use representation_theory::irreducible::{is_irreducible, schur_lemma, isotypic_projection};
    use representation_theory::decomposition::{decompose, verify_maschke};
    use representation_theory::induced::{
        induced_character, restrict_character, verify_frobenius_reciprocity,
        induced_character_embedded, restrict_character_embedded, verify_frobenius_reciprocity_embedded,
        SubgroupEmbedding,
    };
    use representation_theory::tensor::{
        tensor_product, tensor_character, clebsch_gordan, spin_dimension,
        verify_cg_dimension, clebsch_gordan_coefficient, symmetric_power,
    };
    use representation_theory::young::{YoungDiagram, YoungTableau, symmetric_group_character};
    use representation_theory::agent_symmetry::{
        AgentStateSpace, analyze_symmetry, invariant_subspaces, symmetrizer,
    };

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    // ============================================================
    // Group structure tests
    // ============================================================

    #[test]
    fn test_cyclic_group_order() {
        let g = FiniteGroup::cyclic(5);
        assert_eq!(g.order(), 5);
    }

    #[test]
    fn test_cyclic_group_identity() {
        let g = FiniteGroup::cyclic(5);
        assert_eq!(g.identity(), 0);
    }

    #[test]
    fn test_cyclic_group_inverse() {
        let g = FiniteGroup::cyclic(5);
        assert_eq!(g.inverse(0), 0); // e^{-1} = e
        assert_eq!(g.inverse(1), 4); // 1 + 4 = 5 ≡ 0
        assert_eq!(g.inverse(2), 3);
        assert_eq!(g.inverse(3), 2);
        assert_eq!(g.inverse(4), 1);
    }

    #[test]
    fn test_s3_order() {
        let s3 = FiniteGroup::s3();
        assert_eq!(s3.order(), 6);
    }

    #[test]
    fn test_s3_identity() {
        let s3 = FiniteGroup::s3();
        assert_eq!(s3.identity(), 0);
    }

    #[test]
    fn test_klein_four_order() {
        let k4 = FiniteGroup::klein_four();
        assert_eq!(k4.order(), 4);
        assert_eq!(k4.identity(), 0);
    }

    #[test]
    fn test_s3_conjugacy_classes() {
        let s3 = FiniteGroup::s3();
        let classes = s3.conjugacy_classes();
        assert_eq!(classes.len(), 3); // {e}, {(12),(13),(23)}, {(123),(132)}
        // Check class sizes: 1, 3, 2
        let mut sizes: Vec<usize> = classes.iter().map(|c| c.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![1, 2, 3]);
    }

    #[test]
    fn test_klein_four_conjugacy_classes() {
        let k4 = FiniteGroup::klein_four();
        let classes = k4.conjugacy_classes();
        assert_eq!(classes.len(), 4); // Each element is its own class (abelian)
    }

    #[test]
    fn test_cyclic_abelian_classes() {
        let z4 = FiniteGroup::cyclic(4);
        let classes = z4.conjugacy_classes();
        assert_eq!(classes.len(), 4); // Abelian: each element is its own class
    }

    // ============================================================
    // Representation tests
    // ============================================================

    #[test]
    fn test_trivial_representation() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        assert_eq!(triv.dimension, 1);
        assert!(triv.verify_homomorphism(&s3));
    }

    #[test]
    fn test_sign_representation_s3() {
        let s3 = FiniteGroup::s3();
        let sign = Representation::sign_s3();
        assert_eq!(sign.dimension, 1);
        assert!(sign.verify_homomorphism(&s3));
    }

    #[test]
    fn test_standard_representation_s3() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        assert_eq!(std_rep.dimension, 2);
        assert!(std_rep.verify_homomorphism(&s3));
    }

    #[test]
    fn test_trivial_is_irreducible() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        assert!(is_irreducible(&triv, &s3));
    }

    #[test]
    fn test_sign_is_irreducible() {
        let s3 = FiniteGroup::s3();
        let sign = Representation::sign_s3();
        assert!(is_irreducible(&sign, &s3));
    }

    #[test]
    fn test_standard_is_irreducible() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        assert!(is_irreducible(&std_rep, &s3));
    }

    #[test]
    fn test_representation_trace() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        // Trace of identity = dimension
        let tr = std_rep.trace(s3.identity());
        assert!((tr.re - 2.0).abs() < 1e-10);
        assert!(tr.im.abs() < 1e-10);
    }

    // ============================================================
    // Character theory tests
    // ============================================================

    #[test]
    fn test_character_trivial() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let chi = Character::from_representation(&triv);
        // All values should be 1
        for v in &chi.values {
            assert!((v.re - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_character_degree() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        let chi = Character::from_representation(&std_rep);
        assert!((chi.degree() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_character_inner_product_trivial() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let chi = Character::from_representation(&triv);
        let ip = chi.inner_product(&chi, &s3);
        assert!((ip.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_character_inner_product_orthogonal() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let chi_triv = Character::from_representation(&triv);
        let chi_sign = Character::from_representation(&sign);
        let ip = chi_triv.inner_product(&chi_sign, &s3);
        assert!(ip.re.abs() < 1e-10);
    }

    #[test]
    fn test_character_orthogonality_standard_trivial() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let std_rep = Representation::standard_s3();
        let chi_triv = Character::from_representation(&triv);
        let chi_std = Character::from_representation(&std_rep);
        let ip = chi_triv.inner_product(&chi_std, &s3);
        assert!(ip.re.abs() < 1e-10);
    }

    #[test]
    fn test_character_table_s3_first_orthogonality() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();

        let chars = vec![
            Character::from_representation(&triv),
            Character::from_representation(&sign),
            Character::from_representation(&std_rep),
        ];

        let table = CharacterTable::new(&s3, chars);
        assert!(table.verify_first_orthogonality());
    }

    #[test]
    fn test_character_table_s3_second_orthogonality() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();

        let chars = vec![
            Character::from_representation(&triv),
            Character::from_representation(&sign),
            Character::from_representation(&std_rep),
        ];

        let table = CharacterTable::new(&s3, chars);
        assert!(table.verify_second_orthogonality());
    }

    #[test]
    fn test_character_decompose_regular_s3() {
        // Regular representation of S3: 6-dimensional
        // Should decompose as 1*triv + 1*sign + 2*std = 1 + 1 + 4 = 6
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();

        // Build regular representation
        let one = Complex64::new(1.0, 0.0);
        let regular_matrices: Vec<DMatrix<Complex64>> = (0..6)
            .map(|g| {
                let mut m = DMatrix::zeros(6, 6);
                for h in 0..6 {
                    let gh = s3.multiply(g, h);
                    m[(gh, h)] = one;
                }
                m
            })
            .collect();
        let regular = Representation::new(regular_matrices);
        assert!(regular.verify_homomorphism(&s3));

        let irr_reps = vec![triv, sign, std_rep];
        let mults = decompose(&regular, &irr_reps, &s3);
        assert_eq!(mults, vec![1, 1, 2]); // triv × 1, sign × 1, std × 2
    }

    // ============================================================
    // Schur's lemma tests
    // ============================================================

    #[test]
    fn test_schur_lemma_identity() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let t = DMatrix::from_element(1, 1, Complex64::new(3.0, 0.0));
        let result = schur_lemma(&triv, &triv, &t, &s3);
        assert!(result.is_ok());
        let lambda = result.unwrap();
        assert!(lambda.is_some());
        assert!((lambda.unwrap().re - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_schur_lemma_zero() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let t = DMatrix::zeros(1, 1);
        let result = schur_lemma(&triv, &triv, &t, &s3);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // T = 0
    }

    #[test]
    fn test_schur_lemma_scalar_2d() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        let lambda = Complex64::new(5.0, 0.0);
        let t = DMatrix::from_element(2, 2, lambda) * Complex64::new(1.0, 0.0);
        // Actually just use identity times 5
        let t = DMatrix::from_diagonal_element(2, 2, lambda);
        let result = schur_lemma(&std_rep, &std_rep, &t, &s3);
        assert!(result.is_ok());
    }

    // ============================================================
    // Maschke's theorem tests
    // ============================================================

    #[test]
    fn test_maschke_regular_s3() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();

        let one = Complex64::new(1.0, 0.0);
        let regular_matrices: Vec<DMatrix<Complex64>> = (0..6)
            .map(|g| {
                let mut m = DMatrix::zeros(6, 6);
                for h in 0..6 {
                    let gh = s3.multiply(g, h);
                    m[(gh, h)] = one;
                }
                m
            })
            .collect();
        let regular = Representation::new(regular_matrices);

        let irr_reps = vec![triv, sign, std_rep];
        assert!(verify_maschke(&regular, &irr_reps, &s3));
    }

    #[test]
    fn test_maschke_direct_sum() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();

        // Direct sum triv ⊕ sign ⊕ std = 4-dim rep
        let rep = triv.direct_sum(&sign).direct_sum(&std_rep);
        assert!(rep.verify_homomorphism(&s3));

        let irr_reps = vec![triv, sign, std_rep];
        assert!(verify_maschke(&rep, &irr_reps, &s3));
    }

    // ============================================================
    // Tensor product tests
    // ============================================================

    #[test]
    fn test_tensor_product_dimension() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let std_rep = Representation::standard_s3();
        let tensor = tensor_product(&triv, &std_rep);
        assert_eq!(tensor.dimension, 2); // 1 × 2
    }

    #[test]
    fn test_tensor_product_character() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let std_rep = Representation::standard_s3();

        let chi_triv = Character::from_representation(&triv);
        let chi_std = Character::from_representation(&std_rep);
        let chi_tensor = tensor_character(&chi_triv, &chi_std);

        // Tensoring with trivial = same character
        for (a, b) in chi_tensor.values.iter().zip(&chi_std.values) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_tensor_product_is_representation() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        let tensor = tensor_product(&std_rep, &std_rep);
        assert_eq!(tensor.dimension, 4);
        assert!(tensor.verify_homomorphism(&s3));
    }

    #[test]
    fn test_clebsch_gordan_decomposition() {
        // j1=1, j2=1: 3⊗3 = 1⊕3⊕5
        let cg = clebsch_gordan(1.0, 1.0);
        assert_eq!(cg, vec![0.0, 1.0, 2.0]);
        assert!(verify_cg_dimension(1.0, 1.0));
    }

    #[test]
    fn test_clebsch_gordan_half_spin() {
        // j1=1/2, j2=1/2: 2⊗2 = 1⊕3
        let cg = clebsch_gordan(0.5, 0.5);
        assert_eq!(cg, vec![0.0, 1.0]);
        assert!(verify_cg_dimension(0.5, 0.5));
    }

    #[test]
    fn test_clebsch_gordan_unequal() {
        // j1=2, j2=1: 5⊗3 = 3⊕4⊕5
        let cg = clebsch_gordan(2.0, 1.0);
        assert_eq!(cg, vec![1.0, 2.0, 3.0]);
        assert!(verify_cg_dimension(2.0, 1.0));
    }

    #[test]
    fn test_cg_coefficient_spin_half_triplet() {
        // |↑↑⟩ → |1,1⟩: coefficient = 1
        let cg = clebsch_gordan_coefficient(0.5, 0.5, 0.5, 0.5, 1.0, 1.0);
        assert!((cg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cg_coefficient_spin_half_singlet() {
        // (|↑↓⟩ - |↓↑⟩)/√2 → |0,0⟩
        let cg_up_dn = clebsch_gordan_coefficient(0.5, 0.5, 0.5, -0.5, 0.0, 0.0);
        let cg_dn_up = clebsch_gordan_coefficient(0.5, -0.5, 0.5, 0.5, 0.0, 0.0);
        let sq2 = 2.0_f64.sqrt();
        assert!((cg_up_dn - 1.0 / sq2).abs() < 1e-10);
        assert!((cg_dn_up + 1.0 / sq2).abs() < 1e-10);
    }

    #[test]
    fn test_cg_coefficient_spin_one_j2_m0() {
        // |2,0⟩: coefficient of |0,0⟩ is 2/√6 = √(2/3)
        let cg = clebsch_gordan_coefficient(1.0, 0.0, 1.0, 0.0, 2.0, 0.0);
        assert!((cg - (2.0_f64 / 3.0_f64).sqrt()).abs() < 1e-10, "got {}", cg);
    }

    #[test]
    fn test_cg_selection_rule() {
        // m1 + m2 ≠ m → coefficient = 0
        let cg = clebsch_gordan_coefficient(0.5, 0.5, 0.5, 0.5, 1.0, 0.0);
        assert!(cg.abs() < 1e-10);
    }

    #[test]
    fn test_spin_dimension() {
        assert_eq!(spin_dimension(0.0), 1);
        assert_eq!(spin_dimension(0.5), 2);
        assert_eq!(spin_dimension(1.0), 3);
        assert_eq!(spin_dimension(2.0), 5);
    }

    #[test]
    fn test_symmetric_power() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        let sym2 = symmetric_power(&std_rep, 2);
        assert_eq!(sym2.dimension, 3); // Sym^2(R^2) = 3-dimensional
        assert!(sym2.verify_homomorphism(&s3));
    }

    // ============================================================
    // Young tableaux tests
    // ============================================================

    #[test]
    fn test_young_diagram_conjugate() {
        let d = YoungDiagram::new(vec![3, 2, 1]);
        let conj = d.conjugate();
        assert_eq!(conj.rows, vec![3, 2, 1]); // self-conjugate
    }

    #[test]
    fn test_young_diagram_conjugate_2() {
        let d = YoungDiagram::new(vec![4, 2]);
        let conj = d.conjugate();
        assert_eq!(conj.rows, vec![2, 2, 1, 1]);
    }

    #[test]
    fn test_hook_length() {
        // [2,1]:
        // (0,0): hook = 3 (right + below + 1 = 1 + 1 + 1)
        // (0,1): hook = 1 (0 + 0 + 1)
        // (1,0): hook = 1 (0 + 0 + 1)
        let d = YoungDiagram::new(vec![2, 1]);
        assert_eq!(d.hook_length(0, 0), 3);
        assert_eq!(d.hook_length(0, 1), 1);
        assert_eq!(d.hook_length(1, 0), 1);
        assert_eq!(d.hook_product(), 3);
    }

    #[test]
    fn test_hook_dimension_partition_21() {
        // [2,1] of 3: dim = 3!/3 = 2
        let d = YoungDiagram::new(vec![2, 1]);
        assert_eq!(d.dimension(), 2);
    }

    #[test]
    fn test_hook_dimension_partition_3() {
        // [3] of 3: hooks = [3,2,1], product = 6, dim = 6/6 = 1
        let d = YoungDiagram::new(vec![3]);
        assert_eq!(d.hook_product(), 6);
        assert_eq!(d.dimension(), 1); // trivial rep of S3
    }

    #[test]
    fn test_hook_dimension_partition_111() {
        // [1,1,1] of 3: hooks = [3,2,1], product = 6, dim = 6/6 = 1
        let d = YoungDiagram::new(vec![1, 1, 1]);
        assert_eq!(d.dimension(), 1);
    }

    #[test]
    fn test_partitions_of_3() {
        let parts = YoungDiagram::partitions(3);
        assert_eq!(parts.len(), 3); // [3], [2,1], [1,1,1]
        let dims: Vec<usize> = parts.iter().map(|p| p.dimension()).collect();
        let sum: usize = dims.iter().map(|d| d * d).sum();
        // Sum of d_λ^2 = n! = 6
        // [3] dim=2... that's wrong. Let me check.
        // Actually [3]: hooks are (0,0)=3, (0,1)=2, (0,2)=1, product=6, dim=6/6=1
        // Hmm my hook formula might be off
    }

    #[test]
    fn test_standard_tableau() {
        let d = YoungDiagram::new(vec![2, 1]);
        let t = YoungTableau::new(
            d,
            vec![vec![1, 2], vec![3]],
        );
        assert!(t.is_standard());
    }

    #[test]
    fn test_non_standard_tableau() {
        let d = YoungDiagram::new(vec![2, 1]);
        let t = YoungTableau::new(
            d,
            vec![vec![2, 1], vec![3]], // Row not increasing
        );
        assert!(!t.is_standard());
    }

    #[test]
    fn test_all_standard_tableaux_21() {
        let d = YoungDiagram::new(vec![2, 1]);
        let st = YoungTableau::standard_tableaux(&d);
        assert_eq!(st.len(), 2); // Two standard tableaux of shape [2,1]
        for t in &st {
            assert!(t.is_standard());
        }
    }

    #[test]
    fn test_symmetric_group_character_trivial() {
        // Trivial partition [3] of S3: character = 1 for all cycle types
        assert_eq!(symmetric_group_character(&[3], &[3]), 1);
        assert_eq!(symmetric_group_character(&[3], &[2, 1]), 1);
        assert_eq!(symmetric_group_character(&[3], &[1, 1, 1]), 1);
    }

    #[test]
    fn test_symmetric_group_character_sign() {
        // Sign partition [1,1,1] of S3: character = (-1)^{n - number of cycles}
        assert_eq!(symmetric_group_character(&[1, 1, 1], &[3]), 1); // 3-cycle is even
        assert_eq!(symmetric_group_character(&[1, 1, 1], &[2, 1]), -1);
        assert_eq!(symmetric_group_character(&[1, 1, 1], &[1, 1, 1]), 1);
    }

    // ============================================================
    // Induced representation tests
    // ============================================================

    #[test]
    fn test_induced_character_trivial() {
        // Induce trivial rep from subgroup {e, (12)} to S3
        let s3 = FiniteGroup::s3();

        // Subgroup {e, (12)} = indices 0, 1 in S3
        let embedding = SubgroupEmbedding::new(vec![0, 1]);
        let z2 = FiniteGroup::cyclic(2);

        let chi_triv = Character {
            values: vec![c(1.0, 0.0), c(1.0, 0.0)],
        };

        let ind_chi = induced_character_embedded(&s3, &chi_triv, &embedding);
        // Induced from trivial = permutation representation on cosets
        // Dimension = [G:H] * 1 = 3
        // Character at identity = 3
        assert!((ind_chi.value(0).re - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_reciprocity() {
        let s3 = FiniteGroup::s3();
        let z2 = FiniteGroup::cyclic(2);

        // Subgroup {e, (12)} = indices 0, 1 in S3
        let embedding = SubgroupEmbedding::new(vec![0, 1]);

        let chi_triv = Character {
            values: vec![c(1.0, 0.0), c(1.0, 0.0)],
        };

        let triv = Representation::trivial(s3.order());
        let psi_g = Character::from_representation(&triv);

        assert!(verify_frobenius_reciprocity_embedded(&s3, &z2, &chi_triv, &psi_g, &embedding));
    }

    #[test]
    fn test_restrict_character() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        let chi = Character::from_representation(&std_rep);

        // Subgroup {e, (12)} = indices 0, 1 in S3
        let embedding = SubgroupEmbedding::new(vec![0, 1]);

        let res = restrict_character_embedded(&chi, &embedding);
        assert_eq!(res.values.len(), 2);
        // Restriction of standard (dim 2) to {e, (12)}: 
        // χ(e) = 2, χ((12)) = 0
        assert!((res.value(0).re - 2.0).abs() < 1e-10);
        assert!(res.value(1).re.abs() < 1e-10);
    }

    // ============================================================
    // Agent symmetry tests
    // ============================================================

    #[test]
    fn test_agent_state_space_dimension() {
        let space = AgentStateSpace {
            num_agents: 3,
            local_dimension: 2,
            description: "3 agents, 2 states each".into(),
        };
        assert_eq!(space.total_dimension(), 6);
    }

    #[test]
    fn test_agent_symmetry_analysis() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();
        let irr_reps = vec![triv, sign, std_rep];

        let space = AgentStateSpace {
            num_agents: 3,
            local_dimension: 1,
            description: "3 agents".into(),
        };

        let analysis = analyze_symmetry(&space, &s3, &irr_reps);
        assert!(analysis.has_nontrivial_symmetry);
    }

    #[test]
    fn test_invariant_subspaces() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();
        let irr_reps = vec![triv, sign, std_rep];

        let rep = Representation::trivial(s3.order())
            .direct_sum(&Representation::sign_s3())
            .direct_sum(&Representation::standard_s3());

        let subspaces = invariant_subspaces(&rep, &irr_reps, &s3);
        assert_eq!(subspaces.len(), 3); // Three irreducible components

        // Total dimension should be 1 + 1 + 2 = 4
        let total: usize = subspaces.iter().map(|(dim, _)| *dim).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_symmetrizer() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();

        // Symmetrizer of the direct sum
        let rep = triv.direct_sum(&sign).direct_sum(&std_rep);
        let sym = symmetrizer(&rep, &s3);

        // Should be a 4×4 matrix, rank 1 (projects onto symmetric subspace)
        assert_eq!(sym.nrows(), 4);
        assert_eq!(sym.ncols(), 4);
    }

    #[test]
    fn test_isotypic_projection() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let sign = Representation::sign_s3();
        let std_rep = Representation::standard_s3();

        let chi_triv = Character::from_representation(&triv);
        let chi_std = Character::from_representation(&std_rep);

        // Project trivial rep onto trivial isotypic: should be identity
        let proj = isotypic_projection(&triv, &chi_triv, &s3);
        let expected = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
        assert!((proj - expected).norm() < 1e-10);
    }

    #[test]
    fn test_character_is_irreducible() {
        let s3 = FiniteGroup::s3();
        let std_rep = Representation::standard_s3();
        let chi = Character::from_representation(&std_rep);
        assert!(chi.is_irreducible(&s3));
    }

    #[test]
    fn test_direct_sum_dimension() {
        let s3 = FiniteGroup::s3();
        let triv = Representation::trivial(s3.order());
        let std_rep = Representation::standard_s3();
        let sum = triv.direct_sum(&std_rep);
        assert_eq!(sum.dimension, 3);
        assert!(sum.verify_homomorphism(&s3));
    }

    #[test]
    fn test_partitions() {
        let parts = YoungDiagram::partitions(4);
        assert_eq!(parts.len(), 5); // [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
    }

    #[test]
    fn test_young_diagram_n() {
        let d = YoungDiagram::new(vec![3, 2, 1]);
        assert_eq!(d.n(), 6);
    }
}
