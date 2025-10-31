use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq::eval_eq;

use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        small_value_utils::{Accumulators, NUM_SVO_ROUNDS, compute_p_beta},
        sumcheck_single::compute_sumcheck_polynomial,
    },
};

fn precompute_e_in<F: Field>(w: &MultilinearPoint<F>) -> Vec<F> {
    let half_l = w.num_variables() / 2;
    let w_in = w.0[NUM_SVO_ROUNDS..NUM_SVO_ROUNDS + half_l].to_vec();
    eval_eq_in_hypercube(&w_in)
}

fn precompute_e_out<F: Field>(w: &MultilinearPoint<F>) -> [Vec<F>; NUM_SVO_ROUNDS] {
    let half_l = w.num_variables() / 2;
    let w_out_len = w.num_variables() - half_l - 1;

    std::array::from_fn(|round| {
        let mut w_out = Vec::with_capacity(w_out_len);
        w_out.extend_from_slice(&w.0[round + 1..NUM_SVO_ROUNDS]);
        w_out.extend_from_slice(&w.0[half_l + NUM_SVO_ROUNDS..]);
        eval_eq_in_hypercube(&w_out)
    })
}

/// Transposes the polynomial so each `compute_p_beta` call reads a contiguous block of eight values.
/// (This is a performance optimization)
fn transpose_poly_for_svo<F: Field>(
    poly: &EvaluationsList<F>,
    num_variables: usize,
    x_out_num_vars: usize,
    half_l: usize,
) -> Vec<F> {
    let num_x_in = 1 << half_l;
    let _num_x_out = 1 << x_out_num_vars;
    let step_size = 1 << (num_variables - NUM_SVO_ROUNDS);
    let block_size = 8;

    // Pre-allocate the full memory for the transposed data.
    let mut transposed_poly = vec![F::ZERO; 1 << num_variables];
    let x_out_block_size = num_x_in * block_size;

    // Parallelize the transposition work.
    transposed_poly
        .par_chunks_mut(x_out_block_size)
        .enumerate()
        .for_each(|(x_out, chunk)| {
            // Each thread works on a separate `x_out` chunk.
            for x_in in 0..num_x_in {
                let start_index = (x_in << x_out_num_vars) | x_out;

                // The destination index is relative to the start of the current chunk.
                let dest_base_index = x_in * block_size;

                let mut iter = poly.iter().skip(start_index).step_by(step_size);
                for i in 0..block_size {
                    chunk[dest_base_index + i] = *iter.next().unwrap();
                }
            }
        });

    transposed_poly
}
// Procedure 9. Page 37.
fn compute_accumulators<F: Field, EF: ExtensionField<F>>(
    poly: &EvaluationsList<F>,
    e_in: Vec<EF>,
    e_out: [Vec<EF>; NUM_SVO_ROUNDS],
) -> Accumulators<EF> {
    let l = poly.num_variables();
    let half_l = l / 2;

    let x_out_num_variables = half_l - NUM_SVO_ROUNDS + (l % 2);
    debug_assert_eq!(half_l + x_out_num_variables, l - NUM_SVO_ROUNDS);

    // Transpose once so the parallel loop reads contiguous blocks.
    let transposed_poly = transpose_poly_for_svo(poly, l, x_out_num_variables, half_l);

    // Iterate over x_out in parallel.
    (0..1 << x_out_num_variables)
        .into_par_iter()
        .map(|x_out| {
            // Each thread keeps its own accumulators to avoid shared mutable state.
            let mut local_accumulators = Accumulators::<EF>::new_empty();

            let mut temp_accumulators: Vec<EF> = vec![EF::ZERO; 27];
            let mut p_evals_buffer = [F::ZERO; 27];
            let num_x_in = 1 << half_l;

            for x_in in 0..num_x_in {
                // Read the contiguous block for this (x_out, x_in).
                let block_start = (x_out * num_x_in + x_in) * 8;
                let current_evals_arr: [F; 8] = transposed_poly[block_start..block_start + 8]
                    .try_into()
                    .unwrap();

                compute_p_beta(&current_evals_arr, &mut p_evals_buffer);
                let e_in_value = e_in[x_in];

                for (accumulator, &p_eval) in
                    temp_accumulators.iter_mut().zip(p_evals_buffer.iter())
                {
                    *accumulator += e_in_value * p_eval;
                }
            }

            // Hardcoded accumulator distribution

            let temp_acc = &temp_accumulators;
            let e_out_2 = e_out[2][x_out];

            // Cache e_out values for the current x_out.
            let e0_0 = e_out[0][(0 << x_out_num_variables) | x_out];
            let e0_1 = e_out[0][(1 << x_out_num_variables) | x_out];
            let e0_2 = e_out[0][(2 << x_out_num_variables) | x_out];
            let e0_3 = e_out[0][(3 << x_out_num_variables) | x_out];
            let e1_0 = e_out[1][(0 << x_out_num_variables) | x_out];
            let e1_1 = e_out[1][(1 << x_out_num_variables) | x_out];

            // beta_index = 0; b=(0,0,0);
            local_accumulators.accumulate(0, 0, e0_0 * temp_acc[0]); // y=0<<1|0=0
            local_accumulators.accumulate(1, 0, e1_0 * temp_acc[0]); // y=0
            local_accumulators.accumulate(2, 0, e_out_2 * temp_acc[0]);

            // beta_index = 1; b=(0,0,1);
            local_accumulators.accumulate(0, 0, e0_1 * temp_acc[1]); // y=0<<1|1=1
            local_accumulators.accumulate(1, 0, e1_1 * temp_acc[1]); // y=1
            local_accumulators.accumulate(2, 1, e_out_2 * temp_acc[1]);

            // beta_index = 2; b=(0,0,2);
            local_accumulators.accumulate(2, 2, e_out_2 * temp_acc[2]);

            // beta_index = 3; b=(0,1,0);
            local_accumulators.accumulate(0, 0, e0_2 * temp_acc[3]); // y=1<<1|0=2
            local_accumulators.accumulate(1, 1, e1_0 * temp_acc[3]); // y=0
            local_accumulators.accumulate(2, 3, e_out_2 * temp_acc[3]);

            // beta_index = 4; b=(0,1,1);
            local_accumulators.accumulate(0, 0, e0_3 * temp_acc[4]); // y=1<<1|1=3
            local_accumulators.accumulate(1, 1, e1_1 * temp_acc[4]); // y=1
            local_accumulators.accumulate(2, 4, e_out_2 * temp_acc[4]);

            // beta_index = 5; b=(0,1,2);
            local_accumulators.accumulate(2, 5, e_out_2 * temp_acc[5]);

            // beta_index = 6; b=(0,2,0);
            local_accumulators.accumulate(1, 2, e1_0 * temp_acc[6]); // y=0
            local_accumulators.accumulate(2, 6, e_out_2 * temp_acc[6]);

            // beta_index = 7; b=(0,2,1);
            local_accumulators.accumulate(1, 2, e1_1 * temp_acc[7]); // y=1
            local_accumulators.accumulate(2, 7, e_out_2 * temp_acc[7]);

            // beta_index = 8; b=(0,2,2);
            local_accumulators.accumulate(2, 8, e_out_2 * temp_acc[8]);

            // beta_index = 9; b=(1,0,0);
            local_accumulators.accumulate(0, 1, e0_0 * temp_acc[9]); // y=0<<1|0=0
            local_accumulators.accumulate(1, 3, e1_0 * temp_acc[9]); // y=0
            local_accumulators.accumulate(2, 9, e_out_2 * temp_acc[9]);

            // beta_index = 10; b=(1,0,1);
            local_accumulators.accumulate(0, 1, e0_1 * temp_acc[10]); // y=0<<1|1=1
            local_accumulators.accumulate(1, 3, e1_1 * temp_acc[10]); // y=1
            local_accumulators.accumulate(2, 10, e_out_2 * temp_acc[10]);

            // beta_index = 11; b=(1,0,2);
            local_accumulators.accumulate(2, 11, e_out_2 * temp_acc[11]);

            // beta_index = 12; b=(1,1,0);
            local_accumulators.accumulate(0, 1, e0_2 * temp_acc[12]); // y=1<<1|0=2
            local_accumulators.accumulate(1, 4, e1_0 * temp_acc[12]); // y=0
            local_accumulators.accumulate(2, 12, e_out_2 * temp_acc[12]);

            // beta_index = 13; b=(1,1,1);
            local_accumulators.accumulate(0, 1, e0_3 * temp_acc[13]); // y=1<<1|1=3
            local_accumulators.accumulate(1, 4, e1_1 * temp_acc[13]); // y=1
            local_accumulators.accumulate(2, 13, e_out_2 * temp_acc[13]);

            // beta_index = 14; b=(1,1,2);
            local_accumulators.accumulate(2, 14, e_out_2 * temp_acc[14]);

            // beta_index = 15; b=(1,2,0);
            local_accumulators.accumulate(1, 5, e1_0 * temp_acc[15]); // y=0
            local_accumulators.accumulate(2, 15, e_out_2 * temp_acc[15]);

            // beta_index = 16; b=(1,2,1);
            local_accumulators.accumulate(1, 5, e1_1 * temp_acc[16]); // y=1
            local_accumulators.accumulate(2, 16, e_out_2 * temp_acc[16]);

            // beta_index = 17; b=(1,2,2);
            local_accumulators.accumulate(2, 17, e_out_2 * temp_acc[17]);

            // beta_index = 18; b=(2,0,0);
            //local_accumulators.accumulate(0, 2, e0_0 * temp_acc[18]); // y=0<<1|0=0
            local_accumulators.accumulate(1, 6, e1_0 * temp_acc[18]); // y=0
            local_accumulators.accumulate(2, 18, e_out_2 * temp_acc[18]);

            // beta_index = 19; b=(2,0,1);
            //local_accumulators.accumulate(0, 2, e0_1 * temp_acc[19]); // y=0<<1|1=1
            local_accumulators.accumulate(1, 6, e1_1 * temp_acc[19]); // y=1
            local_accumulators.accumulate(2, 19, e_out_2 * temp_acc[19]);

            // beta_index = 20; b=(2,0,2);
            local_accumulators.accumulate(2, 20, e_out_2 * temp_acc[20]);

            // beta_index = 21; b=(2,1,0);
            //local_accumulators.accumulate(0, 2, e0_2 * temp_acc[21]); // y=1<<1|0=2
            local_accumulators.accumulate(1, 7, e1_0 * temp_acc[21]); // y=0
            local_accumulators.accumulate(2, 21, e_out_2 * temp_acc[21]);

            // beta_index = 22; b=(2,1,1);
            //local_accumulators.accumulate(0, 2, e0_3 * temp_acc[22]); // y=1<<1|1=3
            local_accumulators.accumulate(1, 7, e1_1 * temp_acc[22]); // y=1
            local_accumulators.accumulate(2, 22, e_out_2 * temp_acc[22]);

            // beta_index = 23; b=(2,1,2);
            local_accumulators.accumulate(2, 23, e_out_2 * temp_acc[23]);

            // beta_index = 24; b=(2,2,0);
            local_accumulators.accumulate(1, 8, e1_0 * temp_acc[24]); // y=0
            local_accumulators.accumulate(2, 24, e_out_2 * temp_acc[24]);

            // beta_index = 25; b=(2,2,1);
            local_accumulators.accumulate(1, 8, e1_1 * temp_acc[25]); // y=1
            local_accumulators.accumulate(2, 25, e_out_2 * temp_acc[25]);

            // beta_index = 26; b=(2,2,2);
            local_accumulators.accumulate(2, 26, e_out_2 * temp_acc[26]);
            local_accumulators
        })
        // Combine the per-thread results using the Add trait implementation.
        // Note: While we could mutate in-place for slightly better performance,
        // using the Add trait keeps the code cleaner and reuses existing logic.
        .reduce(|| Accumulators::<EF>::new_empty(), |a, b| a + b)
}

pub fn eval_eq_in_hypercube<F: Field>(point: &Vec<F>) -> Vec<F> {
    let n = point.len();
    let mut evals = F::zero_vec(1 << n);
    eval_eq::<_, _, false>(point, &mut evals, F::ONE);
    evals
}

// This function is a copy of eval_eq() in poly/multilinear.rs
pub fn eval_eq_in_point<F: Field>(p: &[F], q: &[F]) -> F {
    let mut acc = F::ONE;
    for (&l, &r) in p.into_iter().zip(q) {
        acc *= F::ONE + l * r.double() - l - r;
    }
    acc
}

pub fn compute_linear_function<F: Field>(w: &[F], r: &[F]) -> [F; 2] {
    let round = w.len();
    debug_assert!(r.len() == round - 1);

    let mut const_eq: F = F::ONE;
    if round != 1 {
        const_eq = eval_eq_in_point(&w[..round - 1], r);
    }
    let w_i = w.last().unwrap();

    // Evaluation of eq(w,X) in [eq(w,0),eq(w,1)]
    [const_eq * (F::ONE - *w_i), const_eq * *w_i]
}

fn get_evals_from_l_and_t<F: Field>(l: &[F; 2], t: &[F]) -> [F; 2] {
    [
        t[0] * l[0],                   // s(0)
        (t[1] - t[0]) * (l[1] - l[0]), //s(inf) -> l(inf) = l(1) - l(0)
    ]
}

// Algorithm 6. Page 19.
// Compute three sumcheck rounds using the small value optimizaition and split-eq accumulators.
// It Returns the three challenges r_1, r_2, r_3
// (TODO: I think it should return also the folded polynomials). [OLD comment]

// Option 2

// This function implements steps 1-4 of the Algorithm 6 (page 19).
///
// 1.  Pre-computes accumulators `A_i(v, u)` (using Proc. 9).
// 2.  For each round `i` from 1 to 3:
//     a.  Calculates the polynomial `t_i(u)` (degree d) using accumulators
//         and Lagrange evals of previous challenges (Eq. 17).
//     b.  Calculates the linear polynomial `l_i(u) = eq(w_i, u) * ...`.
//     c.  Calculates `s_i(u) = t_i(u) * l_i(u)` (Proc. 8).
//     d.  Sends `s_i(0)` and `s_i(inf)` to the `prover_state`.
//     e.  Receives a new challenge `r_i`.
//     f.  Updates the `sum` for the next round based on `s_i(r_i)`.
// Returns the tuple of challenges `(r_1, r_2, r_3)` needed for the transition round.
pub fn small_value_sumcheck_three_rounds_eq<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    poly: &EvaluationsList<F>,
    w: &MultilinearPoint<EF>,
    sum: &mut EF,
) -> (EF, EF, EF)
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let e_in = precompute_e_in(w);
    let e_out = precompute_e_out(w);

    // We compute all the accumulators A_i(v, u).
    let accumulators = compute_accumulators(poly, e_in, e_out);

    // ------------------   Round 1   ------------------

    // 1. For u in {0, 1, inf} compute t_1(u)
    // Recall: In round 1, t_1(u) = A_1(u).
    let t_1_evals = accumulators.get_accumulators_for_round(0);

    // 2. For u in {0, 1, inf} compute S_1(u) = t_1(u) * l_1(u).

    // We compute l_1(0) and l_1(1)
    let linear_1_evals = compute_linear_function(&w.0[..1], &[]);

    // We compute S_1(0) and S_1(inf)
    let round_poly_evals = get_evals_from_l_and_t(&linear_1_evals, &t_1_evals);

    // 3. Send S_1(u) to the verifier.
    prover_state.add_extension_scalars(&round_poly_evals);

    // 4. Receive the challenge r_1 from the verifier.
    let r_1: EF = prover_state.sample();

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_1.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_1
        + round_poly_evals[0];

    // 5. Compute R_2 = [L_0(r_1), L_1(r_1), L_inf(r_1)]
    // L_0 (x) = 1 - x
    // L_1 (x) = x
    // L_inf (x) = (x - 1)x
    let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1];

    // ------------------ Round 2 ------------------

    // 1. For u in {0, 1, inf} compute t_2(u).
    // First we take the accumulators A_2(v, u).
    // There are 9 accumulators, since v in {0, 1, inf} and u in {0, 1, inf}.
    let accumulators_round_2 = accumulators.get_accumulators_for_round(1);

    let mut t_2_evals = [EF::ZERO; 2];

    t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
    t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[3];

    t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
    t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[4];

    // We compute l_2(0) and l_2(inf)
    let linear_2_evals = compute_linear_function(&w.0[..2], &[r_1]);

    // We compute S_2(u)
    let round_poly_evals = get_evals_from_l_and_t(&linear_2_evals, &t_2_evals);
    // debug_assert!(round_poly_evals[2]);

    // 3. Send S_2(u) to the verifier.
    // We only send S_2(0) and S_2(inf). S_2(1) is deduced by the verifier as sum - S_2(0).
    prover_state.add_extension_scalars(&round_poly_evals);

    // 4. Receive the challenge r_2 from the verifier.
    let r_2: EF = prover_state.sample();

    // 5. Compute R_3 = [L_00(r_1, r_2), L_01(r_1, r_2), ..., L_{inf inf}(r_1, r_2)]
    // L_00 (x1, x2) = (1 - x1) * (1 - x2)
    // L_01 (x1, x2) = (1 - x1) * x2
    // ...
    // L_{inf inf} (x1, x2) = (x1 - 1) * x1 * (x2 - 1) * x2

    let [l_0, l_1] = lagrange_evals_r_1;
    let one_minus_r_2 = -r_2 + F::ONE;

    let lagrange_evals_r_2 = [
        l_0 * one_minus_r_2, // L_0 0
        l_0 * r_2,           // L_0 1
        l_1 * one_minus_r_2, // L_1 0
        l_1 * r_2,           // L_1 1
    ];

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_2.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_2
        + round_poly_evals[0];

    // Round 3

    // 1. For u in {0, 1, inf} compute t_3(u).

    // First we take the accumulators A_2(v, u).
    // There are 27 accumulators, since v in {0, 1, inf}^2 and u in {0, 1, inf}.
    let accumulators_round_3 = accumulators.get_accumulators_for_round(2);

    let mut t_3_evals = [EF::ZERO; 2];

    t_3_evals[0] += lagrange_evals_r_2[0] * accumulators_round_3[0]; // (0,0,u=0)
    t_3_evals[0] += lagrange_evals_r_2[1] * accumulators_round_3[3]; // (1,0,u=0)
    t_3_evals[0] += lagrange_evals_r_2[2] * accumulators_round_3[9]; // (0,1,u=0)
    t_3_evals[0] += lagrange_evals_r_2[3] * accumulators_round_3[12]; // (1,1,u=0)

    t_3_evals[1] += lagrange_evals_r_2[0] * accumulators_round_3[1]; // (0,0,u=1)
    t_3_evals[1] += lagrange_evals_r_2[1] * accumulators_round_3[4]; // (1,0,u=1)
    t_3_evals[1] += lagrange_evals_r_2[2] * accumulators_round_3[10]; // (0,1,u=1)
    t_3_evals[1] += lagrange_evals_r_2[3] * accumulators_round_3[13]; // (1,1,u=1)

    // 2. For u in {0, 1, inf} compute S_3(u) = t_3(u) * l_3(u).

    // We compute l_3(0) and l_3(inf)
    let linear_3_evals = compute_linear_function(&w.0[..3], &[r_1, r_2]);

    // We compute S_3(u)
    let round_poly_evals = get_evals_from_l_and_t(&linear_3_evals, &t_3_evals);

    // 3. Send S_3(u) to the verifier.
    // We only send S_3(0) and S_3(inf). S_3(1) is deduced by the verifier as sum - S_3(0).
    prover_state.add_extension_scalars(&round_poly_evals);

    let r_3: EF = prover_state.sample();

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_3.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_3
        + round_poly_evals[0];

    (r_1, r_2, r_3)
}

/// HELPER FUNCTION: Folds evaluations with a set of challenges.
///
/// This function takes a list of evaluations and "folds" or "compresses" them according
/// to the provided challenges `r_1, ..., r_{k}`. The result is a new evaluation table
/// representing p(r_1, ..., r_{k}, x'). This is the core mechanic for the transition.

fn fold_evals_with_challenges<Base, Target>(
    evals: &EvaluationsList<Base>,
    challenges: &[Target],
) -> EvaluationsList<Target>
where
    Base: Field,
    Target: Field + core::ops::Mul<Base, Output = Target>,
{
    let num_challenges = challenges.len();
    let remaining_vars = evals.num_variables() - num_challenges;
    let num_remaining_evals = 1 << remaining_vars;

    let eq_table: Vec<Target> = {
        let mut table = vec![Target::ZERO; 1 << num_challenges];
        eval_eq::<_, _, false>(challenges, &mut table, Target::ONE);
        table
    };

    let folded_evals_flat: Vec<Target> = (0..num_remaining_evals)
        .into_par_iter()
        .map(|i| {
            // Use the multilinear extension formula: p(r, x') = Σ_{b} eq(r, b) * p(b, x')
            eq_table
                .iter()
                .enumerate()
                .fold(Target::ZERO, |acc, (j, &eq_val)| {
                    let original_eval_index = (j * num_remaining_evals) + i;
                    let p_b_x = evals.as_slice()[original_eval_index];
                    acc + eq_val * p_b_x
                })
        })
        .collect();

    EvaluationsList::new(folded_evals_flat)
}

/// TRANSITION ROUND (l_0 + 1):
/// Executes a single round to transition from the SVO phase to the final phase.

// Option 2

/// Implements the "Transition Round" (round `l₀ + 1`) of Algorithm 6 (page 19).
///
/// This round serves as a bridge:
// 1.  It folds the `evals` and `weights` tables (which are in the base field `F`)
//     using the challenges `(r_1, r_2, r_3)` from the SVO phase. This step
//     lifts the evaluations into the extension field `EF`.
// 2.  It then performs one standard sumcheck round
//     on these new, smaller, extension-field tables.
// 3.  It samples the transition challenge `r_{l_0+1}`.

pub(crate) fn run_transition_round_algo2<Challenger, F, EF>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    evals_base: &EvaluationsList<F>,
    weights_base: &EvaluationsList<EF>,
    svo_challenges: &[EF],
    sum: &mut EF,
    pow_bits: usize,
) -> (
    EF,                  // New challenge r_{l_0+1}
    EvaluationsList<EF>, // Folded evaluations for the next phase
    EvaluationsList<EF>, // Folded weights for the next phase
)
where
    F: TwoAdicField + Ord,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut folded_evals = fold_evals_with_challenges(evals_base, svo_challenges);
    let mut folded_weights = fold_evals_with_challenges(weights_base, svo_challenges);

    // How much better would it be to follow strictly the algorithm 2?
    let sumcheck_poly = compute_sumcheck_polynomial(&folded_evals, &folded_weights, *sum);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[0]);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[2]);

    prover_state.pow_grinding(pow_bits);
    let r_transition: EF = prover_state.sample();

    folded_evals.compress(r_transition);
    folded_weights.compress(r_transition);

    // Update the sum
    *sum = sumcheck_poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r_transition]));

    (r_transition, folded_evals, folded_weights)
}

/// FINAL ROUNDS (l_0 + 2 to l): Explicit implementation of Algorithm 5's logic.
///
/// Executes a standard sumcheck round on tables that have already been folded.

// Option 2
/// Implements the "Final Rounds" (rounds `l₀ + 2` to `l`) of Algorithm 6 (page 19).
///
// These are standard sumcheck rounds, identical to Algorithm 1 or 5.
// They operate on the already-folded, extension-field evaluation tables.
pub(crate) fn run_final_round_algo5<Challenger, F, EF>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    folded_evals: &mut EvaluationsList<EF>,
    folded_weights: &mut EvaluationsList<EF>,
    sum: &mut EF,
    pow_bits: usize,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let sumcheck_poly = compute_sumcheck_polynomial(folded_evals, folded_weights, *sum);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[0]);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[2]);

    prover_state.pow_grinding(pow_bits);
    let r: EF = prover_state.sample();

    folded_evals.compress(r);
    folded_weights.compress(r);

    *sum = sumcheck_poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};
    use p3_baby_bear::BabyBear;
    use p3_field::{
        BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField,
        integers::QuotientMap,
    };
    use rand::RngCore;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    // Helper functions for tests
    fn get_random_ef() -> EF {
        let mut rng = rand::rng();

        let r1: u32 = rng.next_u32();
        let r2: u32 = rng.next_u32();
        let r3: u32 = rng.next_u32();
        let r4: u32 = rng.next_u32();

        EF::from_basis_coefficients_slice(&[
            F::from_u32(r1),
            F::from_u32(r2),
            F::from_u32(r3),
            F::from_u32(r4),
        ])
        .unwrap()
    }

    fn naive_sumcheck_verification<F: Field, EF: ExtensionField<F>>(
        w: Vec<EF>,
        poly: EvaluationsList<F>,
    ) -> EF {
        let eq = eval_eq_in_hypercube(&w);
        poly.iter().zip(eq.iter()).map(|(p, e)| *e * *p).sum()
    }

    fn get_evals_from_l_and_t<F: Field>(l: &[F; 2], t: &[F]) -> [F; 2] {
        [
            t[0] * l[0],                   // s(0)
            (t[1] - t[0]) * (l[1] - l[0]), // s(inf) -> l(inf) = l(1) - l(0)
        ]
    }

    // Tests
    #[test]
    fn test_evals_serial_three_vars_matches_new_from_point() {
        let p = vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let point = MultilinearPoint::new(p.to_vec());
        let value = F::from_u64(1);

        let via_method = EvaluationsList::new_from_point(&point, value)
            .into_iter()
            .collect::<Vec<_>>();
        let via_serial = eval_eq_in_hypercube(&p);

        assert_eq!(via_serial, via_method);
    }

    #[test]
    fn test_eq_evals() {
        // r = [p0, p1, p2]
        let p0 = F::from_u64(2);
        let p1 = F::from_u64(3);
        let p2 = F::from_u64(5);

        // Indices: 000, 001, 010, 011, 100, 101, 110, 111
        let expected = vec![
            (F::ONE - p0) * (F::ONE - p1) * (F::ONE - p2), // 000 v[0]
            (F::ONE - p0) * (F::ONE - p1) * p2,            // 001 v[1]
            (F::ONE - p0) * p1 * (F::ONE - p2),            // 010 v[2]
            (F::ONE - p0) * p1 * p2,                       // 011 v[3]
            p0 * (F::ONE - p1) * (F::ONE - p2),            // 100 v[4]
            p0 * (F::ONE - p1) * p2,                       // 101 v[5]
            p0 * p1 * (F::ONE - p2),                       // 110 v[6]
            p0 * p1 * p2,                                  // 111 v[7]
        ];

        let out = eval_eq_in_hypercube(&vec![p0, p1, p2]);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_precompute_e_in() {
        let w: Vec<EF> = (0..10).map(|i| EF::from(F::from_int(i))).collect();
        let w = MultilinearPoint::new(w);

        let e_in = precompute_e_in(&w);

        let w_in = w.0[NUM_SVO_ROUNDS..NUM_SVO_ROUNDS + 5].to_vec();

        assert_eq!(
            w_in,
            vec![
                EF::from(F::from_int(3)),
                EF::from(F::from_int(4)),
                EF::from(F::from_int(5)),
                EF::from(F::from_int(6)),
                EF::from(F::from_int(7)),
            ]
        );

        // e_in should have length 2^5 = 32.
        assert_eq!(e_in.len(), 1 << 5);

        //  e_in[0] should be eq(w_in, 00000)
        let expected_0: EF = w_in.iter().map(|w_i| EF::ONE - *w_i).product();

        assert_eq!(expected_0, e_in[0]);
        assert_eq!(EF::from(-F::from_int(720)), e_in[0]);

        // e_in[5] should be  eq(w_in, 00101)
        let expected_5 =
            (EF::ONE - w_in[0]) * (EF::ONE - w_in[1]) * w_in[2] * (EF::ONE - w_in[3]) * w_in[4];
        assert_eq!(expected_5, e_in[5]);

        // e_in[15] should be eq(w_in, 10000)
        let expected_16: EF = w_in[1..].iter().map(|w_i| EF::ONE - *w_i).product::<EF>() * w_in[0];
        assert_eq!(expected_16, e_in[16]);

        // e_in[31] should be eq(w_in, 11111)
        let expected_31: EF = w_in.iter().map(|w_i| *w_i).product();
        assert_eq!(expected_31, e_in[31]);
    }

    #[test]
    fn test_precompute_e_out() {
        let mut rng = rand::rng();

        let w: Vec<EF> = (0..10)
            .map(|_| {
                let r1: u32 = rng.next_u32();
                let r2: u32 = rng.next_u32();
                let r3: u32 = rng.next_u32();
                let r4: u32 = rng.next_u32();

                EF::from_basis_coefficients_slice(&[
                    F::from_u32(r1),
                    F::from_u32(r2),
                    F::from_u32(r3),
                    F::from_u32(r4),
                ])
                .unwrap()
            })
            .collect();

        let w = MultilinearPoint::new(w);
        let e_out = precompute_e_out(&w);

        // Round 1:
        assert_eq!(e_out[0].len(), 16);

        assert_eq!(
            e_out[0][0],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(
            e_out[0][1],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * (EF::ONE - w[8]) * w[9]
        );
        assert_eq!(
            e_out[0][2],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * w[8] * (EF::ONE - w[9])
        );
        assert_eq!(
            e_out[0][3],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * w[8] * w[9]
        );
        assert_eq!(
            e_out[0][4],
            (EF::ONE - w[1]) * w[2] * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(
            e_out[0][5],
            (EF::ONE - w[1]) * w[2] * (EF::ONE - w[8]) * w[9]
        );
        assert_eq!(
            e_out[0][6],
            (EF::ONE - w[1]) * w[2] * w[8] * (EF::ONE - w[9])
        );
        assert_eq!(e_out[0][7], (EF::ONE - w[1]) * w[2] * w[8] * w[9]);

        assert_eq!(
            e_out[0][8],
            w[1] * (EF::ONE - w[2]) * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(
            e_out[0][9],
            w[1] * (EF::ONE - w[2]) * (EF::ONE - w[8]) * w[9]
        );
        assert_eq!(
            e_out[0][10],
            w[1] * (EF::ONE - w[2]) * w[8] * (EF::ONE - w[9])
        );
        assert_eq!(e_out[0][11], w[1] * (EF::ONE - w[2]) * w[8] * w[9]);
        assert_eq!(
            e_out[0][12],
            w[1] * w[2] * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(e_out[0][13], w[1] * w[2] * (EF::ONE - w[8]) * w[9]);
        assert_eq!(e_out[0][14], w[1] * w[2] * w[8] * (EF::ONE - w[9]));
        assert_eq!(e_out[0][15], w[1] * w[2] * w[8] * w[9]);

        // Round 2:
        assert_eq!(e_out[1].len(), 8);

        assert_eq!(
            e_out[1][0],
            (EF::ONE - w[2]) * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(e_out[1][1], (EF::ONE - w[2]) * (EF::ONE - w[8]) * w[9]);
        assert_eq!(e_out[1][2], (EF::ONE - w[2]) * w[8] * (EF::ONE - w[9]));
        assert_eq!(e_out[1][3], (EF::ONE - w[2]) * w[8] * w[9]);
        assert_eq!(e_out[1][4], w[2] * (EF::ONE - w[8]) * (EF::ONE - w[9]));
        assert_eq!(e_out[1][5], w[2] * (EF::ONE - w[8]) * w[9]);
        assert_eq!(e_out[1][6], w[2] * w[8] * (EF::ONE - w[9]));
        assert_eq!(e_out[1][7], w[2] * w[8] * w[9]);

        // Round 3:
        assert_eq!(e_out[2].len(), 4);

        assert_eq!(e_out[2][0], (EF::ONE - w[8]) * (EF::ONE - w[9]));
        assert_eq!(e_out[2][1], (EF::ONE - w[8]) * w[9]);
        assert_eq!(e_out[2][2], w[8] * (EF::ONE - w[9]));
        assert_eq!(e_out[2][3], w[8] * w[9]);
    }

    #[test]
    fn test_compute_linear_function() {
        // w = [1]
        // r = []
        let w = [EF::from(F::from_int(1))];
        let r = [];
        // l(0) = 0
        // l(1) = 1
        let expected = [EF::from(F::from_int(0)), EF::from(F::from_int(1))];
        let result = compute_linear_function(&w, &r);
        assert_eq!(result, expected);

        // w = [1, 1]
        // r = [1]
        let w = [EF::from(F::from_int(1)), EF::from(F::from_int(1))];
        let r = [EF::from(F::from_int(1))];
        // l(0) = 0
        // l(1) = 1
        let expected = [EF::from(F::from_int(0)), EF::from(F::from_int(1))];
        let result = compute_linear_function(&w, &r);
        assert_eq!(result, expected);

        // w = [w0, w1, w2, w3]
        // r = [r0, r1, r2]
        let w: Vec<EF> = (0..4).map(|_| get_random_ef()).collect();
        let r: Vec<EF> = (0..3).map(|_| get_random_ef()).collect();

        let expected = [
            eval_eq_in_point(&w[..3], &r) * eval_eq_in_point(&w[3..], &[EF::ZERO]),
            eval_eq_in_point(&w[..3], &r) * eval_eq_in_point(&w[3..], &[EF::ONE]),
        ];
        let result = compute_linear_function(&w, &r);
        assert_eq!(result, expected);
    }
}
