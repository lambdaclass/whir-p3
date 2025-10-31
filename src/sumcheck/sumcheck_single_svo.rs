use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        small_value_utils::NUM_SVO_ROUNDS,
        sumcheck_single::SumcheckSingle,
        sumcheck_small_value::{
            run_final_round_algo5, run_transition_round_algo2, small_value_sumcheck_three_rounds_eq,
        },
    },
    whir::statement::Statement,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    pub fn from_base_evals_svo<Challenger>(
        evals: &EvaluationsList<F>,
        statement: &Statement<EF>,
        combination_randomness: EF,
        prover_state: &mut ProverState<F, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> (Self, MultilinearPoint<EF>)
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);
        // Add this assert?
        // assert!(folding_factor > NUM_SVO_ROUNDS);

        let mut challenges = Vec::with_capacity(folding_factor);

        let (weights_init, mut sum) = statement.combine::<F>(combination_randomness);

        // We assume the statement has only one constraint.
        let w = statement.constraints[0].point.0.clone();

        // First three rounds of sumcheck.
        let (r_1, r_2, r_3) =
            small_value_sumcheck_three_rounds_eq(prover_state, evals, &w, &mut sum);
        challenges.push(r_1);
        challenges.push(r_2);
        challenges.push(r_3);

        // Transition Round: l_0 + 1

        let (r_transition, mut folded_evals, mut folded_weights) = run_transition_round_algo2(
            prover_state,
            evals,
            &weights_init,
            &challenges,
            &mut sum,
            pow_bits,
        );
        challenges.push(r_transition);

        // Final Rounds: l_0 + 2 to l

        for _ in (NUM_SVO_ROUNDS + 1)..folding_factor {
            let r_final = run_final_round_algo5(
                prover_state,
                &mut folded_evals,
                &mut folded_weights,
                &mut sum,
                pow_bits,
            );
            challenges.push(r_final);
        }

        challenges[NUM_SVO_ROUNDS..].reverse();

        let sumcheck_instance = SumcheckSingle::<F, EF>::new(folded_evals, folded_weights, sum);
        (sumcheck_instance, MultilinearPoint::new(challenges))
    }
}
