(* Generic sufficient Planck-slope margins. The concrete numerical example is
   only a specialization; finite slopes without these margins are insufficient. *)
From Coq Require Import Reals Psatz Field.
From BlackBox Require Import Algebra OuterDerivatives.
Open Scope R_scope.

Definition heating_uniform_margin epsilon_h := Rmin 1 epsilon_h.
Definition cooling_uniform_margin epsilon_c := Rmin 4 epsilon_c.

Lemma generic_effective_slope_margins p tau epsilon_h epsilon_c :
 0<epsilon_h -> 0<epsilon_c -> 0<=tau ->
 -4+epsilon_c<=p<=1-epsilon_h ->
 0<heating_uniform_margin epsilon_h /\
 0<cooling_uniform_margin epsilon_c /\
 heating_uniform_margin epsilon_h<=1-p/(1+tau) /\
 cooling_uniform_margin epsilon_c<=4+p/(1+tau) /\ Rabs p<=4.
Proof.
 intros Heh Hec Htau Hp.
 assert (Hmh:0<heating_uniform_margin epsilon_h /\
   heating_uniform_margin epsilon_h<=1 /\ heating_uniform_margin epsilon_h<=epsilon_h).
 { unfold heating_uniform_margin,Rmin; destruct Rle_dec; repeat split; lra. }
 assert (Hmc:0<cooling_uniform_margin epsilon_c /\
   cooling_uniform_margin epsilon_c<=4 /\ cooling_uniform_margin epsilon_c<=epsilon_c).
 { unfold cooling_uniform_margin,Rmin; destruct Rle_dec; repeat split; lra. }
 assert (Helh:p/(1+tau)<=1-heating_uniform_margin epsilon_h).
 { apply (Rmult_le_reg_r (1+tau)); [lra|]. field_simplify; nra. }
 assert (Helc:cooling_uniform_margin epsilon_c-4<=p/(1+tau)).
 { apply (Rmult_le_reg_r (1+tau)); [lra|]. field_simplify; nra. }
 repeat split; try tauto; try lra. apply Rabs_le; lra.
Qed.

Theorem planck_slope_contract_implies_conditioning C kap x kp epsilon_h epsilon_c :
 0<C -> 0<kap x -> 0<epsilon_h -> 0<epsilon_c ->
 -4+epsilon_c<=opacity_slope kap x kp<=1-epsilon_h ->
 0<heating_uniform_margin epsilon_h /\
 0<cooling_uniform_margin epsilon_c /\
 heating_uniform_margin epsilon_h<=1-effective_slope C kap x kp /\
 cooling_uniform_margin epsilon_c<=4+effective_slope C kap x kp /\
 Rabs(opacity_slope kap x kp)<=4.
Proof.
 intros HC Hk Heh Hec Hp.
 unfold effective_slope.
 apply generic_effective_slope_margins; try assumption.
 pose proof (optical_depth_positive C kap x HC Hk); lra.
Qed.

Print Assumptions planck_slope_contract_implies_conditioning.
