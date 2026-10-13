(* Uniform-margin extension of the checked operation-graph accuracy theorem.
   The theorem for a closed band also covers the requested open band. The
   explicit epsilon upper bound rules out vacuous oversized bands on a
   degenerate comparison interval. No small-epsilon linearization is used. *)
From Coq Require Import Reals Psatz Field List ZArith.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra LogCalculus GasMap GasBounds OuterDerivatives
 OuterBounds FloatingPoint Guards GuardFloatBridge WidthSafety InnerEvaluator
 Binary64Accuracy GeneralOpacity.
Import ListNotations.
Open Scope R_scope.

Definition epsilon_margin epsilon := Rmin 1 epsilon.
Definition coordinate_budget epsilon := 200*binary64_lambda/epsilon_margin epsilon.
Definition epsilon_gas_budget bx := 53*binary64_lambda+2*bx.
Definition epsilon_radiation_budget epsilon bx := 17*binary64_lambda+(8-epsilon)*bx.

Lemma strict_slope_band_nonempty epsilon p :
 -4+epsilon<p<1-epsilon -> epsilon<5/2.
Proof. intros; lra. Qed.

Lemma epsilon_margin_bounds epsilon : 0<epsilon ->
 0<epsilon_margin epsilon /\ epsilon_margin epsilon<=1 /\ epsilon_margin epsilon<=epsilon.
Proof.
 intro He; unfold epsilon_margin,Rmin; destruct Rle_dec; repeat split; lra.
Qed.

Lemma epsilon_opacity_margins epsilon C kap x kp :
 0<epsilon -> epsilon<=5/2 -> 0<C -> 0<kap x ->
 -4+epsilon<=opacity_slope kap x kp<=1-epsilon ->
 epsilon_margin epsilon<=1-effective_slope C kap x kp /\
 epsilon_margin epsilon<=4+effective_slope C kap x kp /\
 Rabs(opacity_slope kap x kp)<=4-epsilon.
Proof.
 intros He Heu HC Hkap Hp.
 destruct (planck_slope_contract_implies_conditioning C kap x kp epsilon epsilon
   HC Hkap He He Hp) as [Hh [Hc [Hhm [Hcm Habs]]]].
 assert (Hm:epsilon_margin epsilon<=cooling_uniform_margin epsilon).
 { unfold epsilon_margin,cooling_uniform_margin,Rmin.
   destruct Rle_dec; destruct Rle_dec; lra. }
 change (epsilon_margin epsilon<=1-effective_slope C kap x kp) in Hhm.
 split; [assumption|]. split; [lra|]. apply Rabs_le; lra.
Qed.

Lemma coordinate_budget_dominates_width epsilon : 0<epsilon ->
 32*binary64_lambda<=coordinate_budget epsilon.
Proof.
 intro He; destruct (epsilon_margin_bounds epsilon He) as [Hm [Hm1 Hme]].
 unfold coordinate_budget; apply (Rmult_le_reg_r (epsilon_margin epsilon)); [assumption|].
 replace (200*binary64_lambda/epsilon_margin epsilon*epsilon_margin epsilon)
  with (200*binary64_lambda) by (field; lra).
 pose proof binary64_lambda_positive; nra.
Qed.

Lemma branch_inverse_epsilon epsilon (heating:bool) A D T C chi a r kap kp root trial :
 0<epsilon -> epsilon<=5/2 ->
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<root -> 0<trial -> branch_region heating T root trial ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -4+epsilon<=opacity_slope kap t (kp t)<=1-epsilon) ->
 log_error trial root <=
 log_error (branch_balance heating A D T C chi a r kap trial)
  (branch_balance heating A D T C chi a r kap root)/epsilon_margin epsilon.
Proof.
 intros He Heu HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion Hkap Hcont Hder Hslope.
 destruct (epsilon_margin_bounds epsilon He) as [Hm [Hm1 Hme]].
 destruct heating; simpl in Hregion |- *; destruct Hregion as [Hbr Hbt].
 - apply (heating_inverse_log_bound A D T C chi a r kap kp root trial (epsilon_margin epsilon));
   try assumption; try lra.
   intros t Ht; exact (proj1 (epsilon_opacity_margins epsilon C kap t (kp t)
     He Heu HC (Hkap t ltac:(lra)) (Hslope t Ht))).
 - apply (cooling_inverse_log_bound A D T C chi a r kap kp root trial (epsilon_margin epsilon));
   try assumption; try lra.
   intros t Ht; exact (proj1 (proj2 (epsilon_opacity_margins epsilon C kap t (kp t)
     He Heu HC (Hkap t ltac:(lra)) (Hslope t Ht)))).
Qed.

Lemma accepted_coordinate_epsilon_local epsilon choice heating A D T C chi a r kap kp khat root trial th qh :
 0<epsilon -> epsilon<=5/2 ->
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<root -> 0<trial -> branch_region heating T root trial ->
 LogBound (8*binary64_lambda) (khat trial) (kap trial) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -4+epsilon<=opacity_slope kap t (kp t)<=1-epsilon) ->
 branch_balance heating A D T C chi a r kap root=1 ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r root trial ->
 log_error trial root<=coordinate_budget epsilon.
Proof.
 intros He Heu HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion HK Hkap Hcont Hder Hslope Hbalance HI Hacc.
 destruct Hacc as [N W|lb ub Hlb Hlub Hbr Hbt Hns Hnd Nwidth W].
 - assert (Hreg:if heating then T<=trial else trial<=T).
   { destruct heating; simpl in *; tauto. }
   pose proof (inner_outer_graph_concrete_local choice heating A D T C chi a r kap khat
    trial th qh HA HD HT HC Hchi Ha Hr Htrial Hreg HK HI N) as Heval.
   pose proof (outer_exact_window _ _ W (proj2(proj2 Heval))) as Hres.
   pose proof (branch_inverse_epsilon epsilon heating A D T C chi a r kap kp root trial
    He Heu HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion Hkap Hcont Hder Hslope) as Hinv.
   rewrite Hbalance in Hinv; unfold log_error at 2 in Hinv.
   rewrite ln_1,Rminus_0_r in Hinv.
   eapply Rle_trans; [exact Hinv|]. unfold coordinate_budget,Rdiv.
   apply Rmult_le_compat_r; [left; apply Rinv_0_lt_compat; apply (epsilon_margin_bounds epsilon He)|exact Hres].
 - pose proof (rounded_outer_width_safe_RN64 choice lb ub Hlb Hlub Hns Hnd W) as Hwidth.
   pose proof (log_bracket_width lb ub trial root Hlb Hbt Hbr) as Hcoord.
   pose proof (coordinate_budget_dominates_width epsilon He); lra.
Qed.

(* The generic record exposes exact finite exponential bounds. In particular,
   arbitrarily small positive epsilon is not claimed to yield small relative
   error, even though the logarithmic theorem remains finite. *)
Record EpsilonAccuracyBudget64 (epsilon bx trial root gh gs rh rs:R) : Prop := {
 epsilon_gas_positive : 0<gh;
 epsilon_radiation_positive : 0<rh;
 epsilon_dust_log_error : log_error trial root<=bx;
 epsilon_gas_log_error : log_error gh gs<=epsilon_gas_budget bx;
 epsilon_radiation_log_error : log_error rh rs<=epsilon_radiation_budget epsilon bx;
 epsilon_dust_relative_error : Rabs(trial/root-1)<=exp bx-1;
 epsilon_gas_relative_error : Rabs(gh/gs-1)<=exp(epsilon_gas_budget bx)-1;
 epsilon_radiation_relative_error : Rabs(rh/rs-1)<=exp(epsilon_radiation_budget epsilon bx)-1
}.
Definition EpsilonMainAccuracy64 epsilon := EpsilonAccuracyBudget64 epsilon (coordinate_budget epsilon).

Lemma epsilon_accuracy_from_coordinate_local epsilon bx choice A D T C a r cg kap kp khat root trial th qh :
 0<epsilon -> epsilon<=5/2 ->
 0<A -> 0<D -> 0<T -> 0<C -> 0<a -> 0<r -> 0<cg ->
 0<root -> 0<trial ->
 LogBound (8*binary64_lambda) (khat trial) (kap trial) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -4+epsilon<=opacity_slope kap t (kp t)<=1-epsilon) ->
 log_error trial root<=bx ->
 InnerResult choice A D T trial th qh ->
 FiniteNormalNodes choice [cg*th] ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat trial) a trial) ->
 EpsilonAccuracyBudget64 epsilon bx trial root
  (returned_gas choice cg th) (gas_energy cg A D T root)
  (returned_radiation choice r C khat a trial) (OuterDerivatives.radiation C a r kap root).
Proof.
 intros He Heu HA HD HT HC Ha Hr Hcg Hroot Htrial HK Hkap Hcont Hder Hslope Hcoord HI NG NR.
 pose proof (returned_gas_concrete choice cg A D T trial th qh Hcg HA HD HT Htrial HI NG) as HG.
 pose proof (returned_radiation_concrete_local choice C a r kap khat trial HC Ha Hr Htrial HK NR) as HR.
 assert (Habs:forall t, Rmin root trial<t<Rmax root trial ->
   Rabs(opacity_slope kap t (kp t))<=4-epsilon).
 { intros t Ht; exact (proj2 (proj2 (epsilon_opacity_margins epsilon C kap t (kp t)
   He Heu HC (Hkap t ltac:(lra)) (Hslope t Ht)))). }
 pose proof (heating_output_from_coordinate A D T C a r cg kap kp root trial
   HA HD HT HC Ha Hr Hcg Hroot Htrial Hkap Hcont Hder
   (4-epsilon) bx (returned_gas choice cg th)
   (returned_radiation choice r C khat a trial) (53*binary64_lambda) (17*binary64_lambda)
   ltac:(lra) Habs Hcoord (proj2(proj2 HG)) (proj2(proj2 HR))) as [Hgas Hrad].
 assert (Hgas':log_error (returned_gas choice cg th) (gas_energy cg A D T root)<=epsilon_gas_budget bx)
  by (unfold epsilon_gas_budget; lra).
 assert (Hrad':log_error (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap root)<=epsilon_radiation_budget epsilon bx)
  by (unfold epsilon_radiation_budget; nra).
 assert (HGroot:0<gas_energy cg A D T root) by (apply gas_energy_positive; assumption).
 assert (HRroot:0<OuterDerivatives.radiation C a r kap root).
 { apply radiation_positive; try assumption.
   apply Hkap; split; [apply Rmin_l|apply Rmax_l]. }
 constructor; try assumption.
 - exact (proj1 HG).
 - exact (proj1 HR).
 - apply log_error_relative; assumption.
 - apply log_error_relative; [exact (proj1 HG)|assumption|exact Hgas'].
 - apply log_error_relative; [exact (proj1 HR)|assumption|exact Hrad'].
Qed.

Theorem binary64_epsilon_main_accuracy_local epsilon choice heating A D T C chi a r cg kap kp khat root trial th qh :
 0<epsilon -> epsilon<=5/2 ->
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<cg ->
 0<root -> 0<trial -> branch_region heating T root trial ->
 LogBound (8*binary64_lambda) (khat trial) (kap trial) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -4+epsilon<=opacity_slope kap t (kp t)<=1-epsilon) ->
 branch_balance heating A D T C chi a r kap root=1 ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r root trial ->
 FiniteNormalNodes choice [cg*th] ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat trial) a trial) ->
 EpsilonMainAccuracy64 epsilon trial root
  (returned_gas choice cg th) (gas_energy cg A D T root)
  (returned_radiation choice r C khat a trial) (OuterDerivatives.radiation C a r kap root).
Proof.
 intros He Heu HA HD HT HC Hchi Ha Hr Hcg Hroot Htrial Hregion HK Hkap Hcont Hder Hslope
  Hbalance HI Haccept NG NR.
 pose proof (accepted_coordinate_epsilon_local epsilon choice heating A D T C chi a r kap kp khat
  root trial th qh He Heu HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion HK Hkap
  Hcont Hder Hslope Hbalance HI Haccept) as Hcoord.
 exact (epsilon_accuracy_from_coordinate_local epsilon (coordinate_budget epsilon) choice A D T C a r cg
  kap kp khat root trial th qh He Heu HA HD HT HC Ha Hr Hcg Hroot Htrial HK Hkap Hcont Hder Hslope Hcoord HI NG NR).
Qed.

(* A separate narrow-bracket theorem retains its stronger margin-independent
   coordinate bound; no root-balance or inverse-margin premise is needed. *)
Theorem binary64_epsilon_bracket_accuracy_local epsilon choice A D T C a r cg kap kp khat root trial th qh lb ub :
 0<epsilon -> epsilon<=5/2 ->
 0<A -> 0<D -> 0<T -> 0<C -> 0<a -> 0<r -> 0<cg -> 0<root -> 0<trial ->
 LogBound (8*binary64_lambda) (khat trial) (kap trial) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -4+epsilon<=opacity_slope kap t (kp t)<=1-epsilon) ->
 InnerResult choice A D T trial th qh ->
 FiniteNormalNodes choice [cg*th] ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat trial) a trial) ->
 0<lb -> lb<ub -> lb<=root<=ub -> lb<=trial<=ub ->
 SafeDifference64 lb ub -> normal64 (RN64 choice (ub-lb)/lb) ->
 RN64 choice (RN64 choice (ub-lb)/lb)<=16*binary64_u ->
 EpsilonAccuracyBudget64 epsilon (32*binary64_lambda) trial root
  (returned_gas choice cg th) (gas_energy cg A D T root)
  (returned_radiation choice r C khat a trial) (OuterDerivatives.radiation C a r kap root).
Proof.
 intros He Heu HA HD HT HC Ha Hr Hcg Hroot Htrial HK Hkap Hcont Hder Hslope HI NG NR
 Hlb Hlub Hbr Hbt Hsafe Hnormal HW.
 apply (epsilon_accuracy_from_coordinate_local epsilon (32*binary64_lambda) choice A D T C a r cg
  kap kp khat root trial th qh); try assumption.
 pose proof (rounded_outer_width_safe_RN64 choice lb ub Hlb Hlub Hsafe Hnormal HW).
 pose proof (log_bracket_width lb ub trial root Hlb Hbt Hbr); lra.
Qed.

Print Assumptions binary64_epsilon_main_accuracy_local.
Print Assumptions binary64_epsilon_bracket_accuracy_local.

(* Recover the original checked example, and expose the saturated margin for
   epsilon above one rather than incorrectly extending 1/epsilon there. *)
Lemma epsilon_margin_half : epsilon_margin (1/2)=1/2.
Proof. unfold epsilon_margin; apply Rmin_right; lra. Qed.
Lemma epsilon_margin_above_one epsilon : 1<=epsilon -> epsilon_margin epsilon=1.
Proof. intro H; unfold epsilon_margin; apply Rmin_left; assumption. Qed.

Theorem epsilon_half_recovers_binary64_allowances trial root gh gs rh rs :
 0<trial -> 0<root -> 0<gs -> 0<rs ->
 EpsilonMainAccuracy64 (1/2) trial root gh gs rh rs ->
 MainAccuracy64 trial root gh gs rh rs.
Proof.
 intros Htrial Hroot Hgs Hrs H.
 unfold EpsilonMainAccuracy64,coordinate_budget in H.
 rewrite epsilon_margin_half in H.
 replace (200*binary64_lambda/(1/2)) with (400*binary64_lambda) in H by field.
 destruct H as [HG HR HD EG ER RD RG RR].
 unfold epsilon_gas_budget in EG; unfold epsilon_radiation_budget in ER.
 assert (EG':log_error gh gs<=853*binary64_lambda) by lra.
 assert (ER':log_error rh rs<=3017*binary64_lambda) by lra.
 constructor; try assumption.
 - apply binary64_relative_400; assumption.
 - apply binary64_relative_853; assumption.
 - apply binary64_relative_3017; assumption.
Qed.
Print Assumptions epsilon_half_recovers_binary64_allowances.
