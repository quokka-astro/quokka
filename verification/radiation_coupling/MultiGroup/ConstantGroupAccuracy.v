(* Finite constant-group inverse bounds, derived from the physical gas map
   and the pointwise Planck margin. This does not assert an implementation
   meets its explicit evaluator or finite-normal node contracts. *)
From Coq Require Import Reals Psatz Field Lia.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap GasBounds LogCalculus OuterDerivatives OuterBounds
 FloatingPoint Guards GuardFloatBridge WidthSafety.
From MultiGroup Require Import ConstantGroupExistence.
Open Scope R_scope.

Definition mg_heat A D T H (M:R->R) x :=
 (A*(gas_map A D T x-T)+M x)/H.
Definition mg_cool_den A D T H x := A*(T-gas_map A D T x)+H.
Definition mg_cool A D T H (M:R->R) x := M x/mg_cool_den A D T H x.
Definition mg_heat_der A D T H (Mp:R->R) x :=
 (A*gas_derivative A D T x+Mp x)/H.
Definition mg_cool_der A D T H (M Mp:R->R) x :=
 (Mp x*mg_cool_den A D T H x+M x*A*gas_derivative A D T x)/
 (mg_cool_den A D T H x)^2.

Lemma group_sum_scale n f a :
 group_sum n (fun g => a*f g)=a*group_sum n f.
Proof. induction n; simpl; [ring|rewrite IHn; ring]. Qed.

Lemma group_emission_margin n c B Bp x :
 (forall g, (g<n)%nat -> 0<=c g) ->
 (forall g, (g<n)%nat -> B g x<=x*Bp g x) ->
 group_emission n c B x<=x*group_emission n c Bp x.
Proof.
 intros Hc HB; unfold group_emission; rewrite <-group_sum_scale.
 apply group_sum_nondecreasing; intros g Hg.
 specialize (Hc g Hg); specialize (HB g Hg); nra.
Qed.

Lemma group_emission_is_derive n c B Bp x :
 (forall g, (g<n)%nat -> is_derive (B g) x (Bp g x)) ->
 is_derive (group_emission n c B) x (group_emission n c Bp x).
Proof.
 induction n as [|n IH]; intro H; unfold group_emission; simpl.
 - apply (@is_derive_const R_AbsRing R_NormedModule).
 - apply (@is_derive_plus R_AbsRing R_NormedModule).
   + apply IH; intros; apply H; lia.
   + apply is_derive_scal; apply H; lia.
Qed.

Lemma mg_heat_positive A D T H M x :
 0<A -> 0<D -> 0<T -> 0<H -> T<=x -> 0<M x ->
 0<mg_heat A D T H M x.
Proof.
 intros HA HD HT HH Hx HM.
 pose proof (heating_gas_branch A D T x HA HD HT Hx).
 unfold mg_heat; apply Rdiv_lt_0_compat; nra.
Qed.

Lemma mg_cool_positive A D T H M x :
 0<M x -> 0<mg_cool_den A D T H x -> 0<mg_cool A D T H M x.
Proof. intros; unfold mg_cool; apply Rdiv_lt_0_compat; assumption. Qed.

Lemma mg_heat_is_derive A D T H M Mp x :
 0<A -> 0<D -> 0<T -> 0<x -> 0<H ->
 is_derive M x (Mp x) ->
 is_derive (mg_heat A D T H M) x (mg_heat_der A D T H Mp x).
Proof.
 intros HA HD HT Hx HH HM.
 pose proof (gas_map_is_derive A D T x HA HD HT Hx) as HG.
 change (is_derive (gas_map A D T) x (gas_derivative A D T x)) in HG.
 unfold mg_heat, mg_heat_der.
 auto_derive.
 - repeat split; try (exists (gas_derivative A D T x); exact HG); try (exists (Mp x); exact HM); lra.
 - rewrite (is_derive_unique (fun y:R=>gas_map A D T y) x _ HG), (is_derive_unique (fun y:R=>M y) x _ HM); field; lra.
Qed.

Lemma mg_cool_is_derive A D T H M Mp x :
 0<A -> 0<D -> 0<T -> 0<x ->
 0<mg_cool_den A D T H x -> is_derive M x (Mp x) ->
 is_derive (mg_cool A D T H M) x (mg_cool_der A D T H M Mp x).
Proof.
 intros HA HD HT Hx HH HM.
 pose proof (gas_map_is_derive A D T x HA HD HT Hx) as HG.
 change (is_derive (gas_map A D T) x (gas_derivative A D T x)) in HG.
 unfold mg_cool,mg_cool_den,mg_cool_der; unfold mg_cool_den in HH.
 auto_derive.
 - repeat split; try (exists (gas_derivative A D T x); exact HG); try (exists (Mp x); exact HM); lra.
 - rewrite (is_derive_unique (fun y:R=>gas_map A D T y) x _ HG), (is_derive_unique (fun y:R=>M y) x _ HM); unfold mg_cool_den; field; lra.
Qed.

Lemma mg_heat_margin_one A D T H M Mp x :
 0<A -> 0<D -> 0<T -> 0<H -> T<=x -> 0<M x ->
 M x<=x*Mp x ->
 1<=x*mg_heat_der A D T H Mp x/mg_heat A D T H M x.
Proof.
 intros HA HD HT HH Hreg HM Hmargin.
 assert (Hx:0<x) by lra.
 pose proof (heating_gas_branch A D T x HA HD HT Hreg) as Hgas.
 destruct (gas_map_spec A D T x HA HD HT Hx) as [HGpos HGmap].
 pose proof (heating_transfer_derivative_lower A D T (gas_map A D T x) x
  HA HD HT Hgas ltac:(symmetry; exact HGmap)) as Htransfer.
 change (A*(gas_map A D T x-T)<=A*x*gas_derivative A D T x) in Htransfer.
 unfold mg_heat,mg_heat_der.
 assert (Hnum:0<A*(gas_map A D T x-T)+M x) by nra.
 replace (x*((A*gas_derivative A D T x+Mp x)/H)/
  ((A*(gas_map A D T x-T)+M x)/H)) with
  ((A*x*gas_derivative A D T x+x*Mp x)/(A*(gas_map A D T x-T)+M x))
  by (field; split; lra).
 apply (Rmult_le_reg_r (A*(gas_map A D T x-T)+M x)); [exact Hnum|].
 field_simplify; nra.
Qed.

Lemma mg_cool_margin_one A D T H M Mp x :
 0<A -> 0<D -> 0<T -> 0<x -> 0<M x ->
 0<mg_cool_den A D T H x -> M x<=x*Mp x ->
 1<=x*mg_cool_der A D T H M Mp x/mg_cool A D T H M x.
Proof.
 intros HA HD HT Hx HM HH Hmargin.
 pose proof (gas_elasticity_bounds A D T x HA HD HT Hx) as [HG _].
 pose proof (proj1 (gas_map_spec A D T x HA HD HT Hx)) as Ht.
 assert (Hgd:0<gas_derivative A D T x).
 { unfold gas_elasticity in HG.
   apply (Rmult_lt_reg_r (x/gas_map A D T x)); [apply Rdiv_lt_0_compat; assumption|].
   replace (gas_derivative A D T x*(x/gas_map A D T x)) with
    (x*gas_derivative A D T x/gas_map A D T x) by (unfold Rdiv; ring); nra. }
 unfold mg_cool,mg_cool_der.
 set (den:=mg_cool_den A D T H x) in *.
 replace (x*((Mp x*den+M x*A*gas_derivative A D T x)/den^2)/(M x/den)) with
  (x*Mp x/M x+x*A*gas_derivative A D T x/den) by (field; split; lra).
 assert (Hfirst:1<=x*Mp x/M x).
 { apply (Rmult_le_reg_r (M x)); [exact HM|]; field_simplify; nra. }
 assert (Hsecond:0<=x*A*gas_derivative A D T x/den).
 { apply Rdiv_le_0_compat; [apply Rmult_le_pos; [apply Rmult_le_pos|]; lra|exact HH]. }
 lra.
Qed.

Definition mg_balance (heating:bool) A D T H M :=
 if heating then mg_heat A D T H M else mg_cool A D T H M.
Definition mg_balance_der (heating:bool) A D T H M Mp :=
 if heating then mg_heat_der A D T H Mp else mg_cool_der A D T H M Mp.
Definition mg_region (heating:bool) A D T H x :=
 if heating then 0<H /\ T<=x else 0<mg_cool_den A D T H x.

Lemma mg_balance_positive heating A D T H M x :
 0<A -> 0<D -> 0<T -> 0<x -> 0<M x ->
 mg_region heating A D T H x -> 0<mg_balance heating A D T H M x.
Proof.
 intros HA HD HT Hx HM HR; destruct heating; simpl in *.
 - apply mg_heat_positive; tauto.
 - apply mg_cool_positive; assumption.
Qed.

Lemma mg_balance_is_derive heating A D T H M Mp x :
 0<A -> 0<D -> 0<T -> 0<x -> mg_region heating A D T H x ->
 is_derive M x (Mp x) ->
 is_derive (mg_balance heating A D T H M) x (mg_balance_der heating A D T H M Mp x).
Proof.
 intros HA HD HT Hx HR HDer; destruct heating; simpl in *.
 - apply mg_heat_is_derive; tauto.
 - apply mg_cool_is_derive; tauto.
Qed.

Theorem constant_group_margin_one heating A D T H M Mp x :
 0<A -> 0<D -> 0<T -> 0<x -> 0<M x ->
 mg_region heating A D T H x -> M x<=x*Mp x ->
 1<=x*mg_balance_der heating A D T H M Mp x/mg_balance heating A D T H M x.
Proof.
 intros HA HD HT Hx HM HR Hmargin; destruct heating; simpl in *.
 - apply mg_heat_margin_one; tauto.
 - apply mg_cool_margin_one; tauto.
Qed.

Theorem constant_group_inverse_log_bound heating A D T H M Mp root trial :
 0<A -> 0<D -> 0<T -> 0<root -> 0<trial ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<M s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> mg_region heating A D T H s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> is_derive M s (Mp s)) ->
 (forall s, Rmin root trial<s<Rmax root trial -> M s<=s*Mp s) ->
 log_error trial root<=
 log_error (mg_balance heating A D T H M trial) (mg_balance heating A D T H M root).
Proof.
 intros HA HD HT HR HX HM Hreg HDer Hmargin.
 replace (log_error (mg_balance heating A D T H M trial)
  (mg_balance heating A D T H M root)) with
  (log_error (mg_balance heating A D T H M trial)
  (mg_balance heating A D T H M root)/1) by field.
 apply (elasticity_lower_distance (mg_balance heating A D T H M)
  (mg_balance_der heating A D T H M Mp) root trial 1 HR HX ltac:(lra)).
 - intros s Hs; apply mg_balance_positive; try assumption.
   + apply (positive_closed_interval root trial s); assumption.
   + apply HM; assumption.
   + apply Hreg; assumption.
 - intros s Hs; apply (is_derive_continuity_pt _ _ (mg_balance_der heating A D T H M Mp s)).
   apply mg_balance_is_derive; try assumption.
   + apply (positive_closed_interval root trial s); assumption.
   + apply Hreg; assumption.
   + apply HDer; assumption.
 - intros s Hs; apply mg_balance_is_derive; try assumption.
   + apply (positive_closed_interval root trial s); try assumption; lra.
   + apply Hreg; lra.
   + apply HDer; lra.
 - intros s Hs; apply constant_group_margin_one; try assumption.
   + apply (positive_closed_interval root trial s); try assumption; lra.
   + apply HM; lra.
   + apply Hreg; lra.
   + apply Hmargin; assumption.
Qed.

(* This theorem discharges the scalar conditioning for the actual finite
   constant-weight sum, rather than postulating the sum's final margin. *)
Theorem finite_group_inverse_log_bound heating A D T H n c B Bp root trial :
 0<A -> 0<D -> 0<T -> 0<root -> 0<trial ->
 (forall g, (g<n)%nat -> 0<=c g) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<group_emission n c B s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> mg_region heating A D T H s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (B g) s (Bp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> B g s<=s*Bp g s) ->
 log_error trial root<=
 log_error (mg_balance heating A D T H (group_emission n c B) trial)
  (mg_balance heating A D T H (group_emission n c B) root).
Proof.
 intros HA HD HT HR HX Hc HM Hreg HDer Hmargin.
 apply (constant_group_inverse_log_bound heating A D T H
  (group_emission n c B) (group_emission n c Bp) root trial); try assumption.
 - intros s Hs; apply group_emission_is_derive; intros g Hg; apply HDer; assumption.
 - intros s Hs; apply group_emission_margin; [exact Hc|].
   intros g Hg; apply Hmargin; assumption.
Qed.

(* A directly measured positive-ratio guard. No machine logarithm is
   evaluated: 1 +/- 128u has the proved exact-log radius 129 lambda. *)
Theorem finite_group_residual_coordinate64 heating A D T H n c B Bp root trial rhat eta :
 0<A -> 0<D -> 0<T -> 0<root -> 0<trial ->
 (forall g, (g<n)%nat -> 0<=c g) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<group_emission n c B s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> mg_region heating A D T H s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (B g) s (Bp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> B g s<=s*Bp g s) ->
 mg_balance heating A D T H (group_emission n c B) root=1 ->
 LogBound eta rhat (mg_balance heating A D T H (group_emission n c B) trial) ->
 1-128*binary64_u<=rhat<=1+128*binary64_u ->
 log_error trial root<=eta+129*binary64_lambda.
Proof.
 intros HA HD HT HR HX Hc HM Hreg HDer Hmargin Hroot Heval Hguard.
 pose proof (finite_group_inverse_log_bound heating A D T H n c B Bp root trial
  HA HD HT HR HX Hc HM Hreg HDer Hmargin) as Hinv.
 pose proof (outer_measured_window rhat Hguard) as Hmeas.
 destruct Heval as [Hrhat [HRtrial Heval]].
 rewrite Hroot in Hinv.
 pose proof (log_error_triangle (mg_balance heating A D T H (group_emission n c B) trial) rhat 1) as Htri.
 change (log_error rhat (mg_balance heating A D T H (group_emission n c B) trial)<=eta) in Heval.
 rewrite log_error_sym in Heval.
 assert (Hmeas':log_error rhat 1<=129*binary64_lambda).
 { unfold log_error; rewrite ln_1,Rminus_0_r; exact Hmeas. }
 lra.
Qed.

Theorem certified_bracket_coordinate64 choice lo hi root trial :
 0<lo -> lo<hi -> lo<=root<=hi -> lo<=trial<=hi ->
 SafeDifference64 lo hi -> normal64 (RN64 choice (hi-lo)/lo) ->
 RN64 choice (RN64 choice (hi-lo)/lo)<=16*binary64_u ->
 log_error trial root<=32*binary64_lambda.
Proof.
 intros Hl Hlh HR HT HD HN HW.
 pose proof (rounded_outer_width_safe_RN64 choice lo hi Hl Hlh HD HN HW) as Hwidth.
 pose proof (log_bracket_width lo hi trial root Hl HT HR); lra.
Qed.

Lemma multigroup_lower_sign78 rhat r :
 LogBound (78*binary64_lambda) rhat r ->
 rhat<1-128*binary64_u -> r<1.
Proof.
 intros [Hhat [Hr Herr]] Hwin.
 apply (guard_lower_sign 128 78 rhat r); try assumption;
 unfold binary64_u; field_simplify; lra.
Qed.

Lemma multigroup_upper_sign78 rhat r :
 LogBound (78*binary64_lambda) rhat r ->
 1+128*binary64_u<rhat -> 1<r.
Proof.
 intros [Hhat [Hr Herr]] Hwin.
 apply (guard_upper_sign 128 78 rhat r); try assumption;
 unfold binary64_u; field_simplify; lra.
Qed.

(* For the 78-lambda specialization the three plain floating-point guard
   outcomes cover every evaluation: there is no unresolved-sign gap. *)
Theorem multigroup_guard_total78 rhat r :
 LogBound (78*binary64_lambda) rhat r ->
 (rhat<1-128*binary64_u /\ r<1) \/
 (1-128*binary64_u<=rhat<=1+128*binary64_u) \/
 (1+128*binary64_u<rhat /\ 1<r).
Proof.
 intro HE.
 destruct (Rlt_dec rhat (1-128*binary64_u)) as [HL|HL].
 - left; split; [exact HL|apply multigroup_lower_sign78 with rhat; assumption].
 - destruct (Rlt_dec (1+128*binary64_u) rhat) as [HU|HU].
   + right; right; split; [exact HU|apply multigroup_upper_sign78 with rhat; assumption].
   + right; left; lra.
Qed.

Print Assumptions finite_group_inverse_log_bound.
Print Assumptions finite_group_residual_coordinate64.
Print Assumptions certified_bracket_coordinate64.
Print Assumptions multigroup_guard_total78.

Lemma scalar_root_balance_one heating A D T H n c B root :
 mg_region heating A D T H root ->
 constant_group_residual A D T n c B H root=0 ->
 mg_balance heating A D T H (group_emission n c B) root=1.
Proof.
 intros HR HF; destruct heating; simpl in *;
 unfold constant_group_residual in HF.
 - unfold mg_heat; apply (Rmult_eq_reg_r H); [field_simplify; nra|lra].
 - unfold mg_cool,mg_cool_den; unfold mg_cool_den in HR.
   apply (Rmult_eq_reg_r (A*(T-gas_map A D T root)+H));
    [field_simplify; nra|lra].
Qed.

Theorem scalar_root_residual_coordinate64 heating A D T H n c B Bp root trial rhat eta :
 0<A -> 0<D -> 0<T -> 0<root -> 0<trial ->
 (forall g, (g<n)%nat -> 0<=c g) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<group_emission n c B s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> mg_region heating A D T H s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (B g) s (Bp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> B g s<=s*Bp g s) ->
 constant_group_residual A D T n c B H root=0 ->
 LogBound eta rhat (mg_balance heating A D T H (group_emission n c B) trial) ->
 1-128*binary64_u<=rhat<=1+128*binary64_u ->
 log_error trial root<=eta+129*binary64_lambda.
Proof.
 intros HA HD HT HR HX Hc HM Hreg HDer Hmargin Hroot Heval Hguard.
 apply (finite_group_residual_coordinate64 heating A D T H n c B Bp root trial rhat eta);
  try assumption.
 apply scalar_root_balance_one; [apply Hreg; split; [apply Rmin_l|apply Rmax_l]|exact Hroot].
Qed.
