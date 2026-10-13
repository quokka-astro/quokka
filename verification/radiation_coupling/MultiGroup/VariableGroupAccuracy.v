(* Smooth temperature-dependent group coefficients: conditional inverse
   accuracy derived from emission/absorption slopes, not postulated for the
   final balance. All branch, smoothness, positivity and RN node obligations
   are explicit. The constant-opacity sources are imported unchanged. *)
From Coq Require Import Reals Psatz Field Lia List.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap GasBounds LogCalculus OuterDerivatives
 OuterBounds FloatingPoint Guards GuardFloatBridge InnerEvaluator Binary64Accuracy.
From MultiGroup Require Import ConstantGroupExistence ConstantGroupAccuracy
 MultigroupGraph ComponentAccuracy.
Import ListNotations.
Open Scope R_scope.

Definition variable_heat A D T (M H:R->R) x :=
 (A*(gas_map A D T x-T)+M x)/H x.
Definition variable_cool_den A D T (H:R->R) x :=
 A*(T-gas_map A D T x)+H x.
Definition variable_cool A D T (M H:R->R) x :=
 M x/variable_cool_den A D T H x.
Definition variable_heat_der A D T (M Mp H Hp:R->R) x :=
 ((A*gas_derivative A D T x+Mp x)*H x-
   (A*(gas_map A D T x-T)+M x)*Hp x)/(H x)^2.
Definition variable_cool_der A D T (M Mp H Hp:R->R) x :=
 (Mp x*variable_cool_den A D T H x+
  M x*(A*gas_derivative A D T x-Hp x))/(variable_cool_den A D T H x)^2.
Definition variable_balance (heating:bool) A D T M H :=
 if heating then variable_heat A D T M H else variable_cool A D T M H.
Definition variable_balance_der (heating:bool) A D T M Mp H Hp :=
 if heating then variable_heat_der A D T M Mp H Hp
 else variable_cool_der A D T M Mp H Hp.
Definition variable_region (heating:bool) A D T H x :=
 if heating then 0<H x /\ T<=x
 else 0<=H x /\ x<=T /\ 0<variable_cool_den A D T H x.
Definition variable_residual A D T (M H:R->R) x :=
 A*(gas_map A D T x-T)+M x-H x.
Definition variable_margin m v := Rmin 1 m-v.

Lemma physical_gas_derivative_positive A D T x :
 0<A -> 0<D -> 0<T -> 0<x -> 0<gas_derivative A D T x.
Proof.
 intros HA HD HT Hx.
 pose proof (gas_elasticity_bounds A D T x HA HD HT Hx) as [HG _].
 pose proof (proj1 (gas_map_spec A D T x HA HD HT Hx)) as Ht.
 unfold gas_elasticity in HG.
 apply (Rmult_lt_reg_r (x/gas_map A D T x)); [apply Rdiv_lt_0_compat; assumption|].
 replace (gas_derivative A D T x*(x/gas_map A D T x)) with
 (x*gas_derivative A D T x/gas_map A D T x) by (unfold Rdiv; ring); nra.
Qed.

Lemma variable_balance_positive heating A D T M H x :
 0<A -> 0<D -> 0<T -> 0<x -> 0<M x ->
 variable_region heating A D T H x ->
 0<variable_balance heating A D T M H x.
Proof.
 intros HA HD HT Hx HM HR; destruct heating; simpl in *.
 - destruct HR as [HH HX].
   pose proof (heating_gas_branch A D T x HA HD HT HX).
   unfold variable_heat; apply Rdiv_lt_0_compat; nra.
 - unfold variable_cool; apply Rdiv_lt_0_compat; tauto.
Qed.

Lemma variable_heat_is_derive A D T M Mp H Hp x :
 0<A -> 0<D -> 0<T -> 0<x -> 0<H x ->
 is_derive M x (Mp x) -> is_derive H x (Hp x) ->
 is_derive (variable_heat A D T M H) x
 (variable_heat_der A D T M Mp H Hp x).
Proof.
 intros HA HD HT Hx HH HM HDH.
 pose proof (gas_map_is_derive A D T x HA HD HT Hx) as HG.
 change (is_derive (gas_map A D T) x (gas_derivative A D T x)) in HG.
 unfold variable_heat, variable_heat_der.
 auto_derive.
 - repeat split; try (exists (gas_derivative A D T x); exact HG);
   try (exists (Mp x); exact HM); try (exists (Hp x); exact HDH); lra.
 - rewrite (is_derive_unique (fun y:R=>gas_map A D T y) x _ HG),
    (is_derive_unique (fun y:R=>M y) x _ HM),
    (is_derive_unique (fun y:R=>H y) x _ HDH); field; lra.
Qed.

Lemma variable_cool_is_derive A D T M Mp H Hp x :
 0<A -> 0<D -> 0<T -> 0<x -> 0<variable_cool_den A D T H x ->
 is_derive M x (Mp x) -> is_derive H x (Hp x) ->
 is_derive (variable_cool A D T M H) x
 (variable_cool_der A D T M Mp H Hp x).
Proof.
 intros HA HD HT Hx HH HM HDH.
 pose proof (gas_map_is_derive A D T x HA HD HT Hx) as HG.
 change (is_derive (gas_map A D T) x (gas_derivative A D T x)) in HG.
 unfold variable_cool, variable_cool_der,variable_cool_den;
 unfold variable_cool_den in HH.
 auto_derive.
 - repeat split; try (exists (gas_derivative A D T x); exact HG);
   try (exists (Mp x); exact HM); try (exists (Hp x); exact HDH); lra.
 - rewrite (is_derive_unique (fun y:R=>gas_map A D T y) x _ HG),
    (is_derive_unique (fun y:R=>M y) x _ HM),
    (is_derive_unique (fun y:R=>H y) x _ HDH); field; lra.
Qed.

Lemma variable_balance_is_derive heating A D T M Mp H Hp x :
 0<A -> 0<D -> 0<T -> 0<x -> variable_region heating A D T H x ->
 is_derive M x (Mp x) -> is_derive H x (Hp x) ->
 is_derive (variable_balance heating A D T M H) x
 (variable_balance_der heating A D T M Mp H Hp x).
Proof.
 intros HA HD HT Hx HR HM HH; destruct heating; simpl in *.
 - apply variable_heat_is_derive; tauto.
 - apply variable_cool_is_derive; tauto.
Qed.

(* These two algebraic lemmas expose exactly how the varying denominator
   reduces conditioning. No sign assumption on H' is necessary. *)
Lemma heating_quotient_margin q qp M Mp H Hp x m v :
 0<=q -> 0<M -> 0<H -> q<=x*qp -> m*M<=x*Mp -> x*Hp<=v*H ->
 variable_margin m v<=
 x*(((qp+Mp)*H-(q+M)*Hp)/H^2)/((q+M)/H).
Proof.
 intros Hq HM HH Hqp HMp HHp.
 assert (Hnum:0<q+M) by lra.
 pose proof (Rmin_l 1 m) as Hk1.
 pose proof (Rmin_r 1 m) as Hkm.
 assert (Hnumder:Rmin 1 m*(q+M)<=x*qp+x*Mp) by nra.
 assert (Hfirst:Rmin 1 m<=(x*qp+x*Mp)/(q+M)).
 { apply (Rmult_le_reg_r (q+M)); [exact Hnum|]; field_simplify; nra. }
 assert (Hsecond:x*Hp/H<=v).
 { apply (Rmult_le_reg_r H); [exact HH|]; field_simplify; nra. }
 replace (x*(((qp+Mp)*H-(q+M)*Hp)/H^2)/((q+M)/H)) with
 ((x*qp+x*Mp)/(q+M)-x*Hp/H) by (field; split; lra).
 unfold variable_margin; lra.
Qed.

Lemma cooling_quotient_margin q M Mp H Hp gp x m v :
 0<=q -> 0<M -> 0<=H -> 0<q+H -> 0<=x*gp -> 0<=v ->
 m*M<=x*Mp -> x*Hp<=v*H ->
 variable_margin m v<=
 x*((Mp*(q+H)+M*(gp-Hp))/(q+H)^2)/(M/(q+H)).
Proof.
 intros Hq HM HH Hden Hgp Hv HMp HHp.
 assert (Hfirst:m<=x*Mp/M).
 { apply (Rmult_le_reg_r M); [exact HM|]; field_simplify; nra. }
 assert (Hsecond:0<=x*gp/(q+H)).
 { apply Rdiv_le_0_compat; assumption. }
 assert (Hthird:x*Hp/(q+H)<=v).
 { apply (Rmult_le_reg_r (q+H)); [exact Hden|]; field_simplify; nra. }
 replace (x*((Mp*(q+H)+M*(gp-Hp))/(q+H)^2)/(M/(q+H))) with
 (x*Mp/M+x*gp/(q+H)-x*Hp/(q+H)) by (field; split; lra).
 pose proof (Rmin_r 1 m); unfold variable_margin; lra.
Qed.

Theorem variable_group_derived_margin heating A D T M Mp H Hp x m v :
 0<A -> 0<D -> 0<T -> 0<x -> 0<M x ->
 variable_region heating A D T H x -> 0<=v ->
 m*M x<=x*Mp x -> x*Hp x<=v*H x ->
 variable_margin m v<=x*variable_balance_der heating A D T M Mp H Hp x/
 variable_balance heating A D T M H x.
Proof.
 intros HA HD HT Hx HM HR Hv HMp HHp; destruct heating; simpl in *.
 - destruct HR as [HH HX].
   pose proof (heating_gas_branch A D T x HA HD HT HX) as Hgas.
   destruct (gas_map_spec A D T x HA HD HT Hx) as [HGpos HGmap].
   pose proof (heating_transfer_derivative_lower A D T (gas_map A D T x) x
    HA HD HT Hgas ltac:(symmetry; exact HGmap)) as HQ.
   change (A*(gas_map A D T x-T)<=A*x*gas_derivative A D T x) in HQ.
   unfold variable_heat,variable_heat_der; apply heating_quotient_margin;
    try assumption; nra.
 - destruct HR as [HH [HX Hden]].
   pose proof (cooling_gas_branch A D T x HA HD HT Hx HX) as Hgas.
   pose proof (physical_gas_derivative_positive A D T x HA HD HT Hx) as Hgp.
   unfold variable_cool,variable_cool_der,variable_cool_den;
   unfold variable_cool_den in Hden.
   apply cooling_quotient_margin; try assumption.
   + nra.
   + apply Rmult_le_pos; [lra|apply Rmult_le_pos; lra].
Qed.

Lemma variable_branch_agreement A D T M H :
 0<A -> 0<D -> 0<T ->
 variable_heat A D T M H T=variable_cool A D T M H T.
Proof.
 intros; unfold variable_heat,variable_cool,variable_cool_den.
 rewrite gas_map_equilibrium by assumption.
 replace (A*(T-T)) with 0 by ring; rewrite !Rplus_0_l; reflexivity.
Qed.

Lemma variable_root_balance_iff heating A D T M H x :
 variable_region heating A D T H x ->
 (variable_residual A D T M H x=0 <-> variable_balance heating A D T M H x=1).
Proof.
 intro HR; destruct heating; simpl in *; unfold variable_residual;
 [unfold variable_heat|unfold variable_cool,variable_cool_den;
 unfold variable_cool_den in HR]; split; intro HF.
 - apply (Rmult_eq_reg_r (H x)); [field_simplify; nra|lra].
 - apply (f_equal (fun y=>y*H x)) in HF; field_simplify in HF; nra.
 - apply (Rmult_eq_reg_r (A*(T-gas_map A D T x)+H x));
   [field_simplify; nra|lra].
 - apply (f_equal (fun y=>y*(A*(T-gas_map A D T x)+H x))) in HF;
   field_simplify in HF; nra.
Qed.

Theorem variable_group_inverse_log_bound heating A D T M Mp H Hp m v root trial :
 0<A -> 0<D -> 0<T -> 0<root -> 0<trial -> 0<m -> 0<=v ->
 0<variable_margin m v ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<M s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> variable_region heating A D T H s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> is_derive M s (Mp s)) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> is_derive H s (Hp s)) ->
 (forall s, Rmin root trial<s<Rmax root trial -> m*M s<=s*Mp s) ->
 (forall s, Rmin root trial<s<Rmax root trial -> s*Hp s<=v*H s) ->
 log_error trial root<=
 log_error (variable_balance heating A D T M H trial)
 (variable_balance heating A D T M H root)/variable_margin m v.
Proof.
 intros HA HD HT HR HX Hm Hv Hdelta HM Hreg HDM HDH HMp HHp.
 apply (elasticity_lower_distance (variable_balance heating A D T M H)
  (variable_balance_der heating A D T M Mp H Hp) root trial
  (variable_margin m v) HR HX Hdelta).
 - intros s Hs; apply variable_balance_positive; try assumption.
   + apply (positive_closed_interval root trial s); assumption.
   + apply HM; assumption.
   + apply Hreg; assumption.
 - intros s Hs; apply (is_derive_continuity_pt _ _
   (variable_balance_der heating A D T M Mp H Hp s)).
   apply variable_balance_is_derive; try assumption.
   + apply (positive_closed_interval root trial s); assumption.
   + apply Hreg; assumption.
   + apply HDM; assumption.
   + apply HDH; assumption.
 - intros s Hs; apply variable_balance_is_derive; try assumption.
   + apply (positive_closed_interval root trial s); try assumption; lra.
   + apply Hreg; lra.
   + apply HDM; lra.
   + apply HDH; lra.
 - intros s Hs; apply variable_group_derived_margin; try assumption.
   + apply (positive_closed_interval root trial s); try assumption; lra.
   + apply HM; lra.
   + apply Hreg; lra.
   + apply HMp; assumption.
   + apply HHp; assumption.
Qed.

Theorem variable_group_residual_coordinate64 heating A D T M Mp H Hp m v
 root trial rhat eta :
 0<A -> 0<D -> 0<T -> 0<root -> 0<trial -> 0<m -> 0<=v ->
 0<variable_margin m v ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<M s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> variable_region heating A D T H s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> is_derive M s (Mp s)) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> is_derive H s (Hp s)) ->
 (forall s, Rmin root trial<s<Rmax root trial -> m*M s<=s*Mp s) ->
 (forall s, Rmin root trial<s<Rmax root trial -> s*Hp s<=v*H s) ->
 variable_residual A D T M H root=0 ->
 LogBound eta rhat (variable_balance heating A D T M H trial) ->
 1-128*binary64_u<=rhat<=1+128*binary64_u ->
 log_error trial root<=(eta+129*binary64_lambda)/variable_margin m v.
Proof.
 intros HA HD HT HR HX Hm Hv Hdelta HM Hreg HDM HDH HMp HHp HF HE HW.
 pose proof (variable_group_inverse_log_bound heating A D T M Mp H Hp m v root trial
  HA HD HT HR HX Hm Hv Hdelta HM Hreg HDM HDH HMp HHp) as Hinv.
 assert (Hone:variable_balance heating A D T M H root=1).
 { apply (proj1 (variable_root_balance_iff heating A D T M H root
   (Hreg root (conj (Rmin_l _ _) (Rmax_l _ _))))); exact HF. }
 rewrite Hone in Hinv.
 pose proof (outer_measured_window rhat HW) as Hmeas.
 destruct HE as [Hrhat [Hval HE]].
 pose proof (log_error_triangle (variable_balance heating A D T M H trial) rhat 1) as Htri.
 change (log_error rhat (variable_balance heating A D T M H trial)<=eta) in HE.
 rewrite log_error_sym in HE.
 assert (Hmeas':log_error rhat 1<=129*binary64_lambda).
 { unfold log_error; rewrite ln_1,Rminus_0_r; exact Hmeas. }
 eapply Rle_trans; [exact Hinv|].
 apply Rmult_le_compat_r; [left; apply Rinv_0_lt_compat; exact Hdelta|lra].
Qed.

Print Assumptions variable_group_derived_margin.
Print Assumptions variable_group_inverse_log_bound.
Print Assumptions variable_group_residual_coordinate64.

(* Each group uses independently varying absorption a and emission p. The
   data r, timestep h, and reduced-light-speed ratio chi stay fixed. *)
Definition variable_M_term chi h (a p B:R->R) x :=
 chi*(h*p x/(1+h*a x))*B x.
Definition variable_H_term chi h (a:R->R) r x :=
 chi*saturate(h*a x)*r.
Definition variable_M_term_der chi h (a ap p pp B Bp:R->R) x :=
 chi*h*((pp x*B x+p x*Bp x)*(1+h*a x)-p x*B x*h*ap x)/(1+h*a x)^2.
Definition variable_H_term_der chi h (a ap:R->R) r x :=
 chi*h*r*ap x/(1+h*a x)^2.
Definition variable_M_sum n chi h (a p B:nat->R->R) x :=
 group_sum n (fun g=>variable_M_term chi h (a g) (p g) (B g) x).
Definition variable_H_sum n chi h (a:nat->R->R) (r:nat->R) x :=
 group_sum n (fun g=>variable_H_term chi h (a g) (r g) x).
Definition variable_M_sum_der n chi h (a ap p pp B Bp:nat->R->R) x :=
 group_sum n (fun g=>variable_M_term_der chi h (a g) (ap g) (p g) (pp g) (B g) (Bp g) x).
Definition variable_H_sum_der n chi h (a ap:nat->R->R) (r:nat->R) x :=
 group_sum n (fun g=>variable_H_term_der chi h (a g) (ap g) (r g) x).

Lemma variable_den_positive h a : 0<h -> 0<=a -> 0<1+h*a.
Proof. intros; nra. Qed.

Lemma variable_M_term_nonnegative chi h a p B x :
 0<chi -> 0<h -> 0<=a x -> 0<=p x -> 0<=B x ->
 0<=variable_M_term chi h a p B x.
Proof.
 intros Hc Hh Ha Hp HB; unfold variable_M_term.
 apply Rmult_le_pos; [|exact HB].
 apply Rmult_le_pos; [lra|].
 apply Rdiv_le_0_compat; [nra|apply variable_den_positive; assumption].
Qed.

Lemma variable_H_term_nonnegative chi h a r x :
 0<chi -> 0<h -> 0<=a x -> 0<=r ->
 0<=variable_H_term chi h a r x.
Proof.
 intros Hc Hh Ha Hr; unfold variable_H_term,saturate.
 apply Rmult_le_pos; [|exact Hr].
 apply Rmult_le_pos; [lra|].
 apply Rdiv_le_0_compat; [nra|apply variable_den_positive; assumption].
Qed.

Lemma variable_M_term_is_derive chi h a ap p pp B Bp x :
 0<h -> 0<=a x ->
 is_derive a x (ap x) -> is_derive p x (pp x) -> is_derive B x (Bp x) ->
 is_derive (variable_M_term chi h a p B) x
 (variable_M_term_der chi h a ap p pp B Bp x).
Proof.
 intros Hh Ha Hda Hdp HdB.
 pose proof (variable_den_positive h (a x) Hh Ha) as Hden.
 unfold variable_M_term,variable_M_term_der.
 auto_derive.
 - repeat split; try (exists (ap x); exact Hda); try (exists (pp x); exact Hdp);
   try (exists (Bp x); exact HdB); lra.
 - rewrite (is_derive_unique (fun y:R=>a y) x _ Hda),
    (is_derive_unique (fun y:R=>p y) x _ Hdp),
    (is_derive_unique (fun y:R=>B y) x _ HdB); field; lra.
Qed.

(* The cancellation in H' is proved, rather than silently treating H as a
   constant or differentiating only the numerator h*a*r. *)
Lemma variable_H_term_is_derive chi h a ap r x :
 0<h -> 0<=a x -> is_derive a x (ap x) ->
 is_derive (variable_H_term chi h a r) x
 (variable_H_term_der chi h a ap r x).
Proof.
 intros Hh Ha Hda.
 pose proof (variable_den_positive h (a x) Hh Ha) as Hden.
 unfold variable_H_term,variable_H_term_der,saturate.
 auto_derive.
 - repeat split; try (exists (ap x); exact Hda); lra.
 - rewrite (is_derive_unique (fun y:R=>a y) x _ Hda); field; lra.
Qed.

Lemma variable_sum_is_derive n (f fp:nat->R->R) x :
 (forall g, (g<n)%nat -> is_derive (f g) x (fp g x)) ->
 is_derive (fun y=>group_sum n (fun g=>f g y)) x
 (group_sum n (fun g=>fp g x)).
Proof.
 induction n as [|n IH]; intro H; simpl.
 - apply (@is_derive_const R_AbsRing R_NormedModule).
 - apply (@is_derive_plus R_AbsRing R_NormedModule).
   + apply IH; intros; apply H; lia.
   + apply H; lia.
Qed.

Theorem variable_M_sum_is_derive n chi h a ap p pp B Bp x :
 0<h ->
 (forall g, (g<n)%nat -> 0<=a g x) ->
 (forall g, (g<n)%nat -> is_derive (a g) x (ap g x)) ->
 (forall g, (g<n)%nat -> is_derive (p g) x (pp g x)) ->
 (forall g, (g<n)%nat -> is_derive (B g) x (Bp g x)) ->
 is_derive (variable_M_sum n chi h a p B) x
 (variable_M_sum_der n chi h a ap p pp B Bp x).
Proof.
 intros Hh Ha Hda Hdp HdB; unfold variable_M_sum,variable_M_sum_der.
 apply (variable_sum_is_derive n
  (fun g=>variable_M_term chi h (a g) (p g) (B g))
  (fun g=>variable_M_term_der chi h (a g) (ap g) (p g) (pp g) (B g) (Bp g)) x); intros g Hg.
 apply variable_M_term_is_derive; auto.
Qed.

Theorem variable_H_sum_is_derive n chi h a ap r x :
 0<h ->
 (forall g, (g<n)%nat -> 0<=a g x) ->
 (forall g, (g<n)%nat -> is_derive (a g) x (ap g x)) ->
 is_derive (variable_H_sum n chi h a r) x
 (variable_H_sum_der n chi h a ap r x).
Proof.
 intros Hh Ha Hda; unfold variable_H_sum,variable_H_sum_der.
 apply (variable_sum_is_derive n
  (fun g=>variable_H_term chi h (a g) (r g))
  (fun g=>variable_H_term_der chi h (a g) (ap g) (r g)) x); intros g Hg.
 apply variable_H_term_is_derive; auto.
Qed.

Definition variable_fraction h (a:R->R) x := h*a x/(1+h*a x).
Definition variable_emission_slope h (a ap p pp B Bp:R->R) x :=
 x*Bp x/B x+x*pp x/p x-variable_fraction h a x*(x*ap x/a x).
Definition variable_absorption_slope h (a ap:R->R) x :=
 (1-variable_fraction h a x)*(x*ap x/a x).

Lemma variable_fraction_range h a x :
 0<h -> 0<=a x -> 0<=variable_fraction h a x<1.
Proof.
 intros Hh Ha; pose proof (variable_den_positive h (a x) Hh Ha) as Hd.
 unfold variable_fraction; split.
 - apply Rdiv_le_0_compat; nra.
 - apply (Rmult_lt_reg_r (1+h*a x)); [exact Hd|]; field_simplify; nra.
Qed.

(* mu = beta + P - f*A and nu = (1-f)*A are exact identities.
   A=0 or p=0 is handled by direct term-margin contracts below, avoiding
   an undefined logarithmic derivative for inactive coefficients. *)
Lemma variable_M_term_slope_identity chi h a ap p pp B Bp x :
 0<h -> 0<a x -> 0<p x -> 0<B x ->
 x*variable_M_term_der chi h a ap p pp B Bp x=
 variable_emission_slope h a ap p pp B Bp x*variable_M_term chi h a p B x.
Proof.
 intros Hh Ha Hp HB.
 assert (Hd:0<1+h*a x) by nra.
 unfold variable_M_term_der,variable_emission_slope,variable_fraction,variable_M_term.
 field; repeat split; lra.
Qed.

Lemma variable_H_term_slope_identity chi h a ap r x :
 0<h -> 0<a x ->
 x*variable_H_term_der chi h a ap r x=
 variable_absorption_slope h a ap x*variable_H_term chi h a r x.
Proof.
 intros Hh Ha; assert (Hd:0<1+h*a x) by nra.
 unfold variable_H_term_der,variable_absorption_slope,variable_fraction,variable_H_term,saturate.
 field; split; lra.
Qed.

Lemma variable_sum_lower_margin n (f fp:nat->R->R) x m :
 (forall g, (g<n)%nat -> m*f g x<=x*fp g x) ->
 m*group_sum n (fun g=>f g x)<=x*group_sum n (fun g=>fp g x).
Proof.
 intro H; rewrite <- !ConstantGroupAccuracy.group_sum_scale.
 apply group_sum_nondecreasing; exact H.
Qed.

Lemma variable_sum_upper_margin n (f fp:nat->R->R) x v :
 (forall g, (g<n)%nat -> x*fp g x<=v*f g x) ->
 x*group_sum n (fun g=>fp g x)<=v*group_sum n (fun g=>f g x).
Proof.
 intro H; rewrite <- !ConstantGroupAccuracy.group_sum_scale.
 apply group_sum_nondecreasing; exact H.
Qed.

Theorem variable_group_slopes_close n chi h a ap p pp B Bp r x m v :
 0<chi -> 0<h ->
 (forall g, (g<n)%nat -> 0<a g x /\ 0<p g x /\ 0<B g x /\ 0<=r g) ->
 (forall g, (g<n)%nat -> m<=variable_emission_slope h (a g) (ap g) (p g) (pp g) (B g) (Bp g) x) ->
 (forall g, (g<n)%nat -> variable_absorption_slope h (a g) (ap g) x<=v) ->
 m*variable_M_sum n chi h a p B x<=x*variable_M_sum_der n chi h a ap p pp B Bp x /\
 x*variable_H_sum_der n chi h a ap r x<=v*variable_H_sum n chi h a r x.
Proof.
 intros Hc Hh Hpos Hmu Hnu; split.
 - unfold variable_M_sum,variable_M_sum_der; apply (variable_sum_lower_margin n
   (fun g=>variable_M_term chi h (a g) (p g) (B g))
   (fun g=>variable_M_term_der chi h (a g) (ap g) (p g) (pp g) (B g) (Bp g)) x m).
   intros g Hg; destruct (Hpos g Hg) as [Ha [Hp [HB Hr]]].
   rewrite variable_M_term_slope_identity by assumption.
   apply Rmult_le_compat_r; [apply variable_M_term_nonnegative; lra|apply Hmu; exact Hg].
 - unfold variable_H_sum,variable_H_sum_der; apply (variable_sum_upper_margin n
   (fun g=>variable_H_term chi h (a g) (r g))
   (fun g=>variable_H_term_der chi h (a g) (ap g) (r g)) x v).
   intros g Hg; destruct (Hpos g Hg) as [Ha [Hp [HB Hr]]].
   rewrite variable_H_term_slope_identity by assumption.
   apply Rmult_le_compat_r; [apply variable_H_term_nonnegative; lra|apply Hnu; exact Hg].
Qed.

Print Assumptions variable_H_sum_is_derive.
Print Assumptions variable_group_slopes_close.

(* A usable finite-group corollary. All analytic premises refer to the
   coefficient/Planck leaves; the aggregate derivatives and balance margin
   are discharged by the preceding theorems. *)
Theorem finite_variable_group_residual_coordinate64 heating A D T n chi h
 a ap p pp B Bp r m v root trial rhat eta :
 0<A -> 0<D -> 0<T -> 0<chi -> 0<h -> 0<root -> 0<trial ->
 0<m -> 0<=v -> 0<variable_margin m v ->
 (forall s, Rmin root trial<=s<=Rmax root trial ->
  0<variable_M_sum n chi h a p B s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial ->
  variable_region heating A D T (variable_H_sum n chi h a r) s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial ->
  0<a g s /\ 0<p g s /\ 0<B g s /\ 0<=r g) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial ->
  is_derive (a g) s (ap g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial ->
  is_derive (p g) s (pp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial ->
  is_derive (B g) s (Bp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial ->
  m<=variable_emission_slope h (a g) (ap g) (p g) (pp g) (B g) (Bp g) s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial ->
  variable_absorption_slope h (a g) (ap g) s<=v) ->
 variable_residual A D T (variable_M_sum n chi h a p B)
  (variable_H_sum n chi h a r) root=0 ->
 LogBound eta rhat (variable_balance heating A D T
  (variable_M_sum n chi h a p B) (variable_H_sum n chi h a r) trial) ->
 1-128*binary64_u<=rhat<=1+128*binary64_u ->
 log_error trial root<=(eta+129*binary64_lambda)/variable_margin m v.
Proof.
 intros HA HD HT Hchi Hh HR HX Hm Hv Hdelta HM Hreg Hpos Hda Hdp HdB Hmu Hnu HF HE HW.
 apply (variable_group_residual_coordinate64 heating A D T
  (variable_M_sum n chi h a p B) (variable_M_sum_der n chi h a ap p pp B Bp)
  (variable_H_sum n chi h a r) (variable_H_sum_der n chi h a ap r)
  m v root trial rhat eta); try assumption.
 - intros s Hs; apply variable_M_sum_is_derive; try assumption.
   + intros g Hg; pose proof (Hpos g s Hg Hs); lra.
   + intros g Hg; apply Hda; assumption.
   + intros g Hg; apply Hdp; assumption.
   + intros g Hg; apply HdB; assumption.
 - intros s Hs; apply variable_H_sum_is_derive; try assumption.
   + intros g Hg; pose proof (Hpos g s Hg Hs); lra.
   + intros g Hg; apply Hda; assumption.
 - intros s Hs; apply (proj1 (variable_group_slopes_close n chi h a ap p pp B Bp r s m v
   Hchi Hh ltac:(intros; apply Hpos; try assumption; lra)
   ltac:(intros; apply Hmu; assumption) ltac:(intros; apply Hnu; assumption))).
 - intros s Hs; apply (proj2 (variable_group_slopes_close n chi h a ap p pp B Bp r s m v
   Hchi Hh ltac:(intros; apply Hpos; try assumption; lra)
   ltac:(intros; apply Hmu; assumption) ltac:(intros; apply Hnu; assumption))).
Qed.

(* Pointwise varying coefficients use exactly the same already-checked
   arithmetic graph. The ea,ep,eb budgets are fixed leaf-error contracts;
   no opacity derivative enters this local RN64 statement. *)
Theorem variable_pointwise_RN64_budget choice (heating:bool) eq ea ep eb chi h qh q
 (ah ph bh:nat->R) (a p B:nat->R->R) (r:nat->R) tM tH x :
 0<=ea -> 0<chi -> 0<h -> ZLogBound eq qh q ->
 (forall g, In g (tree_leaves tM++tree_leaves tH) -> ZLogBound ea (ah g) (a g x)) ->
 (forall g, In g (tree_leaves tM) -> ZLogBound ep (ph g) (p g x)) ->
 (forall g, In g (tree_leaves tM) -> ZLogBound eb (bh g) (B g x)) ->
 (forall g, In g (tree_leaves tH) -> 0<=r g) ->
 (if heating then
  0<q+tree_value (fun g=>variable_M_term chi h (a g) (p g) (B g) x) tM /\
  0<tree_value (fun g=>variable_H_term chi h (a g) (r g) x) tH
 else 0<tree_value (fun g=>variable_M_term chi h (a g) (p g) (B g) x) tM /\
  0<q+tree_value (fun g=>variable_H_term chi h (a g) (r g) x) tH) ->
 ZFiniteNormalNodes choice (mg_all_nodes (RN64 choice) heating chi h qh ah ph bh r tM tH) ->
 LogBound
 (mg_outer_budget heating eq (ea+ep+eb+(6+INR(tree_depth tM))*lambda64)
  (ea+(5+INR(tree_depth tH))*lambda64) lambda64)
 (mg_total_eval (RN64 choice) heating chi h qh ah ph bh r tM tH)
 (mg_outer_exact heating q
  (tree_value (fun g=>variable_M_term chi h (a g) (p g) (B g) x) tM)
  (tree_value (fun g=>variable_H_term chi h (a g) (r g) x) tH)).
Proof.
 exact (multigroup_formed_outer_RN64_budget choice heating eq ea ep eb chi h qh q
  ah (fun g=>a g x) ph (fun g=>p g x) bh (fun g=>B g x) r tM tH).
Qed.

(* Simple independently checkable slope certificates. With common opacity,
   mu=beta+nu, allowing stronger one-sided bounds than the distinct case. *)
Lemma variable_distinct_slope_certificate beta P A f pe aa :
 1<=beta -> 0<=f<=1 -> 0<=pe -> 0<=aa ->
 -pe<=P<=pe -> -aa<=A<=aa ->
 1-pe-aa<=beta+P-f*A /\ (1-f)*A<=aa.
Proof. intros; split; nra. Qed.

Lemma variable_distinct_positive_margin pe aa :
 0<=pe -> 0<=aa -> pe+2*aa<1 ->
 0<1-pe-aa /\ 0<=aa /\
 variable_margin (1-pe-aa) aa=1-pe-2*aa /\
 0<variable_margin (1-pe-aa) aa.
Proof.
 intros; unfold variable_margin; rewrite Rmin_right by lra; repeat split; lra.
Qed.

Lemma variable_common_slope_identity beta P f :
 beta+P-f*P=beta+(1-f)*P.
Proof. ring. Qed.

Lemma variable_common_twosided_certificate beta P f s :
 1<=beta -> 0<=f<=1 -> 0<=s -> -s<=P<=s ->
 1-s<=beta+(1-f)*P /\ (1-f)*P<=s.
Proof. intros; split; nra. Qed.

Lemma variable_common_increasing_certificate beta P f s :
 1<=beta -> 0<=f<=1 -> 0<=P<=s ->
 1<=beta+(1-f)*P /\ (1-f)*P<=s.
Proof. intros; split; nra. Qed.

Lemma variable_common_decreasing_certificate beta P f s :
 1<=beta -> 0<=f<=1 -> -s<=P<=0 ->
 1-s<=beta+(1-f)*P /\ (1-f)*P<=0.
Proof. intros; split; nra. Qed.

Print Assumptions finite_variable_group_residual_coordinate64.
Print Assumptions variable_pointwise_RN64_budget.

(* Fully finite dust and gas consequences. The checked inner solve still
   contributes 52 lambda and gas-energy reconstruction one further lambda;
   only the outer coordinate contribution is amplified by 1/delta. *)
Theorem variable_residual_dust_gas64 choice heating cg A D T M Mp H Hp m v
 root trial th qh rhat eta :
 0<cg -> 0<A -> 0<D -> 0<T -> 0<root -> 0<trial -> 0<m -> 0<=v ->
 0<variable_margin m v ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<M s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> variable_region heating A D T H s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> is_derive M s (Mp s)) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> is_derive H s (Hp s)) ->
 (forall s, Rmin root trial<s<Rmax root trial -> m*M s<=s*Mp s) ->
 (forall s, Rmin root trial<s<Rmax root trial -> s*Hp s<=v*H s) ->
 variable_residual A D T M H root=0 ->
 LogBound eta rhat (variable_balance heating A D T M H trial) ->
 1-128*binary64_u<=rhat<=1+128*binary64_u ->
 InnerResult choice A D T trial th qh -> FiniteNormalNodes choice [cg*th] ->
 let b:=(eta+129*binary64_lambda)/variable_margin m v in
 log_error trial root<=b /\ Rabs(trial/root-1)<=exp b-1 /\
 LogBound (53*binary64_lambda+2*b)
  (returned_gas choice cg th) (gas_energy cg A D T root) /\
 Rabs(returned_gas choice cg th/gas_energy cg A D T root-1)
  <=exp(53*binary64_lambda+2*b)-1.
Proof.
 intros Hcg HA HD HT HR HX Hm Hv Hdelta HM Hreg HDM HDH HMp HHp HF HE HW HI HN b.
 assert (Hcoord:log_error trial root<=b).
 { unfold b; apply (variable_group_residual_coordinate64 heating A D T M Mp H Hp m v
   root trial rhat eta); assumption. }
 split; [exact Hcoord|]; split.
 - apply dust_finite_accuracy; assumption.
 - apply (gas_finite_accuracy64 choice cg A D T root trial th qh b); assumption.
Qed.

Print Assumptions variable_residual_dust_gas64.
