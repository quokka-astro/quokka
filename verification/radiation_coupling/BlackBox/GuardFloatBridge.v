(* Bridges from the concrete real guard constants to Flocq's binary64 model.
   FLT has an unbounded upper exponent; finite/overflow assumptions remain part
   of the implementation contract, as in FloatingPoint.v. *)
From Coq Require Import Reals Psatz Field ZArith.
Require Import Flocq.Core.Core Flocq.Prop.Relative.
From BlackBox Require Import Guards FloatingPoint Brackets.
Open Scope R_scope.
Local Instance bridge_prec64 : Prec_gt_0 53. Proof. unfold Prec_gt_0; lia. Qed.

Lemma guard_u64 : binary64_u=u64.
Proof. unfold binary64_u,u64; reflexivity. Qed.
Lemma guard_lambda64 : binary64_lambda=lambda64.
Proof. unfold binary64_lambda,lambda64,log_unit; rewrite guard_u64; reflexivity. Qed.

Definition format64 := generic_format radix2 (FLT_exp (-1074) 53).
Definition successor64 := succ radix2 (FLT_exp (-1074) 53).

Lemma successor64_positive_gap a : 0<a -> normal64 a ->
  0<successor64 a-a<=2*binary64_u*a.
Proof.
 intros Ha Hn.
 pose proof (ulp_FLT_le radix2 (-1074) 53 a Hn) as Hu.
 rewrite Rabs_pos_eq in Hu by lra.
 change (ulp radix2 (FLT_exp (-1074) 53) a<=a*bpow radix2 (-52)) in Hu.
 replace (bpow radix2 (-52)) with (2*binary64_u) in Hu.
 - unfold successor64; rewrite succ_eq_pos by lra.
   pose proof (bpow_gt_0 radix2 (cexp radix2 (FLT_exp (-1074) 53) a)) as Hp.
   rewrite <- ulp_neq_0 in Hp by lra; lra.
 - rewrite guard_u64; unfold u64.
   replace (-52)%Z with (1+(-53))%Z by lia.
   rewrite bpow_plus; reflexivity.
Qed.

Lemma adjacent_format64_is_successor a b :
  0<a -> format64 a -> format64 b -> a<b ->
  (forall x, format64 x -> ~ (a<x<b)) -> b=successor64 a.
Proof.
 intros Ha Hfa Hfb Hab Hnone.
 pose proof (succ_le_lt radix2 (FLT_exp (-1074) 53) a b Hfa Hfb Hab) as Hle.
 pose proof (succ_gt_id radix2 (FLT_exp (-1074) 53) a ltac:(lra)) as Hgt.
 pose proof (generic_format_succ radix2 (FLT_exp (-1074) 53) a Hfa) as Hfs.
 unfold successor64.
 destruct Hle as [Hlt|Heq]; [exfalso; apply (Hnone _ Hfs); split; assumption|lra].
Qed.

Lemma adjacent_format64_gap a b :
  0<a -> normal64 a -> format64 a -> format64 b -> a<b ->
  (forall x, format64 x -> ~ (a<x<b)) -> b-a<=2*binary64_u*a.
Proof.
 intros Ha Hn Hfa Hfb Hab Hnone.
 rewrite (adjacent_format64_is_successor a b Ha Hfa Hfb Hab Hnone).
 exact (proj2 (successor64_positive_gap a Ha Hn)).
Qed.

Lemma RN64_relative_factor choice x : 0<x -> normal64 x ->
  exists e, Rabs e<=binary64_u /\ RN64 choice x=x*(1+e).
Proof.
 intros Hx Hn; exists ((RN64 choice x-x)/x); split.
 - pose proof (RN64_relative choice x Hn) as Hr.
   rewrite (Rabs_pos_eq x) in Hr by lra.
   rewrite guard_u64.
   unfold Rdiv; rewrite Rabs_mult,Rabs_inv,(Rabs_pos_eq x) by lra.
   apply (Rmult_le_reg_r x); [lra|].
   replace (Rabs(RN64 choice x-x)* /x*x) with (Rabs(RN64 choice x-x))
     by (field; lra).
   exact Hr.
 - field; lra.
Qed.

Theorem rounded_bracket_width_RN64 choice a b :
  0<a -> a<b -> normal64 (b-a) ->
  normal64 (RN64 choice (b-a)/a) ->
  RN64 choice (RN64 choice (b-a)/a)<=16*binary64_u ->
  ln (b/a)<32*binary64_lambda.
Proof.
 intros Ha Hab Hns Hnd Htest.
 destruct (RN64_relative_factor choice (b-a) ltac:(lra) Hns) as [es [Hes Hsub]].
 assert (Hsubp:0<RN64 choice (b-a)).
 { apply guard_abs_bounds in Hes; pose proof binary64_u_bounds; rewrite Hsub; nra. }
 assert (Hdivp:0<RN64 choice (b-a)/a) by (apply Rdiv_lt_0_compat; lra).
 destruct (RN64_relative_factor choice (RN64 choice (b-a)/a) Hdivp Hnd) as [ed [Hed Hdiv]].
 apply (rounded_bracket_width a b es ed (RN64 choice (RN64 choice (b-a)/a)));
 try assumption; try lra.
 rewrite Hdiv,Hsub; reflexivity.
Qed.

(* An adjacent normal bracket is always much narrower than the outer guard.
   This lemma covers the rounded test, conditional on its two relative errors. *)
Lemma small_gap_passes_rounded_width a b es ed width :
  0<a -> a<=b -> b-a<=2*binary64_u*a ->
  Rabs es<=binary64_u -> Rabs ed<=binary64_u ->
  width=((b-a)*(1+es)/a)*(1+ed) -> width<=16*binary64_u.
Proof.
 intros Ha Hab Hgap Hes Hed Hwidth.
 pose proof binary64_u_bounds as Hu.
 apply guard_abs_bounds in Hes; apply guard_abs_bounds in Hed.
 set (w:=(b-a)/a).
 assert (Hw:0<=w<=2*binary64_u).
 { unfold w; split.
   - unfold Rdiv; apply Rmult_le_pos; [lra|left; apply Rinv_0_lt_compat; lra].
   - apply (Rmult_le_reg_r a); [lra|].
     replace ((b-a)/a*a) with (b-a) by (field; lra); exact Hgap. }
 assert (Hwidth':width=w*(1+es)*(1+ed)) by (unfold w; rewrite Hwidth; field; lra).
 assert (Hprod:0<(1+es)*(1+ed)<4).
 { assert (0<1+es<2) by lra; assert (0<1+ed<2) by lra; nra. }
 nra.
Qed.

Theorem adjacent_normal_bracket_passes a b es ed width :
  0<a -> normal64 a -> format64 a -> format64 b -> a<b ->
  (forall x, format64 x -> ~ (a<x<b)) ->
  Rabs es<=binary64_u -> Rabs ed<=binary64_u ->
  width=((b-a)*(1+es)/a)*(1+ed) -> width<=16*binary64_u.
Proof.
 intros Ha Hn Hfa Hfb Hab Hnone Hes Hed Hw.
 apply (small_gap_passes_rounded_width a b es ed width); try assumption; try lra.
 apply adjacent_format64_gap; assumption.
Qed.

Print Assumptions adjacent_normal_bracket_passes.
Print Assumptions rounded_bracket_width_RN64.

Theorem rounded_inner_bracket_width_RN64 choice a b :
  0<a -> a<b -> normal64 (b-a) ->
  normal64 (RN64 choice (b-a)/a) ->
  RN64 choice (RN64 choice (b-a)/a)<=4*binary64_u ->
  ln (b/a)<8*binary64_lambda.
Proof.
 intros Ha Hab Hns Hnd Htest.
 destruct (RN64_relative_factor choice (b-a) ltac:(lra) Hns) as [es [Hes Hsub]].
 assert (Hsubp:0<RN64 choice (b-a)).
 { apply guard_abs_bounds in Hes; pose proof binary64_u_bounds; rewrite Hsub; nra. }
 assert (Hdivp:0<RN64 choice (b-a)/a) by (apply Rdiv_lt_0_compat; lra).
 destruct (RN64_relative_factor choice (RN64 choice (b-a)/a) Hdivp Hnd) as [ed [Hed Hdiv]].
 apply (rounded_inner_bracket_width a b es ed (RN64 choice (RN64 choice (b-a)/a)));
 try assumption; try lra.
 rewrite Hdiv,Hsub; reflexivity.
Qed.

Lemma small_gap_passes_inner_width a b es ed width :
  0<a -> a<=b -> b-a<=2*binary64_u*a ->
  Rabs es<=binary64_u -> Rabs ed<=binary64_u ->
  width=((b-a)*(1+es)/a)*(1+ed) -> width<=4*binary64_u.
Proof.
 intros Ha Hab Hgap Hes Hed Hwidth.
 pose proof binary64_u_bounds as Hu.
 apply guard_abs_bounds in Hes; apply guard_abs_bounds in Hed.
 set (w:=(b-a)/a).
 assert (Hw:0<=w<=2*binary64_u).
 { unfold w; split.
   - unfold Rdiv; apply Rmult_le_pos; [lra|left; apply Rinv_0_lt_compat; lra].
   - apply (Rmult_le_reg_r a); [lra|].
     replace ((b-a)/a*a) with (b-a) by (field; lra); exact Hgap. }
 assert (Hwidth':width=w*(1+es)*(1+ed)) by (unfold w; rewrite Hwidth; field; lra).
 assert (Hprod:0<(1+es)*(1+ed)<2).
 { assert (0<1+es) by lra; assert (0<1+ed) by lra.
   assert ((1+es)*(1+ed)<=(1+binary64_u)^2) by nra.
   assert ((1+binary64_u)^2<2) by (unfold binary64_u; field_simplify; lra).
   nra. }
 nra.
Qed.

Theorem adjacent_normal_inner_bracket_passes a b es ed width :
  0<a -> normal64 a -> format64 a -> format64 b -> a<b ->
  (forall x, format64 x -> ~ (a<x<b)) ->
  Rabs es<=binary64_u -> Rabs ed<=binary64_u ->
  width=((b-a)*(1+es)/a)*(1+ed) -> width<=4*binary64_u.
Proof.
 intros Ha Hn Hfa Hfb Hab Hnone Hes Hed Hw.
 apply (small_gap_passes_inner_width a b es ed width); try assumption; try lra.
 apply adjacent_format64_gap; assumption.
Qed.
