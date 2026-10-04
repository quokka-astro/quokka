(* Width arithmetic at close floating-point endpoints: Sterbenz subtraction
   is exact even when the difference is subnormal. No scaling is hidden. *)
From Coq Require Import Reals Psatz Field ZArith Lia.
Require Import Flocq.Core.Core Flocq.Prop.Sterbenz.
From BlackBox Require Import FloatingPoint Guards GuardFloatBridge.
Open Scope R_scope.
Local Instance width_prec64 : Prec_gt_0 53. Proof. unfold Prec_gt_0; lia. Qed.

Theorem RN64_sterbenz_difference choice a b :
 0<a -> format64 a -> format64 b -> a<=b<=2*a ->
 RN64 choice (b-a)=b-a.
Proof.
 intros Ha Hfa Hfb Hab. unfold RN64.
 apply round_generic; [typeclasses eauto|].
 unfold format64 in *.
 apply (sterbenz radix2 (FLT_exp (-1074) 53)); try typeclasses eauto; try assumption; lra.
Qed.

Lemma RN64_sterbenz_relative_factor choice a b :
 0<a -> format64 a -> format64 b -> a<=b<=2*a ->
 Rabs 0<=binary64_u /\ RN64 choice(b-a)=(b-a)*(1+0).
Proof.
 intros Ha Hfa Hfb Hab; split.
 - rewrite Rabs_R0. pose proof binary64_u_bounds; lra.
 - rewrite (RN64_sterbenz_difference choice a b Ha Hfa Hfb Hab); ring.
Qed.

Definition SafeDifference64 (a b:R) : Prop :=
 normal64(b-a) \/ (format64 a /\ format64 b /\ b<=2*a).

Lemma safe_difference_relative_factor choice a b :
 0<a -> a<b -> SafeDifference64 a b ->
 exists es, Rabs es<=binary64_u /\ RN64 choice(b-a)=(b-a)*(1+es).
Proof.
 intros Ha Hab [Hnormal|[Hfa [Hfb Hclose]]].
 - apply RN64_relative_factor; lra || assumption.
 - exists 0; apply RN64_sterbenz_relative_factor; assumption || lra.
Qed.

(* For any pair of positive normal formatted endpoints, the near case is
   exact and the far case has a normal difference. Thus no hypothesis that
   every subtraction result is normal is needed on a normal grid. *)
Lemma normal_formatted_endpoints_safe a b :
 0<a -> normal64 a -> format64 a -> format64 b -> a<=b ->
 SafeDifference64 a b.
Proof.
 intros Ha Hna Hfa Hfb Hab.
 destruct (Rle_dec b (2*a)) as [Hclose|Hfar].
 - right; auto.
 - left; unfold normal64 in *.
   rewrite Rabs_pos_eq in Hna by lra.
   rewrite Rabs_pos_eq by lra. lra.
Qed.

Lemma safe_width_relative_factors choice a b :
 0<a -> a<b -> SafeDifference64 a b ->
 normal64(RN64 choice(b-a)/a) ->
 exists es ed, Rabs es<=binary64_u /\ Rabs ed<=binary64_u /\
 RN64 choice(RN64 choice(b-a)/a)=((b-a)*(1+es)/a)*(1+ed).
Proof.
 intros Ha Hab Hsafe Hdivn.
 destruct (safe_difference_relative_factor choice a b Ha Hab Hsafe) as [es [Hes Hsub]].
 assert (Hsubp:0<RN64 choice(b-a)).
 { pose proof binary64_u_bounds. pose proof (guard_abs_bounds es binary64_u Hes).
   rewrite Hsub; nra. }
 assert (Hdivp:0<RN64 choice(b-a)/a) by (apply Rdiv_lt_0_compat; assumption).
 destruct (RN64_relative_factor choice _ Hdivp Hdivn) as [ed [Hed Hdiv]].
 exists es,ed; repeat split; try assumption. rewrite Hdiv,Hsub; reflexivity.
Qed.

Theorem rounded_inner_width_safe_RN64 choice a b :
 0<a -> a<b -> SafeDifference64 a b ->
 normal64(RN64 choice(b-a)/a) ->
 RN64 choice(RN64 choice(b-a)/a)<=4*binary64_u ->
 ln(b/a)<8*binary64_lambda.
Proof.
 intros Ha Hab Hsafe Hdivn Htest.
 destruct (safe_width_relative_factors choice a b Ha Hab Hsafe Hdivn) as [es [ed [Hes [Hed Heq]]]].
 apply (rounded_inner_bracket_width a b es ed (RN64 choice(RN64 choice(b-a)/a))); assumption || lra.
Qed.

Theorem rounded_outer_width_safe_RN64 choice a b :
 0<a -> a<b -> SafeDifference64 a b ->
 normal64(RN64 choice(b-a)/a) ->
 RN64 choice(RN64 choice(b-a)/a)<=16*binary64_u ->
 ln(b/a)<32*binary64_lambda.
Proof.
 intros Ha Hab Hsafe Hdivn Htest.
 destruct (safe_width_relative_factors choice a b Ha Hab Hsafe Hdivn) as [es [ed [Hes [Hed Heq]]]].
 apply (rounded_bracket_width a b es ed (RN64 choice(RN64 choice(b-a)/a))); assumption || lra.
Qed.

Corollary normal_grid_width_relative_factors choice a b :
 0<a -> normal64 a -> format64 a -> format64 b -> a<b ->
 normal64(RN64 choice(b-a)/a) ->
 exists es ed, Rabs es<=binary64_u /\ Rabs ed<=binary64_u /\
 RN64 choice(RN64 choice(b-a)/a)=((b-a)*(1+es)/a)*(1+ed).
Proof.
 intros; apply safe_width_relative_factors; try assumption.
 apply normal_formatted_endpoints_safe; assumption || lra.
Qed.

Print Assumptions RN64_sterbenz_difference.
Print Assumptions rounded_inner_width_safe_RN64.
Print Assumptions normal_grid_width_relative_factors.

Corollary normal_grid_inner_width_RN64 choice a b :
 0<a -> normal64 a -> format64 a -> format64 b -> a<b ->
 normal64(RN64 choice(b-a)/a) ->
 RN64 choice(RN64 choice(b-a)/a)<=4*binary64_u ->
 ln(b/a)<8*binary64_lambda.
Proof.
 intros; apply (rounded_inner_width_safe_RN64 choice a b); try assumption.
 apply normal_formatted_endpoints_safe; assumption || lra.
Qed.
Corollary normal_grid_outer_width_RN64 choice a b :
 0<a -> normal64 a -> format64 a -> format64 b -> a<b ->
 normal64(RN64 choice(b-a)/a) ->
 RN64 choice(RN64 choice(b-a)/a)<=16*binary64_u ->
 ln(b/a)<32*binary64_lambda.
Proof.
 intros; apply (rounded_outer_width_safe_RN64 choice a b); try assumption.
 apply normal_formatted_endpoints_safe; assumption || lra.
Qed.
