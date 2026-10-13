(* Concrete binary64 acceptance guards. All results are conditional only on
   the stated real-valued evaluation-error and normal-rounding contracts. *)
From Coq Require Import Reals Psatz Field.
Open Scope R_scope.

Definition binary64_u : R := / 9007199254740992.
Definition binary64_lambda : R := - ln (1-binary64_u).

Lemma binary64_u_power : binary64_u = / (2^53).
Proof. unfold binary64_u; f_equal; ring. Qed.
Lemma binary64_u_bounds : 0 < binary64_u /\ binary64_u < / 1000000.
Proof. unfold binary64_u; split; apply Rinv_0_lt_compat || idtac;
  try lra; apply Rinv_lt_contravar; lra. Qed.

Lemma guard_abs_bounds x b : Rabs x<=b -> -b<=x<=b.
Proof. unfold Rabs; destruct (Rcase_abs x); lra. Qed.

Lemma guard_ln_le x y : 0<x -> x<=y -> ln x<=ln y.
Proof. intros Hx Hxy; destruct Hxy as [Hxy|Hxy].
 - left; apply ln_increasing; assumption.
 - subst; reflexivity.
Qed.

Lemma guard_ln_upper x : 0<x -> ln x<=x-1.
Proof.
 intro Hx; pose proof (exp_ineq1_le (ln x)).
 rewrite exp_ln in H by assumption; lra.
Qed.

Lemma guard_ln_lower x : 0<x -> (x-1)/x<=ln x.
Proof.
 intro Hx; pose proof (guard_ln_upper (/x) (Rinv_0_lt_compat x Hx)).
 rewrite ln_Rinv in H by assumption.
 replace ((x-1)/x) with (1-/x) by (field; lra).
 lra.
Qed.

Lemma binary64_lambda_bounds :
  binary64_u <= binary64_lambda <= binary64_u/(1-binary64_u).
Proof.
 destruct binary64_u_bounds as [Hu Hu']; unfold binary64_lambda.
 pose proof (guard_ln_upper (1-binary64_u) ltac:(lra)).
 pose proof (guard_ln_lower (1-binary64_u) ltac:(lra)).
 replace ((1-binary64_u-1)/(1-binary64_u)) with
   (- (binary64_u/(1-binary64_u))) in H0 by (field; lra).
 lra.
Qed.

Lemma binary64_lambda_positive : 0<binary64_lambda.
Proof. pose proof binary64_u_bounds; pose proof binary64_lambda_bounds; lra. Qed.

(* The arithmetic side conditions state exactly how much headroom is needed. *)
Lemma guard_window_general k y :
  0<=k -> k*(k+1)*binary64_u<=1 -> 0<1-k*binary64_u ->
  1-k*binary64_u<=y<=1+k*binary64_u ->
  Rabs (ln y)<=(k+1)*binary64_lambda.
Proof.
 intros Hk Hsmall Hpos Hy.
 assert (Hup:ln y<=k*binary64_u).
 { pose proof (guard_ln_upper y ltac:(lra)); lra. }
 assert (Hlow:-(k*binary64_u/(1-k*binary64_u))<=ln y).
 { pose proof (guard_ln_lower (1-k*binary64_u) Hpos).
   pose proof (guard_ln_le (1-k*binary64_u) y Hpos ltac:(lra)).
   replace ((1-k*binary64_u-1)/(1-k*binary64_u)) with
     (-(k*binary64_u/(1-k*binary64_u))) in H by (field; lra).
   lra. }
 assert (Hfrac:k*binary64_u/(1-k*binary64_u)<=(k+1)*binary64_u).
 { apply (Rmult_le_reg_r (1-k*binary64_u)); [lra|].
   field_simplify; pose proof (proj1 binary64_u_bounds); nra. }
 pose proof binary64_lambda_bounds.
 pose proof binary64_u_bounds.
 apply Rabs_le; split; nra.
Qed.

Lemma initial_measured_window zhat :
  1-64*binary64_u<=zhat<=1+64*binary64_u ->
  Rabs (ln zhat)<=65*binary64_lambda.
Proof.
 intro H; replace 65 with (64+1) by ring; apply (guard_window_general 64); try assumption;
 unfold binary64_u; field_simplify; lra.
Qed.

Lemma inner_measured_window hhat :
  1-16*binary64_u<=hhat<=1+16*binary64_u ->
  Rabs (ln hhat)<=17*binary64_lambda.
Proof.
 intro H; replace 17 with (16+1) by ring; apply (guard_window_general 16); try assumption;
 unfold binary64_u; field_simplify; lra.
Qed.

Lemma outer_measured_window shat :
  1-128*binary64_u<=shat<=1+128*binary64_u ->
  Rabs (ln shat)<=129*binary64_lambda.
Proof.
 intro H; replace 129 with (128+1) by ring; apply (guard_window_general 128); try assumption;
 unfold binary64_u; field_simplify; lra.
Qed.

Lemma guard_error_transfer measured exact rho beta :
  Rabs (ln measured)<=rho ->
  Rabs (ln measured-ln exact)<=beta ->
  Rabs (ln exact)<=rho+beta.
Proof.
 intros Hm He; replace (ln exact) with
   (ln measured + -(ln measured-ln exact)) by ring.
 eapply Rle_trans; [apply Rabs_triang|].
 rewrite Rabs_Ropp; lra.
Qed.

Lemma initial_exact_window zhat z :
  1-64*binary64_u<=zhat<=1+64*binary64_u ->
  Rabs (ln zhat-ln z)<=5*binary64_lambda ->
  Rabs (ln z)<=70*binary64_lambda.
Proof.
 intros Hw He; pose proof (initial_measured_window zhat Hw).
 pose proof (guard_error_transfer zhat z _ _ H He); lra.
Qed.

Lemma inner_exact_window hhat h :
  1-16*binary64_u<=hhat<=1+16*binary64_u ->
  Rabs (ln hhat-ln h)<=8*binary64_lambda ->
  Rabs (ln h)<=25*binary64_lambda.
Proof.
 intros Hw He; pose proof (inner_measured_window hhat Hw).
 pose proof (guard_error_transfer hhat h _ _ H He); lra.
Qed.

Lemma outer_exact_window shat s :
  1-128*binary64_u<=shat<=1+128*binary64_u ->
  Rabs (ln shat-ln s)<=71*binary64_lambda ->
  Rabs (ln s)<=200*binary64_lambda.
Proof.
 intros Hw He; pose proof (outer_measured_window shat Hw).
 pose proof (guard_error_transfer shat s _ _ H He); lra.
Qed.

Lemma guard_lower_sign k m measured exact :
  0<=k -> 0<=m -> m/(1-binary64_u)<k ->
  0<measured -> 0<exact -> measured<1-k*binary64_u ->
  Rabs (ln measured-ln exact)<=m*binary64_lambda -> exact<1.
Proof.
 intros Hk Hm Hgap Hme Hex Hwin Herr.
 pose proof binary64_u_bounds as Hu.
 pose proof binary64_lambda_bounds as Hl.
 pose proof (guard_ln_upper measured Hme) as Hlog.
 apply guard_abs_bounds in Herr.
 assert (Hbudget:m*binary64_lambda<k*binary64_u).
 { assert (m*(binary64_u/(1-binary64_u)) =
           binary64_u*(m/(1-binary64_u))) by (field; lra).
   nra. }
 apply (ln_lt_inv exact 1); try lra.
 rewrite ln_1; lra.
Qed.

Lemma guard_upper_sign k m measured exact :
  0<=k -> 0<=m -> 0<1+k*binary64_u ->
  m/(1-binary64_u)<k/(1+k*binary64_u) ->
  0<measured -> 0<exact -> 1+k*binary64_u<measured ->
  Rabs (ln measured-ln exact)<=m*binary64_lambda -> 1<exact.
Proof.
 intros Hk Hm Hpos Hgap Hme Hex Hwin Herr.
 pose proof binary64_u_bounds as Hu.
 pose proof binary64_lambda_bounds as Hl.
 pose proof (guard_ln_lower (1+k*binary64_u) Hpos) as Hlog.
 pose proof (ln_increasing (1+k*binary64_u) measured Hpos Hwin) as Hstrict.
 replace ((1+k*binary64_u-1)/(1+k*binary64_u)) with
   (k*binary64_u/(1+k*binary64_u)) in Hlog by (field; lra).
 apply guard_abs_bounds in Herr.
 assert (Hbudget:m*binary64_lambda<k*binary64_u/(1+k*binary64_u)).
 { assert (m*(binary64_u/(1-binary64_u)) =
           binary64_u*(m/(1-binary64_u))) by (field; lra).
   assert (k*binary64_u/(1+k*binary64_u) =
           binary64_u*(k/(1+k*binary64_u))) by (field; lra).
   nra. }
 apply (ln_lt_inv 1 exact); try lra.
 rewrite ln_1; lra.
Qed.

Lemma initial_lower_sign zhat z :
  0<zhat -> 0<z -> zhat<1-64*binary64_u ->
  Rabs (ln zhat-ln z)<=5*binary64_lambda -> z<1.
Proof. apply (guard_lower_sign 64 5); unfold binary64_u; field_simplify; lra. Qed.
Lemma initial_upper_sign zhat z :
  0<zhat -> 0<z -> 1+64*binary64_u<zhat ->
  Rabs (ln zhat-ln z)<=5*binary64_lambda -> 1<z.
Proof. apply (guard_upper_sign 64 5); unfold binary64_u; field_simplify; lra. Qed.
Lemma inner_lower_sign hhat h :
  0<hhat -> 0<h -> hhat<1-16*binary64_u ->
  Rabs (ln hhat-ln h)<=8*binary64_lambda -> h<1.
Proof. apply (guard_lower_sign 16 8); unfold binary64_u; field_simplify; lra. Qed.
Lemma inner_upper_sign hhat h :
  0<hhat -> 0<h -> 1+16*binary64_u<hhat ->
  Rabs (ln hhat-ln h)<=8*binary64_lambda -> 1<h.
Proof. apply (guard_upper_sign 16 8); unfold binary64_u; field_simplify; lra. Qed.
Lemma outer_lower_sign shat s :
  0<shat -> 0<s -> shat<1-128*binary64_u ->
  Rabs (ln shat-ln s)<=71*binary64_lambda -> s<1.
Proof. apply (guard_lower_sign 128 71); unfold binary64_u; field_simplify; lra. Qed.
Lemma outer_upper_sign shat s :
  0<shat -> 0<s -> 1+128*binary64_u<shat ->
  Rabs (ln shat-ln s)<=71*binary64_lambda -> 1<s.
Proof. apply (guard_upper_sign 128 71); unfold binary64_u; field_simplify; lra. Qed.

(* Two explicitly separate relative rounding errors: subtraction then division.
   They are valid for normal round-to-nearest results; exact subtraction uses 0. *)
Lemma rounded_bracket_width a b es ed width :
  0<a -> a<=b -> Rabs es<=binary64_u -> Rabs ed<=binary64_u ->
  width=((b-a)*(1+es)/a)*(1+ed) -> width<=16*binary64_u ->
  ln (b/a)<32*binary64_lambda.
Proof.
 intros Ha Hab Hes Hed Hwidth Htest.
 pose proof binary64_u_bounds as Hu.
 pose proof binary64_lambda_bounds as Hl.
 apply guard_abs_bounds in Hes; apply guard_abs_bounds in Hed.
 set (w:=(b-a)/a).
 assert (Hw:0<=w) by (unfold w; unfold Rdiv; apply Rmult_le_pos; [lra|left; apply Rinv_0_lt_compat; lra]).
 assert (Hwidth':width=w*(1+es)*(1+ed)) by (unfold w; rewrite Hwidth; field; lra).
 assert (Hprod:(1-binary64_u)^2<=(1+es)*(1+ed)).
 { assert (0<1+es) by lra.
   assert (0<1+ed) by lra.
   assert (0<1-binary64_u) by lra.
   nra. }
 assert (Hhalf: /2<(1-binary64_u)^2).
 { unfold binary64_u; field_simplify; lra. }
 assert (Hbound:w<32*binary64_u) by nra.
 assert (Hratio:0<b/a) by (apply Rdiv_lt_0_compat; lra).
 pose proof (guard_ln_upper (b/a) Hratio).
 assert (b/a-1=w) by (unfold w; field; lra).
 nra.
Qed.

Print Assumptions initial_exact_window.
Print Assumptions outer_upper_sign.
Print Assumptions rounded_bracket_width.

Lemma binary64_lambda_small : binary64_lambda<= /1024.
Proof.
 pose proof binary64_lambda_bounds as [_ H].
 assert (binary64_u/(1-binary64_u)<= /1024).
 { unfold binary64_u; field_simplify; lra. }
 lra.
Qed.

Lemma binary64_boundary_small : 25*binary64_lambda<=ln(4/3).
Proof.
 pose proof binary64_lambda_small.
 pose proof (guard_ln_lower (4/3) ltac:(lra)).
 replace ((4/3-1)/(4/3)) with (1/4) in H0 by field.
 lra.
Qed.

Lemma rounded_inner_bracket_width a b es ed width :
  0<a -> a<=b -> Rabs es<=binary64_u -> Rabs ed<=binary64_u ->
  width=((b-a)*(1+es)/a)*(1+ed) -> width<=4*binary64_u ->
  ln (b/a)<8*binary64_lambda.
Proof.
 intros Ha Hab Hes Hed Hwidth Htest.
 pose proof binary64_u_bounds as Hu.
 pose proof binary64_lambda_bounds as Hl.
 apply guard_abs_bounds in Hes; apply guard_abs_bounds in Hed.
 set (w:=(b-a)/a).
 assert (Hw:0<=w) by (unfold w,Rdiv; apply Rmult_le_pos; [lra|left; apply Rinv_0_lt_compat; lra]).
 assert (Hwidth':width=w*(1+es)*(1+ed)) by (unfold w; rewrite Hwidth; field; lra).
 assert (Hprod:(1-binary64_u)^2<=(1+es)*(1+ed)).
 { assert (0<1+es) by lra; assert (0<1+ed) by lra.
   assert (0<1-binary64_u) by lra; nra. }
 assert (Hhalf: /2<(1-binary64_u)^2).
 { unfold binary64_u; field_simplify; lra. }
 assert (Hbound:w<8*binary64_u) by nra.
 assert (Hratio:0<b/a) by (apply Rdiv_lt_0_compat; lra).
 pose proof (guard_ln_upper (b/a) Hratio).
 assert (b/a-1=w) by (unfold w; field; lra).
 nra.
Qed.
