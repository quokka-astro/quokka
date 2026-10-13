(* Optional sharpened component bounds: the emitted-energy fraction multiplies
   Planck sensitivity.  Every factor is bounded over the entire interval. *)
From Coq Require Import Reals Psatz Field.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import LogCalculus FloatingPoint.
From MultiGroup Require Import ComponentAccuracy.
Open Scope R_scope.

Definition emitted_fraction h p r Bx := h*p*Bx/(r+h*p*Bx).

Lemma emitted_fraction_bounds h p r Bx :
 0<h -> 0<=p -> 0<=r -> 0<Bx -> (0<r \/ 0<p) ->
 0<=emitted_fraction h p r Bx<=1.
Proof.
 intros Hh Hp Hr HB Hactive.
 pose proof (band_numerator_positive h p r Bx Hh Hp Hr HB Hactive) as Hnum.
 assert (Hprod:0<=h*p*Bx) by (repeat apply Rmult_le_pos; lra).
 unfold emitted_fraction; split.
 - unfold Rdiv; apply Rmult_le_pos; [exact Hprod|left; apply Rinv_0_lt_compat; assumption].
 - apply (Rmult_le_reg_r (r+h*p*Bx)); [exact Hnum|]. field_simplify; nra.
Qed.

Lemma emitted_fraction_nondecreasing h p r Bx By :
 0<h -> 0<=p -> 0<=r -> 0<Bx -> Bx<=By -> (0<r \/ 0<p) ->
 emitted_fraction h p r Bx<=emitted_fraction h p r By.
Proof.
 intros Hh Hp Hr HB Hxy Hactive.
 assert (HBy:0<By) by lra.
 pose proof (band_numerator_positive h p r Bx Hh Hp Hr HB Hactive) as Hnumx.
 pose proof (band_numerator_positive h p r By Hh Hp Hr HBy Hactive) as Hnumy.
 assert (Hhp:0<=h*p) by nra.
 assert (Hdiff:0<=r*(h*p)*(By-Bx)) by (repeat apply Rmult_le_pos; lra).
 unfold emitted_fraction.
 apply (Rmult_le_reg_r ((r+h*p*Bx)*(r+h*p*By))); [nra|].
 field_simplify; nra.
Qed.

Lemma band_energy_elasticity_factor h alpha p r B Bp x :
 0<h -> 0<=alpha -> 0<=p -> 0<=r -> 0<B x -> (0<r \/ 0<p) ->
 x*band_energy_derivative h alpha p Bp x/band_energy h alpha p r B x =
 emitted_fraction h p r (B x)*(x*Bp x/B x).
Proof.
 intros Hh Ha Hp Hr HB Hactive.
 pose proof (band_numerator_positive h p r (B x) Hh Hp Hr HB Hactive) as Hnum.
 assert (Hden:0<1+h*alpha) by nra.
 unfold band_energy_derivative,band_energy,emitted_fraction.
 field; split; nra.
Qed.

Theorem radiation_band_diluted_finite_accuracy h alpha p r B Bp root trial eh eta W L b :
 0<h -> 0<=alpha -> 0<=p -> 0<=r -> (0<r \/ 0<p) ->
 0<root -> 0<trial -> 0<=W -> 0<=L ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<B t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt B t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive B t (Bp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial -> Rabs(t*Bp t/B t)<=L) ->
 (forall t, Rmin root trial<t<Rmax root trial -> emitted_fraction h p r (B t)<=W) ->
 log_error trial root<=b -> LogBound eta eh (band_energy h alpha p r B trial) ->
 LogBound (eta+W*L*b) eh (band_energy h alpha p r B root) /\
 Rabs(eh/band_energy h alpha p r B root-1)<=exp(eta+W*L*b)-1.
Proof.
 intros Hh Ha Hp Hr Hactive Hroot Htrial HW HL HB Hcont Hder Hslope Hfrac Hcoord Heval.
 apply (positive_component_finite_transfer (band_energy h alpha p r B)
   (band_energy_derivative h alpha p Bp) root trial eh eta (W*L) b);
   try assumption; try nra.
 - intros t Ht; apply band_energy_positive; try assumption; apply HB; exact Ht.
 - intros t Ht; apply band_energy_continuous; try assumption; apply Hcont; exact Ht.
 - intros t Ht; apply band_energy_is_derive; try assumption; apply Hder; exact Ht.
 - intros t Ht.
   assert (HBt:0<B t) by (apply HB; lra).
   rewrite (band_energy_elasticity_factor h alpha p r B Bp t Hh Ha Hp Hr HBt Hactive), Rabs_mult.
   pose proof (emitted_fraction_bounds h p r (B t) Hh Hp Hr HBt Hactive) as Hwb.
   rewrite (Rabs_pos_eq (emitted_fraction h p r (B t))) by lra.
   pose proof (Hslope t Ht); pose proof (Hfrac t Ht).
   eapply Rle_trans; [apply Rmult_le_compat_l; eassumption || lra|nra].
Qed.

(* U may be the positive upper bracket endpoint.  Monotonicity of B on the
   bracket establishes Hupper, which is weaker than a global monotonicity axiom. *)
Theorem radiation_band_endpoint_diluted_finite_accuracy h alpha p r B Bp root trial U eh eta L b :
 0<h -> 0<=alpha -> 0<=p -> 0<=r -> (0<r \/ 0<p) ->
 0<root -> 0<trial -> 0<=L -> 0<B U ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<B t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> B t<=B U) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt B t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive B t (Bp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial -> Rabs(t*Bp t/B t)<=L) ->
 log_error trial root<=b -> LogBound eta eh (band_energy h alpha p r B trial) ->
 LogBound (eta+emitted_fraction h p r (B U)*L*b) eh (band_energy h alpha p r B root) /\
 Rabs(eh/band_energy h alpha p r B root-1)
  <=exp(eta+emitted_fraction h p r (B U)*L*b)-1.
Proof.
 intros Hh Ha Hp Hr Hactive Hroot Htrial HL HBU HB Hupper Hcont Hder Hslope Hcoord Heval.
 apply (radiation_band_diluted_finite_accuracy h alpha p r B Bp root trial eh eta
  (emitted_fraction h p r (B U)) L b); try assumption.
 - exact (proj1 (emitted_fraction_bounds h p r (B U) Hh Hp Hr HBU Hactive)).
 - intros t Ht; apply emitted_fraction_nondecreasing; try assumption.
   + apply HB; lra.
   + apply Hupper; lra.
Qed.

(* Degenerate emission is a constant-output branch.  It needs no positive
   Planck band and incurs no coordinate-sensitivity error.  The local LogBound
   supplies positivity of the surviving component itself. *)
Theorem coordinate_independent_band_finite_accuracy h alpha p r B root trial eh eta :
 (p=0 \/ B trial=B root) ->
 LogBound eta eh (band_energy h alpha p r B trial) ->
 LogBound eta eh (band_energy h alpha p r B root) /\
 Rabs(eh/band_energy h alpha p r B root-1)<=exp eta-1.
Proof.
 intros Hconstant Heval.
 assert (HE:band_energy h alpha p r B trial=band_energy h alpha p r B root).
 { unfold band_energy; destruct Hconstant as [Hp|HB].
   - subst p; unfold Rdiv; ring.
   - rewrite HB; reflexivity. }
 rewrite HE in Heval.
 split; [exact Heval|].
 destruct Heval as [Heh [HEpos Hlog]].
 apply log_error_relative; assumption.
Qed.

Lemma identically_zero_band h alpha p B x :
 (p=0 \/ B x=0) -> band_energy h alpha p 0 B x=0.
Proof.
 intro H; unfold band_energy; destruct H as [Hp|HB];
 rewrite Hp || rewrite HB; unfold Rdiv; ring.
Qed.

Print Assumptions radiation_band_diluted_finite_accuracy.
Print Assumptions radiation_band_endpoint_diluted_finite_accuracy.
