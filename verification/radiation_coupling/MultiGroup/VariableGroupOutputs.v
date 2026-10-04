(* Variable coefficient output sensitivity, independently derived from the
   physical positive quotient and the actual pointwise RN64 output graph. *)
From Coq Require Import Reals Psatz Field.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import FloatingPoint LogCalculus Guards GuardFloatBridge.
From MultiGroup Require Import MultigroupGraph ComponentAccuracy.
Open Scope R_scope.

Definition variable_energy h r (alpha p B:R->R) x :=
 (r+h*p x*B x)/(1+h*alpha x).
Definition variable_energy_der h r (alpha ap p pp B Bp:R->R) x :=
 ((h*(pp x*B x+p x*Bp x))*(1+h*alpha x)-
  (r+h*p x*B x)*h*ap x)/(1+h*alpha x)^2.
Definition emitted_share h r (p B:R->R) x := h*p x*B x/(r+h*p x*B x).
Definition absorption_share h (alpha:R->R) x := h*alpha x/(1+h*alpha x).

Lemma variable_energy_positive h r alpha p B x :
 0<h -> 0<=r -> 0<alpha x -> 0<p x -> 0<B x ->
 0<variable_energy h r alpha p B x.
Proof.
 intros Hh Hr Ha Hp HB; unfold variable_energy.
 assert (0<h*p x*B x) by (repeat apply Rmult_lt_0_compat; assumption).
 apply Rdiv_lt_0_compat; nra.
Qed.

Lemma variable_energy_is_derive h r alpha ap p pp B Bp x :
 0<h -> 0<alpha x ->
 is_derive alpha x (ap x) -> is_derive p x (pp x) -> is_derive B x (Bp x) ->
 is_derive (variable_energy h r alpha p B) x (variable_energy_der h r alpha ap p pp B Bp x).
Proof.
 intros Hh Ha HaD HpD HBD; unfold variable_energy,variable_energy_der.
 auto_derive.
 - repeat split; try (exists (ap x); exact HaD); try (exists (pp x); exact HpD);
    try (exists (Bp x); exact HBD); nra.
 - rewrite (is_derive_unique (fun y:R=>alpha y) x _ HaD),
    (is_derive_unique (fun y:R=>p y) x _ HpD),
    (is_derive_unique (fun y:R=>B y) x _ HBD); field; nra.
Qed.

Lemma variable_energy_elasticity_identity h r alpha ap p pp B Bp x :
 0<h -> 0<=r -> 0<alpha x -> 0<p x -> 0<B x ->
 x*variable_energy_der h r alpha ap p pp B Bp x/variable_energy h r alpha p B x=
 emitted_share h r p B x*(x*pp x/p x+x*Bp x/B x)-
 absorption_share h alpha x*(x*ap x/alpha x).
Proof.
 intros Hh Hr Ha Hp HB.
 assert (Hprod:0<h*p x*B x) by (repeat apply Rmult_lt_0_compat; assumption).
 assert (HN:0<r+h*p x*B x) by lra.
 assert (HD:0<1+h*alpha x) by nra.
 unfold variable_energy_der,variable_energy,emitted_share,absorption_share.
 field; repeat split; lra.
Qed.

Lemma variable_output_share_bounds h r alpha p B x :
 0<h -> 0<=r -> 0<alpha x -> 0<p x -> 0<B x ->
 0<=emitted_share h r p B x<=1 /\ 0<=absorption_share h alpha x<=1.
Proof.
 intros Hh Hr Ha Hp HB.
 assert (Hprod:0<h*p x*B x) by (repeat apply Rmult_lt_0_compat; assumption).
 assert (HN:0<r+h*p x*B x) by lra.
 assert (HD:0<1+h*alpha x) by nra.
 unfold emitted_share,absorption_share.
 split; split.
 - apply Rdiv_le_0_compat; nra.
 - apply (Rmult_le_reg_r (r+h*p x*B x)); [exact HN|]; field_simplify; nra.
 - apply Rdiv_le_0_compat; nra.
 - apply (Rmult_le_reg_r (1+h*alpha x)); [exact HD|]; field_simplify; nra.
Qed.

Lemma weighted_variable_slope_bound s f P K beta Pmax Kmax Bmax :
 0<=s<=1 -> 0<=f<=1 -> Rabs P<=Pmax -> Rabs K<=Kmax -> Rabs beta<=Bmax ->
 Rabs(s*(P+beta)-f*K)<=Pmax+Kmax+Bmax.
Proof.
 intros Hs Hf HP HK HB.
 assert (HP0:0<=Pmax) by (pose proof (Rabs_pos P); lra).
 assert (HK0:0<=Kmax) by (pose proof (Rabs_pos K); lra).
 assert (HB0:0<=Bmax) by (pose proof (Rabs_pos beta); lra).
 pose proof (Rabs_triang (s*(P+beta)) (-f*K)) as Htri.
 rewrite Rabs_mult, (Rabs_pos_eq s) in Htri by lra.
 replace (-f*K) with (-(f*K)) in Htri by ring.
 rewrite Rabs_Ropp,Rabs_mult,(Rabs_pos_eq f) in Htri by lra.
 pose proof (Rabs_triang P beta) as Hsum.
 assert (HsumBound:Rabs(P+beta)<=Pmax+Bmax) by lra.
 assert (HsBound:s*Rabs(P+beta)<=Pmax+Bmax).
 { eapply Rle_trans; [apply Rmult_le_compat_l; [lra|exact HsumBound]|nra]. }
 assert (HfBound:f*Rabs K<=Kmax).
 { eapply Rle_trans; [apply Rmult_le_compat_l; [lra|exact HK]|nra]. }
 unfold Rminus; lra.
Qed.

Lemma variable_energy_elasticity_bound h r alpha ap p pp B Bp x Pmax Kmax Bmax :
 0<h -> 0<=r -> 0<alpha x -> 0<p x -> 0<B x ->
 Rabs(x*pp x/p x)<=Pmax -> Rabs(x*ap x/alpha x)<=Kmax -> Rabs(x*Bp x/B x)<=Bmax ->
 Rabs(x*variable_energy_der h r alpha ap p pp B Bp x/variable_energy h r alpha p B x)
 <=Pmax+Kmax+Bmax.
Proof.
 intros Hh Hr Ha Hp HB HP HK HBeta.
 rewrite variable_energy_elasticity_identity by assumption.
 destruct (variable_output_share_bounds h r alpha p B x Hh Hr Ha Hp HB) as [HS HF].
 apply weighted_variable_slope_bound; assumption.
Qed.

Theorem variable_band_finite_transfer h r alpha ap p pp B Bp root trial eh eta b Pmax Kmax Bmax :
 0<h -> 0<=r -> 0<root -> 0<trial -> 0<=Pmax -> 0<=Kmax -> 0<=Bmax ->
 (forall x, Rmin root trial<=x<=Rmax root trial -> 0<alpha x /\ 0<p x /\ 0<B x) ->
 (forall x, Rmin root trial<=x<=Rmax root trial -> is_derive alpha x (ap x) /\ is_derive p x (pp x) /\ is_derive B x (Bp x)) ->
 (forall x, Rmin root trial<x<Rmax root trial ->
  Rabs(x*pp x/p x)<=Pmax /\ Rabs(x*ap x/alpha x)<=Kmax /\ Rabs(x*Bp x/B x)<=Bmax) ->
 log_error trial root<=b -> LogBound eta eh (variable_energy h r alpha p B trial) ->
 LogBound (eta+(Pmax+Kmax+Bmax)*b) eh (variable_energy h r alpha p B root) /\
 Rabs(eh/variable_energy h r alpha p B root-1)<=exp(eta+(Pmax+Kmax+Bmax)*b)-1.
Proof.
 intros Hh Hr HR HX HP HK HB Hpos HDer Hslope Hcoord HE.
 apply (positive_component_finite_transfer (variable_energy h r alpha p B)
  (variable_energy_der h r alpha ap p pp B Bp) root trial eh eta (Pmax+Kmax+Bmax) b); try assumption; try lra.
 - intros x Hx; destruct (Hpos x Hx) as [Ha [Hp HBx]]; apply variable_energy_positive; assumption.
 - intros x Hx; destruct (Hpos x Hx) as [Ha [Hp HBx]]; destruct (HDer x Hx) as [Had [Hpd HBd]].
   apply (is_derive_continuity_pt _ _ (variable_energy_der h r alpha ap p pp B Bp x)).
   apply variable_energy_is_derive; assumption.
 - intros x Hx; destruct (Hpos x ltac:(lra)) as [Ha [Hp HBx]]; destruct (HDer x ltac:(lra)) as [Had [Hpd HBd]].
   apply variable_energy_is_derive; assumption.
 - intros x Hx; destruct (Hpos x ltac:(lra)) as [Ha [Hp HBx]]; destruct (Hslope x Hx) as [HPx [HKx HBx']].
   apply variable_energy_elasticity_bound; assumption.
Qed.

Theorem variable_band_RN64_finite_accuracy choice h r alpha ap p pp B Bp root trial ah ph bh
 ea ep eb b Pmax Kmax Bmax :
 0<h -> 0<=r -> 0<root -> 0<trial -> 0<=ea -> 0<=ep -> 0<=eb ->
 0<=Pmax -> 0<=Kmax -> 0<=Bmax ->
 (forall x, Rmin root trial<=x<=Rmax root trial -> 0<alpha x /\ 0<p x /\ 0<B x) ->
 (forall x, Rmin root trial<=x<=Rmax root trial -> is_derive alpha x (ap x) /\ is_derive p x (pp x) /\ is_derive B x (Bp x)) ->
 (forall x, Rmin root trial<x<Rmax root trial ->
  Rabs(x*pp x/p x)<=Pmax /\ Rabs(x*ap x/alpha x)<=Kmax /\ Rabs(x*Bp x/B x)<=Bmax) ->
 log_error trial root<=b ->
 ZLogBound ea ah (alpha trial) -> ZLogBound ep ph (p trial) -> ZLogBound eb bh (B trial) ->
 ZFiniteNormalNodes choice (E_nodes (RN64 choice) h ah ph bh r) ->
 LogBound (ea+ep+eb+6*binary64_lambda+(Pmax+Kmax+Bmax)*b)
 (g_E (RN64 choice) h ah ph bh r) (variable_energy h r alpha p B root) /\
 Rabs(g_E (RN64 choice) h ah ph bh r/variable_energy h r alpha p B root-1)
 <=exp(ea+ep+eb+6*binary64_lambda+(Pmax+Kmax+Bmax)*b)-1.
Proof.
 intros Hh Hr HR HX Hea Hep Heb HP HK HB Hpos HDer Hslope Hcoord Ha Hp HBval HN.
 assert (Htrial:Rmin root trial<=trial<=Rmax root trial) by (split; [apply Rmin_r|apply Rmax_r]).
 destruct (Hpos trial Htrial) as [Halpha [Hpval HBpos]].
 pose proof (group_E_RN64_budget choice ea ep eb 0 h ah (alpha trial) ph (p trial)
  bh (B trial) r r Hea Hh Ha Hp HBval (ZLogBound_exact 0 r ltac:(lra) Hr) HN) as HE.
 rewrite <-guard_lambda64,Rmax_right in HE by (pose proof binary64_lambda_positive; lra).
 replace (ep+eb+2*binary64_lambda+ea+4*binary64_lambda)
  with (ea+ep+eb+6*binary64_lambda) in HE by ring.
 assert (Houtput:0<variable_energy h r alpha p B trial) by (apply variable_energy_positive; assumption).
 apply ZLogBound_positive in HE; [|exact Houtput].
 apply (variable_band_finite_transfer h r alpha ap p pp B Bp root trial
  (g_E (RN64 choice) h ah ph bh r) (ea+ep+eb+6*binary64_lambda) b Pmax Kmax Bmax); assumption.
Qed.

Print Assumptions variable_band_RN64_finite_accuracy.
