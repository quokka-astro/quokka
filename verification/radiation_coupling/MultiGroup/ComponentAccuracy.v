(* Finite component accuracy for the constant-opacity multigroup reduction.
   No desired final component error is a hypothesis.  The hypotheses describe
   coordinate error, local evaluator graphs, and analytic Planck-band behavior
   on the ENTIRE root/trial interval.  Exponentials retain finite errors. *)
From Coq Require Import Reals Psatz Field Lia List.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import LogCalculus GasMap GasBounds FloatingPoint
 Binary64Accuracy.
From MultiGroup Require Import ConstantGroupExistence FullGroupReconstruction ConstantGroupAccuracy.
Import ListNotations.
Open Scope R_scope.

Lemma local_logbound_error eta yh y :
 LogBound eta yh y -> log_error yh y <= eta.
Proof. intros [_ [_ H]]; exact H. Qed.

(* Generic derivative-to-finite-output transfer, with local evaluation error. *)
Theorem positive_component_finite_transfer f df root trial yh eta L b :
 0<root -> 0<trial -> 0<=L ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<f t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt f t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive f t (df t)) ->
 (forall t, Rmin root trial<t<Rmax root trial -> Rabs(t*df t/f t)<=L) ->
 log_error trial root<=b -> LogBound eta yh (f trial) ->
 LogBound (eta+L*b) yh (f root) /\
 Rabs(yh/f root-1)<=exp(eta+L*b)-1.
Proof.
 intros Hr Ht HL Hpos Hcont Hder Hslope Hcoord Heval.
 pose proof (elasticity_upper_distance f df root trial L Hr Ht HL
   Hpos Hcont Hder Hslope) as Hsens.
 pose proof (local_logbound_error _ _ _ Heval) as Hlocal.
 pose proof (log_error_triangle yh (f trial) (f root)) as Htri.
 assert (Hroot:0<f root).
 { apply Hpos; split; [apply Rmin_l|apply Rmax_l]. }
 assert (Hfinal:log_error yh (f root)<=eta+L*b) by nra.
 assert (Hy:0<yh) by exact (proj1 Heval).
 split; [repeat split; assumption|].
 apply log_error_relative; assumption.
Qed.

Theorem dust_finite_accuracy root trial b :
 0<root -> 0<trial -> log_error trial root<=b ->
 Rabs(trial/root-1)<=exp b-1.
Proof. intros; apply log_error_relative; assumption. Qed.

(* The 53 units are charged, not assumed: 52 from the accepted gray inner
   solve and one from the final gas-energy multiplication. *)
Theorem gas_finite_accuracy64 choice cg A D T root trial th qh b :
 0<cg -> 0<A -> 0<D -> 0<T -> 0<root -> 0<trial ->
 InnerEvaluator.InnerResult choice A D T trial th qh ->
 FiniteNormalNodes choice [cg*th] -> log_error trial root<=b ->
 LogBound (53*Guards.binary64_lambda+2*b)
  (returned_gas choice cg th) (gas_energy cg A D T root) /\
 Rabs(returned_gas choice cg th/gas_energy cg A D T root-1)
  <=exp(53*Guards.binary64_lambda+2*b)-1.
Proof.
 intros Hcg HA HD HT Hr Ht HI Hnodes Hcoord.
 pose proof (returned_gas_concrete choice cg A D T trial th qh
   Hcg HA HD HT Ht HI Hnodes) as Heval.
 pose proof (gas_energy_log_lipschitz cg A D T root trial
   Hcg HA HD HT Hr Ht) as Hsens.
 pose proof (local_logbound_error _ _ _ Heval) as Hlocal.
 pose proof (log_error_triangle (returned_gas choice cg th)
  (gas_energy cg A D T trial) (gas_energy cg A D T root)) as Htri.
 assert (Hy:0<returned_gas choice cg th) by exact (proj1 Heval).
 assert (Hroot:0<gas_energy cg A D T root)
  by (apply gas_energy_positive; assumption).
 assert (Hfinal:log_error (returned_gas choice cg th)
   (gas_energy cg A D T root)<=53*Guards.binary64_lambda+2*b) by lra.
 split; [repeat split; assumption|].
 apply log_error_relative; assumption.
Qed.

(* A scalar form makes the coefficient algebra transparent.  This is exactly
   FullGroupReconstruction.group_energy for any selected group. *)
Definition band_energy h alpha p r (B:R->R) x :=
 (r+h*p*B x)/(1+h*alpha).
Definition band_energy_derivative h alpha p (Bp:R->R) x :=
 h*p*Bp x/(1+h*alpha).

Lemma band_energy_matches_group h alpha p r B g x :
 band_energy h (alpha g) (p g) (r g) (B g) x =
 group_energy h alpha p r B g x.
Proof. reflexivity. Qed.

Lemma band_numerator_positive h p r Bx :
 0<h -> 0<=p -> 0<=r -> 0<Bx -> (0<r \/ 0<p) -> 0<r+h*p*Bx.
Proof.
 intros Hh Hp Hr HB Hactive.
 assert (Hprod:0<=h*p*Bx) by (repeat apply Rmult_le_pos; lra).
 destruct Hactive; [lra|].
 assert (Hpos:0<h*p*Bx) by (repeat apply Rmult_lt_0_compat; assumption).
 lra.
Qed.

Lemma band_energy_positive h alpha p r B x :
 0<h -> 0<=alpha -> 0<=p -> 0<=r -> 0<B x ->
 (0<r \/ 0<p) -> 0<band_energy h alpha p r B x.
Proof.
 intros Hh Ha Hp Hr HB Hactive; unfold band_energy.
 apply Rdiv_lt_0_compat; [apply band_numerator_positive; assumption|nra].
Qed.

Lemma band_energy_continuous h alpha p r B x :
 0<h -> 0<=alpha -> continuity_pt B x ->
 continuity_pt (band_energy h alpha p r B) x.
Proof. intros; unfold band_energy; reg; assumption || nra. Qed.

Lemma band_energy_is_derive h alpha p r B Bp x :
 0<h -> 0<=alpha -> is_derive B x (Bp x) ->
 is_derive (band_energy h alpha p r B) x
  (band_energy_derivative h alpha p Bp x).
Proof.
 intros Hh Ha HD; unfold band_energy,band_energy_derivative; auto_derive.
 - repeat split; try (exists (Bp x); exact HD); nra.
 - rewrite (is_derive_unique (fun y:R=>B y) x (Bp x) HD); field; nra.
Qed.

(* Dilution by preexisting radiation can only reduce the Planck elasticity.
   In particular a constant L must bound the whole intervening interval;
   a slope evaluated only at the returned temperature is insufficient. *)
Lemma band_energy_elasticity h alpha p r B Bp x L :
 0<h -> 0<=alpha -> 0<=p -> 0<=r -> 0<B x ->
 (0<r \/ 0<p) -> 0<=L -> Rabs(x*Bp x/B x)<=L ->
 Rabs(x*band_energy_derivative h alpha p Bp x/
       band_energy h alpha p r B x)<=L.
Proof.
 intros Hh Ha Hp Hr HB Hactive HL Hslope.
 assert (Hnum:0<r+h*p*B x) by (apply band_numerator_positive; assumption).
 assert (Hden:0<1+h*alpha) by nra.
 assert (Hprod:0<=h*p*B x) by (repeat apply Rmult_le_pos; lra).
 set (w := h*p*B x/(r+h*p*B x)).
 assert (Hw0:0<=w).
 { unfold w,Rdiv; apply Rmult_le_pos; [exact Hprod|left; apply Rinv_0_lt_compat; assumption]. }
 assert (Hw1:w<=1).
 { unfold w; apply (Rmult_le_reg_r (r+h*p*B x)); [exact Hnum|].
   field_simplify; nra. }
 assert (HE:x*band_energy_derivative h alpha p Bp x/
        band_energy h alpha p r B x = w*(x*Bp x/B x)).
 { unfold band_energy_derivative,band_energy,w; field; split; nra. }
 rewrite HE,Rabs_mult,(Rabs_pos_eq w Hw0).
 eapply Rle_trans; [apply Rmult_le_compat_l; eassumption|nra].
Qed.

Theorem radiation_band_finite_accuracy h alpha p r B Bp root trial eh eta L b :
 0<h -> 0<=alpha -> 0<=p -> 0<=r -> (0<r \/ 0<p) ->
 0<root -> 0<trial -> 0<=L ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<B t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt B t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive B t (Bp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial -> Rabs(t*Bp t/B t)<=L) ->
 log_error trial root<=b ->
 LogBound eta eh (band_energy h alpha p r B trial) ->
 LogBound (eta+L*b) eh (band_energy h alpha p r B root) /\
 Rabs(eh/band_energy h alpha p r B root-1)<=exp(eta+L*b)-1.
Proof.
 intros Hh Ha Hp Hr Hactive Hroot Htrial HL HB Hcont Hder Hslope Hcoord Heval.
 apply (positive_component_finite_transfer (band_energy h alpha p r B)
  (band_energy_derivative h alpha p Bp) root trial eh eta L b);
  try assumption.
 - intros t Ht; apply band_energy_positive; try assumption; apply HB; exact Ht.
 - intros t Ht; apply band_energy_continuous; try assumption; apply Hcont; exact Ht.
 - intros t Ht; apply band_energy_is_derive; try assumption; apply Hder; exact Ht.
 - intros t Ht; apply band_energy_elasticity; try assumption.
   + apply HB; lra.
   + apply Hslope; assumption.
Qed.

(* Exact endpoint enclosures avoid any logarithmic derivative hypothesis. *)
Theorem positive_output_endpoint_certificate yh y lo hi :
 0<=yh -> 0<lo -> lo<=y<=hi ->
 Rabs(yh/y-1)<=Rmax (yh/lo-1) (1-yh/hi).
Proof.
 intros Hyh Hlo Hy.
 assert (Hyp:0<y) by lra.
 assert (Hhi:0<hi) by lra.
 assert (Hlow:yh/hi<=yh/y).
 { apply (Rmult_le_reg_r (hi*y)); [nra|]. field_simplify; nra. }
 assert (Hup:yh/y<=yh/lo).
 { apply (Rmult_le_reg_r (lo*y)); [nra|]. field_simplify; nra. }
 pose proof (Rmax_l (yh/lo-1) (1-yh/hi)).
 pose proof (Rmax_r (yh/lo-1) (1-yh/hi)).
 apply Rabs_le; lra.
Qed.

Theorem monotone_output_endpoint_certificate (f:R->R) root yh a z lo hi :
 a<=root<=z ->
 (forall s t, a<=s -> s<=t -> t<=z -> f s<=f t) ->
 0<=yh -> 0<lo -> lo<=f a -> f z<=hi ->
 Rabs(yh/f root-1)<=Rmax (yh/lo-1) (1-yh/hi).
Proof.
 intros Hroot Hmono Hyh Hlo Hleft Hright.
 apply positive_output_endpoint_certificate; try assumption.
 pose proof (Hmono a root ltac:(lra) ltac:(lra) ltac:(lra)).
 pose proof (Hmono root z ltac:(lra) ltac:(lra) ltac:(lra)).
 lra.
Qed.

(* The endpoint certificate can be fed rigorously rounded endpoint evaluations.
   No exact evaluation of lo or hi is needed in an implementation. *)
Lemma logbound_endpoint_enclosure eta yh y :
 LogBound eta yh y -> exp(-eta)*yh<=y<=exp eta*yh.
Proof.
 intro H.
 split; [|apply LogBound_exp_lower; exact H].
 pose proof (LogBound_exp_upper _ _ _ H) as Hu.
 apply (Rmult_le_reg_l (exp eta)); [apply exp_pos|].
 rewrite <- Rmult_assoc, <- exp_plus.
 replace (eta + -eta) with 0 by ring; rewrite exp_0; lra.
Qed.

Lemma group_sum_dominates n f g :
 (g<n)%nat -> (forall j, (j<n)%nat -> 0<=f j) -> f g<=group_sum n f.
Proof.
 induction n as [|n IH]; intros Hg Hf; [lia|].
 simpl; destruct (Nat.eq_dec g n) as [->|Hne].
 - pose proof (group_sum_nonnegative n f ltac:(intros; apply Hf; lia)); lra.
 - assert (Hg':(g<n)%nat) by lia.
   pose proof (IH Hg' ltac:(intros; apply Hf; lia)).
   pose proof (Hf n ltac:(lia)); lra.
Qed.

(* A total-energy norm bound yields a componentwise certificate only after
   division by that component's exact positive energy share. *)
Theorem component_energy_share_certificate n exact approx total eps g :
 0<total -> (g<n)%nat -> 0<exact g ->
 group_sum n (fun j => Rabs(approx j-exact j))/total<=eps ->
 Rabs(approx g/exact g-1)<=eps/(exact g/total).
Proof.
 intros Htotal Hg Hex Hglobal.
 pose proof (group_sum_dominates n (fun j=>Rabs(approx j-exact j)) g Hg
  ltac:(intros; apply Rabs_pos)) as Hdom.
 assert (Habs:Rabs(approx g-exact g)<=eps*total).
 { assert (Hmul:group_sum n (fun j=>Rabs(approx j-exact j))/total*total<=eps*total)
      by (apply Rmult_le_compat_r; lra).
   replace (group_sum n (fun j=>Rabs(approx j-exact j))/total*total)
     with (group_sum n (fun j=>Rabs(approx j-exact j))) in Hmul by (field; lra).
   lra. }
 replace (approx g/exact g-1) with ((approx g-exact g)/exact g) by (field; lra).
 rewrite Rabs_div,(Rabs_pos_eq (exact g)) by lra.
 apply (Rmult_le_reg_r (exact g)); [exact Hex|].
 field_simplify; nra.
Qed.

Theorem component_energy_share_floor n exact approx total eps sigma g :
 0<total -> (g<n)%nat -> 0<exact g -> 0<=eps -> 0<sigma ->
 sigma<=exact g/total ->
 group_sum n (fun j => Rabs(approx j-exact j))/total<=eps ->
 Rabs(approx g/exact g-1)<=eps/sigma.
Proof.
 intros Htotal Hg Hex Heps Hsigma Hshare Hglobal.
 eapply Rle_trans; [apply (component_energy_share_certificate n exact approx total eps g); eassumption|].
 assert (Hsharepos:0<exact g/total) by (apply Rdiv_lt_0_compat; assumption).
 apply (Rmult_le_reg_r (sigma*(exact g/total))); [nra|].
 field_simplify; nra.
Qed.

(* Concrete accepted-width specializations.  The constants do not contain
   group count; local eta may contain whatever group evaluator graph charges. *)
Theorem dust_bracket_finite_accuracy64 choice lo hi root trial :
 0<lo -> lo<hi -> lo<=root<=hi -> lo<=trial<=hi ->
 WidthSafety.SafeDifference64 lo hi -> normal64 (RN64 choice (hi-lo)/lo) ->
 RN64 choice (RN64 choice (hi-lo)/lo)<=16*Guards.binary64_u ->
 Rabs(trial/root-1)<=exp(32*Guards.binary64_lambda)-1 /\
 Rabs(trial/root-1)<=33*Guards.binary64_u.
Proof.
 intros Hl Hlh Hr Ht HD HN HW.
 pose proof (certified_bracket_coordinate64 choice lo hi root trial
  Hl Hlh Hr Ht HD HN HW) as Hcoord.
 assert (Hrel:Rabs(trial/root-1)<=exp(32*Guards.binary64_lambda)-1)
  by (apply dust_finite_accuracy; lra).
 split; [exact Hrel|].
 eapply Rle_trans; [exact Hrel|].
 apply (binary64_exp_budget 32 33); try lra.
 unfold Guards.binary64_u; field_simplify; lra.
Qed.

Theorem gas_bracket_finite_accuracy64 choice cg A D T root trial th qh lo hi :
 0<cg -> 0<A -> 0<D -> 0<T ->
 InnerEvaluator.InnerResult choice A D T trial th qh ->
 FiniteNormalNodes choice [cg*th] ->
 0<lo -> lo<hi -> lo<=root<=hi -> lo<=trial<=hi ->
 WidthSafety.SafeDifference64 lo hi -> normal64 (RN64 choice (hi-lo)/lo) ->
 RN64 choice (RN64 choice (hi-lo)/lo)<=16*Guards.binary64_u ->
 LogBound (117*Guards.binary64_lambda)
  (returned_gas choice cg th) (gas_energy cg A D T root) /\
 Rabs(returned_gas choice cg th/gas_energy cg A D T root-1)
  <=exp(117*Guards.binary64_lambda)-1 /\
 Rabs(returned_gas choice cg th/gas_energy cg A D T root-1)
  <=128*Guards.binary64_u.
Proof.
 intros Hcg HA HD HT HI Hnodes Hl Hlh Hr Ht Hdiff HN HW.
 pose proof (certified_bracket_coordinate64 choice lo hi root trial
  Hl Hlh Hr Ht Hdiff HN HW) as Hcoord.
 pose proof (gas_finite_accuracy64 choice cg A D T root trial th qh
  (32*Guards.binary64_lambda) Hcg HA HD HT ltac:(lra) ltac:(lra)
  HI Hnodes Hcoord) as [Hlog Hrel].
 replace (53*Guards.binary64_lambda+2*(32*Guards.binary64_lambda))
  with (117*Guards.binary64_lambda) in Hlog,Hrel by ring.
 split; [exact Hlog|]; split; [exact Hrel|].
 eapply Rle_trans; [exact Hrel|].
 apply (binary64_exp_budget 117 128); try lra.
 unfold Guards.binary64_u; field_simplify; lra.
Qed.

Theorem radiation_band_bracket_finite_accuracy64 choice lo hi
 h alpha p r B Bp root trial eh eta L :
 0<h -> 0<=alpha -> 0<=p -> 0<=r -> (0<r \/ 0<p) -> 0<=L ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<B t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt B t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive B t (Bp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial -> Rabs(t*Bp t/B t)<=L) ->
 LogBound eta eh (band_energy h alpha p r B trial) ->
 0<lo -> lo<hi -> lo<=root<=hi -> lo<=trial<=hi ->
 WidthSafety.SafeDifference64 lo hi -> normal64 (RN64 choice (hi-lo)/lo) ->
 RN64 choice (RN64 choice (hi-lo)/lo)<=16*Guards.binary64_u ->
 LogBound (eta+32*L*Guards.binary64_lambda) eh (band_energy h alpha p r B root) /\
 Rabs(eh/band_energy h alpha p r B root-1)
  <=exp(eta+32*L*Guards.binary64_lambda)-1.
Proof.
 intros Hh Ha Hp Hr Hactive HL HB Hcont Hder Hslope Heval
  Hl Hlh Hroot Htrial Hdiff HN HW.
 pose proof (certified_bracket_coordinate64 choice lo hi root trial
  Hl Hlh Hroot Htrial Hdiff HN HW) as Hcoord.
 replace (eta+32*L*Guards.binary64_lambda)
  with (eta+L*(32*Guards.binary64_lambda)) by ring.
 apply (radiation_band_finite_accuracy h alpha p r B Bp root trial eh eta L
  (32*Guards.binary64_lambda)); try assumption; lra.
Qed.

(* Finite exponential-to-unit-roundoff conversions used by the concrete
   multigroup operation-graph specializations. *)
Lemma component_relative_binary64_budget k K yh y :
 0<=k -> 0<=K ->
 k*(1+K*Guards.binary64_u)<=K*(1-Guards.binary64_u) ->
 LogBound (k*Guards.binary64_lambda) yh y ->
 Rabs(yh/y-1)<=K*Guards.binary64_u.
Proof.
 intros Hk HK Hroom [Hyh [Hy Hlog]].
 eapply Rle_trans; [apply log_error_relative; eassumption|].
 apply binary64_exp_budget; assumption.
Qed.

Lemma component_relative_207 yh y :
 LogBound (207*Guards.binary64_lambda) yh y ->
 Rabs(yh/y-1)<=256*Guards.binary64_u.
Proof.
 intro H; apply (component_relative_binary64_budget 207 256); try lra; try assumption.
 unfold Guards.binary64_u; field_simplify; lra.
Qed.

Lemma component_relative_467 yh y :
 LogBound (467*Guards.binary64_lambda) yh y ->
 Rabs(yh/y-1)<=512*Guards.binary64_u.
Proof.
 intro H; apply (component_relative_binary64_budget 467 512); try lra; try assumption.
 unfold Guards.binary64_u; field_simplify; lra.
Qed.

Lemma component_relative_842 yh y :
 LogBound (842*Guards.binary64_lambda) yh y ->
 Rabs(yh/y-1)<=1024*Guards.binary64_u.
Proof.
 intro H; apply (component_relative_binary64_budget 842 1024); try lra; try assumption.
 unfold Guards.binary64_u; field_simplify; lra.
Qed.

Lemma component_relative_32 yh y :
 LogBound (32*Guards.binary64_lambda) yh y ->
 Rabs(yh/y-1)<=64*Guards.binary64_u.
Proof.
 intro H; apply (component_relative_binary64_budget 32 64); try lra; try assumption.
 unfold Guards.binary64_u; field_simplify; lra.
Qed.

Lemma component_relative_117 yh y :
 LogBound (117*Guards.binary64_lambda) yh y ->
 Rabs(yh/y-1)<=128*Guards.binary64_u.
Proof.
 intro H; apply (component_relative_binary64_budget 117 128); try lra; try assumption.
 unfold Guards.binary64_u; field_simplify; lra.
Qed.

Lemma component_relative_142 yh y :
 LogBound (142*Guards.binary64_lambda) yh y ->
 Rabs(yh/y-1)<=256*Guards.binary64_u.
Proof.
 intro H; apply (component_relative_binary64_budget 142 256); try lra; try assumption.
 unfold Guards.binary64_u; field_simplify; lra.
Qed.

Print Assumptions positive_component_finite_transfer.
Print Assumptions gas_finite_accuracy64.
Print Assumptions radiation_band_finite_accuracy.
Print Assumptions monotone_output_endpoint_certificate.
Print Assumptions component_energy_share_certificate.
