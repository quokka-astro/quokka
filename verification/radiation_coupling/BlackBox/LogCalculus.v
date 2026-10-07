(* Finite logarithmic error estimates derived from actual derivatives and MVT.
   Differentiability and slope bounds are needed only in the open interval;
   endpoint continuity is stated separately. *)
From Coq Require Import Reals Psatz Field.
From Coquelicot Require Import Coquelicot.
Open Scope R_scope.

Definition log_error (x y : R) : R := Rabs (ln x - ln y).
Definition log_lift (f : R -> R) (s : R) : R := ln (f (exp s)).

(* MVT_gen returns a point in the closed interval. Extending the derivative
   by any admissible value at endpoints permits genuinely interior bounds. *)
Lemma mvt_interior_property (F dF : R -> R) a b (P : R -> Prop) d :
 P d ->
 (forall s, Rmin a b < s < Rmax a b -> is_derive F s (dF s)) ->
 (forall s, Rmin a b <= s <= Rmax a b -> continuity_pt F s) ->
 (forall s, Rmin a b < s < Rmax a b -> P (dF s)) ->
 exists v, P v /\ F b - F a = v * (b-a).
Proof.
 intros Hd Hder Hcont HP.
 set (D := fun s => if Rlt_dec (Rmin a b) s then
                     if Rlt_dec s (Rmax a b) then dF s else d
                   else d).
 assert (HD : forall s, Rmin a b < s < Rmax a b ->
                        is_derive F s (D s)).
 { intros s Hs; unfold D.
   destruct (Rlt_dec (Rmin a b) s); [|lra].
   destruct (Rlt_dec s (Rmax a b)); [apply Hder; exact Hs|lra]. }
 destruct (MVT_gen F a b D HD Hcont) as [c [Hc Heq]].
 exists (D c); split; [|exact Heq].
 unfold D.
 destruct (Rlt_dec (Rmin a b) c); [|exact Hd].
 destruct (Rlt_dec c (Rmax a b)); [apply HP; split; assumption|exact Hd].
Qed.

Lemma derivative_lower_distance (F dF : R -> R) a b mu :
 0 < mu ->
 (forall s, Rmin a b < s < Rmax a b -> is_derive F s (dF s)) ->
 (forall s, Rmin a b <= s <= Rmax a b -> continuity_pt F s) ->
 (forall s, Rmin a b < s < Rmax a b -> mu <= dF s) ->
 Rabs (b-a) <= Rabs (F b-F a) / mu.
Proof.
 intros Hmu Hder Hcont Hbound.
 destruct (mvt_interior_property F dF a b (fun v => mu<=v) mu
            (Rle_refl _) Hder Hcont Hbound) as [v [Hv Heq]].
 rewrite Heq, Rabs_mult.
 rewrite (Rabs_right v) by lra.
 apply (Rmult_le_reg_r mu); [exact Hmu|].
 replace (v * Rabs (b-a) / mu * mu) with (v * Rabs (b-a))
   by (field; lra).
 pose proof (Rabs_pos (b-a)); nra.
Qed.

Lemma derivative_upper_distance (F dF : R -> R) a b M :
 0 <= M ->
 (forall s, Rmin a b < s < Rmax a b -> is_derive F s (dF s)) ->
 (forall s, Rmin a b <= s <= Rmax a b -> continuity_pt F s) ->
 (forall s, Rmin a b < s < Rmax a b -> Rabs (dF s) <= M) ->
 Rabs (F b-F a) <= M * Rabs (b-a).
Proof.
 intros HM Hder Hcont Hbound.
 assert (H0 : Rabs 0 <= M) by (rewrite Rabs_R0; exact HM).
 destruct (mvt_interior_property F dF a b (fun v => Rabs v<=M) 0
            H0 Hder Hcont Hbound) as [v [Hv Heq]].
 rewrite Heq, Rabs_mult.
 apply Rmult_le_compat_r; [apply Rabs_pos|exact Hv].
Qed.

Lemma is_derive_log_lift (f : R -> R) s df :
 0 < f (exp s) -> is_derive f (exp s) df ->
 is_derive (log_lift f) s (exp s * df / f (exp s)).
Proof.
 intros Hpos Hdf.
 pose proof (is_derive_comp ln f (exp s) (/f(exp s)) df
              (is_derive_ln _ Hpos) Hdf) as Hlf.
 pose proof (is_derive_comp (fun x => ln(f x)) exp s
              (df */f(exp s)) (exp s) Hlf (is_derive_exp s)) as H.
 unfold log_lift.
 change (is_derive (fun x => ln(f(exp x))) s
   (exp s * (df * / f(exp s)))) in H.
 unfold Rdiv; rewrite Rmult_assoc; exact H.
Qed.

Lemma log_lift_at_ln (f : R -> R) x :
 0 < x -> log_lift f (ln x) = ln (f x).
Proof. intro Hx; unfold log_lift; rewrite exp_ln; auto. Qed.

Lemma log_error_nonnegative x y : 0 <= log_error x y.
Proof. apply Rabs_pos. Qed.

Lemma log_error_sym x y : log_error x y = log_error y x.
Proof. unfold log_error; apply Rabs_minus_sym. Qed.

Lemma log_error_triangle x y z :
 log_error x z <= log_error x y + log_error y z.
Proof.
 unfold log_error.
 replace (ln x-ln z) with ((ln x-ln y)+(ln y-ln z)) by ring.
 apply Rabs_triang.
Qed.

Lemma log_error_ratio x y : 0<x -> 0<y ->
 log_error x y = Rabs (ln (x/y)).
Proof. intros; unfold log_error; rewrite ln_div; auto. Qed.

Lemma residual_with_evaluator_error shat s beta rho :
 Rabs (shat-s) <= beta -> Rabs shat <= rho ->
 Rabs s <= rho+beta.
Proof.
 intros Herr Haccept.
 replace s with (shat + (s-shat)) by ring.
 eapply Rle_trans; [apply Rabs_triang|].
 rewrite Rabs_minus_sym; lra.
Qed.

Lemma is_derive_continuity_pt (f : R -> R) x df :
 is_derive f x df -> continuity_pt f x.
Proof.
 intro Hd; apply derivable_continuous_pt; exists df.
 apply is_derive_Reals; exact Hd.
Qed.

Lemma continuity_log_lift (f : R -> R) s :
 0 < f(exp s) -> continuity_pt f (exp s) ->
 continuity_pt (log_lift f) s.
Proof.
 intros Hpos Hf.
 unfold log_lift.
 change (continuity_pt (comp ln (comp f exp)) s).
 apply continuity_pt_comp.
 - apply continuity_pt_comp; [|exact Hf].
   eapply is_derive_continuity_pt; apply is_derive_exp.
 - eapply is_derive_continuity_pt; apply is_derive_ln; exact Hpos.
Qed.

Lemma exp_le_mono a b : a<=b -> exp a<=exp b.
Proof.
 intro H; destruct H as [H|H]; [left; apply exp_increasing; exact H|].
 subst; apply Rle_refl.
Qed.

Lemma exp_closed_log_interval x y s : 0<x -> 0<y ->
 Rmin (ln x) (ln y) <= s <= Rmax (ln x) (ln y) ->
 Rmin x y <= exp s <= Rmax x y.
Proof.
 intros Hx Hy Hs.
 destruct (Rle_dec x y) as [Hxy|Hyx].
 - pose proof (ln_le x y Hx Hxy) as Hln.
   rewrite Rmin_left, Rmax_right in Hs by exact Hln.
   rewrite Rmin_left, Rmax_right by exact Hxy.
   destruct Hs as [Ha Hb]; split.
   + rewrite <- (exp_ln x Hx); apply exp_le_mono; exact Ha.
   + rewrite <- (exp_ln y Hy); apply exp_le_mono; exact Hb.
 - assert (Hxy : y<=x) by lra.
   pose proof (ln_le y x Hy Hxy) as Hln.
   rewrite Rmin_right, Rmax_left in Hs by exact Hln.
   rewrite Rmin_right, Rmax_left by exact Hxy.
   destruct Hs as [Ha Hb]; split.
   + rewrite <- (exp_ln y Hy); apply exp_le_mono; exact Ha.
   + rewrite <- (exp_ln x Hx); apply exp_le_mono; exact Hb.
Qed.

Lemma exp_open_log_interval x y s : 0<x -> 0<y ->
 Rmin (ln x) (ln y) < s < Rmax (ln x) (ln y) ->
 Rmin x y < exp s < Rmax x y.
Proof.
 intros Hx Hy Hs.
 destruct (Rle_dec x y) as [Hxy|Hyx].
 - pose proof (ln_le x y Hx Hxy) as Hln.
   rewrite Rmin_left, Rmax_right in Hs by exact Hln.
   rewrite Rmin_left, Rmax_right by exact Hxy.
   destruct Hs as [Ha Hb]; split.
   + rewrite <- (exp_ln x Hx); apply exp_increasing; exact Ha.
   + rewrite <- (exp_ln y Hy); apply exp_increasing; exact Hb.
 - assert (Hxy : y<=x) by lra.
   pose proof (ln_le y x Hy Hxy) as Hln.
   rewrite Rmin_right, Rmax_left in Hs by exact Hln.
   rewrite Rmin_right, Rmax_left by exact Hxy.
   destruct Hs as [Ha Hb]; split.
   + rewrite <- (exp_ln y Hy); apply exp_increasing; exact Ha.
   + rewrite <- (exp_ln x Hx); apply exp_increasing; exact Hb.
Qed.

Lemma elasticity_upper_distance (f df : R -> R) x y M :
 0<x -> 0<y -> 0<=M ->
 (forall t, Rmin x y <= t <= Rmax x y -> 0<f t) ->
 (forall t, Rmin x y <= t <= Rmax x y -> continuity_pt f t) ->
 (forall t, Rmin x y < t < Rmax x y -> is_derive f t (df t)) ->
 (forall t, Rmin x y < t < Rmax x y -> Rabs (t*df t/f t)<=M) ->
 log_error (f y) (f x) <= M * log_error y x.
Proof.
 intros Hx Hy HM Hpos Hcont Hder Hbound.
 unfold log_error.
 rewrite <- (log_lift_at_ln f y Hy), <- (log_lift_at_ln f x Hx).
 apply (derivative_upper_distance (log_lift f)
          (fun s => exp s*df(exp s)/f(exp s)) (ln x) (ln y) M HM).
 - intros s Hs; apply is_derive_log_lift.
   + apply Hpos; apply exp_closed_log_interval; try assumption; lra.
   + apply Hder; apply exp_open_log_interval; assumption.
 - intros s Hs; apply continuity_log_lift.
   + apply Hpos; apply exp_closed_log_interval; assumption.
   + apply Hcont; apply exp_closed_log_interval; assumption.
 - intros s Hs; apply Hbound; apply exp_open_log_interval; assumption.
Qed.

Lemma elasticity_lower_distance (f df : R -> R) x y mu :
 0<x -> 0<y -> 0<mu ->
 (forall t, Rmin x y <= t <= Rmax x y -> 0<f t) ->
 (forall t, Rmin x y <= t <= Rmax x y -> continuity_pt f t) ->
 (forall t, Rmin x y < t < Rmax x y -> is_derive f t (df t)) ->
 (forall t, Rmin x y < t < Rmax x y -> mu<=t*df t/f t) ->
 log_error y x <= log_error (f y) (f x) / mu.
Proof.
 intros Hx Hy Hmu Hpos Hcont Hder Hbound.
 unfold log_error.
 rewrite <- (log_lift_at_ln f y Hy), <- (log_lift_at_ln f x Hx).
 apply (derivative_lower_distance (log_lift f)
          (fun s => exp s*df(exp s)/f(exp s)) (ln x) (ln y) mu Hmu).
 - intros s Hs; apply is_derive_log_lift.
   + apply Hpos; apply exp_closed_log_interval; try assumption; lra.
   + apply Hder; apply exp_open_log_interval; assumption.
 - intros s Hs; apply continuity_log_lift.
   + apply Hpos; apply exp_closed_log_interval; assumption.
   + apply Hcont; apply exp_closed_log_interval; assumption.
 - intros s Hs; apply Hbound; apply exp_open_log_interval; assumption.
Qed.

Lemma log_bracket_width a b x y :
 0<a -> a<=x<=b -> a<=y<=b ->
 log_error x y <= ln(b/a).
Proof.
 intros Ha Hx Hy.
 assert (Hb:0<b) by lra.
 assert (Hxp:0<x) by lra.
 assert (Hyp:0<y) by lra.
 pose proof (ln_le a x Ha (proj1 Hx)).
 pose proof (ln_le x b Hxp (proj2 Hx)).
 pose proof (ln_le a y Ha (proj1 Hy)).
 pose proof (ln_le y b Hyp (proj2 Hy)).
 unfold log_error; rewrite ln_div by assumption.
 apply Rabs_le; split; lra.
Qed.

Lemma exp_above_tangent t : 1+t<=exp t.
Proof.
 destruct (Req_dec t 0) as [H|H].
 - subst; rewrite exp_0; lra.
 - pose proof (exp_ineq1 t H); lra.
Qed.

Lemma log_error_relative x y b : 0<x -> 0<y ->
 log_error x y <= b -> Rabs (x/y-1) <= exp b-1.
Proof.
 intros Hx Hy Hbound.
 assert (Hr:0<x/y) by (apply Rdiv_lt_0_compat; assumption).
 rewrite log_error_ratio in Hbound by assumption.
 assert (Hup:ln(x/y)<=b).
 { eapply Rle_trans; [apply Rle_abs|exact Hbound]. }
 assert (Hlo:-b<=ln(x/y)).
 { pose proof (Rle_abs (-ln(x/y))) as H.
   rewrite Rabs_Ropp in H; lra. }
 pose proof (exp_le_mono _ _ Hup) as Hexpup.
 pose proof (exp_le_mono _ _ Hlo) as Hexplo.
 rewrite exp_ln in Hexpup, Hexplo by exact Hr.
 pose proof (exp_above_tangent b).
 pose proof (exp_above_tangent (-b)).
 apply Rabs_le; split; lra.
Qed.

Lemma derivative_absolute_lower_distance (F dF : R -> R) a b mu :
 0 < mu ->
 (forall s, Rmin a b < s < Rmax a b -> is_derive F s (dF s)) ->
 (forall s, Rmin a b <= s <= Rmax a b -> continuity_pt F s) ->
 (forall s, Rmin a b < s < Rmax a b -> mu <= Rabs (dF s)) ->
 Rabs (b-a) <= Rabs (F b-F a) / mu.
Proof.
 intros Hmu Hder Hcont Hbound.
 assert (Hdefault:mu<=Rabs mu) by apply Rle_abs.
 destruct (mvt_interior_property F dF a b (fun v => mu<=Rabs v) mu
            Hdefault Hder Hcont Hbound) as [v [Hv Heq]].
 rewrite Heq, Rabs_mult.
 apply (Rmult_le_reg_r mu); [exact Hmu|].
 replace (Rabs v * Rabs (b-a) / mu * mu)
   with (Rabs v * Rabs (b-a)) by (field; lra).
 pose proof (Rabs_pos (b-a)); nra.
Qed.

Lemma elasticity_absolute_lower_distance (f df : R -> R) x y mu :
 0<x -> 0<y -> 0<mu ->
 (forall t, Rmin x y <= t <= Rmax x y -> 0<f t) ->
 (forall t, Rmin x y <= t <= Rmax x y -> continuity_pt f t) ->
 (forall t, Rmin x y < t < Rmax x y -> is_derive f t (df t)) ->
 (forall t, Rmin x y < t < Rmax x y -> mu<=Rabs(t*df t/f t)) ->
 log_error y x <= log_error (f y) (f x) / mu.
Proof.
 intros Hx Hy Hmu Hpos Hcont Hder Hbound.
 unfold log_error.
 rewrite <- (log_lift_at_ln f y Hy), <- (log_lift_at_ln f x Hx).
 apply (derivative_absolute_lower_distance (log_lift f)
          (fun s => exp s*df(exp s)/f(exp s)) (ln x) (ln y) mu Hmu).
 - intros s Hs; apply is_derive_log_lift.
   + apply Hpos; apply exp_closed_log_interval; try assumption; lra.
   + apply Hder; apply exp_open_log_interval; assumption.
 - intros s Hs; apply continuity_log_lift.
   + apply Hpos; apply exp_closed_log_interval; assumption.
   + apply Hcont; apply exp_closed_log_interval; assumption.
 - intros s Hs; apply Hbound; apply exp_open_log_interval; assumption.
Qed.

Lemma elasticity_negative_lower_distance (f df : R -> R) x y mu :
 0<x -> 0<y -> 0<mu ->
 (forall t, Rmin x y <= t <= Rmax x y -> 0<f t) ->
 (forall t, Rmin x y <= t <= Rmax x y -> continuity_pt f t) ->
 (forall t, Rmin x y < t < Rmax x y -> is_derive f t (df t)) ->
 (forall t, Rmin x y < t < Rmax x y -> mu<= -(t*df t/f t)) ->
 log_error y x <= log_error (f y) (f x) / mu.
Proof.
 intros Hx Hy Hmu Hpos Hcont Hder Hbound.
 apply (elasticity_absolute_lower_distance f df x y mu Hx Hy Hmu
           Hpos Hcont Hder).
 intros t Ht; eapply Rle_trans; [apply Hbound; exact Ht|].
 rewrite <- Rabs_Ropp; apply Rle_abs.
Qed.

Lemma residual_acceptance_distance (F dF : R -> R) root trial mu shat beta rho :
 0<mu -> F root=0 ->
 (forall s, Rmin root trial<s<Rmax root trial -> is_derive F s (dF s)) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> continuity_pt F s) ->
 (forall s, Rmin root trial<s<Rmax root trial -> mu<=Rabs(dF s)) ->
 Rabs (shat-F trial)<=beta -> Rabs shat<=rho ->
 Rabs(trial-root)<=(rho+beta)/mu.
Proof.
 intros Hmu Hroot Hder Hcont Hbound Herr Haccept.
 eapply Rle_trans.
 - apply (derivative_absolute_lower_distance F dF root trial mu
            Hmu Hder Hcont Hbound).
 - rewrite Hroot, Rminus_0_r.
   unfold Rdiv; apply Rmult_le_compat_r.
   + left; apply Rinv_0_lt_compat; exact Hmu.
   + apply (residual_with_evaluator_error shat (F trial) beta rho Herr Haccept).
Qed.

Lemma balance_acceptance_log_error (f df : R -> R)
    root trial mu shat beta rho :
 0<root -> 0<trial -> 0<mu -> f root=1 ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<f t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt f t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive f t (df t)) ->
 (forall t, Rmin root trial<t<Rmax root trial -> mu<=Rabs(t*df t/f t)) ->
 Rabs (shat-ln(f trial))<=beta -> Rabs shat<=rho ->
 log_error trial root<=(rho+beta)/mu.
Proof.
 intros Hr Ht Hmu Hroot Hpos Hcont Hder Hbound Herr Haccept.
 eapply Rle_trans.
 - apply (elasticity_absolute_lower_distance f df root trial mu
            Hr Ht Hmu Hpos Hcont Hder Hbound).
 - unfold log_error; rewrite Hroot, ln_1, Rminus_0_r.
   unfold Rdiv; apply Rmult_le_compat_r.
   + left; apply Rinv_0_lt_compat; exact Hmu.
   + apply (residual_with_evaluator_error shat (ln(f trial)) beta rho Herr Haccept).
Qed.

Lemma log_error_triangle_bound x y z a b :
 log_error x y<=a -> log_error y z<=b -> log_error x z<=a+b.
Proof. intros; pose proof (log_error_triangle x y z); lra. Qed.

Lemma log_error_relative_fraction x y b : 0<x -> 0<y ->
 log_error x y<=b -> Rabs(x-y)/y<=exp b-1.
Proof.
 intros Hx Hy Hbound.
 pose proof (log_error_relative x y b Hx Hy Hbound) as H.
 replace (x/y-1) with ((x-y)/y) in H by (field; lra).
 rewrite Rabs_div, (Rabs_right y) in H by lra.
 exact H.
Qed.
