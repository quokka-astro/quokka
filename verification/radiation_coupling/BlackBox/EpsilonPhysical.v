(* Positive physical branch derivatives for the full opacity slope interval.
   The effective-slope margins and monotonicity are derived here from the
   original derivative formulae, rather than supplied as contracts. *)
From Coq Require Import Reals Psatz Field List.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra GasMap GasBounds LogCalculus OuterDerivatives
 OuterBounds Brackets SolverLoop PhysicalLoop Binary64Accuracy EndToEnd
 FloatingPoint Guards GuardFloatBridge InnerEvaluator PhysicalRoot UniqueRoot
 Initialization EpsilonAccuracy.
Import ListNotations.
Open Scope R_scope.

Lemma epsilon_physical_margins eps C kap x kp :
 0<eps -> 0<C -> 0<kap x ->
 -4+eps<=opacity_slope kap x kp<=1-eps ->
 Rmin 1 eps<=1-effective_slope C kap x kp /\
 Rmin 1 eps<=4+effective_slope C kap x kp.
Proof.
 intros Heps HC Hk Hp.
 pose proof (Rmin_l 1 eps) as Hmu1.
 pose proof (Rmin_r 1 eps) as Hmueps.
 pose proof (optical_depth_positive C kap x HC Hk) as Htau.
 assert (HH:opacity_slope kap x kp/(1+optical_depth C kap x)<=1-Rmin 1 eps).
 { apply (Rmult_le_reg_r (1+optical_depth C kap x)); [lra|].
   replace (opacity_slope kap x kp/(1+optical_depth C kap x)*
    (1+optical_depth C kap x)) with (opacity_slope kap x kp) by (field; lra).
   nra. }
 assert (HCool:-4+Rmin 1 eps<=opacity_slope kap x kp/(1+optical_depth C kap x)).
 { apply (Rmult_le_reg_r (1+optical_depth C kap x)); [lra|].
   replace (opacity_slope kap x kp/(1+optical_depth C kap x)*
    (1+optical_depth C kap x)) with (opacity_slope kap x kp) by (field; lra).
   nra. }
 unfold effective_slope; split; lra.
Qed.

Lemma positive_elasticity_derivative_margin f x df mu :
 0<x -> 0<f x -> 0<mu -> mu<=elasticity f x df -> 0<df.
Proof.
 intros Hx Hf Hmu Hbound.
 unfold elasticity in Hbound.
 pose proof (Rmult_le_compat_r (f x) mu (x*df/f x) ltac:(lra) Hbound) as H.
 replace (x*df/f x*f x) with (x*df) in H by (field; lra).
 nra.
Qed.

Lemma epsilon_physical_oriented_derivative_positive eps heating A D T C chi a r kap kp x :
 0<eps -> 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<kap x -> physical_region heating T x ->
 -4+eps<=opacity_slope kap x (kp x)<=1-eps ->
 0<(if heating then physical_balance_derivative heating A D T C chi a r kap kp x
 else -physical_balance_derivative heating A D T C chi a r kap kp x).
Proof.
 intros Heps HA HD HT HC Hchi Ha Hr Hx Hk Hregion Hp.
 destruct (epsilon_physical_margins eps C kap x (kp x) Heps HC Hk Hp) as [Hh Hc].
 assert (Hmu:0<Rmin 1 eps) by (apply Rmin_glb_lt; lra).
 pose proof (Rmin_l 1 eps) as Hmu1.
 pose proof (physical_balance_positive heating A D T C chi a r kap x
  HA HD HT HC Hchi Ha Hr Hx Hk Hregion) as Hpos.
 destruct heating; simpl in Hregion,Hpos |- *.
 - apply (positive_elasticity_derivative_margin (physical_heating A D T C chi a r kap) x _ (Rmin 1 eps)); try assumption.
   change (Rmin 1 eps<=elasticity (OuterDerivatives.heating_balance A T C chi a r kap (gas_map A D T)) x
    (heating_derivative A T C chi a r kap (gas_map A D T) x (kp x)
     (physical_gas_derivative A D T (gas_map A D T x)))).
   apply heating_conditioning; try assumption; try lra.
   + apply heating_gas_branch; assumption.
   + symmetry; apply (proj2 (gas_map_spec A D T x HA HD HT Hx)).
 - apply (positive_elasticity_derivative_margin (physical_cooling A D T C chi a r kap) x _ (Rmin 1 eps)); try assumption.
   replace (elasticity (physical_cooling A D T C chi a r kap) x
    (-physical_cooling_derivative A D T C chi a r kap kp x)) with
    (-elasticity (physical_cooling A D T C chi a r kap) x
    (physical_cooling_derivative A D T C chi a r kap kp x)) by (unfold elasticity,Rdiv; ring).
   change (Rmin 1 eps<= -elasticity (OuterDerivatives.cooling_balance A T C chi a r kap (gas_map A D T)) x
    (cooling_derivative A T C chi a r kap (gas_map A D T) x (kp x)
     (physical_gas_derivative A D T (gas_map A D T x)))).
   apply cooling_conditioning; try assumption; try lra.
   + apply (proj1 (gas_map_spec A D T x HA HD HT Hx)).
   + apply cooling_gas_branch; assumption.
Qed.

Theorem epsilon_physical_outer_monotonicity eps heating A D T C chi a r kap kp lo hi :
 0<eps -> 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<lo ->
 (forall x, lo<=x<=hi -> physical_region heating T x) ->
 (forall x, lo<=x<=hi -> 0<kap x /\ continuity_pt kap x) ->
 (forall x, lo<x<hi -> is_derive kap x (kp x) /\
  -4+eps<=opacity_slope kap x (kp x)<=1-eps) ->
 nondecreasing_on (oriented_balance heating (physical_balance heating A D T C chi a r kap)) lo hi.
Proof.
 intros Heps HA HD HT HC Hchi Ha Hr Hlo Hregion Hkap Hder.
 apply (derivative_positive_nondecreasing _
 (fun x=>if heating then physical_balance_derivative heating A D T C chi a r kap kp x
  else -physical_balance_derivative heating A D T C chi a r kap kp x)).
 - intros x Hx.
   assert (Hcont:continuity_pt (physical_balance heating A D T C chi a r kap) x).
   { apply physical_balance_continuous; try assumption; try lra; apply (Hkap x Hx). }
   unfold oriented_balance; destruct heating; reg; assumption.
 - intros x Hx; apply physical_oriented_is_derive; try assumption; try lra.
   + apply (Hkap x ltac:(lra)).
   + apply (Hder x Hx).
 - intros x Hx; apply (epsilon_physical_oriented_derivative_positive eps); try assumption; try lra.
   + apply (Hkap x ltac:(lra)).
   + apply Hregion; lra.
   + apply (Hder x Hx).
Qed.

Print Assumptions epsilon_physical_outer_monotonicity.

(* Physical reference solutions are constructed from the original equations;
   epsilon appears only in smoothness/conditioning, not in their definition. *)
Definition smooth_epsilon_planck_on eps (kap kp:R->R) lo hi : Prop :=
 (forall x, lo<=x<=hi -> 0<kap x /\ continuity_pt kap x) /\
 (forall x, lo<x<hi -> is_derive kap x (kp x) /\
  -4+eps<=opacity_slope kap x (kp x)<=1-eps).

Section EpsilonPhysicalAccuracy.
Variable eps : R.
Variables A D T C chi a r : R.
Hypotheses (Heps:0<eps) (Heps_upper:eps<=5/2).
Hypotheses (HA:0<A) (HD:0<D) (HT:0<T) (HC:0<C) (Hchi:0<chi) (Ha:0<a) (Hr:0<r).
Variables kap kp khat : R->R.
Variables lo hi : R.
Hypothesis Hphysical_domain : lo<=T<=hi /\ lo<=radiation_temperature a r<=hi.
Hypothesis Hsmooth : smooth_epsilon_planck_on eps kap kp lo hi.

Lemma reference_epsilon_dust_spec :
 0<exact_dust A D T C chi a r kap /\
 lo<=exact_dust A D T C chi a r kap<=hi /\
 positive_physical_solution A D T C chi a r kap
  (exact_dust A D T C chi a r kap) (exact_gas A D T C chi a r kap).
Proof.
 assert (HK:forall x,
 Rmin T (radiation_temperature a r)<=x<=Rmax T (radiation_temperature a r) ->
 0<kap x /\ continuity_pt kap x).
 { intros x Hx; apply (proj1 Hsmooth).
   apply (interval_endpoints_containment lo hi T (radiation_temperature a r) x); tauto. }
 destruct (exact_dust_spec A D T C chi a r kap HA HD HT HC Hchi Ha Hr HK)
   as [Hxp [Hb _]].
 split; [exact Hxp|]. split.
 - apply (interval_endpoints_containment lo hi T (radiation_temperature a r) _); tauto.
 - apply exact_physical_solution; assumption.
Qed.

Theorem physical_binary64_epsilon_main_accuracy_local choice (heating:bool) trial th qh :
 0<trial -> lo<=trial<=hi ->
 (if heating then T<=radiation_temperature a r /\ T<=trial
  else radiation_temperature a r<=T /\ trial<=T) ->
 LogBound (8*binary64_lambda) (khat trial) (kap trial) ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r
  (exact_dust A D T C chi a r kap) trial ->
 FiniteNormalNodes choice [A*th] ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat trial) a trial) ->
 EpsilonMainAccuracy64 eps trial (exact_dust A D T C chi a r kap)
  (returned_gas choice A th) (A*exact_gas A D T C chi a r kap)
  (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intros Htrial Htdom Hbranch HK HI HO NG NR.
 destruct reference_epsilon_dust_spec as [Hroot [Hrdom HS]].
 destruct HS as [_ [Hg [Hkroot [HE Hcol]]]].
 assert (Hrootphysical:Rmin T (radiation_temperature a r)<=exact_dust A D T C chi a r kap<=
   Rmax T (radiation_temperature a r)).
 { apply (source_root_in_physical_interval A D T C chi a r kap
     (radiation_temperature a r) (exact_dust A D T C chi a r kap)
     HA HD HT HC Hchi Ha Hr (proj1 (radiation_temperature_spec a r Ha Hr))
     (proj2 (radiation_temperature_spec a r Ha Hr)) Hroot Hkroot HE). }
 assert (Hreg:branch_region heating T (exact_dust A D T C chi a r kap) trial).
 { unfold branch_region; destruct heating.
   - rewrite Rmin_left,Rmax_right in Hrootphysical by lra; lra.
   - rewrite Rmin_right,Rmax_left in Hrootphysical by lra; lra. }
 assert (HB:branch_balance heating A D T C chi a r kap (exact_dust A D T C chi a r kap)=1).
 { unfold branch_balance; destruct heating; [unfold physical_heating|unfold physical_cooling].
   - apply (proj2 (heating_balance_root_equiv A T C chi a r kap (gas_map A D T)
       (exact_dust A D T C chi a r kap) HC Hchi Hkroot Hr)); exact HE.
   - apply (proj2 (cooling_balance_root_equiv A T C chi a r kap (gas_map A D T)
       (exact_dust A D T C chi a r kap) HC Hchi Hkroot Ha Hroot)); exact HE. }
 change (EpsilonMainAccuracy64 eps trial (exact_dust A D T C chi a r kap)
  (returned_gas choice A th) (gas_energy A A D T (exact_dust A D T C chi a r kap))
  (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap))).
 apply (binary64_epsilon_main_accuracy_local eps choice heating A D T C chi a r A kap kp khat
   (exact_dust A D T C chi a r kap) trial th qh); try assumption.
 - intros x Hx. apply (proj1 (proj1 Hsmooth x
     (interval_endpoints_containment lo hi _ _ x Hrdom Htdom Hx))).
 - intros x Hx. apply (proj2 (proj1 Hsmooth x
     (interval_endpoints_containment lo hi _ _ x Hrdom Htdom Hx))).
 - intros x Hx. apply (proj1 (proj2 Hsmooth x
     (subinterval_open lo hi _ _ x Hrdom Htdom Hx))).
 - intros x Hx. apply (proj2 (proj2 Hsmooth x
     (subinterval_open lo hi _ _ x Hrdom Htdom Hx))).
Qed.

Theorem epsilon_physical_solution_unique x t :
 positive_physical_solution A D T C chi a r kap x t ->
 x=exact_dust A D T C chi a r kap /\ t=exact_gas A D T C chi a r kap.
Proof.
 intro HS.
 assert (Hmu:0<Rmin 1 eps) by (apply Rmin_glb_lt; lra).
 pose proof (Rmin_l 1 eps) as Hmu1.
 destruct reference_epsilon_dust_spec as [Hx [Hb Href]].
 destruct (radiation_temperature_spec a r Ha Hr) as [HTr HB].
 destruct Hphysical_domain as [HTdom HTrdom].
 destruct (Rle_dec T (radiation_temperature a r)) as [Hheat|Hcool].
 - apply (heating_physical_solution_unique A D T C chi a r kap kp
     (radiation_temperature a r) (Rmin 1 eps) x t
     (exact_dust A D T C chi a r kap) (exact_gas A D T C chi a r kap));
     try assumption; try lra.
   + intros z Hz; apply (proj1 Hsmooth); lra.
   + intros z Hz; apply (proj1 (proj2 Hsmooth z ltac:(lra))).
   + intros z Hz.
     apply (proj1 (epsilon_physical_margins eps C kap z (kp z) Heps HC
       (proj1 (proj1 Hsmooth z ltac:(lra)))
       (proj2 (proj2 Hsmooth z ltac:(lra))))).
 - apply (cooling_physical_solution_unique A D T C chi a r kap kp
     (radiation_temperature a r) (Rmin 1 eps) x t
     (exact_dust A D T C chi a r kap) (exact_gas A D T C chi a r kap));
     try assumption; try lra.
   + intros z Hz; apply (proj1 Hsmooth); lra.
   + intros z Hz; apply (proj1 (proj2 Hsmooth z ltac:(lra))).
   + intros z Hz.
     apply (proj2 (epsilon_physical_margins eps C kap z (kp z) Heps HC
       (proj1 (proj1 Hsmooth z ltac:(lra)))
       (proj2 (proj2 Hsmooth z ltac:(lra))))).
Qed.


End EpsilonPhysicalAccuracy.

(* Early equilibrium keeps its slope-independent physical enclosure proof.
   This wrapper merely expresses those stronger small budgets in the common
   epsilon-parametric output format used by the nested branch solver. *)
Theorem physical_binary64_epsilon_early_accuracy eps A D T C chi a r kap choice :
 0<eps -> eps<=5/2 ->
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 (forall x, Rmin T (radiation_temperature a r)<=x<=Rmax T (radiation_temperature a r) ->
  0<kap x /\ continuity_pt kap x) ->
 FiniteNormalNodes choice (initial_nodes (RN64 choice) a T r) ->
 FiniteNormalNodes choice [A*T] ->
 1-64*binary64_u<=eval_initial_ratio (RN64 choice) a T r<=1+64*binary64_u ->
 EpsilonMainAccuracy64 eps T (exact_dust A D T C chi a r kap)
  (RN64 choice (A*T)) (A*exact_gas A D T C chi a r kap)
  r (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intros Heps Heps_upper HA HD HT HC Hchi Ha Hr HK NI NG HW.
 pose proof (initial_early_finite64_accuracy A D T C chi a r kap
   HA HD HT HC Hchi Ha Hr HK choice NI NG HW) as [HDust [HGas HRad]].
 destruct (exact_dust_spec A D T C chi a r kap HA HD HT HC Hchi Ha Hr HK)
  as [Hroot [Hdom [HE Hcol]]].
 pose proof (proj1 (HK _ Hdom)) as Hkr.
 pose proof (proj1 (gas_map_spec A D T (exact_dust A D T C chi a r kap)
  HA HD HT Hroot)) as Hgr.
 change (0<exact_gas A D T C chi a r kap) in Hgr.
 pose proof binary64_lambda_positive as Hlam.
 pose proof (coordinate_budget_dominates_width eps Heps) as Hbudget.
 assert (HEp:0<RN64 choice (A*T)).
 { unfold FiniteNormalNodes in NG.
   apply List.Forall_forall with (x:=A*T) in NG; [tauto|simpl; auto]. }
 assert (HEref:0<A*exact_gas A D T C chi a r kap) by nra.
 assert (HRref:0<OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
 { apply OuterDerivatives.radiation_positive; assumption. }
 assert (Hd:log_error T (exact_dust A D T C chi a r kap)<=coordinate_budget eps) by lra.
 assert (Hg:log_error (RN64 choice (A*T)) (A*exact_gas A D T C chi a r kap)<=
  epsilon_gas_budget (coordinate_budget eps)) by (unfold epsilon_gas_budget; lra).
 assert (Hrad:log_error r (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap))<=
  epsilon_radiation_budget eps (coordinate_budget eps)) by (unfold epsilon_radiation_budget; nra).
 constructor; try assumption.
 - apply log_error_relative; assumption.
 - apply log_error_relative; assumption.
 - apply log_error_relative; assumption.
Qed.

Print Assumptions physical_binary64_epsilon_main_accuracy_local.
Print Assumptions epsilon_physical_solution_unique.
Print Assumptions physical_binary64_epsilon_early_accuracy.
