(* Public physical-model accuracy theorems. Reference outputs are defined from
   existence of the original equations, not supplied as accuracy assumptions.
   The loop modules establish the acceptance certificates used here. *)
From Coq Require Import Reals Psatz Field List.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra GasMap GasBounds LogCalculus FloatingPoint
 Guards GuardFloatBridge OuterDerivatives PhysicalRoot UniqueRoot OuterBounds
 InnerEvaluator Binary64Accuracy EarlyEquilibrium Initialization.
Import ListNotations.
Open Scope R_scope.

Definition smooth_planck_on (kap kp:R->R) lo hi : Prop :=
 (forall x, lo<=x<=hi -> 0<kap x /\ continuity_pt kap x) /\
 (forall x, lo<x<hi -> is_derive kap x (kp x) /\
  -(7/2)<=opacity_slope kap x (kp x)<=1/2).

Lemma interval_endpoints_containment lo hi a b x :
 lo<=a<=hi -> lo<=b<=hi -> Rmin a b<=x<=Rmax a b -> lo<=x<=hi.
Proof.
 intros Ha Hb Hx. unfold Rmin,Rmax in Hx; destruct Rle_dec; lra.
Qed.

Section PhysicalAccuracy.
Variables A D T C chi a r : R.
Hypotheses (HA:0<A) (HD:0<D) (HT:0<T) (HC:0<C) (Hchi:0<chi) (Ha:0<a) (Hr:0<r).
Variables kap kp khat : R->R.
Variables lo hi : R.
Hypothesis Hphysical_domain : lo<=T<=hi /\ lo<=radiation_temperature a r<=hi.
Hypothesis Hsmooth : smooth_planck_on kap kp lo hi.

Lemma reference_dust_spec :
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

Theorem physical_binary64_main_accuracy choice (heating:bool) trial th qh :
 0<trial -> lo<=trial<=hi ->
 (if heating then T<=radiation_temperature a r /\ T<=trial
  else radiation_temperature a r<=T /\ trial<=T) ->
 PlanckValueContract kap khat (8*binary64_lambda) ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r
  (exact_dust A D T C chi a r kap) trial ->
 FiniteNormalNodes choice [A*th] ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat trial) a trial) ->
 MainAccuracy64 trial (exact_dust A D T C chi a r kap)
  (returned_gas choice A th) (A*exact_gas A D T C chi a r kap)
  (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intros Htrial Htdom Hbranch HK HI HO NG NR.
 destruct reference_dust_spec as [Hroot [Hrdom HS]].
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
 change (MainAccuracy64 trial (exact_dust A D T C chi a r kap)
  (returned_gas choice A th) (gas_energy A A D T (exact_dust A D T C chi a r kap))
  (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap))).
 apply (binary64_main_accuracy choice heating A D T C chi a r A kap kp khat
   (exact_dust A D T C chi a r kap) trial th qh); try assumption.
 - intros x Hx. apply (proj2 (proj1 Hsmooth x
     (interval_endpoints_containment lo hi _ _ x Hrdom Htdom Hx))).
 - intros x Hx. apply (proj1 (proj2 Hsmooth x
     (subinterval_open lo hi _ _ x Hrdom Htdom Hx))).
 - intros x Hx. apply (proj2 (proj2 Hsmooth x
     (subinterval_open lo hi _ _ x Hrdom Htdom Hx))).
Qed.

Theorem physical_binary64_main_accuracy_local choice (heating:bool) trial th qh :
 0<trial -> lo<=trial<=hi ->
 (if heating then T<=radiation_temperature a r /\ T<=trial
  else radiation_temperature a r<=T /\ trial<=T) ->
 LogBound (8*binary64_lambda) (khat trial) (kap trial) ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r
  (exact_dust A D T C chi a r kap) trial ->
 FiniteNormalNodes choice [A*th] ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat trial) a trial) ->
 MainAccuracy64 trial (exact_dust A D T C chi a r kap)
  (returned_gas choice A th) (A*exact_gas A D T C chi a r kap)
  (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intros Htrial Htdom Hbranch HK HI HO NG NR.
 destruct reference_dust_spec as [Hroot [Hrdom HS]].
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
 change (MainAccuracy64 trial (exact_dust A D T C chi a r kap)
  (returned_gas choice A th) (gas_energy A A D T (exact_dust A D T C chi a r kap))
  (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap))).
 apply (binary64_main_accuracy_local choice heating A D T C chi a r A kap kp khat
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

Theorem physical_binary64_early_accuracy choice :
 FiniteNormalNodes choice (initial_nodes (RN64 choice) a T r) ->
 FiniteNormalNodes choice [A*T] ->
 1-64*binary64_u<=eval_initial_ratio (RN64 choice) a T r<=1+64*binary64_u ->
 MainAccuracy64 T (exact_dust A D T C chi a r kap)
  (RN64 choice (A*T)) (A*exact_gas A D T C chi a r kap)
  r (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intros NI NG HW.
 assert (HK:forall x,
 Rmin T (radiation_temperature a r)<=x<=Rmax T (radiation_temperature a r) ->
 0<kap x /\ continuity_pt kap x).
 { intros x Hx; apply (proj1 Hsmooth).
   apply (interval_endpoints_containment lo hi T (radiation_temperature a r) x); tauto. }
 pose proof (initial_early_finite64_accuracy A D T C chi a r kap
   HA HD HT HC Hchi Ha Hr HK choice NI NG HW) as [HDust [HGas HRad]].
 destruct reference_dust_spec as [Hroot [Hdom [_ [Hgr [Hkr _]]]]].
 pose proof binary64_lambda_positive as Hlam.
 assert (HEp:0<RN64 choice (A*T)).
 { unfold FiniteNormalNodes in NG.
   apply List.Forall_forall with (x:=A*T) in NG; [tauto|simpl; auto]. }
 assert (HEref:0<A*exact_gas A D T C chi a r kap) by nra.
 assert (HRref:0<OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
 { apply OuterDerivatives.radiation_positive; assumption. }
 assert (Hd:log_error T (exact_dust A D T C chi a r kap)<=400*binary64_lambda) by lra.
 assert (Hg:log_error (RN64 choice (A*T)) (A*exact_gas A D T C chi a r kap)<=853*binary64_lambda) by lra.
 assert (Hrad:log_error r (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap))<=3017*binary64_lambda) by lra.
 constructor; try assumption.
 - apply binary64_relative_400; assumption.
 - apply binary64_relative_853; assumption.
 - apply binary64_relative_3017; assumption.
Qed.
Theorem physical_solution_unique x t :
 positive_physical_solution A D T C chi a r kap x t ->
 x=exact_dust A D T C chi a r kap /\ t=exact_gas A D T C chi a r kap.
Proof.
 intro HS.
 destruct reference_dust_spec as [Hx [Hb Href]].
 destruct (radiation_temperature_spec a r Ha Hr) as [HTr HB].
 destruct Hphysical_domain as [HTdom HTrdom].
 destruct (Rle_dec T (radiation_temperature a r)) as [Hheat|Hcool].
 - apply (heating_physical_solution_unique A D T C chi a r kap kp
     (radiation_temperature a r) (1/2) x t
     (exact_dust A D T C chi a r kap) (exact_gas A D T C chi a r kap));
     try assumption; try lra.
   + intros z Hz; apply (proj1 Hsmooth); lra.
   + intros z Hz; apply (proj1 (proj2 Hsmooth z ltac:(lra))).
   + intros z Hz.
     apply (proj1 (concrete_opacity_margins C kap z (kp z) HC
       (proj1 (proj1 Hsmooth z ltac:(lra)))
       (proj2 (proj2 Hsmooth z ltac:(lra))))).
 - apply (cooling_physical_solution_unique A D T C chi a r kap kp
     (radiation_temperature a r) (1/2) x t
     (exact_dust A D T C chi a r kap) (exact_gas A D T C chi a r kap));
     try assumption; try lra.
   + intros z Hz; apply (proj1 Hsmooth); lra.
   + intros z Hz; apply (proj1 (proj2 Hsmooth z ltac:(lra))).
   + intros z Hz.
     apply (proj1 (proj2 (concrete_opacity_margins C kap z (kp z) HC
       (proj1 (proj1 Hsmooth z ltac:(lra)))
       (proj2 (proj2 Hsmooth z ltac:(lra)))))).
Qed.

End PhysicalAccuracy.

Print Assumptions physical_binary64_main_accuracy.
Print Assumptions physical_binary64_early_accuracy.

Print Assumptions physical_solution_unique.

Print Assumptions physical_binary64_main_accuracy_local.
