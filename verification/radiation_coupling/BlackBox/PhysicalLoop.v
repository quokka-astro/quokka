(* Concrete physical instantiation of the outer safeguarded loop. The per-trial
   inner interface is a proved chart acceptance certificate, not an assumed
   aggregate evaluator error. Opacity and range contracts cover the full,
   possibly outward-expanded, initial grid interval. *)
From Coq Require Import Reals Psatz Field Lia List ZArith.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra GasMap GasBounds LogCalculus OuterDerivatives OuterBounds
 FloatingPoint Guards GuardFloatBridge WidthSafety InnerEvaluator Brackets SolverLoop
 PhysicalRoot UniqueRoot Initialization.
Open Scope R_scope.

Definition physical_balance (heating:bool) A D T C chi a r kap :=
 if heating then physical_heating A D T C chi a r kap
 else physical_cooling A D T C chi a r kap.
Definition physical_balance_derivative (heating:bool) A D T C chi a r kap kp x :=
 if heating then physical_heating_derivative A D T C chi a r kap kp x
 else physical_cooling_derivative A D T C chi a r kap kp x.
Definition physical_region (heating:bool) T x := if heating then T<=x else x<=T.

Lemma physical_balance_positive heating A D T C chi a r kap x :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<kap x -> physical_region heating T x ->
 0<physical_balance heating A D T C chi a r kap x.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hk Hregion.
 destruct heating; simpl in Hregion |- *.
 - apply OuterDerivatives.heating_balance_positive; try assumption.
   apply heating_gas_branch; assumption.
 - apply cooling_balance_positive; try assumption.
   apply cooling_gas_branch; assumption.
Qed.

Lemma physical_balance_continuous heating A D T C chi a r kap x :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<kap x -> continuity_pt kap x ->
 continuity_pt (physical_balance heating A D T C chi a r kap) x.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hk Hcont.
 destruct heating; simpl.
 - apply heating_balance_continuous; try assumption.
   apply gas_map_continuous; assumption.
 - apply cooling_balance_continuous; try assumption.
   apply gas_map_continuous; assumption.
Qed.

Lemma physical_balance_is_derive heating A D T C chi a r kap kp x :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<kap x -> is_derive kap x (kp x) ->
 is_derive (physical_balance heating A D T C chi a r kap) x
 (physical_balance_derivative heating A D T C chi a r kap kp x).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hk Hder.
 destruct heating; simpl.
 - apply heating_balance_is_derive; try assumption; apply gas_map_is_derive; assumption.
 - apply cooling_balance_is_derive; try assumption; apply gas_map_is_derive; assumption.
Qed.

Lemma physical_concrete_margins C kap x kp :
 0<C -> 0<kap x -> -(7/2)<=opacity_slope kap x kp<=1/2 ->
 1/2<=1-effective_slope C kap x kp /\ 1/2<=4+effective_slope C kap x kp.
Proof.
 intros HC Hk Hp.
 pose proof (example_effective_slope (opacity_slope kap x kp) (optical_depth C kap x) Hp
  ltac:(pose proof (optical_depth_positive C kap x HC Hk); lra)) as H.
 unfold effective_slope; split; lra.
Qed.

Lemma positive_elasticity_derivative f x df :
 0<x -> 0<f x -> 1/2<=elasticity f x df -> 0<df.
Proof.
 intros Hx Hf Hbound.
 unfold elasticity in Hbound.
 pose proof (Rmult_le_compat_r (f x) (1/2) (x*df/f x) ltac:(lra) Hbound) as H.
 replace (x*df/f x*f x) with (x*df) in H by (field; lra).
 nra.
Qed.

Lemma physical_oriented_derivative_positive heating A D T C chi a r kap kp x :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<kap x -> physical_region heating T x ->
 -(7/2)<=opacity_slope kap x (kp x)<=1/2 ->
 0<(if heating then physical_balance_derivative heating A D T C chi a r kap kp x
 else -physical_balance_derivative heating A D T C chi a r kap kp x).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hk Hregion Hp.
 destruct (physical_concrete_margins C kap x (kp x) HC Hk Hp) as [Hh Hc].
 pose proof (physical_balance_positive heating A D T C chi a r kap x
  HA HD HT HC Hchi Ha Hr Hx Hk Hregion) as Hpos.
 destruct heating; simpl in Hregion,Hpos |- *.
 - apply (positive_elasticity_derivative (physical_heating A D T C chi a r kap) x); try assumption.
   change (1/2<=elasticity (OuterDerivatives.heating_balance A T C chi a r kap (gas_map A D T)) x
    (heating_derivative A T C chi a r kap (gas_map A D T) x (kp x)
     (physical_gas_derivative A D T (gas_map A D T x)))).
   apply heating_conditioning; try assumption; try lra.
   + apply heating_gas_branch; assumption.
   + symmetry; apply (proj2 (gas_map_spec A D T x HA HD HT Hx)).
 - apply (positive_elasticity_derivative (physical_cooling A D T C chi a r kap) x); try assumption.
   replace (elasticity (physical_cooling A D T C chi a r kap) x
    (-physical_cooling_derivative A D T C chi a r kap kp x)) with
    (-elasticity (physical_cooling A D T C chi a r kap) x
    (physical_cooling_derivative A D T C chi a r kap kp x)) by (unfold elasticity,Rdiv; ring).
   change (1/2<= -elasticity (OuterDerivatives.cooling_balance A T C chi a r kap (gas_map A D T)) x
    (cooling_derivative A T C chi a r kap (gas_map A D T) x (kp x)
     (physical_gas_derivative A D T (gas_map A D T x)))).
   apply cooling_conditioning; try assumption; try lra.
   + apply (proj1 (gas_map_spec A D T x HA HD HT Hx)).
   + apply cooling_gas_branch; assumption.
Qed.

Lemma physical_oriented_is_derive heating A D T C chi a r kap kp x :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<kap x -> is_derive kap x (kp x) ->
 is_derive (oriented_balance heating (physical_balance heating A D T C chi a r kap)) x
 (if heating then physical_balance_derivative heating A D T C chi a r kap kp x
  else -physical_balance_derivative heating A D T C chi a r kap kp x).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hk Hder.
 pose proof (physical_balance_is_derive heating A D T C chi a r kap kp x
  HA HD HT HC Hchi Ha Hr Hx Hk Hder) as H.
 destruct heating; unfold oriented_balance.
 - replace (physical_balance_derivative true A D T C chi a r kap kp x) with
   (physical_balance_derivative true A D T C chi a r kap kp x-0) by ring.
   apply (@is_derive_minus R_AbsRing R_NormedModule); [exact H|apply (@is_derive_const R_AbsRing R_NormedModule)].
 - replace (-physical_balance_derivative false A D T C chi a r kap kp x) with
   (0-physical_balance_derivative false A D T C chi a r kap kp x) by ring.
   apply (@is_derive_minus R_AbsRing R_NormedModule); [apply (@is_derive_const R_AbsRing R_NormedModule)|exact H].
Qed.

Lemma derivative_positive_nondecreasing f df lo hi :
 (forall x, lo<=x<=hi -> continuity_pt f x) ->
 (forall x, lo<x<hi -> is_derive f x (df x)) ->
 (forall x, lo<x<hi -> 0<df x) -> nondecreasing_on f lo hi.
Proof.
 intros Hcont Hder Hpos x y Hlx Hxy Hyh.
 destruct (Rle_lt_or_eq_dec x y Hxy) as [Hlt|Heq]; [|subst; reflexivity].
 destruct (mvt_interior_property f df x y (fun d=>0<d) 1 ltac:(lra)) as [v [Hv He]].
 - intros t Ht; apply Hder; rewrite Rmin_left,Rmax_right in Ht by lra; lra.
 - intros t Ht; apply Hcont; rewrite Rmin_left,Rmax_right in Ht by lra; lra.
 - intros t Ht; apply Hpos; rewrite Rmin_left,Rmax_right in Ht by lra; lra.
 - nra.
Qed.

Theorem physical_outer_monotonicity heating A D T C chi a r kap kp lo hi :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<lo ->
 (forall x, lo<=x<=hi -> physical_region heating T x) ->
 (forall x, lo<=x<=hi -> 0<kap x /\ continuity_pt kap x) ->
 (forall x, lo<x<hi -> is_derive kap x (kp x) /\ -(7/2)<=opacity_slope kap x (kp x)<=1/2) ->
 nondecreasing_on (oriented_balance heating (physical_balance heating A D T C chi a r kap)) lo hi.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hlo Hregion Hkap Hder.
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
 - intros x Hx; apply physical_oriented_derivative_positive; try assumption; try lra.
   + apply (Hkap x ltac:(lra)).
   + apply Hregion; lra.
   + apply (Hder x Hx).
Qed.

Lemma physical_exact_outer heating A D T C chi a r kap x :
 0<A -> 0<D -> 0<T -> 0<x -> physical_region heating T x ->
 exact_outer heating (A*Rabs(gas_map A D T x-T))
 (chi*saturate(C*kap x)) (a*x^4) r = physical_balance heating A D T C chi a r kap x.
Proof.
 intros HA HD HT Hx Hregion.
 assert (HV:chi*saturate(C*kap x)=coupling C chi kap x) by
  (unfold saturate,coupling,optical_depth,Rdiv; ring).
 rewrite HV; destruct heating; simpl in Hregion |- *;
 unfold physical_heating,physical_cooling,OuterDerivatives.heating_balance,
 OuterDerivatives.cooling_balance,emission.
 - pose proof (heating_gas_branch A D T x HA HD HT Hregion).
   rewrite Rabs_right by lra; reflexivity.
 - pose proof (cooling_gas_branch A D T x HA HD HT Hx Hregion).
   rewrite Rabs_left1 by lra.
   replace (-(gas_map A D T x-T)) with (T-gas_map A D T x) by ring; reflexivity.
Qed.

Lemma physical_outer_beta :
 outer_beta (52*binary64_lambda) (8*binary64_lambda) lambda64=71*binary64_lambda.
Proof.
 rewrite <- guard_lambda64; unfold outer_beta.
 rewrite Rmax_left by (pose proof binary64_lambda_positive; lra); ring.
Qed.

Theorem physical_evaluator_budget choice heating A D T C chi a r kap khat x th qh :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<x ->
 physical_region heating T x ->
 PlanckValueContract kap khat (8*binary64_lambda) ->
 InnerResult choice A D T x th qh ->
 FiniteNormalNodes choice (outer_nodes (RN64 choice) heating qh C (khat x) chi a x r) ->
 LogBound (71*binary64_lambda)
 (eval_outer (RN64 choice) heating qh C (khat x) chi a x r)
 (physical_balance heating A D T C chi a r kap x).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hregion HK HI N.
 destruct (inner_result_accuracy choice A D T x th qh HA HD HT Hx HI) as [Hth [[He Hq0]|Hq]].
 - subst x qh.
   pose proof (planck_outer_zero_finite64_budget choice (52*binary64_lambda)
    (8*binary64_lambda) kap khat heating C chi a T r HC Hchi Ha HT Hr HK N) as H.
   rewrite physical_outer_beta in H.
   assert (Hq:A*Rabs(gas_map A D T T-T)=0).
   { rewrite gas_map_equilibrium by assumption; rewrite Rminus_diag_eq,Rabs_R0 by reflexivity; ring. }
   rewrite <- Hq in H at 2.
   rewrite physical_exact_outer in H by assumption; exact H.
 - pose proof (planck_outer_finite64_budget choice (52*binary64_lambda)
    (8*binary64_lambda) kap khat heating qh (A*Rabs(gas_map A D T x-T))
    C chi a x r HC Hchi Ha Hx Hr HK Hq N) as H.
   rewrite physical_outer_beta,physical_exact_outer in H by assumption; exact H.
Qed.

Lemma physical_evaluator_budget_local choice heating A D T C chi a r kap khat x th qh :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<x ->
 physical_region heating T x ->
 LogBound (8*binary64_lambda) (khat x) (kap x) ->
 InnerResult choice A D T x th qh ->
 FiniteNormalNodes choice (outer_nodes (RN64 choice) heating qh C (khat x) chi a x r) ->
 LogBound (71*binary64_lambda)
 (eval_outer (RN64 choice) heating qh C (khat x) chi a x r)
 (physical_balance heating A D T C chi a r kap x).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hreg HK HI N.
 assert (Hconst:PlanckValueContract (fun _=>kap x) (fun _=>khat x) (8*binary64_lambda)).
 { intros y Hy; exact HK. }
 pose proof (physical_evaluator_budget choice heating A D T C chi a r
 (fun _=>kap x) (fun _=>khat x) x th qh HA HD HT HC Hchi Ha Hr Hx Hreg Hconst HI N) as H.
 destruct heating; exact H.
Qed.

Lemma physical_reference_balance heating A D T C chi a r kap :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 (forall x, Rmin T (radiation_temperature a r)<=x<=Rmax T (radiation_temperature a r) ->
  0<kap x /\ continuity_pt kap x) ->
 physical_balance heating A D T C chi a r kap (exact_dust A D T C chi a r kap)=1.
Proof.
 intros HA HD HT HC Hchi Ha Hr HK.
 destruct (exact_dust_spec A D T C chi a r kap HA HD HT HC Hchi Ha Hr HK)
  as [Hx [Hrange [HE Hcol]]].
 pose proof (proj1 (HK _ Hrange)) as Hk.
 destruct heating; simpl; unfold physical_heating,physical_cooling.
 - apply (proj2 (heating_balance_root_equiv A T C chi a r kap (gas_map A D T) _ HC Hchi Hk Hr)); exact HE.
 - apply (proj2 (cooling_balance_root_equiv A T C chi a r kap (gas_map A D T) _ HC Hchi Hk Ha Hx)); exact HE.
Qed.

Definition physical_width choice (value:nat->R) i j :=
 RN64 choice (RN64 choice (value j-value i)/value i).
Definition physical_measured choice heating (A D T C chi a r:R) khat
 (value qhat:nat->R) i :=
 eval_outer (RN64 choice) heating (qhat i) C (khat (value i)) chi a (value i) r.

Lemma physical_width_arithmetic choice value i j :
 0<value i -> value i<value j -> SafeDifference64 (value i) (value j) ->
 normal64 (RN64 choice (value j-value i)/value i) ->
 exists es ed, Rabs es<=binary64_u /\ Rabs ed<=binary64_u /\
 physical_width choice value i j=((value j-value i)*(1+es)/value i)*(1+ed).
Proof. unfold physical_width; apply safe_width_relative_factors. Qed.

Section ConcretePhysicalLoop.
Variables A D T C chi a r : R.
Variables kap khat kp : R->R.
Variable choice : Z->bool.
Variable heating : bool.
Variables value that qhat : nat->R.
Variable proposal : nat->nat->nat.
Variables il iu : nat.
Hypotheses (HA:0<A) (HD:0<D) (HT:0<T) (HC:0<C) (Hchi:0<chi) (Ha:0<a) (Hr:0<r).
Hypothesis Hreference : forall x,
 Rmin T (radiation_temperature a r)<=x<=Rmax T (radiation_temperature a r) ->
 0<kap x /\ continuity_pt kap x.
Hypothesis Hopacity : forall x, value il<=x<=value iu -> 0<kap x /\ continuity_pt kap x.
Hypothesis Hslope : forall x, value il<x<value iu ->
 is_derive kap x (kp x) /\ -(7/2)<=opacity_slope kap x (kp x)<=1/2.
Hypothesis HPlanck : forall i, (il<=i<=iu)%nat ->
 LogBound (8*binary64_lambda) (khat(value i)) (kap(value i)).
Hypothesis Hinitial_nodes : FiniteNormalNodes choice (initial_nodes (RN64 choice) a T r).
Hypothesis Houtward_nodes : FiniteNormalNodes choice (outward_nodes (RN64 choice) heating a r).
Hypothesis Hbranch_guard : if heating
 then eval_initial_ratio (RN64 choice) a T r<1-64*binary64_u
 else 1+64*binary64_u<eval_initial_ratio (RN64 choice) a T r.
Hypothesis Hindices : (il<iu)%nat.
Hypothesis Hlower : value il=if heating then T else outward_endpoint (RN64 choice) heating a r.
Hypothesis Hupper : value iu=if heating then outward_endpoint (RN64 choice) heating a r else T.
Hypothesis Gincreasing : forall i j, (il<=i)%nat -> (i<j)%nat -> (j<=iu)%nat -> value i<value j.
Hypothesis Gformat : forall i, (il<=i<=iu)%nat -> format64 (value i).
Hypothesis Gnormal : forall i, (il<=i<=iu)%nat -> normal64 (value i).
Hypothesis Gcomplete : forall i, (il<=i)%nat -> (S i<=iu)%nat ->
 forall x, format64 x -> ~(value i<x<value (S i)).
Hypothesis Hwidth_nodes : forall i j, (il<=i)%nat -> (i<j)%nat -> (j<=iu)%nat ->
 normal64 (RN64 choice (value j-value i)/value i).
Hypothesis Hinner : forall i, (il<=i<=iu)%nat ->
 InnerResult choice A D T (value i) (that i) (qhat i).
Hypothesis Houter_nodes : forall i, (il<=i<=iu)%nat ->
 FiniteNormalNodes choice
 (outer_nodes (RN64 choice) heating (qhat i) C (khat (value i)) chi a (value i) r).

Lemma concrete_initial_bracket : 0<value il /\ value il<value iu /\
 root_bracket (value il) (value iu) (exact_dust A D T C chi a r kap).
Proof.
 unfold root_bracket; rewrite Hlower,Hupper; destruct heating; simpl in *.
 - pose proof (heating_initial_bracket A D T C chi a r kap HA HD HT HC Hchi Ha Hr Hreference
    choice Hinitial_nodes Houtward_nodes Hbranch_guard); tauto.
 - pose proof (cooling_initial_bracket A D T C chi a r kap HA HD HT HC Hchi Ha Hr Hreference
    choice Hinitial_nodes Houtward_nodes Hbranch_guard); tauto.
Qed.

Lemma concrete_grid_le i j : (il<=i)%nat -> (i<=j)%nat -> (j<=iu)%nat -> value i<=value j.
Proof.
 intros Hi Hij Hj; destruct (Nat.eq_dec i j) as [He|Hne].
 - subst; reflexivity.
 - left; apply Gincreasing; lia.
Qed.
Lemma concrete_grid_positive i : (il<=i<=iu)%nat -> 0<value i.
Proof.
 intro Hi; pose proof (proj1 concrete_initial_bracket) as Hlo.
 pose proof (concrete_grid_le il i ltac:(lia) ltac:(lia) ltac:(lia)); lra.
Qed.
Lemma concrete_grid_range i : (il<=i<=iu)%nat -> value il<=value i<=value iu.
Proof. intros; split; apply concrete_grid_le; lia. Qed.
Lemma concrete_region x : value il<=x<=value iu -> physical_region heating T x.
Proof.
 intro Hx; unfold physical_region; destruct heating;
 rewrite Hlower,Hupper in Hx; simpl in Hx; lra.
Qed.

Lemma concrete_evaluator i : (il<=i<=iu)%nat ->
 0<physical_measured choice heating A D T C chi a r khat value qhat i /\
 0<physical_balance heating A D T C chi a r kap (value i) /\
 Rabs(ln(physical_measured choice heating A D T C chi a r khat value qhat i)-
   ln(physical_balance heating A D T C chi a r kap (value i)))<=71*binary64_lambda.
Proof.
 intro Hi.
 apply (physical_evaluator_budget_local choice heating A D T C chi a r kap khat (value i) (that i) (qhat i));
 try assumption.
 - apply concrete_grid_positive; assumption.
 - apply concrete_region,concrete_grid_range; assumption.
 - apply HPlanck; assumption.
 - apply Hinner; assumption.
 - apply Houter_nodes; assumption.
Qed.
Lemma concrete_monotonicity :
 nondecreasing_on (oriented_balance heating (physical_balance heating A D T C chi a r kap))
 (value il) (value iu).
Proof.
 apply (physical_outer_monotonicity heating A D T C chi a r kap kp); try assumption.
 - apply concrete_grid_positive; lia.
 - exact concrete_region.
Qed.
Lemma concrete_width_arithmetic i j : (il<=i)%nat -> (i<j)%nat -> (j<=iu)%nat ->
 exists es ed, Rabs es<=binary64_u /\ Rabs ed<=binary64_u /\
 physical_width choice value i j=((value j-value i)*(1+es)/value i)*(1+ed).
Proof.
 intros Hi Hij Hj; apply physical_width_arithmetic.
 - apply concrete_grid_positive; lia.
 - apply Gincreasing; assumption.
 - apply normal_formatted_endpoints_safe.
   + apply concrete_grid_positive; lia.
   + apply Gnormal; lia.
   + apply Gformat; lia.
   + apply Gformat; lia.
   + left; apply Gincreasing; assumption.
 - apply (Hwidth_nodes i j); assumption.
Qed.
Lemma concrete_exact_root :
 physical_balance heating A D T C chi a r kap (exact_dust A D T C chi a r kap)=1.
Proof. apply physical_reference_balance; assumption. Qed.
Lemma concrete_initial_state : loop_state value il iu (exact_dust A D T C chi a r kap) il iu.
Proof.
 split; [lia|]. exact (proj2 (proj2 concrete_initial_bracket)).
Qed.

Theorem physical_scalar_solve_certificate fuel :
 result_certificate value (physical_balance heating A D T C chi a r kap) il iu
 (exact_dust A D T C chi a r kap)
 (scalar_solve fuel heating
  (physical_measured choice heating A D T C chi a r khat value qhat)
  (physical_width choice value) proposal il iu).
Proof.
 apply (scalar_solve_certificate heating value (physical_balance heating A D T C chi a r kap)
 (physical_measured choice heating A D T C chi a r khat value qhat)
 (physical_width choice value) proposal il iu (exact_dust A D T C chi a r kap));
 auto using concrete_grid_positive,concrete_width_arithmetic,concrete_evaluator,
 concrete_monotonicity,concrete_exact_root,concrete_initial_state.
Qed.

Theorem physical_scalar_solve_sufficient_fuel fuel : (iu-il<=fuel)%nat ->
 successful (scalar_solve fuel heating
  (physical_measured choice heating A D T C chi a r khat value qhat)
  (physical_width choice value) proposal il iu).
Proof.
 intro Hfuel.
 apply (scalar_solve_sufficient_fuel heating value (physical_balance heating A D T C chi a r kap)
 (physical_measured choice heating A D T C chi a r khat value qhat)
 (physical_width choice value) proposal il iu (exact_dust A D T C chi a r kap));
 auto using concrete_grid_positive,concrete_width_arithmetic,concrete_evaluator,
 concrete_monotonicity,concrete_exact_root,concrete_initial_state.
Qed.
End ConcretePhysicalLoop.

Print Assumptions physical_evaluator_budget.
Print Assumptions physical_outer_monotonicity.
Print Assumptions physical_scalar_solve_certificate.
Print Assumptions physical_scalar_solve_sufficient_fuel.
