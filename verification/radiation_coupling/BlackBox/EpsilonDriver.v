(* Epsilon-parametric verified nested solver, reusing the actual operational
   driver and all failure constructors from NestedDriver. Range, initialization,
   node, grid, and finite-fuel contracts are retained verbatim. *)
From Coq Require Import Reals Psatz Field Lia List ZArith.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra GasMap GasBounds LogCalculus OuterDerivatives OuterBounds
 FloatingPoint Guards GuardFloatBridge WidthSafety InnerEvaluator Brackets SolverLoop
 PhysicalRoot UniqueRoot Initialization PhysicalLoop Binary64Accuracy EndToEnd
 NormalGrid InnerLoop NestedInnerSolve StoredInputs NestedDriver EpsilonAccuracy EpsilonPhysical.
Import ListNotations.
Open Scope R_scope.

Section VerifiedEpsilonNestedDriver.
Variable eps : R.
Hypotheses (Heps:0<eps) (Heps_upper:eps<=5/2).
Variables A D T C chi a r : R.
Variables kap khat kp : R->R.
Variable choice : Z->bool.
Variable heating : bool.
Let value := grid64.
Variable proposal : nat->nat->nat.
Variables il iu inner_fuel : nat.
Variable inner_value : nat->inner_chart->nat->R.
Variable inner_proposal : nat->inner_chart->nat->nat->nat.
Variables inner_lower inner_upper : nat->inner_chart->nat.
Variable inner_range : nat->inner_chart->bool.
Variable boundary_range : nat->bool.
Hypotheses (HA:0<A) (HD:0<D) (HT:0<T) (HC:0<C) (Hchi:0<chi) (Ha:0<a) (Hr:0<r).
Hypothesis Hreference : forall x,
 Rmin T (radiation_temperature a r)<=x<=Rmax T (radiation_temperature a r) ->
 0<kap x /\ continuity_pt kap x.
Hypothesis Hopacity : forall x, value il<=x<=value iu -> 0<kap x /\ continuity_pt kap x.
Hypothesis Hslope : forall x, value il<x<value iu ->
 is_derive kap x (kp x) /\ -4+eps<=opacity_slope kap x (kp x)<=1-eps.
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
Hypothesis Hwidth_nodes : forall i j, (il<=i)%nat -> (i<j)%nat -> (j<=iu)%nat ->
 normal64 (RN64 choice (value j-value i)/value i).
Hypothesis Hwidth_finite : forall i j, (il<=i)%nat -> (i<j)%nat -> (j<=iu)%nat ->
 finite64 (RN64 choice (value j-value i)) /\
 FiniteNormalNodes choice [RN64 choice (value j-value i)/value i].
Hypothesis Hinner_grid : forall i, (il<=i<=iu)%nat -> forall c,
 chart_physical c A D T (value i) -> inner_range i c=true ->
 InnerGrid choice c A D T (value i) (inner_value i c) (inner_lower i c) (inner_upper i c).
Hypothesis Hboundary_nodes : forall i, (il<=i<=iu)%nat -> boundary_range i=true ->
 FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T (value i) (T/2)).

Definition driver_inner i := solve_nested_inner inner_fuel choice A D T (value i)
 (inner_value i) (inner_proposal i) (inner_lower i) (inner_upper i) (inner_range i) (boundary_range i).
Definition driver_measure i qh := eval_outer (RN64 choice) heating qh C (khat(value i)) chi a (value i) r.
Definition nested_solve fuel := run_nested_driver fuel heating driver_inner driver_measure
 (physical_width choice value) proposal il iu.

(* These are primitive node contracts, conditional only on which output the
   actual inner call returns. They are not whole-evaluator error assumptions. *)
Hypothesis Houter_nodes : forall i th qh, (il<=i<=iu)%nat -> driver_inner i=InnerConverged th qh ->
 FiniteNormalNodes choice (outer_nodes (RN64 choice) heating qh C (khat(value i)) chi a (value i) r).
Hypothesis Hgas_nodes : forall i th qh, (il<=i<=iu)%nat -> driver_inner i=InnerConverged th qh ->
 FiniteNormalNodes choice [A*th].
Hypothesis Hrad_nodes : forall i, (il<=i<=iu)%nat ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat(value i)) a (value i)).

Lemma driver_initial_domain : 0<value il /\ value il<=T<=value iu /\
 value il<=radiation_temperature a r<=value iu /\
 (if heating then T<radiation_temperature a r else radiation_temperature a r<T).
Proof.
 pose proof (initial_ratio_finite64_budget choice a T r Ha HT Hr Hinitial_nodes) as Hratio.
 pose proof (outward_endpoint_finite64_encloses choice heating a r Ha Hr Houtward_nodes) as Hend.
 rewrite Hlower,Hupper; destruct heating; simpl in *.
 - pose proof (initial_heating_order a T r _ Ha HT Hr Hratio Hbranch_guard); repeat split; lra.
 - pose proof (initial_cooling_order a T r _ Ha HT Hr Hratio Hbranch_guard); repeat split; lra.
Qed.
Lemma driver_initial_state : loop_state value il iu (exact_dust A D T C chi a r kap) il iu.
Proof.
 split; [lia|]. unfold root_bracket; rewrite Hlower,Hupper; destruct heating; simpl in *.
 - pose proof (heating_initial_bracket A D T C chi a r kap HA HD HT HC Hchi Ha Hr Hreference
    choice Hinitial_nodes Houtward_nodes Hbranch_guard); tauto.
 - pose proof (cooling_initial_bracket A D T C chi a r kap HA HD HT HC Hchi Ha Hr Hreference
    choice Hinitial_nodes Houtward_nodes Hbranch_guard); tauto.
Qed.
Lemma driver_grid_range i : (il<=i<=iu)%nat -> value il<=value i<=value iu.
Proof. intro Hi; split; apply grid64_le; lia. Qed.
Lemma driver_region x : value il<=x<=value iu -> physical_region heating T x.
Proof. intro Hx; unfold physical_region; destruct heating; rewrite Hlower,Hupper in Hx; simpl in Hx; lra. Qed.
Lemma driver_root : physical_balance heating A D T C chi a r kap (exact_dust A D T C chi a r kap)=1.
Proof. apply physical_reference_balance; assumption. Qed.
Lemma driver_monotonicity :
 nondecreasing_on (oriented_balance heating (physical_balance heating A D T C chi a r kap))
 (value il) (value iu).
Proof.
 apply (epsilon_physical_outer_monotonicity eps heating A D T C chi a r kap kp); try assumption.
 - apply grid64_positive.
 - exact driver_region.
Qed.
Lemma driver_inner_certificate i th qh : (il<=i<=iu)%nat ->
 driver_inner i=InnerConverged th qh -> InnerResult choice A D T (value i) th qh.
Proof.
 intros Hi HR; unfold driver_inner in HR.
 eapply solve_nested_inner_success; try eassumption.
 - apply grid64_positive.
 - intros; apply (Hinner_grid i Hi); assumption.
 - apply Hboundary_nodes; assumption.
Qed.
Lemma driver_evaluation i th qh : (il<=i<=iu)%nat -> driver_inner i=InnerConverged th qh ->
 LogBound (71*binary64_lambda) (driver_measure i qh)
 (physical_balance heating A D T C chi a r kap (value i)).
Proof.
 intros Hi HR; unfold driver_measure.
 apply (physical_evaluator_budget_local choice heating A D T C chi a r kap khat (value i) th qh);
 try assumption.
 - apply grid64_positive.
 - apply driver_region,driver_grid_range; assumption.
 - apply HPlanck; assumption.
 - apply driver_inner_certificate; assumption.
 - apply (Houter_nodes i th qh); assumption.
Qed.

Lemma driver_local_monotonicity l h : loop_state value il iu (exact_dust A D T C chi a r kap) l h ->
 nondecreasing_on (oriented_balance heating (physical_balance heating A D T C chi a r kap))
 (value l) (value h).
Proof.
 intros [[Hl [Hlh Hh]] Hroot] x y Hlx Hxy Hyh.
 pose proof (grid64_le il l Hl); pose proof (grid64_le h iu Hh); fold value in H,H0.
 apply driver_monotonicity; lra.
Qed.

Lemma driver_trial_inside l h : loop_state value il iu (exact_dust A D T C chi a r kap) l h ->
 width_window(physical_width choice value l h)=false ->
 (l<safeguarded_index l h (proposal l h)<h)%nat.
Proof.
 intros [[Hl [Hlh Hh]] Hroot] Hw; apply safeguarded_index_inside.
 destruct (Nat.lt_ge_cases (l+1) h) as [Hlt|Hge]; [assumption|].
 assert (He:h=S l) by lia; subst h.
 pose proof (Hwidth_nodes l (S l) Hl ltac:(lia) Hh) as Hd.
 assert (Hsafe:SafeDifference64 (value l) (value (S l))).
 { apply normal_formatted_endpoints_safe; auto using grid64_positive,grid64_normal,grid64_format.
   left; apply grid64_increasing; lia. }
 destruct (physical_width_arithmetic choice value l (S l) (grid64_positive l)
 (grid64_increasing l (S l) ltac:(lia)) Hsafe Hd) as [es [ed [Hes [Hed He]]]].
 assert (Hpass:physical_width choice value l (S l)<=16*binary64_u).
 { apply (adjacent_normal_bracket_passes (value l) (value (S l)) es ed (physical_width choice value l (S l)));
   try assumption.
   - apply grid64_positive.
   - apply grid64_normal.
   - apply grid64_format.
   - apply grid64_format.
   - apply grid64_increasing; lia.
   - apply grid64_adjacent_complete. }
 apply width_window_spec in Hpass; congruence.
Qed.

Lemma driver_lower_state l h i th qh : loop_state value il iu (exact_dust A D T C chi a r kap) l h ->
 (l<i<h)%nat -> driver_inner i=InnerConverged th qh ->
 residual_window(driver_measure i qh)=false -> lower_side heating (driver_measure i qh)=true ->
 loop_state value il iu (exact_dust A D T C chi a r kap) i h.
Proof.
 intros HS Hi HI HW Hside.
 pose proof (driver_local_monotonicity l h HS) as HM.
 destruct HS as [[Hl [Hlh Hh]] Hroot].
 destruct (driver_evaluation i th qh ltac:(lia) HI) as [Hme [Hex Herr]].
 pose proof (guarded_lower_side heating _ _ Hme Hex Herr HW Hside) as Hsign.
 split; [lia|].
 apply (lower_update_preserves_root (oriented_balance heating (physical_balance heating A D T C chi a r kap))
  (value l) (value h) (exact_dust A D T C chi a r kap) (value i)); try assumption.
 - split; left; apply grid64_increasing; lia.
 - pose proof driver_root as HReq; unfold oriented_balance; destruct heating; simpl in HReq |- *; lra.
Qed.
Lemma driver_upper_state l h i th qh : loop_state value il iu (exact_dust A D T C chi a r kap) l h ->
 (l<i<h)%nat -> driver_inner i=InnerConverged th qh ->
 residual_window(driver_measure i qh)=false -> lower_side heating (driver_measure i qh)=false ->
 loop_state value il iu (exact_dust A D T C chi a r kap) l i.
Proof.
 intros HS Hi HI HW Hside.
 pose proof (driver_local_monotonicity l h HS) as HM.
 destruct HS as [[Hl [Hlh Hh]] Hroot].
 destruct (driver_evaluation i th qh ltac:(lia) HI) as [Hme [Hex Herr]].
 pose proof (guarded_upper_side heating _ _ Hme Hex Herr HW Hside) as Hsign.
 split; [lia|].
 apply (upper_update_preserves_root (oriented_balance heating (physical_balance heating A D T C chi a r kap))
  (value l) (value h) (exact_dust A D T C chi a r kap) (value i)); try assumption.
 - split; left; apply grid64_increasing; lia.
 - pose proof driver_root as HReq; unfold oriented_balance; destruct heating; simpl in HReq |- *; lra.
Qed.

Theorem driver_accepted_certificate fuel l h i th qh :
 loop_state value il iu (exact_dust A D T C chi a r kap) l h ->
 run_nested_driver fuel heating driver_inner driver_measure (physical_width choice value) proposal l h
  =DriverAccepted i th qh ->
 (il<=i<=iu)%nat /\ driver_inner i=InnerConverged th qh /\
 OuterAccepted64 choice heating qh C khat chi a r (exact_dust A D T C chi a r kap) (value i).
Proof.
 revert l h; induction fuel as [|fuel IH]; intros l h HS HR; simpl in HR; [discriminate|].
 destruct (width_window(physical_width choice value l h)) eqn:HW.
 - destruct (driver_inner l) as [t q|a0 b0| |] eqn:HI; simpl in HR; try discriminate.
   inversion HR; subst i th qh.
   destruct HS as [[Hl [Hlh Hh]] Hroot].
   split; [lia|]. split; [exact HI|].
   apply (OuterBracket64 choice heating q C khat chi a r
    (exact_dust A D T C chi a r kap) (value l) (value l) (value h)).
   + apply grid64_positive.
   + apply grid64_increasing; lia.
   + exact Hroot.
   + split; [reflexivity|left; apply grid64_increasing; lia].
   + apply normal_formatted_endpoints_safe; auto using grid64_positive,grid64_normal,grid64_format.
     left; apply grid64_increasing; lia.
   + apply (Hwidth_nodes l h); assumption.
   + apply Hwidth_finite; assumption.
   + apply width_window_spec; exact HW.
 - set (j:=safeguarded_index l h (proposal l h)) in *.
   assert (Hj:(l<j<h)%nat) by (unfold j; apply driver_trial_inside; assumption).
   destruct (driver_inner j) as [t q|a0 b0| |] eqn:HI; simpl in HR; try discriminate.
   destruct (residual_window(driver_measure j q)) eqn:Hres.
   + inversion HR; subst i th qh.
     destruct HS as [[Hl [Hlh Hh]] Hroot].
     split; [lia|]. split; [exact HI|].
     apply OuterResidual64.
     * apply (Houter_nodes j t q); [lia|exact HI].
     * apply residual_window_spec; exact Hres.
   + destruct (lower_side heating (driver_measure j q)) eqn:Hside.
     * apply (IH j h); [|exact HR]. eapply driver_lower_state; eauto.
     * apply (IH l j); [|exact HR]. eapply driver_upper_state; eauto.
Qed.

Theorem nested_driver_accepted_accuracy fuel i th qh :
 nested_solve fuel=DriverAccepted i th qh ->
 EpsilonMainAccuracy64 eps (value i) (exact_dust A D T C chi a r kap)
 (returned_gas choice A th) (A*exact_gas A D T C chi a r kap)
 (returned_radiation choice r C khat a (value i))
 (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intro HR; unfold nested_solve in HR.
 destruct (driver_accepted_certificate fuel il iu i th qh driver_initial_state HR)
  as [Hi [HI HO]].
 destruct driver_initial_domain as [Hlo [HTdom [HTrdom Hbranch]]].
 pose proof (driver_grid_range i Hi) as Hrange.
 apply (physical_binary64_epsilon_main_accuracy_local eps A D T C chi a r Heps Heps_upper HA HD HT HC Hchi Ha Hr
  kap kp khat (value il) (value iu) ltac:(auto) ltac:(split; assumption)
  choice heating (value i) th qh); try assumption.
 - apply grid64_positive.
 - destruct heating; simpl in *; rewrite Hlower,Hupper in Hrange; simpl in Hrange; lra.
 - apply HPlanck; assumption.
 - apply driver_inner_certificate; assumption.
 - apply (Hgas_nodes i th qh); assumption.
 - apply Hrad_nodes; assumption.
Qed.
(* Stored-input companion: the implicit half and all threshold constants are
   exactly representable under explicit format/range premises. *)
Theorem nested_driver_stored_input_accuracy fuel i th qh :
 List.Forall format64 [A;D;T;C;chi;a;r] -> normal64 (T/2) -> finite64 (T/2) ->
 nested_solve fuel=DriverAccepted i th qh ->
 RN64 choice (T/2)=T/2 /\
 List.Forall (fun c=>RN64 choice c=c) stored_guard_constants64 /\
 EpsilonMainAccuracy64 eps (value i) (exact_dust A D T C chi a r kap)
 (returned_gas choice A th) (A*exact_gas A D T C chi a r kap)
 (returned_radiation choice r C khat a (value i))
 (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intros Hformat Hnormal Hfinite HR; split.
 - apply stored_half_exact64; try assumption.
   apply (proj1 (List.Forall_forall _ _) Hformat T); simpl; auto.
 - split; [apply stored_guard_constants_exact64|].
   apply (nested_driver_accepted_accuracy fuel i th qh); exact HR.
Qed.

(* Sufficient readiness conditions for successful completion. Each is a
   primitive chart/range/fuel or analytic seed condition, not inner accuracy
   or a successful execution assumption. *)
Hypothesis Hinner_ready : forall i, (il<=i<=iu)%nat -> forall c,
 chart_physical c A D T (value i) ->
 inner_range i c=true /\
 chart_seed_bracket c A D T (value i)
   (inner_value i c (inner_lower i c)) (inner_value i c (inner_upper i c)) /\
 (inner_upper i c-inner_lower i c<=inner_fuel)%nat.
Hypothesis Hboundary_ready : forall i, (il<=i<=iu)%nat -> boundary_range i=true.

Lemma driver_inner_converges i : (il<=i<=iu)%nat ->
 exists th qh, driver_inner i=InnerConverged th qh.
Proof.
 intro Hi; unfold driver_inner.
 destruct (solve_nested_inner_converges inner_fuel choice A D T (value i)
  (inner_value i) (inner_proposal i) (inner_lower i) (inner_upper i)
  (inner_range i) (boundary_range i) HA HD HT (grid64_positive i)) as [th [qh [HR HCer]]].
 - intros c HP; destruct (Hinner_ready i Hi c HP) as [Hrange [Hseed Hfuel]].
   split; [assumption|]. split.
   + apply (Hinner_grid i Hi c HP Hrange).
   + split; [|assumption]. apply chart_seed_bracket_contains; assumption || apply grid64_positive.
 - apply Hboundary_ready; assumption.
 - apply Hboundary_nodes; [assumption|apply Hboundary_ready; assumption].
 - exists th,qh; exact HR.
Qed.

Lemma driver_converges_from_state fuel l h :
 loop_state value il iu (exact_dust A D T C chi a r kap) l h -> (h-l<=fuel)%nat ->
 exists i th qh,
 run_nested_driver fuel heating driver_inner driver_measure (physical_width choice value) proposal l h
  =DriverAccepted i th qh.
Proof.
 revert l h; induction fuel as [|fuel IH]; intros l h HS HF.
 - destruct HS as [[Hl [Hlh Hh]] Hroot]; lia.
 - simpl; destruct (width_window(physical_width choice value l h)) eqn:HW.
   + destruct HS as [[Hl [Hlh Hh]] Hroot].
     destruct (driver_inner_converges l ltac:(lia)) as [th [qh HI]].
     rewrite HI; exists l,th,qh; reflexivity.
   + set (j:=safeguarded_index l h (proposal l h)).
     assert (Hj:(l<j<h)%nat) by (unfold j; apply driver_trial_inside; assumption).
     assert (HJ:(il<=j<=iu)%nat) by (destruct HS as [[Hl [Hlh Hh]] Hroot]; lia).
     destruct (driver_inner_converges j HJ) as [th [qh HI]].
     fold j; rewrite HI.
     destruct (residual_window(driver_measure j qh)) eqn:HWres.
     * exists j,th,qh; reflexivity.
     * destruct (lower_side heating (driver_measure j qh)) eqn:Hside.
       -- apply (IH j h); [|lia]. eapply driver_lower_state; eauto.
       -- apply (IH l j); [|lia]. eapply driver_upper_state; eauto.
Qed.

Theorem nested_driver_converges fuel : (iu-il<=fuel)%nat ->
 exists i th qh, nested_solve fuel=DriverAccepted i th qh /\
 EpsilonMainAccuracy64 eps (value i) (exact_dust A D T C chi a r kap)
 (returned_gas choice A th) (A*exact_gas A D T C chi a r kap)
 (returned_radiation choice r C khat a (value i))
 (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intro HF.
 destruct (driver_converges_from_state fuel il iu driver_initial_state HF) as [i [th [qh HR]]].
 exists i,th,qh; split; [exact HR|].
 apply (nested_driver_accepted_accuracy fuel i th qh); exact HR.
Qed.
End VerifiedEpsilonNestedDriver.

(* These are the original operational failure-origin results. Generalizing
   the analytic slope contract cannot erase or reinterpret failures. *)
Definition epsilon_driver_range_failure_origin := nested_driver_range_failure_origin.
Definition epsilon_driver_sign_failure_origin := nested_driver_sign_failure_origin.
Definition epsilon_driver_inner_exhaustion_origin := nested_driver_inner_exhaustion_origin.

Print Assumptions nested_driver_accepted_accuracy.
Print Assumptions nested_driver_stored_input_accuracy.
Print Assumptions nested_driver_converges.
Print Assumptions epsilon_driver_range_failure_origin.
Print Assumptions epsilon_driver_sign_failure_origin.
Print Assumptions epsilon_driver_inner_exhaustion_origin.
