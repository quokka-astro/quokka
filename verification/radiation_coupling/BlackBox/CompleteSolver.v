(* Accepted executions of the concrete outer loop inherit the component
   accuracy theorem. The output is reevaluated at the returned grid point. *)
From Coq Require Import Reals Psatz Field Lia List ZArith.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra GasMap GasBounds LogCalculus OuterDerivatives OuterBounds
 FloatingPoint Guards GuardFloatBridge WidthSafety InnerEvaluator Brackets SolverLoop
 PhysicalRoot UniqueRoot Initialization PhysicalLoop Binary64Accuracy EndToEnd
 NormalGrid NestedInnerSolve.
Import ListNotations.
Open Scope R_scope.

Lemma outer_run_residual_guard fuel rising measured width proposal lo hi a b i :
 scalar_solve fuel rising measured width proposal lo hi=ResidualAccepted a b i ->
 residual_window(measured i)=true.
Proof.
 revert lo hi; induction fuel; intros lo hi H; simpl in H; [discriminate|].
 destruct (width_window(width lo hi)) eqn:Hw; [discriminate|].
 destruct (residual_window(measured(safeguarded_index lo hi (proposal lo hi)))) eqn:Hr.
 - inversion H; subst; exact Hr.
 - destruct (lower_side rising (measured(safeguarded_index lo hi (proposal lo hi)))); eapply IHfuel; eauto.
Qed.
Lemma outer_run_width_guard fuel rising measured width proposal lo hi a b i :
 scalar_solve fuel rising measured width proposal lo hi=BracketAccepted a b i ->
 width_window(width a b)=true /\ i=a.
Proof.
 revert lo hi; induction fuel; intros lo hi H; simpl in H; [discriminate|].
 destruct (width_window(width lo hi)) eqn:Hw.
 - inversion H; subst; auto.
 - destruct (residual_window(measured(safeguarded_index lo hi (proposal lo hi)))); [discriminate|].
   destruct (lower_side rising (measured(safeguarded_index lo hi (proposal lo hi)))); eapply IHfuel; eauto.
Qed.

Section CompletePhysicalSolver.
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

Hypothesis Hwidth_finite : forall i j, (il<=i)%nat -> (i<j)%nat -> (j<=iu)%nat ->
 finite64 (RN64 choice (value j-value i)) /\
 FiniteNormalNodes choice [RN64 choice (value j-value i)/value i].
Hypothesis Hgas_output : forall i, (il<=i<=iu)%nat -> FiniteNormalNodes choice [A*that i].
Hypothesis Hrad_output : forall i, (il<=i<=iu)%nat ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat(value i)) a (value i)).

Let run fuel := scalar_solve fuel heating
 (physical_measured choice heating A D T C chi a r khat value qhat)
 (physical_width choice value) proposal il iu.

Lemma complete_loop_certificate fuel :
 result_certificate value (physical_balance heating A D T C chi a r kap) il iu
 (exact_dust A D T C chi a r kap) (run fuel).
Proof.
 unfold run; apply (physical_scalar_solve_certificate A D T C chi a r kap khat kp
 choice heating value that qhat proposal il iu); assumption.
Qed.

Lemma complete_initial_domain : 0<value il /\
 value il<=T<=value iu /\
 value il<=radiation_temperature a r<=value iu /\
 (if heating then T<radiation_temperature a r else radiation_temperature a r<T).
Proof.
 pose proof (initial_ratio_finite64_budget choice a T r Ha HT Hr Hinitial_nodes) as Hratio.
 pose proof (outward_endpoint_finite64_encloses choice heating a r Ha Hr Houtward_nodes) as Hend.
 rewrite Hlower,Hupper; destruct heating; simpl in *.
 - pose proof (initial_heating_order a T r _ Ha HT Hr Hratio Hbranch_guard); repeat split; lra.
 - pose proof (initial_cooling_order a T r _ Ha HT Hr Hratio Hbranch_guard); repeat split; lra.
Qed.
Lemma complete_grid_range i : (il<=i<=iu)%nat -> value il<=value i<=value iu.
Proof.
 intro Hi; split.
 - destruct (Nat.eq_dec il i); [subst; reflexivity|left; apply Gincreasing; lia].
 - destruct (Nat.eq_dec i iu); [subst; reflexivity|left; apply Gincreasing; lia].
Qed.
Lemma complete_grid_positive i : (il<=i<=iu)%nat -> 0<value i.
Proof. intro Hi; pose proof (complete_grid_range i Hi); pose proof (proj1 complete_initial_domain); lra. Qed.

Theorem loop_return_outer_accepted fuel l h i :
 (run fuel=ResidualAccepted l h i \/ run fuel=BracketAccepted l h i) ->
 (il<=i<=iu)%nat /\
 OuterAccepted64 choice heating (qhat i) C khat chi a r
 (exact_dust A D T C chi a r kap) (value i).
Proof.
 intro Hrun; pose proof (complete_loop_certificate fuel) as Hcert.
 destruct Hrun as [Hrun|Hrun]; rewrite Hrun in Hcert; simpl in Hcert.
 - destruct Hcert as [[[Hl [Hlh Hh]] Hroot] [Hi Hres]].
   assert (Hir:(il<=i<=iu)%nat) by lia; split; [assumption|].
   apply OuterResidual64; [apply Houter_nodes; assumption|].
   apply residual_window_spec.
   exact (outer_run_residual_guard fuel heating
    (physical_measured choice heating A D T C chi a r khat value qhat)
    (physical_width choice value) proposal il iu l h i Hrun).
 - destruct Hcert as [[[Hl [Hlh Hh]] Hroot] [Hi Hres]].
   subst i; assert (Hlr:(il<=l<=iu)%nat) by lia; split; [assumption|].
   apply (OuterBracket64 choice heating (qhat l) C khat chi a r
    (exact_dust A D T C chi a r kap) (value l) (value l) (value h)).
   + apply complete_grid_positive; assumption.
   + apply Gincreasing; assumption.
   + exact Hroot.
   + split; [reflexivity|left; apply Gincreasing; assumption].
   + apply normal_formatted_endpoints_safe.
     * apply complete_grid_positive; assumption.
     * apply Gnormal; lia.
     * apply Gformat; lia.
     * apply Gformat; lia.
     * left; apply Gincreasing; assumption.
   + apply (Hwidth_nodes l h); assumption.
   + apply Hwidth_finite; assumption.
   + apply width_window_spec.
     exact (proj1 (outer_run_width_guard fuel heating
      (physical_measured choice heating A D T C chi a r khat value qhat)
      (physical_width choice value) proposal il iu l h l Hrun)).
Qed.

Definition returned_point_accuracy i :=
 MainAccuracy64 (value i) (exact_dust A D T C chi a r kap)
 (returned_gas choice A (that i)) (A*exact_gas A D T C chi a r kap)
 (returned_radiation choice r C khat a (value i))
 (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).

Theorem complete_solver_accepted_accuracy fuel l h i :
 (run fuel=ResidualAccepted l h i \/ run fuel=BracketAccepted l h i) ->
 returned_point_accuracy i.
Proof.
 intro Hrun.
 destruct (loop_return_outer_accepted fuel l h i Hrun) as [Hi HO].
 destruct complete_initial_domain as [Hlo [HTdom [HTrdom Hbranch]]].
 pose proof (complete_grid_range i Hi) as Hrange.
 unfold returned_point_accuracy.
 apply (physical_binary64_main_accuracy_local A D T C chi a r HA HD HT HC Hchi Ha Hr
  kap kp khat (value il) (value iu) ltac:(auto) ltac:(split; assumption)
  choice heating (value i) (that i) (qhat i)); try assumption.
 - apply complete_grid_positive; assumption.
 - destruct heating; simpl in *; rewrite Hlower,Hupper in Hrange; simpl in Hrange; lra.
 - apply HPlanck; assumption.
 - apply Hinner; assumption.
 - apply Hgas_output; assumption.
 - apply Hrad_output; assumption.
Qed.

Theorem complete_solver_sufficient_fuel_accuracy fuel : (iu-il<=fuel)%nat ->
 exists l h i, (run fuel=ResidualAccepted l h i \/ run fuel=BracketAccepted l h i) /\
 returned_point_accuracy i.
Proof.
 intro Hfuel.
 assert (Hsuccess:successful (run fuel)).
 { unfold run; apply (physical_scalar_solve_sufficient_fuel A D T C chi a r kap khat kp
   choice heating value that qhat proposal il iu); assumption. }
 destruct (run fuel) as [l h i|l h i|l h] eqn:Hrun; [| |contradiction].
 - exists l,h,i; split; [auto|apply (complete_solver_accepted_accuracy fuel l h i); auto].
 - exists l,h,i; split; [auto|apply (complete_solver_accepted_accuracy fuel l h i); auto].
Qed.
End CompletePhysicalSolver.

Print Assumptions complete_solver_accepted_accuracy.
Print Assumptions complete_solver_sufficient_fuel_accuracy.


(* A recorded concrete inner invocation, retaining all primitive chart/range
   conditions and the actual successful execution equation. *)
Inductive ExecutedInner (choice:Z->bool) (A D T x th qh:R) : Prop :=
| executed_inner_call : forall fuel values proposals lower upper range_ok boundary_range,
 (forall c, chart_physical c A D T x -> range_ok c=true ->
   InnerGrid choice c A D T x (values c) (lower c) (upper c)) ->
 (boundary_range=true -> FiniteNormalNodes choice
   (inner_strong_nodes (RN64 choice) A D T x (T/2))) ->
 solve_nested_inner fuel choice A D T x values proposals lower upper range_ok boundary_range
   =InnerConverged th qh -> ExecutedInner choice A D T x th qh.

Lemma executed_inner_certificate choice A D T x th qh :
 0<A -> 0<D -> 0<T -> 0<x -> ExecutedInner choice A D T x th qh ->
 InnerResult choice A D T x th qh.
Proof.
 intros HA HD HT Hx Hrun; destruct Hrun.
 eapply solve_nested_inner_success; eauto.
Qed.

Section ConcreteSuccessorGrid.
Variables A D T C chi a r : R.
Variables kap khat kp : R->R.
Variable choice : Z->bool.
Variable heating : bool.
Variables that qhat : nat->R.
Let value := grid64.
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
Hypothesis Hwidth_nodes : forall i j, (il<=i)%nat -> (i<j)%nat -> (j<=iu)%nat ->
 normal64 (RN64 choice (value j-value i)/value i).
Hypothesis Hnested : forall i, (il<=i<=iu)%nat ->
 ExecutedInner choice A D T (value i) (that i) (qhat i).
Hypothesis Houter_nodes : forall i, (il<=i<=iu)%nat ->
 FiniteNormalNodes choice
 (outer_nodes (RN64 choice) heating (qhat i) C (khat (value i)) chi a (value i) r).

Hypothesis Hwidth_finite : forall i j, (il<=i)%nat -> (i<j)%nat -> (j<=iu)%nat ->
 finite64 (RN64 choice (value j-value i)) /\
 FiniteNormalNodes choice [RN64 choice (value j-value i)/value i].
Hypothesis Hgas_output : forall i, (il<=i<=iu)%nat -> FiniteNormalNodes choice [A*that i].
Hypothesis Hrad_output : forall i, (il<=i<=iu)%nat ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat(value i)) a (value i)).


Let run fuel := scalar_solve fuel heating
 (physical_measured choice heating A D T C chi a r khat value qhat)
 (physical_width choice value) proposal il iu.

Lemma nested_grid_inner i : (il<=i<=iu)%nat -> InnerResult choice A D T (value i) (that i) (qhat i).
Proof.
 intro Hi; apply executed_inner_certificate; try assumption.
 - apply grid64_positive.
 - apply Hnested; assumption.
Qed.

Theorem successor_grid_nested_accepted_accuracy fuel l h i :
 (run fuel=ResidualAccepted l h i \/ run fuel=BracketAccepted l h i) ->
 MainAccuracy64 (value i) (exact_dust A D T C chi a r kap)
 (returned_gas choice A (that i)) (A*exact_gas A D T C chi a r kap)
 (returned_radiation choice r C khat a (value i))
 (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intro HR.
 eapply (complete_solver_accepted_accuracy A D T C chi a r kap khat kp choice heating
 value that qhat proposal il iu); try assumption.
 - intros; apply grid64_increasing; assumption.
 - intros; apply grid64_format.
 - intros; apply grid64_normal.
 - intros; apply grid64_adjacent_complete; assumption.
 - exact nested_grid_inner.
 - exact HR.
Qed.

Theorem successor_grid_nested_sufficient_fuel_accuracy fuel : (iu-il<=fuel)%nat ->
 exists l h i, (run fuel=ResidualAccepted l h i \/ run fuel=BracketAccepted l h i) /\
 MainAccuracy64 (value i) (exact_dust A D T C chi a r kap)
 (returned_gas choice A (that i)) (A*exact_gas A D T C chi a r kap)
 (returned_radiation choice r C khat a (value i))
 (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap)).
Proof.
 intro Hfuel.
 eapply (complete_solver_sufficient_fuel_accuracy A D T C chi a r kap khat kp choice heating
 value that qhat proposal il iu); try assumption.
 - intros; apply grid64_increasing; assumption.
 - intros; apply grid64_format.
 - intros; apply grid64_normal.
 - intros; apply grid64_adjacent_complete; assumption.
 - exact nested_grid_inner.
Qed.
End ConcreteSuccessorGrid.

Print Assumptions successor_grid_nested_accepted_accuracy.
Print Assumptions successor_grid_nested_sufficient_fuel_accuracy.
