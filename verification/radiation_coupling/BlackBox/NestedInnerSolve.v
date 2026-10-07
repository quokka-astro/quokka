(* A concrete nested-inner execution wrapper. Success is derived from an actual
   safeguarded run and primitive finite binary64 graphs. Failure is a status. *)
From Coq Require Import Reals Psatz Field List ZArith Lia.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import LogCalculus GasMap InnerCharts FloatingPoint Guards
 GuardFloatBridge WidthSafety Brackets SolverLoop InnerLoop InnerEvaluator.
Import ListNotations.
Open Scope R_scope.

Inductive inner_chart := HeatingChart | WeakChart | StrongChart.
Definition chart_rising c := match c with StrongChart=>false | _=>true end.
Definition chart_balance c A D T x := match c with
 | HeatingChart=>Hh T A D (x-T)
 | WeakChart=>Hw T A D (T-x)
 | StrongChart=>Hs T A D x end.
Definition chart_measured choice c A D T x z := match c with
 | HeatingChart=>heating_balance (RN64 choice) A D T x z
 | WeakChart=>weak_balance (RN64 choice) A D T x z
 | StrongChart=>eval_inner_strong (RN64 choice) A D T x z end.
Definition chart_nodes choice c A D T x z := match c with
 | HeatingChart=>heating_nodes (RN64 choice) A D T x z
 | WeakChart=>weak_nodes (RN64 choice) A D T x z
 | StrongChart=>inner_strong_nodes (RN64 choice) A D T x z end.
Definition chart_temperature choice c T z := match c with
 | HeatingChart=>RN64 choice(T+z)
 | WeakChart=>RN64 choice(T-z)
 | StrongChart=>z end.
Definition chart_transfer choice c A T z := match c with
 | HeatingChart | WeakChart=>RN64 choice(A*z)
 | StrongChart=>RN64 choice(A*RN64 choice(T-z)) end.
Definition chart_root c A D T x := match c with
 | HeatingChart=>gas_map A D T x-T
 | WeakChart=>T-gas_map A D T x
 | StrongChart=>gas_map A D T x end.
Definition chart_physical c A D T x := match c with
 | HeatingChart=>T<x
 | WeakChart=>x<T /\ T/2<=gas_map A D T x
 | StrongChart=>x<T /\ gas_map A D T x<=T/2 end.
Definition chart_domain c T z := match c with
 | HeatingChart=>0<z
 | _=>0<z<=T/2 end.
Definition chart_width choice (value:nat->R) a b :=
 RN64 choice(RN64 choice(value b-value a)/value a).

Lemma chart_domain_positive c T z : chart_domain c T z -> 0<z.
Proof. destruct c; simpl; tauto. Qed.
Lemma chart_domain_interval c T lo hi z :
 chart_domain c T lo -> chart_domain c T hi -> lo<=z<=hi -> chart_domain c T z.
Proof. destruct c; simpl; intros; lra. Qed.

Lemma chart_exact_root c A D T x :
 0<A -> 0<D -> 0<T -> 0<x -> chart_physical c A D T x ->
 chart_domain c T (chart_root c A D T x) /\
 chart_balance c A D T x (chart_root c A D T x)=1.
Proof.
 intros HA HD HT Hx Hp; destruct c; simpl in *.
 - pose proof (gas_map_heating_order A D T x HA HD HT Hp).
   split; [lra|apply gas_map_heating_chart; assumption].
 - pose proof (gas_map_cooling_order A D T x HA HD Hx ltac:(lra)).
   split; [lra|apply gas_map_weak_chart; tauto].
 - pose proof (gas_map_cooling_order A D T x HA HD Hx ltac:(lra)).
   split; [lra|apply gas_map_strong_chart; assumption].
Qed.

Lemma chart_evaluator_bound choice c A D T x z :
 0<A -> 0<D -> 0<T -> 0<x -> chart_physical c A D T x -> chart_domain c T z ->
 FiniteNormalNodes choice (chart_nodes choice c A D T x z) ->
 LogBound (8*binary64_lambda) (chart_measured choice c A D T x z) (chart_balance c A D T x z).
Proof.
 intros HA HD HT Hx Hp Hz N; pose proof binary64_lambda_positive as Hl.
 destruct c; simpl in *.
 - exact (proj1 (heating_complete_graph (RN64 choice) binary64_lambda A D T x z ltac:(lra) HA HD Hz (normal_nodes_round choice _ N))).
 - exact (proj1 (weak_complete_graph (RN64 choice) binary64_lambda A D T x z ltac:(lra) HA HD ltac:(lra) (normal_nodes_round choice _ N))).
 - exact (proj1 (strong_complete_graph (RN64 choice) binary64_lambda A D T x z ltac:(lra) HA HD Hx ltac:(lra) (normal_nodes_round choice _ N))).
Qed.

Lemma chart_residual_result choice c A D T x z :
 chart_physical c A D T x -> chart_domain c T z ->
 FiniteNormalNodes choice (chart_nodes choice c A D T x z) ->
 inner_residual_window (chart_measured choice c A D T x z)=true ->
 InnerResult choice A D T x (chart_temperature choice c T z) (chart_transfer choice c A T z).
Proof.
 intros Hp Hz N Haccept; apply inner_residual_window_spec in Haccept.
 destruct c; simpl in *.
 - apply IR_heating_residual; assumption.
 - apply IR_weak_residual; unfold inner_window; tauto.
 - apply IR_strong_residual; unfold inner_window; tauto.
Qed.

Lemma finite_nodes_subset choice l1 l2 :
 (forall y, In y l1 -> In y l2) -> FiniteNormalNodes choice l2 -> FiniteNormalNodes choice l1.
Proof.
 intros Hsub H; unfold FiniteNormalNodes in *.
 rewrite Forall_forall in *; intros; apply H,Hsub; assumption.
Qed.
Lemma chart_bracket_result choice c A D T x lo hi z :
 chart_physical c A D T x -> 0<lo -> lo<=z<=hi -> chart_domain c T hi ->
 lo<=chart_root c A D T x<=hi -> InnerNarrow choice lo hi ->
 FiniteNormalNodes choice (chart_nodes choice c A D T x z) ->
 InnerResult choice A D T x (chart_temperature choice c T z) (chart_transfer choice c A T z).
Proof.
 intros Hp Hl Hz Hhi Hr Hw N; destruct c; simpl in *.
 - apply (IR_heating_bracket choice A D T x lo hi z); try assumption.
   eapply finite_nodes_subset; [|exact N]; simpl; tauto.
 - apply (IR_weak_bracket choice A D T x lo hi z); try tauto.
   eapply finite_nodes_subset; [|exact N]; simpl; tauto.
 - apply (IR_strong_bracket choice A D T x lo hi z); try tauto.
   eapply finite_nodes_subset; [|exact N]; simpl; tauto.
Qed.

Lemma chart_monotone c A D T x lo hi :
 0<A -> 0<D -> 0<T -> 0<x -> chart_physical c A D T x ->
 chart_domain c T lo -> chart_domain c T hi -> lo<=hi ->
 nondecreasing_on (inner_oriented_balance (chart_rising c) (chart_balance c A D T x)) lo hi.
Proof.
 intros HA HD HT Hx Hp Hl Hh Hlh; destruct c; simpl in *.
 - apply inner_heating_chart_monotone; lra.
 - apply inner_weak_chart_monotone; lra.
 - apply inner_strong_chart_monotone; lra.
Qed.

Lemma chart_root_unique c A D T x z :
 0<A -> 0<D -> 0<T -> 0<x -> chart_physical c A D T x -> chart_domain c T z ->
 chart_balance c A D T x z=1 -> z=chart_root c A D T x.
Proof.
 intros HA HD HT Hx Hp Hz Hroot.
 destruct (chart_exact_root c A D T x HA HD HT Hx Hp) as [Hdom HE].
 assert (Hlog:log_error z (chart_root c A D T x)<=0).
 { destruct c; simpl in *.
   - pose proof (heating_coordinate_error T A D (x-T) (gas_map A D T x-T) z HT HA HD ltac:(lra) Hdom Hz HE) as H.
     rewrite Hroot,ln_1,Rabs_R0 in H; lra.
   - pose proof (weak_coordinate_error T A D (T-x) (T-gas_map A D T x) z HT HA HD ltac:(lra) Hdom Hz HE) as H.
     rewrite Hroot,ln_1,Rabs_R0 in H; lra.
   - pose proof (strong_coordinate_error T A D x (gas_map A D T x) z HA HD Hx ltac:(lra) ltac:(lra) HE) as H.
     rewrite Hroot,ln_1,Rabs_R0 in H; lra. }
 unfold log_error in Hlog.
 assert (ln z=ln(chart_root c A D T x)) by (pose proof (Rle_abs (ln z-ln(chart_root c A D T x))); pose proof (Rle_abs (-(ln z-ln(chart_root c A D T x)))); rewrite Rabs_Ropp in H0; lra).
 apply ln_inv; [exact (chart_domain_positive c T z Hz)|exact (chart_domain_positive c T _ Hdom)|assumption].
Qed.

Lemma inner_run_residual_guard fuel rising measured width proposal lo hi a b i :
 inner_scalar_solve fuel rising measured width proposal lo hi=ResidualAccepted a b i ->
 inner_residual_window(measured i)=true.
Proof.
 revert lo hi; induction fuel; intros lo hi H; simpl in H; [discriminate|].
 destruct (inner_width_window(width lo hi)) eqn:Hw; [discriminate|].
 destruct (inner_residual_window(measured(safeguarded_index lo hi (proposal lo hi)))) eqn:Hr.
 - inversion H; subst; exact Hr.
 - destruct (inner_lower_side rising (measured(safeguarded_index lo hi (proposal lo hi)))); eapply IHfuel; eauto.
Qed.
Lemma inner_run_width_guard fuel rising measured width proposal lo hi a b i :
 inner_scalar_solve fuel rising measured width proposal lo hi=BracketAccepted a b i ->
 inner_width_window(width a b)=true /\ i=a.
Proof.
 revert lo hi; induction fuel; intros lo hi H; simpl in H; [discriminate|].
 destruct (inner_width_window(width lo hi)) eqn:Hw.
 - inversion H; subst; auto.
 - destruct (inner_residual_window(measured(safeguarded_index lo hi (proposal lo hi)))); [discriminate|].
   destruct (inner_lower_side rising (measured(safeguarded_index lo hi (proposal lo hi)))); eapply IHfuel; eauto.
Qed.

Lemma chart_continuous c A D T x z :
 0<A -> 0<D -> 0<T -> 0<x -> chart_physical c A D T x -> chart_domain c T z ->
 continuity_pt (inner_oriented_balance (chart_rising c) (chart_balance c A D T x)) z.
Proof.
 intros HA HD HT Hx Hp Hz; destruct c; simpl in *.
 - apply (is_derive_continuity_pt _ _ (Hh' T A D (x-T) z)).
   unfold inner_oriented_balance; simpl.
   replace (Hh' T A D (x-T) z) with (Hh' T A D (x-T) z-0) by ring.
   apply (@is_derive_minus R_AbsRing R_NormedModule (Hh T A D (x-T)) (fun _=>1) z (Hh' T A D (x-T) z) 0); [apply Hh_derivative; lra|exact (@is_derive_const R_AbsRing R_NormedModule 1 z)].
 - apply (is_derive_continuity_pt _ _ (Hw' T A D (T-x) z)).
   unfold inner_oriented_balance; simpl.
   replace (Hw' T A D (T-x) z) with (Hw' T A D (T-x) z-0) by ring.
   apply (@is_derive_minus R_AbsRing R_NormedModule (Hw T A D (T-x)) (fun _=>1) z (Hw' T A D (T-x) z) 0); [apply Hw_derivative; lra|exact (@is_derive_const R_AbsRing R_NormedModule 1 z)].
 - apply (is_derive_continuity_pt _ _ (-Hs' T A D x z)).
   unfold inner_oriented_balance; simpl.
   replace (-Hs' T A D x z) with (0-Hs' T A D x z) by ring.
   apply (@is_derive_minus R_AbsRing R_NormedModule (fun _=>1) (Hs T A D x) z 0 (Hs' T A D x z)); [exact (@is_derive_const R_AbsRing R_NormedModule 1 z)|apply Hs_derivative; lra].
Qed.

Lemma chart_initial_root_checked choice c A D T x value lo hi range_ok :
 0<A -> 0<D -> 0<T -> 0<x -> chart_physical c A D T x ->
 value lo<value hi -> chart_domain c T (value lo) -> chart_domain c T (value hi) ->
 FiniteNormalNodes choice (chart_nodes choice c A D T x (value lo)) ->
 FiniteNormalNodes choice (chart_nodes choice c A D T x (value hi)) ->
 initialize_inner range_ok (chart_rising c)
  (fun i=>chart_measured choice c A D T x (value i)) lo hi=InnerInitialBracket lo hi ->
 root_bracket (value lo) (value hi) (chart_root c A D T x).
Proof.
 intros HA HD HT Hx Hp Hlh Hlo Hhi Nlo Nhi Hinit.
 pose proof (chart_evaluator_bound choice c A D T x (value lo) HA HD HT Hx Hp Hlo Nlo) as [Hml [Hbl Hel]].
 pose proof (chart_evaluator_bound choice c A D T x (value hi) HA HD HT Hx Hp Hhi Nhi) as [Hmh [Hbh Heh]].
 destruct (initialize_inner_bracket_checked range_ok (chart_rising c)
 (fun i=>chart_measured choice c A D T x (value i)) lo hi (value lo) (value hi)
 (chart_balance c A D T x) Hinit Hlh) as [r [Hr Hroot]].
 - intros z Hz; apply chart_continuous; try assumption.
   apply (chart_domain_interval c T (value lo) (value hi) z); assumption.
 - exact Hml.
 - exact Hmh.
 - exact Hbl.
 - exact Hbh.
 - exact Hel.
 - exact Heh.
 - assert (Hrdom:chart_domain c T r).
   { apply (chart_domain_interval c T (value lo) (value hi) r); assumption. }
   pose proof (chart_root_unique c A D T x r HA HD HT Hx Hp Hrdom Hroot); now subst r.
Qed.

Lemma initialize_inner_accepted_guard range_ok rising measured lo hi i :
 initialize_inner range_ok rising measured lo hi=InnerInitialAccepted i ->
 (i=lo \/ i=hi) /\ inner_residual_window(measured i)=true.
Proof.
 unfold initialize_inner; destruct range_ok; [|discriminate].
 destruct (inner_residual_window(measured lo)) eqn:Hl.
 - intro H; inversion H; subst; auto.
 - destruct (inner_residual_window(measured hi)) eqn:Hh.
   + intro H; inversion H; subst; auto.
   + destruct (inner_lower_side rising (measured lo)); [destruct (inner_lower_side rising (measured hi))|]; discriminate.
Qed.
Lemma initialize_inner_bracket_indices range_ok rising measured lo hi a b :
 initialize_inner range_ok rising measured lo hi=InnerInitialBracket a b -> a=lo /\ b=hi.
Proof.
 unfold initialize_inner; destruct range_ok; [|discriminate].
 destruct inner_residual_window; [discriminate|].
 destruct inner_residual_window; [discriminate|].
 destruct inner_lower_side; [destruct inner_lower_side|]; intro H; inversion H; auto.
Qed.

Lemma chart_width_arithmetic choice value a b :
 0<value a -> value a<value b -> SafeDifference64 (value a) (value b) ->
 normal64 (RN64 choice(value b-value a)/value a) ->
 exists es ed, Rabs es<=binary64_u /\ Rabs ed<=binary64_u /\
 chart_width choice value a b=((value b-value a)*(1+es)/value a)*(1+ed).
Proof.
 intros Ha Hab Hsafe Hnd.
 exact (safe_width_relative_factors choice (value a) (value b) Ha Hab Hsafe Hnd).
Qed.

Record InnerGrid (choice:Z->bool) c A D T x (value:nat->R) (lo hi:nat) : Prop := {
 inner_grid_order : (lo<hi)%nat;
 inner_grid_domain : forall i, (lo<=i<=hi)%nat -> chart_domain c T (value i);
 inner_grid_increasing : forall i j, (lo<=i)%nat -> (i<j)%nat -> (j<=hi)%nat -> value i<value j;
 inner_grid_format : forall i, (lo<=i<=hi)%nat -> format64(value i);
 inner_grid_normal : forall i, (lo<=i<=hi)%nat -> normal64(value i);
 inner_grid_finite : forall i, (lo<=i<=hi)%nat -> finite64(value i);
 inner_grid_complete : forall i, (lo<=i)%nat -> (S i<=hi)%nat ->
   forall y, format64 y -> ~(value i<y<value(S i));
 inner_grid_width : forall a b, (lo<=a)%nat -> (a<b)%nat -> (b<=hi)%nat ->
   SafeDifference64 (value a) (value b) /\ normal64(RN64 choice(value b-value a)/value a) /\
   finite64(RN64 choice(value b-value a)) /\ finite64(chart_width choice value a b);
 inner_grid_nodes : forall i, (lo<=i<=hi)%nat ->
   FiniteNormalNodes choice (chart_nodes choice c A D T x (value i))
}.

Inductive inner_execution : Type :=
| InnerConverged (temperature transfer:R)
| InnerExhausted (lower upper:nat)
| InnerRangeFailure
| InnerSignFailure.
Definition recover_inner_run choice c A T value result := match result with
 | ResidualAccepted a b i | BracketAccepted a b i=>
   InnerConverged (chart_temperature choice c T (value i)) (chart_transfer choice c A T (value i))
 | BudgetExhausted a b=>InnerExhausted a b end.
Definition solve_inner_chart fuel choice c A D T x (value:nat->R) proposal lo hi range_ok :=
 match initialize_inner range_ok (chart_rising c)
 (fun i=>chart_measured choice c A D T x (value i)) lo hi with
 | InnerInitialAccepted i=>InnerConverged (chart_temperature choice c T (value i)) (chart_transfer choice c A T (value i))
 | InnerInitialBracket a b=>recover_inner_run choice c A T value
   (inner_scalar_solve fuel (chart_rising c)
    (fun i=>chart_measured choice c A D T x (value i))
    (chart_width choice value) proposal a b)
 | InnerInitialRangeFailure=>InnerRangeFailure
 | InnerInitialSignFailure=>InnerSignFailure end.

Section ChartExecution.
Variables (choice:Z->bool) (c:inner_chart) (A D T x:R) (value:nat->R)
 (proposal:nat->nat->nat) (lo hi:nat).
Hypotheses (HA:0<A) (HD:0<D) (HT:0<T) (Hx:0<x) (Hp:chart_physical c A D T x).
Hypothesis Grid:InnerGrid choice c A D T x value lo hi.

Lemma chart_loop_success fuel th qh :
 root_bracket (value lo) (value hi) (chart_root c A D T x) ->
 recover_inner_run choice c A T value
 (inner_scalar_solve fuel (chart_rising c)
 (fun i=>chart_measured choice c A D T x (value i)) (chart_width choice value) proposal lo hi)
 =InnerConverged th qh -> InnerResult choice A D T x th qh.
Proof.
 intros Hbr Hout.
 destruct Grid as [Hlohi Hdom Hincr Hfmt Hnorm Hfinite Hcomplete Hwidth Hnodes].
 assert (Hpos:forall i, (lo<=i<=hi)%nat ->0<value i).
 { intros i Hi; exact (chart_domain_positive c T _ (Hdom i Hi)). }
 assert (Hwa:forall a b, (lo<=a)%nat -> (a<b)%nat -> (b<=hi)%nat ->
 exists es ed, Rabs es<=binary64_u /\ Rabs ed<=binary64_u /\
 chart_width choice value a b=((value b-value a)*(1+es)/value a)*(1+ed)).
 { intros a b Hal Hab Hbh. destruct (Hwidth a b Hal Hab Hbh) as [Hn [Hd' _]].
   apply chart_width_arithmetic; try assumption; [apply Hpos; lia|apply Hincr; assumption]. }
 assert (Heval:forall i, (lo<=i<=hi)%nat ->
  0<chart_measured choice c A D T x (value i) /\ 0<chart_balance c A D T x (value i) /\
  Rabs(ln(chart_measured choice c A D T x (value i))-ln(chart_balance c A D T x (value i)))<=8*binary64_lambda).
 { intros i Hi. exact (chart_evaluator_bound choice c A D T x (value i) HA HD HT Hx Hp (Hdom i Hi) (Hnodes i Hi)). }
 assert (Hmono:nondecreasing_on (inner_oriented_balance (chart_rising c) (chart_balance c A D T x)) (value lo) (value hi)).
 { apply chart_monotone; try assumption; [apply Hdom; lia|apply Hdom; lia|left;apply Hincr; lia]. }
 pose proof (proj2 (chart_exact_root c A D T x HA HD HT Hx Hp)) as Hroot.
 assert (Hstate:inner_loop_state value lo hi (chart_root c A D T x) lo hi).
 { split; [lia|exact Hbr]. }
 pose proof (inner_scalar_solve_certificate (chart_rising c) value (chart_balance c A D T x)
 (fun i=>chart_measured choice c A D T x (value i)) (chart_width choice value) proposal lo hi
 (chart_root c A D T x) Hpos Hincr Hfmt Hnorm Hcomplete Hwa Heval Hmono Hroot fuel lo hi Hstate) as Hcert.
 destruct (inner_scalar_solve fuel (chart_rising c)
 (fun i=>chart_measured choice c A D T x (value i)) (chart_width choice value) proposal lo hi)
 as [a b i|a b i|a b] eqn:Hrun; simpl in Hcert,Hout.
 - destruct Hcert as [[[Hal [Hab Hbh]] Hbr'] [Hi Hres]].
   inversion Hout; subst th qh. apply chart_residual_result; try assumption.
   + apply Hdom; lia.
   + apply Hnodes; lia.
   + exact (inner_run_residual_guard fuel (chart_rising c) (fun j=>chart_measured choice c A D T x (value j)) (chart_width choice value) proposal lo hi a b i Hrun).
 - destruct Hcert as [[[Hal [Hab Hbh]] Hbr'] [Hi Herr]].
   inversion Hout; subst th qh. subst i.
   pose proof (inner_run_width_guard _ _ _ _ _ _ _ _ _ _ Hrun) as [Hwg _].
   apply inner_width_window_spec in Hwg.
   apply (chart_bracket_result choice c A D T x (value a) (value b) (value a)); try assumption.
   + apply Hpos; lia.
   + split; [lra|left;apply Hincr; lia].
   + apply Hdom; lia.
   + right; destruct (Hwidth a b Hal Hab Hbh) as [Hn [Hnd _]]. repeat split; try assumption.
     apply Hincr; assumption.
   + apply Hnodes; lia.
 - discriminate.
Qed.

Theorem solve_inner_chart_success fuel range_ok th qh :
 solve_inner_chart fuel choice c A D T x value proposal lo hi range_ok=InnerConverged th qh ->
 InnerResult choice A D T x th qh.
Proof.
 intro Hrun; unfold solve_inner_chart in Hrun.
 destruct (initialize_inner range_ok (chart_rising c)
 (fun i=>chart_measured choice c A D T x (value i)) lo hi) as [i|a b| |] eqn:Hinit;
 try discriminate.
 - destruct Grid as [Hlohi Hdom Hincr Hfmt Hnorm Hfinite Hcomplete Hwidth Hnodes].
   destruct (initialize_inner_accepted_guard _ _ _ _ _ _ Hinit) as [Hi Hw].
   inversion Hrun; subst th qh. apply chart_residual_result; try assumption.
   + apply Hdom; destruct Hi; subst; lia.
   + apply Hnodes; destruct Hi; subst; lia.
 - destruct (initialize_inner_bracket_indices _ _ _ _ _ _ _ Hinit) as [Ha Hb]; subst a b.
   apply (chart_loop_success fuel th qh); [|exact Hrun].
   destruct Grid as [Hlohi Hdom Hincr Hfmt Hnorm Hfinite Hcomplete Hwidth Hnodes].
   apply (chart_initial_root_checked choice c A D T x value lo hi range_ok); try assumption.
   + apply Hincr; lia.
   + apply Hdom; lia.
   + apply Hdom; lia.
   + apply Hnodes; lia.
   + apply Hnodes; lia.
Qed.
End ChartExecution.

Theorem solve_inner_chart_range_failure fuel choice c A D T x value proposal lo hi :
 solve_inner_chart fuel choice c A D T x value proposal lo hi false=InnerRangeFailure.
Proof. reflexivity. Qed.

(* Chart selection is executable and uses the same measured 16u guard at the
   boundary. An uncertain boundary is accepted, never used as a sign. *)
Definition solve_nested_inner fuel choice A D T x
 (value:inner_chart->nat->R) (proposal:inner_chart->nat->nat->nat)
 (lo hi:inner_chart->nat) (range_ok:inner_chart->bool) (boundary_range:bool) :=
 let run c:=solve_inner_chart fuel choice c A D T x (value c) (proposal c) (lo c) (hi c) (range_ok c) in
 if Rlt_dec T x then run HeatingChart
 else if Req_EM_T x T then InnerConverged T 0
 else if Rlt_dec (T/2) x then run WeakChart
 else if boundary_range then
   let h:=eval_inner_strong (RN64 choice) A D T x (T/2) in
   if inner_residual_window h then
     InnerConverged (T/2) (RN64 choice(A*RN64 choice(T-T/2)))
   else if Rlt_dec h 1 then run StrongChart else run WeakChart
 else InnerRangeFailure.

Theorem solve_nested_inner_success fuel choice A D T x value proposal lo hi range_ok boundary_range th qh :
 0<A -> 0<D -> 0<T -> 0<x ->
 (forall c, chart_physical c A D T x -> range_ok c=true ->
   InnerGrid choice c A D T x (value c) (lo c) (hi c)) ->
 (boundary_range=true -> FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T x (T/2))) ->
 solve_nested_inner fuel choice A D T x value proposal lo hi range_ok boundary_range=InnerConverged th qh ->
 InnerResult choice A D T x th qh.
Proof.
 intros HA HD HT Hx HG HB Hrun.
 assert (Hchart:forall c, chart_physical c A D T x ->
   solve_inner_chart fuel choice c A D T x (value c) (proposal c) (lo c) (hi c) (range_ok c)=InnerConverged th qh ->
   InnerResult choice A D T x th qh).
 { intros c Hp Hr. destruct (range_ok c) eqn:Hrange.
   - apply (solve_inner_chart_success choice c A D T x (value c) (proposal c) (lo c) (hi c)
     HA HD HT Hx Hp (HG c Hp Hrange) fuel true th qh); exact Hr.
   - rewrite solve_inner_chart_range_failure in Hr; discriminate. }
 unfold solve_nested_inner in Hrun.
 destruct (Rlt_dec T x) as [Hheat|Hheat].
 - apply (Hchart HeatingChart); assumption.
 - destruct (Req_EM_T x T) as [Heq|Hne].
   + inversion Hrun; subst th qh; apply IR_equilibrium; exact Heq.
   + assert (Hcool:x<T) by lra.
     pose proof (gas_map_cooling_order A D T x HA HD Hx Hcool) as Horder.
     destruct (Rlt_dec (T/2) x) as [Hweak|Hweak].
     * apply (Hchart WeakChart); [simpl; lra|exact Hrun].
     * destruct boundary_range eqn:Hbr; [|discriminate].
       specialize (HB eq_refl).
       destruct (inner_residual_window(eval_inner_strong (RN64 choice) A D T x (T/2))) eqn:Hguard.
       -- inversion Hrun; subst th qh; apply IR_boundary_residual; try assumption.
          apply inner_residual_window_spec; exact Hguard.
       -- pose proof (RN64_boundary_selects_chart choice A D T x HA HD Hx Hcool HB) as [Hs Hw].
          pose proof (inner_residual_window_reject _ Hguard) as Hreject.
          pose proof binary64_u_bounds as Hu.
          destruct (Rlt_dec (eval_inner_strong (RN64 choice) A D T x (T/2)) 1) as [Hlow|Hhigh].
          ++ apply (Hchart StrongChart); [simpl;split;[exact Hcool|]|exact Hrun].
             left; apply Hs; destruct Hreject; lra.
          ++ apply (Hchart WeakChart); [simpl;split;[exact Hcool|]|exact Hrun].
             left; apply Hw; destruct Hreject; lra.
Qed.

Corollary nested_inner_accuracy fuel choice A D T x value proposal lo hi range_ok boundary_range th qh :
 0<A -> 0<D -> 0<T -> 0<x ->
 (forall c, chart_physical c A D T x -> range_ok c=true ->
   InnerGrid choice c A D T x (value c) (lo c) (hi c)) ->
 (boundary_range=true -> FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T x (T/2))) ->
 solve_nested_inner fuel choice A D T x value proposal lo hi range_ok boundary_range=InnerConverged th qh ->
 LogBound (52*binary64_lambda) th (gas_map A D T x) /\
 ((x=T /\ qh=0) \/ LogBound (52*binary64_lambda) qh (A*Rabs(gas_map A D T x-T))).
Proof.
 intros HA HD HT Hx HG HB Hrun.
 apply (inner_result_accuracy choice A D T x th qh HA HD HT Hx).
 eapply solve_nested_inner_success; eauto.
Qed.

Print Assumptions solve_nested_inner_success.
Print Assumptions nested_inner_accuracy.

Definition inner_not_exhausted r := match r with InnerExhausted _ _=>False | _=>True end.
Theorem solve_inner_chart_sufficient_fuel fuel choice c A D T x value proposal lo hi range_ok :
 0<A -> 0<D -> 0<T -> 0<x -> chart_physical c A D T x ->
 InnerGrid choice c A D T x value lo hi -> (hi-lo<=fuel)%nat ->
 inner_not_exhausted (solve_inner_chart fuel choice c A D T x value proposal lo hi range_ok).
Proof.
 intros HA HD HT Hx Hp Grid Hfuel.
 unfold solve_inner_chart.
 destruct (initialize_inner range_ok (chart_rising c)
 (fun i=>chart_measured choice c A D T x (value i)) lo hi) as [i|a b| |] eqn:Hinit;
 try exact I.
 destruct (initialize_inner_bracket_indices _ _ _ _ _ _ _ Hinit) as [Ha Hb]; subst a b.
 destruct Grid as [Hlohi Hdom Hincr Hfmt Hnorm Hfinite Hcomplete Hwidth Hnodes].
 assert (Hbr:root_bracket (value lo) (value hi) (chart_root c A D T x)).
 { apply (chart_initial_root_checked choice c A D T x value lo hi range_ok); try assumption.
   - apply Hincr; lia.
   - apply Hdom; lia.
   - apply Hdom; lia.
   - apply Hnodes; lia.
   - apply Hnodes; lia. }
 assert (Hpos:forall i, (lo<=i<=hi)%nat ->0<value i).
 { intros i Hi; exact (chart_domain_positive c T _ (Hdom i Hi)). }
 assert (Hwa:forall a b, (lo<=a)%nat -> (a<b)%nat -> (b<=hi)%nat ->
 exists es ed, Rabs es<=binary64_u /\ Rabs ed<=binary64_u /\
 chart_width choice value a b=((value b-value a)*(1+es)/value a)*(1+ed)).
 { intros a b Hal Hab Hbh. destruct (Hwidth a b Hal Hab Hbh) as [Hn [Hd' _]].
   apply chart_width_arithmetic; try assumption; [apply Hpos; lia|apply Hincr; assumption]. }
 assert (Heval:forall i, (lo<=i<=hi)%nat ->
  0<chart_measured choice c A D T x (value i) /\ 0<chart_balance c A D T x (value i) /\
  Rabs(ln(chart_measured choice c A D T x (value i))-ln(chart_balance c A D T x (value i)))<=8*binary64_lambda).
 { intros i Hi. exact (chart_evaluator_bound choice c A D T x (value i) HA HD HT Hx Hp (Hdom i Hi) (Hnodes i Hi)). }
 assert (Hmono:nondecreasing_on (inner_oriented_balance (chart_rising c) (chart_balance c A D T x)) (value lo) (value hi)).
 { apply chart_monotone; try assumption; [apply Hdom; lia|apply Hdom; lia|left;apply Hincr; lia]. }
 pose proof (proj2 (chart_exact_root c A D T x HA HD HT Hx Hp)) as Hroot.
 assert (Hstate:inner_loop_state value lo hi (chart_root c A D T x) lo hi).
 { split; [lia|exact Hbr]. }
 pose proof (inner_scalar_solve_sufficient_fuel (chart_rising c) value (chart_balance c A D T x)
 (fun i=>chart_measured choice c A D T x (value i)) (chart_width choice value) proposal lo hi
 (chart_root c A D T x) Hpos Hincr Hfmt Hnorm Hcomplete Hwa Heval Hmono Hroot fuel lo hi Hstate Hfuel) as Hsuccess.
 destruct (inner_scalar_solve fuel (chart_rising c)
 (fun i=>chart_measured choice c A D T x (value i)) (chart_width choice value) proposal lo hi);
 simpl in *; assumption || exact I.
Qed.

From BlackBox Require Import NormalGrid.
Lemma grid64_inner_contract choice c A D T x lo hi :
 (lo<hi)%nat -> chart_domain c T (grid64 lo) -> chart_domain c T (grid64 hi) ->
 finite64 (grid64 hi) ->
 (forall a b, (lo<=a)%nat -> (a<b)%nat -> (b<=hi)%nat ->
   SafeDifference64 (grid64 a) (grid64 b) /\ normal64(RN64 choice(grid64 b-grid64 a)/grid64 a) /\
   finite64(RN64 choice(grid64 b-grid64 a)) /\ finite64(chart_width choice grid64 a b)) ->
 (forall i, (lo<=i<=hi)%nat -> FiniteNormalNodes choice (chart_nodes choice c A D T x (grid64 i))) ->
 InnerGrid choice c A D T x grid64 lo hi.
Proof.
 intros Hlohi Hlo Hhi Hfinite Hwidth Hnodes.
 constructor; try assumption.
 - intros i Hi; apply (chart_domain_interval c T (grid64 lo) (grid64 hi) (grid64 i)); try assumption.
   split; apply grid64_le; lia.
 - intros; apply grid64_increasing; assumption.
 - intros; apply grid64_format.
 - intros; apply grid64_normal.
 - intros i Hi; unfold finite64 in *.
   rewrite Rabs_pos_eq in Hfinite by (left; apply grid64_positive).
   rewrite Rabs_pos_eq by (left; apply grid64_positive).
   pose proof (grid64_le i hi ltac:(lia)); lra.
 - intros; apply grid64_adjacent_complete; assumption.
Qed.

Lemma normal_grid_inner_contract choice c A D T x base lo hi :
 0<base -> normal64 base -> format64 base -> (lo<hi)%nat ->
 chart_domain c T (normal_grid_from base lo) -> chart_domain c T (normal_grid_from base hi) ->
 finite64 (normal_grid_from base hi) ->
 (forall a b, (lo<=a)%nat -> (a<b)%nat -> (b<=hi)%nat ->
   SafeDifference64 (normal_grid_from base a) (normal_grid_from base b) /\
   normal64(RN64 choice(normal_grid_from base b-normal_grid_from base a)/normal_grid_from base a) /\
   finite64(RN64 choice(normal_grid_from base b-normal_grid_from base a)) /\
   finite64(chart_width choice (normal_grid_from base) a b)) ->
 (forall i, (lo<=i<=hi)%nat -> FiniteNormalNodes choice (chart_nodes choice c A D T x (normal_grid_from base i))) ->
 InnerGrid choice c A D T x (normal_grid_from base) lo hi.
Proof.
 intros Hb Hn Hf Hlohi Hlo Hhi Hfinite Hwidth Hnodes.
 constructor; try assumption.
 - intros i Hi; apply (chart_domain_interval c T (normal_grid_from base lo) (normal_grid_from base hi) (normal_grid_from base i)); try assumption.
   split; apply normal_grid_le; assumption || lia.
 - intros; apply normal_grid_increasing; assumption.
 - intros; apply normal_grid_format; assumption.
 - intros; apply normal_grid_normal; assumption.
 - intros i Hi; unfold finite64 in *.
   rewrite Rabs_pos_eq in Hfinite by (left; apply normal_grid_positive; assumption).
   rewrite Rabs_pos_eq by (left; apply normal_grid_positive; assumption).
   pose proof (normal_grid_le base i hi Hb ltac:(lia)); lra.
 - intros; apply normal_grid_adjacent_complete; assumption.
Qed.

Print Assumptions solve_inner_chart_sufficient_fuel.
Print Assumptions grid64_inner_contract.

Lemma initialized_root_bracket_succeeds rising measured balance value lo hi root :
 root_bracket (value lo) (value hi) root ->
 nondecreasing_on (inner_oriented_balance rising balance) (value lo) (value hi) ->
 balance root=1 ->
 0<measured lo -> 0<measured hi -> 0<balance(value lo) -> 0<balance(value hi) ->
 Rabs(ln(measured lo)-ln(balance(value lo)))<=8*binary64_lambda ->
 Rabs(ln(measured hi)-ln(balance(value hi)))<=8*binary64_lambda ->
 (exists i, initialize_inner true rising measured lo hi=InnerInitialAccepted i) \/
 initialize_inner true rising measured lo hi=InnerInitialBracket lo hi.
Proof.
 intros Hbr Hmono Hroot Hml Hmh Hbl Hbh Hel Heh.
 assert (Hzero:inner_oriented_balance rising balance root=0).
 { unfold inner_oriented_balance; destruct rising; rewrite Hroot; ring. }
 assert (Hlower:inner_oriented_balance rising balance (value lo)<=0).
 { rewrite <- Hzero. apply Hmono; unfold root_bracket in Hbr; lra. }
 assert (Hupper:0<=inner_oriented_balance rising balance (value hi)).
 { rewrite <- Hzero. apply Hmono; unfold root_bracket in Hbr; lra. }
 unfold initialize_inner.
 destruct (inner_residual_window(measured lo)) eqn:Hl; [left;exists lo;reflexivity|].
 destruct (inner_residual_window(measured hi)) eqn:Hh; [left;exists hi;reflexivity|].
 assert (Hslo:inner_lower_side rising (measured lo)=true).
 { destruct (inner_lower_side rising (measured lo)) eqn:Hs; [reflexivity|].
   pose proof (inner_guarded_upper_side rising (measured lo) (balance(value lo)) Hml Hbl Hel Hl Hs).
   unfold inner_oriented_balance in Hlower; lra. }
 assert (Hshi:inner_lower_side rising (measured hi)=false).
 { destruct (inner_lower_side rising (measured hi)) eqn:Hs; [|reflexivity].
   pose proof (inner_guarded_lower_side rising (measured hi) (balance(value hi)) Hmh Hbh Heh Hh Hs).
   unfold inner_oriented_balance in Hupper; lra. }
 rewrite Hslo,Hshi; auto.
Qed.

Theorem solve_inner_chart_converges fuel choice c A D T x value proposal lo hi :
 0<A -> 0<D -> 0<T -> 0<x -> chart_physical c A D T x ->
 InnerGrid choice c A D T x value lo hi ->
 root_bracket (value lo) (value hi) (chart_root c A D T x) -> (hi-lo<=fuel)%nat ->
 exists th qh,
 solve_inner_chart fuel choice c A D T x value proposal lo hi true=InnerConverged th qh /\
 InnerResult choice A D T x th qh.
Proof.
 intros HA HD HT Hx Hp Grid Hbr Hfuel.
 pose proof (solve_inner_chart_sufficient_fuel fuel choice c A D T x value proposal lo hi true
 HA HD HT Hx Hp Grid Hfuel) as Hnotex.
 assert (Hinit:(exists i, initialize_inner true (chart_rising c)
  (fun i=>chart_measured choice c A D T x (value i)) lo hi=InnerInitialAccepted i) \/
  initialize_inner true (chart_rising c)
  (fun i=>chart_measured choice c A D T x (value i)) lo hi=InnerInitialBracket lo hi).
 { destruct Grid as [Hlohi Hdom Hincr Hfmt Hnorm Hfinite Hcomplete Hwidth Hnodes].
   pose proof (chart_evaluator_bound choice c A D T x (value lo) HA HD HT Hx Hp (Hdom lo ltac:(lia)) (Hnodes lo ltac:(lia))) as [Hml [Hbl Hel]].
   pose proof (chart_evaluator_bound choice c A D T x (value hi) HA HD HT Hx Hp (Hdom hi ltac:(lia)) (Hnodes hi ltac:(lia))) as [Hmh [Hbh Heh]].
   apply (initialized_root_bracket_succeeds (chart_rising c)
    (fun i=>chart_measured choice c A D T x (value i)) (chart_balance c A D T x) value lo hi (chart_root c A D T x)); try assumption.
   - apply chart_monotone; try assumption; [apply Hdom;lia|apply Hdom;lia|left;apply Hincr;lia].
   - exact (proj2 (chart_exact_root c A D T x HA HD HT Hx Hp)). }
 assert (Hallowed:match solve_inner_chart fuel choice c A D T x value proposal lo hi true with
  InnerRangeFailure|InnerSignFailure=>False | _=>True end).
 { unfold solve_inner_chart; destruct Hinit as [[i Hinit]|Hinit]; rewrite Hinit; [exact I|].
   destruct inner_scalar_solve; exact I. }
 destruct (solve_inner_chart fuel choice c A D T x value proposal lo hi true) as [th qh|a b| |] eqn:Hrun;
 simpl in Hnotex,Hallowed; try contradiction.
 exists th,qh; split; [reflexivity|].
 apply (solve_inner_chart_success choice c A D T x value proposal lo hi HA HD HT Hx Hp Grid fuel true th qh); exact Hrun.
Qed.

Print Assumptions solve_inner_chart_converges.

Theorem solve_nested_inner_converges fuel choice A D T x value proposal lo hi range_ok boundary_range :
 0<A -> 0<D -> 0<T -> 0<x ->
 (forall c, chart_physical c A D T x ->
   range_ok c=true /\ InnerGrid choice c A D T x (value c) (lo c) (hi c) /\
   root_bracket (value c (lo c)) (value c (hi c)) (chart_root c A D T x) /\
   (hi c-lo c<=fuel)%nat) ->
 boundary_range=true ->
 FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T x (T/2)) ->
 exists th qh,
 solve_nested_inner fuel choice A D T x value proposal lo hi range_ok boundary_range=InnerConverged th qh /\
 InnerResult choice A D T x th qh.
Proof.
 intros HA HD HT Hx HG Hbr HB.
 assert (Hchart:forall c, chart_physical c A D T x ->
 exists th qh,
 solve_inner_chart fuel choice c A D T x (value c) (proposal c) (lo c) (hi c) (range_ok c)=InnerConverged th qh /\
 InnerResult choice A D T x th qh).
 { intros c Hp. destruct (HG c Hp) as [Hrange [Grid [Hroot Hfuel]]].
   rewrite Hrange. apply solve_inner_chart_converges; assumption. }
 unfold solve_nested_inner.
 destruct (Rlt_dec T x) as [Hheat|Hheat].
 - apply (Hchart HeatingChart); exact Hheat.
 - destruct (Req_EM_T x T) as [Heq|Hne].
   + exists T,0; split; [reflexivity|apply IR_equilibrium; assumption].
   + assert (Hcool:x<T) by lra.
     pose proof (gas_map_cooling_order A D T x HA HD Hx Hcool) as Horder.
     destruct (Rlt_dec (T/2) x) as [Hweak|Hweak].
     * apply (Hchart WeakChart); simpl; lra.
     * rewrite Hbr.
       destruct (inner_residual_window(eval_inner_strong (RN64 choice) A D T x (T/2))) eqn:Hguard.
       -- exists (T/2),(RN64 choice(A*RN64 choice(T-T/2))); split; [reflexivity|].
          apply IR_boundary_residual; try assumption.
          apply inner_residual_window_spec; exact Hguard.
       -- pose proof (RN64_boundary_selects_chart choice A D T x HA HD Hx Hcool HB) as [Hs Hw].
          pose proof (inner_residual_window_reject _ Hguard) as Hreject.
          pose proof binary64_u_bounds as Hu.
          destruct (Rlt_dec (eval_inner_strong (RN64 choice) A D T x (T/2)) 1) as [Hlow|Hhigh].
          ++ apply (Hchart StrongChart); simpl;split;[exact Hcool|].
             left; apply Hs; destruct Hreject; lra.
          ++ apply (Hchart WeakChart); simpl;split;[exact Hcool|].
             left; apply Hw; destruct Hreject; lra.
Qed.

Print Assumptions solve_nested_inner_converges.
