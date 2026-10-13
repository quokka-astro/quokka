(* Binary64 inner acceptance composed from primitive round-to-nearest nodes,
   measured guards, the real collision charts, and stable output recovery. *)
From Coq Require Import Reals Psatz Field List ZArith.
From BlackBox Require Import LogCalculus GasMap InnerCharts FloatingPoint Guards GuardFloatBridge WidthSafety.
Import ListNotations.
Open Scope R_scope.

Definition heating_nodes rnd A D T x z :=
 (T+z)::(x-T)::inner_increment_nodes rnd A D z (rnd(T+z)) (rnd(x-T)).
Definition weak_nodes rnd A D T x z :=
 (T-z)::(T-x)::inner_increment_nodes rnd A D z (rnd(T-z)) (rnd(T-x)).
Definition heating_balance rnd A D T x z :=
 eval_inner_increment rnd A D z (rnd(T+z)) (rnd(x-T)).
Definition weak_balance rnd A D T x z :=
 eval_inner_increment rnd A D z (rnd(T-z)) (rnd(T-x)).
Definition inner_window h := 1-16*binary64_u<=h<=1+16*binary64_u.

Lemma heating_complete_graph rnd l A D T x z :
 0<=l -> 0<A -> 0<D -> 0<z -> RoundNodes rnd l (heating_nodes rnd A D T x z) ->
 LogBound (8*l) (heating_balance rnd A D T x z) (Hh T A D (x-T) z) /\
 LogBound l (rnd(T+z)) (T+z) /\ LogBound l (rnd(A*z)) (A*z).
Proof.
 intros Hl HA HD Hz N.
 assert (Ht:LogBound l (rnd(T+z)) (T+z)) by (apply N; simpl; auto).
 assert (Hd:LogBound l (rnd(x-T)) (x-T)) by (apply N; simpl; auto).
 assert (Hq:LogBound l (rnd(A*z)) (A*z)) by (apply N; simpl; auto).
 assert (Ni:RoundNodes rnd l (inner_increment_nodes rnd A D z (rnd(T+z)) (rnd(x-T)))).
 { intros y Hy. apply N. simpl. right; right; exact Hy. }
 pose proof (heating_inner_graph_budget rnd l A D T x z Hl HA HD Hz Ht Hd Ni) as H.
 split; [exact H|auto].
Qed.

Lemma weak_complete_graph rnd l A D T x z :
 0<=l -> 0<A -> 0<D -> 0<z -> RoundNodes rnd l (weak_nodes rnd A D T x z) ->
 LogBound (8*l) (weak_balance rnd A D T x z) (Hw T A D (T-x) z) /\
 LogBound l (rnd(T-z)) (T-z) /\ LogBound l (rnd(A*z)) (A*z).
Proof.
 intros Hl HA HD Hz N.
 assert (Ht:LogBound l (rnd(T-z)) (T-z)) by (apply N; simpl; auto).
 assert (Hd:LogBound l (rnd(T-x)) (T-x)) by (apply N; simpl; auto).
 assert (Hq:LogBound l (rnd(A*z)) (A*z)) by (apply N; simpl; auto).
 assert (Ni:RoundNodes rnd l (inner_increment_nodes rnd A D z (rnd(T-z)) (rnd(T-x)))).
 { intros y Hy. apply N. simpl. right; right; exact Hy. }
 pose proof (weak_cooling_inner_graph_budget rnd l A D T x z Hl HA HD Hz Ht Hd Ni) as H.
 split; [exact H|auto].
Qed.

Lemma strong_complete_graph rnd l A D T x z :
 0<=l -> 0<A -> 0<D -> 0<x -> 0<z<T ->
 RoundNodes rnd l (inner_strong_nodes rnd A D T x z) ->
 LogBound (8*l) (eval_inner_strong rnd A D T x z) (Hs T A D x z) /\
 LogBound (2*l) (rnd(A*rnd(T-z))) (A*(T-z)).
Proof.
 intros Hl HA HD Hx Hz N.
 pose proof (inner_strong_graph_budget rnd l A D T x z Hl HA HD Hx ltac:(lra) ltac:(lra) N) as Hb.
 assert (Hv:LogBound l (rnd(T-z)) (T-z)) by (apply N; simpl; auto).
 assert (Hq:LogBound l (rnd(A*rnd(T-z))) (A*rnd(T-z))) by (apply N; simpl; auto).
 pose proof (rounded_mult_budget rnd l 0 l A (rnd(T-z)) A (T-z) (LogBound_exact A HA) Hv Hq) as Ho.
 split; [exact Hb|]. eapply LogBound_mono; [|exact Ho]; lra.
Qed.

Lemma LogBound_error e h x : LogBound e h x -> log_error h x<=e.
Proof. intros (_&_&H); exact H. Qed.
Lemma logs_positive_to_inner_bound A D T x th qh :
 0<A -> 0<D -> 0<T -> 0<x -> 0<th -> 0<qh ->
 log_error th (gas_map A D T x)<=52*binary64_lambda ->
 log_error qh (A*Rabs(gas_map A D T x-T))<=52*binary64_lambda ->
 0<A*Rabs(gas_map A D T x-T) ->
 LogBound (52*binary64_lambda) th (gas_map A D T x) /\
 LogBound (52*binary64_lambda) qh (A*Rabs(gas_map A D T x-T)).
Proof.
 intros HA HD HT Hx Hth Hqh Ht Hq Hp.
 pose proof (proj1 (gas_map_spec A D T x HA HD HT Hx)) as Hg.
 split; repeat split; assumption.
Qed.

Lemma normal_nodes_round choice nodes : FiniteNormalNodes choice nodes ->
 RoundNodes (RN64 choice) binary64_lambda nodes.
Proof. rewrite guard_lambda64; intros H; apply RN64_rounded_nodes, finite_nodes_rounded_normal; exact H. Qed.

Definition InnerBound A D T x th qh :=
 LogBound (52*binary64_lambda) th (gas_map A D T x) /\
 LogBound (52*binary64_lambda) qh (A*Rabs(gas_map A D T x-T)).

Theorem heating_RN64_accepted choice A D T x z :
 0<A -> 0<D -> 0<T -> T<x -> 0<z ->
 FiniteNormalNodes choice (heating_nodes (RN64 choice) A D T x z) ->
 inner_window (heating_balance (RN64 choice) A D T x z) ->
 InnerBound A D T x (RN64 choice (T+z)) (RN64 choice (A*z)).
Proof.
 intros HA HD HT Hx Hz N W.
 pose proof binary64_lambda_positive as Hl.
 destruct (heating_complete_graph (RN64 choice) binary64_lambda A D T x z
 ltac:(lra) HA HD Hz (normal_nodes_round choice _ N)) as [HB [HTout HQout]].
 pose proof (inner_exact_window _ _ W (proj2(proj2 HB))) as Hres.
 pose proof (gas_map_heating_residual_outputs A D T x z binary64_lambda
 (RN64 choice (T+z)) (RN64 choice (A*z)) HA HD HT Hx Hz ltac:(lra) Hres
 ltac:(pose proof (LogBound_error _ _ _ HTout); lra)
 ltac:(pose proof (LogBound_error _ _ _ HQout); lra)) as [Ht Hq].
 pose proof (gas_map_heating_order A D T x HA HD HT Hx) as Hg.
 unfold InnerBound; apply logs_positive_to_inner_bound; try assumption; try lra;
 try (exact (proj1 HTout)); try (exact (proj1 HQout));
 rewrite Rabs_pos_eq by lra; assumption || nra.
Qed.

Theorem weak_RN64_accepted choice A D T x z :
 0<A -> 0<D -> 0<x -> x<T -> T/2<=gas_map A D T x -> 0<z<=T/2 ->
 FiniteNormalNodes choice (weak_nodes (RN64 choice) A D T x z) ->
 inner_window (weak_balance (RN64 choice) A D T x z) ->
 InnerBound A D T x (RN64 choice (T-z)) (RN64 choice (A*z)).
Proof.
 intros HA HD Hx HT Hweak Hz N W.
 pose proof binary64_lambda_positive as Hl.
 destruct (weak_complete_graph (RN64 choice) binary64_lambda A D T x z
 ltac:(lra) HA HD ltac:(lra) (normal_nodes_round choice _ N)) as [HB [HTout HQout]].
 pose proof (inner_exact_window _ _ W (proj2(proj2 HB))) as Hres.
 pose proof (gas_map_weak_residual_outputs A D T x z binary64_lambda
 (RN64 choice (T-z)) (RN64 choice (A*z)) HA HD Hx HT Hweak Hz ltac:(lra) Hres
 ltac:(pose proof (LogBound_error _ _ _ HTout); lra)
 ltac:(pose proof (LogBound_error _ _ _ HQout); lra)) as [Ht Hq].
 pose proof (gas_map_cooling_order A D T x HA HD Hx HT) as Hg.
 assert (HE:Rabs(gas_map A D T x-T)=T-gas_map A D T x).
 { rewrite Rabs_left1 by lra; ring. }
 unfold InnerBound; apply logs_positive_to_inner_bound; try assumption; try lra;
 try (exact (proj1 HTout)); try (exact (proj1 HQout)); rewrite HE; assumption || nra.
Qed.

Theorem strong_RN64_accepted choice A D T x z :
 0<A -> 0<D -> 0<x -> x<T -> gas_map A D T x<=T/2 -> 0<z<=T/2 ->
 FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T x z) ->
 inner_window (eval_inner_strong (RN64 choice) A D T x z) ->
 InnerBound A D T x z (RN64 choice (A*RN64 choice (T-z))).
Proof.
 intros HA HD Hx HT Hstrong Hz N W.
 pose proof binary64_lambda_positive as Hl.
 destruct (strong_complete_graph (RN64 choice) binary64_lambda A D T x z
 ltac:(lra) HA HD Hx ltac:(lra) (normal_nodes_round choice _ N)) as [HB HQout].
 pose proof (inner_exact_window _ _ W (proj2(proj2 HB))) as Hres.
 assert (Hzz:log_error z z<=2*binary64_lambda).
 { unfold log_error; rewrite Rminus_diag_eq,Rabs_R0; lra. }
 pose proof (gas_map_strong_residual_outputs A D T x z binary64_lambda z
 (RN64 choice (A*RN64 choice (T-z))) HA HD Hx HT Hstrong Hz ltac:(lra) Hres Hzz
 (LogBound_error _ _ _ HQout)) as [Ht Hq].
 pose proof (gas_map_cooling_order A D T x HA HD Hx HT) as Hg.
 assert (HE:Rabs(gas_map A D T x-T)=T-gas_map A D T x).
 { rewrite Rabs_left1 by lra; ring. }
 unfold InnerBound; apply logs_positive_to_inner_bound; try assumption; try lra;
 try (exact (proj1 HQout)); rewrite HE; assumption || nra.
Qed.

Theorem boundary_RN64_accepted choice A D T x :
 0<A -> 0<D -> 0<x -> x<T ->
 FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T x (T/2)) ->
 inner_window (eval_inner_strong (RN64 choice) A D T x (T/2)) ->
 InnerBound A D T x (T/2) (RN64 choice (A*RN64 choice (T-T/2))).
Proof.
 intros HA HD Hx HT N W.
 pose proof binary64_lambda_positive as Hl.
 destruct (strong_complete_graph (RN64 choice) binary64_lambda A D T x (T/2)
 ltac:(lra) HA HD Hx ltac:(lra) (normal_nodes_round choice _ N)) as [HB HQout].
 pose proof (inner_exact_window _ _ W (proj2(proj2 HB))) as Hres.
 assert (Hzz:log_error (T/2) (T/2)<=2*binary64_lambda).
 { unfold log_error; rewrite Rminus_diag_eq,Rabs_R0; lra. }
 pose proof (gas_map_boundary_residual_outputs A D T x binary64_lambda (T/2)
 (RN64 choice (A*RN64 choice (T-T/2))) HA HD Hx HT ltac:(lra) Hres binary64_boundary_small Hzz
 (LogBound_error _ _ _ HQout)) as [Ht Hq].
 pose proof (gas_map_cooling_order A D T x HA HD Hx HT) as Hg.
 assert (HE:Rabs(gas_map A D T x-T)=T-gas_map A D T x).
 { rewrite Rabs_left1 by lra; ring. }
 unfold InnerBound; apply logs_positive_to_inner_bound; try assumption; try lra;
 try (exact (proj1 HQout)); rewrite HE; assumption || nra.
Qed.

Lemma strong_recovery_nodes rnd l A T z : 0<A ->
 RoundNodes rnd l [T-z; A*rnd(T-z)] ->
 LogBound (2*l) (rnd(A*rnd(T-z))) (A*(T-z)).
Proof.
 intros HA N.
 assert (Hv:LogBound l (rnd(T-z)) (T-z)) by (apply N; simpl; auto).
 assert (Hq:LogBound l (rnd(A*rnd(T-z))) (A*rnd(T-z))) by (apply N; simpl; auto).
 pose proof (rounded_mult_budget rnd l 0 l A (rnd(T-z)) A (T-z) (LogBound_exact A HA) Hv Hq) as H.
 eapply LogBound_mono; [|exact H]; lra.
Qed.

Theorem heating_RN64_bracket choice A D T x lo hi z :
 0<A -> 0<D -> 0<T -> T<x -> 0<lo -> lo<=z<=hi ->
 lo<=gas_map A D T x-T<=hi -> ln(hi/lo)<=8*binary64_lambda ->
 FiniteNormalNodes choice [T+z; A*z] ->
 InnerBound A D T x (RN64 choice (T+z)) (RN64 choice (A*z)).
Proof.
 intros HA HD HT Hx Hl Hz Hg Hwidth N.
 pose proof binary64_lambda_positive as Hlam.
 pose proof (normal_nodes_round choice _ N) as RN.
 assert (Ht:LogBound binary64_lambda (RN64 choice(T+z)) (T+z)) by (apply RN; simpl; auto).
 assert (Hq:LogBound binary64_lambda (RN64 choice(A*z)) (A*z)) by (apply RN; simpl; auto).
 pose proof (heating_bracket_outputs T A lo hi (gas_map A D T x-T) z binary64_lambda
 (RN64 choice(T+z)) (RN64 choice(A*z)) HT HA Hl Hg Hz ltac:(lra) Hwidth
 ltac:(pose proof (LogBound_error _ _ _ Ht); lra)
 ltac:(pose proof (LogBound_error _ _ _ Hq); lra)) as [Et Eq].
 replace (T+(gas_map A D T x-T)) with (gas_map A D T x) in Et by ring.
 unfold InnerBound; apply logs_positive_to_inner_bound; try assumption; try lra;
 try (exact (proj1 Ht)); try (exact (proj1 Hq));
 rewrite Rabs_pos_eq by lra; assumption || nra.
Qed.

Theorem weak_RN64_bracket choice A D T x lo hi z :
 0<A -> 0<D -> 0<x -> x<T -> 0<lo -> lo<=z<=hi -> hi<=T/2 ->
 lo<=T-gas_map A D T x<=hi -> ln(hi/lo)<=8*binary64_lambda ->
 FiniteNormalNodes choice [T-z; A*z] ->
 InnerBound A D T x (RN64 choice (T-z)) (RN64 choice (A*z)).
Proof.
 intros HA HD Hx HT Hl Hz Hhi Hg Hwidth N.
 pose proof binary64_lambda_positive as Hlam.
 pose proof (normal_nodes_round choice _ N) as RN.
 assert (Ht:LogBound binary64_lambda (RN64 choice(T-z)) (T-z)) by (apply RN; simpl; auto).
 assert (Hq:LogBound binary64_lambda (RN64 choice(A*z)) (A*z)) by (apply RN; simpl; auto).
 pose proof (weak_bracket_outputs T A lo hi (T-gas_map A D T x) z binary64_lambda
 (RN64 choice(T-z)) (RN64 choice(A*z)) ltac:(lra) HA Hl Hg Hz Hhi ltac:(lra) Hwidth
 ltac:(pose proof (LogBound_error _ _ _ Ht); lra)
 ltac:(pose proof (LogBound_error _ _ _ Hq); lra)) as [Et Eq].
 replace (T-(T-gas_map A D T x)) with (gas_map A D T x) in Et by ring.
 assert (HE:Rabs(gas_map A D T x-T)=T-gas_map A D T x).
 { rewrite Rabs_left1 by lra; ring. }
 unfold InnerBound; apply logs_positive_to_inner_bound; try assumption; try lra;
 try (exact (proj1 Ht)); try (exact (proj1 Hq)); rewrite HE; assumption || nra.
Qed.

Theorem strong_RN64_bracket choice A D T x lo hi z :
 0<A -> 0<D -> 0<x -> x<T -> 0<lo -> lo<=z<=hi -> hi<=T/2 ->
 lo<=gas_map A D T x<=hi -> ln(hi/lo)<=8*binary64_lambda ->
 FiniteNormalNodes choice [T-z; A*RN64 choice(T-z)] ->
 InnerBound A D T x z (RN64 choice (A*RN64 choice(T-z))).
Proof.
 intros HA HD Hx HT Hl Hz Hhi Hg Hwidth N.
 pose proof binary64_lambda_positive as Hlam.
 pose proof (normal_nodes_round choice _ N) as RN.
 pose proof (strong_recovery_nodes (RN64 choice) binary64_lambda A T z HA RN) as Hq.
 assert (Hzz:log_error z z<=2*binary64_lambda).
 { unfold log_error; rewrite Rminus_diag_eq,Rabs_R0; lra. }
 pose proof (strong_bracket_outputs T A lo hi (gas_map A D T x) z binary64_lambda
 z (RN64 choice (A*RN64 choice(T-z))) ltac:(lra) HA Hl Hg Hz Hhi ltac:(lra) Hwidth Hzz
 (LogBound_error _ _ _ Hq)) as [Et Eq].
 assert (HE:Rabs(gas_map A D T x-T)=T-gas_map A D T x).
 { rewrite Rabs_left1 by lra; ring. }
 unfold InnerBound; apply logs_positive_to_inner_bound; try assumption; try lra;
 try (exact (proj1 Hq)); rewrite HE; assumption || nra.
Qed.

Lemma strong_balance_lower_forward A D T x t : 0<t ->
 Hs T A D x t<1 -> x<forward_map A D T t.
Proof.
 intros Ht Hb. unfold Hs in Hb.
 apply (Rmult_lt_compat_r t) in Hb; [|lra].
 replace ((x+A*(T-t)/(D*sqrt t))/t*t) with (x+A*(T-t)/(D*sqrt t)) in Hb by (remember (x+A*(T-t)/(D*sqrt t)) as n; field; lra).
 unfold forward_map, Rdiv in *; nra.
Qed.
Lemma strong_balance_upper_forward A D T x t : 0<t ->
 1<Hs T A D x t -> forward_map A D T t<x.
Proof.
 intros Ht Hb. unfold Hs in Hb.
 apply (Rmult_lt_compat_r t) in Hb; [|lra].
 replace ((x+A*(T-t)/(D*sqrt t))/t*t) with (x+A*(T-t)/(D*sqrt t)) in Hb by (remember (x+A*(T-t)/(D*sqrt t)) as n; field; lra).
 unfold forward_map, Rdiv in *; nra.
Qed.

Lemma strong_boundary_selects_chart A D T x :
 0<A -> 0<D -> 0<x -> x<T -> Hs T A D x (T/2)<1 -> gas_map A D T x<T/2.
Proof.
 intros HA HD Hx HT Hb.
 pose proof (strong_balance_lower_forward A D T x (T/2) ltac:(lra) Hb) as Hf.
 destruct (gas_map_spec A D T x HA HD ltac:(lra) Hx) as [Hg HF].
 apply Rnot_le_lt; intro Hle.
 destruct (Rle_lt_or_eq_dec _ _ Hle) as [Hlt|Heq].
 - pose proof (forward_increasing A D T (T/2) (gas_map A D T x) HA HD ltac:(lra) ltac:(lra) Hlt); lra.
 - rewrite Heq in Hf; lra.
Qed.
Lemma weak_boundary_selects_chart A D T x :
 0<A -> 0<D -> 0<x -> x<T -> 1<Hs T A D x (T/2) -> T/2<gas_map A D T x.
Proof.
 intros HA HD Hx HT Hb.
 pose proof (strong_balance_upper_forward A D T x (T/2) ltac:(lra) Hb) as Hf.
 destruct (gas_map_spec A D T x HA HD ltac:(lra) Hx) as [Hg HF].
 apply Rnot_le_lt; intro Hle.
 destruct (Rle_lt_or_eq_dec _ _ Hle) as [Hlt|Heq].
 - pose proof (forward_increasing A D T (gas_map A D T x) (T/2) HA HD ltac:(lra) Hg Hlt); lra.
 - rewrite <- Heq in Hf; lra.
Qed.

Theorem RN64_boundary_selects_chart choice A D T x :
 0<A -> 0<D -> 0<x -> x<T ->
 FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T x (T/2)) ->
 (eval_inner_strong (RN64 choice) A D T x (T/2)<1-16*binary64_u ->
  gas_map A D T x<T/2) /\
 (1+16*binary64_u<eval_inner_strong (RN64 choice) A D T x (T/2) ->
  T/2<gas_map A D T x).
Proof.
 intros HA HD Hx HT N.
 pose proof binary64_lambda_positive as Hl.
 destruct (strong_complete_graph (RN64 choice) binary64_lambda A D T x (T/2)
 ltac:(lra) HA HD Hx ltac:(lra) (normal_nodes_round choice _ N)) as [HB HQout].
 destruct HB as [Hbp [Hep Herr]].
 split; intro Hsign.
 - apply strong_boundary_selects_chart; try assumption.
   exact (inner_lower_sign _ _ Hbp Hep Hsign Herr).
 - apply weak_boundary_selects_chart; try assumption.
   exact (inner_upper_sign _ _ Hbp Hep Hsign Herr).
Qed.

(* A true inner width is obtained from two actual rounded operations. A zero
   width is handled directly rather than assuming a normal zero difference. *)
Definition InnerNarrow choice lo hi :=
 lo=hi \/
 (lo<hi /\ SafeDifference64 lo hi /\ normal64 (RN64 choice (hi-lo)/lo) /\
  RN64 choice (RN64 choice (hi-lo)/lo)<=4*binary64_u).
Lemma inner_narrow_log_width choice lo hi :
 0<lo -> InnerNarrow choice lo hi -> ln(hi/lo)<=8*binary64_lambda.
Proof.
 intros Hl [He|[Hlt [Hn [Hd Hw]]]].
 - subst hi. replace (lo/lo) with 1 by (field; lra).
   rewrite ln_1. pose proof binary64_lambda_positive; lra.
 - left; apply (rounded_inner_width_safe_RN64 choice lo hi); assumption.
Qed.

Inductive InnerResult (choice:Z->bool) (A D T x:R) : R -> R -> Prop :=
| IR_equilibrium : x=T -> InnerResult choice A D T x T 0
| IR_heating_residual : forall z,
 T<x -> 0<z ->
 FiniteNormalNodes choice (heating_nodes (RN64 choice) A D T x z) ->
 inner_window (heating_balance (RN64 choice) A D T x z) ->
 InnerResult choice A D T x (RN64 choice(T+z)) (RN64 choice(A*z))
| IR_weak_residual : forall z,
 x<T -> T/2<=gas_map A D T x -> 0<z<=T/2 ->
 FiniteNormalNodes choice (weak_nodes (RN64 choice) A D T x z) ->
 inner_window (weak_balance (RN64 choice) A D T x z) ->
 InnerResult choice A D T x (RN64 choice(T-z)) (RN64 choice(A*z))
| IR_strong_residual : forall z,
 x<T -> gas_map A D T x<=T/2 -> 0<z<=T/2 ->
 FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T x z) ->
 inner_window (eval_inner_strong (RN64 choice) A D T x z) ->
 InnerResult choice A D T x z (RN64 choice(A*RN64 choice(T-z)))
| IR_boundary_residual :
 x<T -> FiniteNormalNodes choice (inner_strong_nodes (RN64 choice) A D T x (T/2)) ->
 inner_window (eval_inner_strong (RN64 choice) A D T x (T/2)) ->
 InnerResult choice A D T x (T/2) (RN64 choice(A*RN64 choice(T-T/2)))
| IR_heating_bracket : forall lo hi z,
 T<x -> 0<lo -> lo<=z<=hi -> lo<=gas_map A D T x-T<=hi ->
 InnerNarrow choice lo hi -> FiniteNormalNodes choice [T+z; A*z] ->
 InnerResult choice A D T x (RN64 choice(T+z)) (RN64 choice(A*z))
| IR_weak_bracket : forall lo hi z,
 x<T -> 0<lo -> lo<=z<=hi -> hi<=T/2 -> lo<=T-gas_map A D T x<=hi ->
 InnerNarrow choice lo hi -> FiniteNormalNodes choice [T-z; A*z] ->
 InnerResult choice A D T x (RN64 choice(T-z)) (RN64 choice(A*z))
| IR_strong_bracket : forall lo hi z,
 x<T -> 0<lo -> lo<=z<=hi -> hi<=T/2 -> lo<=gas_map A D T x<=hi ->
 InnerNarrow choice lo hi -> FiniteNormalNodes choice [T-z; A*RN64 choice(T-z)] ->
 InnerResult choice A D T x z (RN64 choice(A*RN64 choice(T-z))).

Theorem inner_result_accuracy choice A D T x th qh :
 0<A -> 0<D -> 0<T -> 0<x -> InnerResult choice A D T x th qh ->
 LogBound (52*binary64_lambda) th (gas_map A D T x) /\
 ((x=T /\ qh=0) \/ LogBound (52*binary64_lambda) qh (A*Rabs(gas_map A D T x-T))).
Proof.
 intros HA HD HT Hx HR.
 destruct HR as [HE|z HX Hz N W|z HX Hchart Hz N W|z HX Hchart Hz N W|
 HX N W|lo hi z HX Hl Hz Hg HW N|lo hi z HX Hl Hz Hhi Hg HW N|
 lo hi z HX Hl Hz Hhi Hg HW N].
 - subst x. rewrite (gas_map_equilibrium A D T HA HD HT).
   split; [|left; auto].
   eapply LogBound_mono; [|apply LogBound_exact; exact HT].
   pose proof binary64_lambda_positive; lra.
 - destruct (heating_RN64_accepted choice A D T x z HA HD HT HX Hz N W) as [Ht Hq]. auto.
 - destruct (weak_RN64_accepted choice A D T x z HA HD Hx HX Hchart Hz N W) as [Ht Hq]. auto.
 - destruct (strong_RN64_accepted choice A D T x z HA HD Hx HX Hchart Hz N W) as [Ht Hq]. auto.
 - destruct (boundary_RN64_accepted choice A D T x HA HD Hx HX N W) as [Ht Hq]. auto.
 - destruct (heating_RN64_bracket choice A D T x lo hi z HA HD HT HX Hl Hz Hg
   (inner_narrow_log_width choice lo hi Hl HW) N) as [Ht Hq]. auto.
 - destruct (weak_RN64_bracket choice A D T x lo hi z HA HD Hx HX Hl Hz Hhi Hg
   (inner_narrow_log_width choice lo hi Hl HW) N) as [Ht Hq]. auto.
 - destruct (strong_RN64_bracket choice A D T x lo hi z HA HD Hx HX Hl Hz Hhi Hg
   (inner_narrow_log_width choice lo hi Hl HW) N) as [Ht Hq]. auto.
Qed.

Corollary inner_result_nonzero_accuracy choice A D T x th qh :
 0<A -> 0<D -> 0<T -> 0<x -> x<>T -> InnerResult choice A D T x th qh ->
 InnerBound A D T x th qh.
Proof.
 intros HA HD HT Hx Hne HR.
 destruct (inner_result_accuracy choice A D T x th qh HA HD HT Hx HR) as [Ht [[HE Hq]|Hq]];
 [contradiction|split; assumption].
Qed.

Corollary inner_result_positive choice A D T x th qh :
 0<A -> 0<D -> 0<T -> 0<x -> InnerResult choice A D T x th qh ->
 0<th /\ 0<=qh /\ (x<>T -> 0<qh).
Proof.
 intros HA HD HT Hx HR.
 destruct (inner_result_accuracy choice A D T x th qh HA HD HT Hx HR) as [Ht [[HE Hq]|Hq]].
 - destruct Ht as [Htp _]. subst qh; split; [exact Htp|]. split; [lra|]. intro Hne; exfalso; apply Hne; exact HE.
 - destruct Ht as [Htp _]. destruct Hq as [Hqp _]. repeat split; intros; lra.
Qed.

Corollary inner_result_exact_zero choice A D T x th qh :
 0<A -> 0<D -> 0<T -> x=T -> InnerResult choice A D T x th qh -> th=T /\ qh=0.
Proof.
 intros HA HD HT HE HR. destruct HR; try lra; auto.
Qed.

Print Assumptions inner_result_accuracy.
Print Assumptions inner_result_nonzero_accuracy.
Print Assumptions RN64_boundary_selects_chart.
