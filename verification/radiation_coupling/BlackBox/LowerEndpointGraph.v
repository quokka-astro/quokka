(* Concrete positive inner lower endpoints, formed with ordinary nearest
   rounding and a conservatively rounded multiplicative outward step. *)
From Coq Require Import Reals Psatz Field List.
From BlackBox Require Import FloatingPoint Guards GuardFloatBridge.
Open Scope R_scope.
Import ListNotations.

Definition exact_positive_lower A D d tb := d/(1+A/(D*sqrt tb)).
Definition eval_lower_raw (rnd:R->R) A D d th :=
 let dh:=rnd d in let sh:=rnd(sqrt th) in let ds:=rnd(D*sh) in
 let ratio:=rnd(A/ds) in let den:=rnd(1+ratio) in rnd(dh/den).
Definition lower_raw_nodes (rnd:R->R) A D d th :=
 let dh:=rnd d in let sh:=rnd(sqrt th) in let ds:=rnd(D*sh) in
 let ratio:=rnd(A/ds) in let den:=rnd(1+ratio) in
 [d;sqrt th;D*sh;A/ds;1+ratio;dh/den].
Definition lower_outward_factor := 1-32*binary64_u.
Definition eval_lower_outward (rnd:R->R) A D d th := rnd(eval_lower_raw rnd A D d th*lower_outward_factor).
Definition lower_outward_nodes (rnd:R->R) A D d th :=
 lower_raw_nodes rnd A D d th ++ [eval_lower_raw rnd A D d th*lower_outward_factor].

Lemma lower_raw_graph_budget rnd l A D d th tb :
 0<=l -> 0<A -> 0<D -> LogBound l th tb ->
 RoundNodes rnd l (lower_raw_nodes rnd A D d th) ->
 LogBound (7*l) (eval_lower_raw rnd A D d th) (exact_positive_lower A D d tb).
Proof.
 intros Hl HA HD HT N; unfold RoundNodes,lower_raw_nodes in N.
 pose proof (N d ltac:(simpl; tauto)) as Hd.
 pose proof (N (sqrt th) ltac:(simpl; tauto)) as Hs.
 pose proof (N (D*rnd(sqrt th)) ltac:(simpl; tauto)) as Hds.
 pose proof (N (A/rnd(D*rnd(sqrt th))) ltac:(simpl; tauto)) as Hr.
 pose proof (N (1+rnd(A/rnd(D*rnd(sqrt th)))) ltac:(simpl; tauto)) as Hden.
 pose proof (N (rnd d/rnd(1+rnd(A/rnd(D*rnd(sqrt th))))) ltac:(simpl; tauto)) as Ho.
 pose proof (rounded_sqrt_budget _ _ _ _ _ HT Hs) as HS.
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ (LogBound_exact D HD) HS Hds) as HDS.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ (LogBound_exact A HA) HDS Hr) as HR.
 pose proof (rounded_add_budget _ _ _ _ _ _ _ _ (LogBound_exact 1 ltac:(lra)) HR Hden) as HDEN.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ Hd HDEN Ho) as HO.
 unfold eval_lower_raw,exact_positive_lower; eapply LogBound_mono; [|exact HO].
 assert (Rmax 0 (0+(0+(l/2+l)+l)+l)<=7*l/2) by (apply Rmax_lub; lra); lra.
Qed.
Lemma lower_outward_log_margin : 0<lower_outward_factor /\ ln lower_outward_factor<= -16*binary64_lambda.
Proof.
 pose proof binary64_u_bounds as Hu; pose proof binary64_lambda_bounds as Hl.
 assert (Hfrac:binary64_u/(1-binary64_u)<=2*binary64_u).
 { apply (Rmult_le_reg_r (1-binary64_u)); [lra|]. field_simplify; nra. }
 unfold lower_outward_factor.
 pose proof (guard_ln_upper (1-32*binary64_u) ltac:(lra)); split; lra.
Qed.
Theorem lower_outward_graph_encloses rnd A D d th tb :
 0<A -> 0<D -> LogBound binary64_lambda th tb ->
 RoundNodes rnd binary64_lambda (lower_outward_nodes rnd A D d th) ->
 0<eval_lower_outward rnd A D d th<exact_positive_lower A D d tb.
Proof.
 intros HA HD HT N; unfold lower_outward_nodes in N.
 apply RoundNodes_app in N as [Nraw Nout].
 pose proof binary64_lambda_positive as Hl.
 pose proof (lower_raw_graph_budget rnd binary64_lambda A D d th tb ltac:(lra) HA HD HT Nraw) as Hraw.
 destruct Hraw as [Hrp [Hlp Hr]].
 pose proof (Nout _ ltac:(simpl; auto)) as [Hop [Hpp Ho]].
 destruct lower_outward_log_margin as [Hfp Hfl].
 unfold eval_lower_outward; split; [exact Hop|].
 apply ln_lt_inv; [exact Hop|exact Hlp|].
 rewrite ln_mult in Ho by assumption.
 apply guard_abs_bounds in Ho; apply guard_abs_bounds in Hr; lra.
Qed.

Theorem lower_outward_finite64_encloses choice A D d th tb :
 0<A -> 0<D -> LogBound lambda64 th tb ->
 FiniteNormalNodes choice (lower_outward_nodes (RN64 choice) A D d th) ->
 0<eval_lower_outward (RN64 choice) A D d th<exact_positive_lower A D d tb.
Proof.
 intros HA HD HT N; apply lower_outward_graph_encloses; auto.
 - rewrite guard_lambda64; apply RN64_rounded_nodes; now apply finite_nodes_rounded_normal.
Qed.

Definition eval_heating_lower rnd A D T0 x := eval_lower_outward rnd A D (x-T0) T0.
Definition heating_lower_nodes rnd A D T0 x := lower_outward_nodes rnd A D (x-T0) T0.
Theorem heating_lower_finite64_encloses choice A D T0 x :
 0<A -> 0<D -> 0<T0 ->
 FiniteNormalNodes choice (heating_lower_nodes (RN64 choice) A D T0 x) ->
 0<eval_heating_lower (RN64 choice) A D T0 x<exact_positive_lower A D (x-T0) T0.
Proof.
 intros HA HD HT N; unfold eval_heating_lower,heating_lower_nodes in *.
 apply lower_outward_finite64_encloses; auto.
 eapply LogBound_mono; [|apply LogBound_exact; exact HT].
 rewrite <- guard_lambda64; pose proof binary64_lambda_positive; lra.
Qed.

Definition eval_weak_lower rnd A D T0 x := eval_lower_outward rnd A D (T0-x) (rnd(T0/2)).
Definition weak_lower_nodes rnd A D T0 x := (T0/2)::lower_outward_nodes rnd A D (T0-x) (rnd(T0/2)).
Theorem weak_lower_finite64_encloses choice A D T0 x :
 0<A -> 0<D ->
 FiniteNormalNodes choice (weak_lower_nodes (RN64 choice) A D T0 x) ->
 0<eval_weak_lower (RN64 choice) A D T0 x<exact_positive_lower A D (T0-x) (T0/2).
Proof.
 intros HA HD N; unfold weak_lower_nodes in N.
 pose proof (RN64_rounded_nodes choice _ (finite_nodes_rounded_normal choice _ N)) as R.
 assert (HT:LogBound lambda64 (RN64 choice (T0/2)) (T0/2)) by (apply R; simpl; auto).
 assert (NR:FiniteNormalNodes choice (lower_outward_nodes (RN64 choice) A D (T0-x) (RN64 choice (T0/2)))).
 { unfold FiniteNormalNodes in *; inversion N; assumption. }
 unfold eval_weak_lower; now apply lower_outward_finite64_encloses.
Qed.

Print Assumptions lower_outward_finite64_encloses.
Print Assumptions heating_lower_finite64_encloses.
Print Assumptions weak_lower_finite64_encloses.
