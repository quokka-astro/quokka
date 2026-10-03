(* Concrete outward-rounded inner initialization. True root containment is
   derived from the primitive arithmetic graph and exact chart equations;
   it is never an initialization hypothesis. Range failures remain explicit. *)
From Coq Require Import Reals Psatz Field List Lia.
Require Import Flocq.Core.Core.
From BlackBox Require Import FloatingPoint Guards GuardFloatBridge Brackets
  InnerCharts InnerLoop LowerEndpointGraph StoredInputs.
Open Scope R_scope.
Import ListNotations.
Local Instance init_precision64 : Prec_gt_0 53. Proof. unfold Prec_gt_0; lia. Qed.

Definition upper_endpoint_factor : R := 1+32*binary64_u.
Definition eval_upper_outward (rnd:R->R) d := rnd(rnd d*upper_endpoint_factor).
Definition upper_outward_nodes (rnd:R->R) d := [d;rnd d*upper_endpoint_factor].
Definition eval_heating_upper rnd T x := eval_upper_outward rnd (x-T).
Definition eval_weak_upper rnd T x := Rmin (rnd(T/2)) (eval_upper_outward rnd (T-x)).

Lemma upper_endpoint_factor_format : format64 upper_endpoint_factor.
Proof.
 unfold upper_endpoint_factor; replace (1+32*binary64_u) with
   (1+2*IZR 16*binary64_u) by (simpl; ring).
 apply stored_even_guard_format64; lia.
Qed.
Lemma upper_endpoint_factor_exact choice : RN64 choice upper_endpoint_factor=upper_endpoint_factor.
Proof. apply stored_format64_exact,upper_endpoint_factor_format. Qed.

Lemma upper_endpoint_log_margin :
  0<upper_endpoint_factor /\ 8*binary64_lambda<=ln upper_endpoint_factor.
Proof.
 pose proof binary64_u_bounds as Hu; pose proof binary64_lambda_bounds as Hl.
 assert (Hfrac:binary64_u/(1-binary64_u)<=2*binary64_u).
 { apply (Rmult_le_reg_r (1-binary64_u)); [lra|]. field_simplify; nra. }
 assert (Hpos:0<1+32*binary64_u) by lra.
 pose proof (guard_ln_lower (1+32*binary64_u) Hpos) as Hlog.
 replace ((1+32*binary64_u-1)/(1+32*binary64_u)) with
   (32*binary64_u/(1+32*binary64_u)) in Hlog by (field; lra).
 assert (Hmargin:16*binary64_u<=32*binary64_u/(1+32*binary64_u)).
 { apply (Rmult_le_reg_r (1+32*binary64_u)); [lra|].
   field_simplify; nra. }
 unfold upper_endpoint_factor; split; lra.
Qed.

Lemma upper_outward_graph_encloses rnd d :
  RoundNodes rnd binary64_lambda (upper_outward_nodes rnd d) ->
  0<d<eval_upper_outward rnd d.
Proof.
 intro N; unfold RoundNodes,upper_outward_nodes in N.
 destruct (N d ltac:(simpl; auto)) as [Hdh [Hd He]].
 destruct (N (rnd d*upper_endpoint_factor) ltac:(simpl; auto)) as [Hup [Hp Hu]].
 destruct upper_endpoint_log_margin as [Hfp Hfl].
 pose proof binary64_lambda_positive as Hl.
 split; [exact Hd|].
 unfold eval_upper_outward; apply ln_lt_inv; try assumption.
 rewrite ln_mult in Hu by assumption.
 apply guard_abs_bounds in He; apply guard_abs_bounds in Hu; lra.
Qed.

Theorem upper_outward_finite64_encloses choice d :
  FiniteNormalNodes choice (upper_outward_nodes (RN64 choice) d) ->
  0<d<eval_upper_outward (RN64 choice) d.
Proof.
 intro N; apply upper_outward_graph_encloses.
 rewrite guard_lambda64; apply RN64_rounded_nodes; now apply finite_nodes_rounded_normal.
Qed.

Lemma rounded_node_properties choice nodes z :
  FiniteNormalNodes choice nodes -> In z nodes ->
  0<RN64 choice z /\ normal64 (RN64 choice z) /\
  finite64 (RN64 choice z) /\ format64 (RN64 choice z).
Proof.
 intros N Hz; unfold FiniteNormalNodes in N.
 apply Forall_forall with (x:=z) in N; [|exact Hz].
 destruct N as [Hp [Hn Hf]]; repeat split; try assumption.
 unfold format64,RN64; apply generic_format_round; typeclasses eauto.
Qed.

Lemma heating_lower_node_properties choice A D T x :
  FiniteNormalNodes choice (heating_lower_nodes (RN64 choice) A D T x) ->
  0<eval_heating_lower (RN64 choice) A D T x /\
  normal64 (eval_heating_lower (RN64 choice) A D T x) /\
  finite64 (eval_heating_lower (RN64 choice) A D T x) /\
  format64 (eval_heating_lower (RN64 choice) A D T x).
Proof.
 intro N; unfold eval_heating_lower,eval_lower_outward.
 apply (rounded_node_properties choice (heating_lower_nodes (RN64 choice) A D T x)); [exact N|].
 unfold heating_lower_nodes,lower_outward_nodes; apply in_or_app; right; simpl; auto.
Qed.

Lemma weak_lower_node_properties choice A D T x :
  FiniteNormalNodes choice (weak_lower_nodes (RN64 choice) A D T x) ->
  0<eval_weak_lower (RN64 choice) A D T x /\
  normal64 (eval_weak_lower (RN64 choice) A D T x) /\
  finite64 (eval_weak_lower (RN64 choice) A D T x) /\
  format64 (eval_weak_lower (RN64 choice) A D T x).
Proof.
 intro N; unfold eval_weak_lower,eval_lower_outward.
 apply (rounded_node_properties choice (weak_lower_nodes (RN64 choice) A D T x)); [exact N|].
 unfold weak_lower_nodes; right; unfold lower_outward_nodes.
 apply in_or_app; right; simpl; auto.
Qed.

Lemma upper_node_properties choice d :
  FiniteNormalNodes choice (upper_outward_nodes (RN64 choice) d) ->
  0<eval_upper_outward (RN64 choice) d /\
  normal64 (eval_upper_outward (RN64 choice) d) /\
  finite64 (eval_upper_outward (RN64 choice) d) /\
  format64 (eval_upper_outward (RN64 choice) d).
Proof.
 intro N; unfold eval_upper_outward.
 apply (rounded_node_properties choice (upper_outward_nodes (RN64 choice) d)); [exact N|].
 unfold upper_outward_nodes; simpl; auto.
Qed.

(* Heating chart: the rounded lower endpoint is below the analytic positive
   seed, while the two-operation upper graph is strictly above x-T. *)
Theorem heating_rounded_initial_bracket choice A D T x z :
  0<T -> 0<A -> 0<D -> T<x -> 0<z -> Hh T A D (x-T) z=1 ->
  FiniteNormalNodes choice (heating_lower_nodes (RN64 choice) A D T x) ->
  FiniteNormalNodes choice (upper_outward_nodes (RN64 choice) (x-T)) ->
  0<eval_heating_lower (RN64 choice) A D T x /\
  eval_heating_lower (RN64 choice) A D T x<z<eval_heating_upper (RN64 choice) T x.
Proof.
 intros HT HA HD Hx Hz Hroot NL NU.
 pose proof (heating_lower_finite64_encloses choice A D T x HA HD HT NL) as Hl.
 pose proof (inner_heating_seed_brackets_root T A D (x-T) z HT HA HD ltac:(lra) Hz Hroot) as Hr.
 pose proof (upper_outward_finite64_encloses choice (x-T) NU) as Hu.
 change (0<eval_heating_lower (RN64 choice) A D T x<positive_chart_seed T A D (x-T)) in Hl.
 unfold eval_heating_upper; lra.
Qed.

(* Weak cooling chart: division by two is exact for the stored T under the
   explicit normal/finite condition. min chooses one stored endpoint. *)
Theorem weak_rounded_initial_bracket choice A D T x z :
  0<T -> 0<A -> 0<D -> x<T -> 0<z<=T/2 -> Hw T A D (T-x) z=1 ->
  format64 T -> normal64 (T/2) -> finite64 (T/2) ->
  FiniteNormalNodes choice (weak_lower_nodes (RN64 choice) A D T x) ->
  FiniteNormalNodes choice (upper_outward_nodes (RN64 choice) (T-x)) ->
  0<eval_weak_lower (RN64 choice) A D T x /\
  eval_weak_lower (RN64 choice) A D T x<z /\ z<=eval_weak_upper (RN64 choice) T x.
Proof.
 intros HT HA HD Hx Hz Hroot HTf HThn HThf NL NU.
 pose proof (weak_lower_finite64_encloses choice A D T x HA HD NL) as Hl.
 pose proof (inner_weak_seed_brackets_root T A D (T-x) z HT HA HD ltac:(lra) Hz Hroot) as Hr.
 pose proof (upper_outward_finite64_encloses choice (T-x) NU) as Hu.
 change (0<eval_weak_lower (RN64 choice) A D T x<positive_chart_seed (T/2) A D (T-x)) in Hl.
 pose proof (Rmin_l (T-x) (T/2)).
 split; [lra|split; [lra|]].
 unfold eval_weak_upper; rewrite (stored_half_exact64 choice T HTf HThn HThf).
 apply Rmin_glb; lra.
Qed.

Theorem strong_stored_initial_bracket choice A D T x z :
  0<T -> 0<A -> 0<D -> 0<x -> 0<z<=T/2 -> Hs T A D x z=1 ->
  format64 T -> normal64 (T/2) -> finite64 (T/2) ->
  0<x /\ x<z<=RN64 choice (T/2).
Proof.
 intros HT HA HD Hx Hz Hroot HTf HThn HThf.
 pose proof (inner_strong_brackets_root T A D x z HT HA HD Hx Hz Hroot) as Hr.
 rewrite (stored_half_exact64 choice T HTf HThn HThf); lra.
Qed.

Lemma weak_upper_node_properties choice T x :
  0<T -> format64 T -> normal64 (T/2) -> finite64 (T/2) ->
  FiniteNormalNodes choice (upper_outward_nodes (RN64 choice) (T-x)) ->
  0<eval_weak_upper (RN64 choice) T x /\
  normal64 (eval_weak_upper (RN64 choice) T x) /\
  finite64 (eval_weak_upper (RN64 choice) T x) /\
  format64 (eval_weak_upper (RN64 choice) T x).
Proof.
 intros HT HTf HThn HThf N.
 pose proof (upper_node_properties choice (T-x) N) as Hu.
 unfold eval_weak_upper; rewrite (stored_half_exact64 choice T HTf HThn HThf).
 destruct (Rle_dec (T/2) (eval_upper_outward (RN64 choice) (T-x))) as [Hle|Hgt].
 - rewrite Rmin_left by exact Hle; repeat split; try assumption; try lra.
   apply stored_half_format64; assumption.
 - rewrite Rmin_right by lra; exact Hu.
Qed.

Print Assumptions heating_rounded_initial_bracket.
Print Assumptions weak_rounded_initial_bracket.
Print Assumptions strong_stored_initial_bracket.
