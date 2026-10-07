(* A concrete representable grid built from Flocq successors. Every structural
   grid contract of SolverLoop/InnerLoop is proved here, not postulated.
   This specification is deliberately unary; an efficient binary64 bit/rank
   implementation requires a separate refinement proof. *)
From Coq Require Import Reals Psatz Lia Arith ZArith.
Require Import Flocq.Core.Core.
From BlackBox Require Import FloatingPoint GuardFloatBridge.
Open Scope R_scope.
Local Instance grid_precision64 : Prec_gt_0 53. Proof. unfold Prec_gt_0; lia. Qed.

Fixpoint normal_grid_from (base:R) (n:nat) : R :=
  match n with O=>base | S k=>successor64 (normal_grid_from base k) end.
Definition minnormal64 : R := bpow radix2 (-1022).
Definition grid64 : nat->R := normal_grid_from minnormal64.
Definition maxfinite64 : R := bpow radix2 1024-bpow radix2 971.

Lemma normal_grid_step base n :
  normal_grid_from base (S n)=successor64 (normal_grid_from base n).
Proof. reflexivity. Qed.

Lemma normal_grid_positive base n : 0<base -> 0<normal_grid_from base n.
Proof.
 intro Hb; induction n as [|n IH]; [exact Hb|].
 simpl; pose proof (succ_gt_id radix2 (FLT_exp (-1074) 53) (normal_grid_from base n) ltac:(lra)).
 unfold successor64; lra.
Qed.

Lemma normal_grid_step_strict base n :
  0<base -> normal_grid_from base n<normal_grid_from base (S n).
Proof.
 intro Hb; simpl; unfold successor64; apply succ_gt_id.
 pose proof (normal_grid_positive base n Hb); lra.
Qed.

Lemma normal_grid_base_le base n : 0<base -> base<=normal_grid_from base n.
Proof.
 intro Hb; induction n as [|n IH]; [simpl; lra|].
 pose proof (normal_grid_step_strict base n Hb); lra.
Qed.

Lemma normal_grid_le base i j :
  0<base -> (i<=j)%nat -> normal_grid_from base i<=normal_grid_from base j.
Proof.
 intros Hb Hij; induction Hij as [|j Hij IH]; [reflexivity|].
 pose proof (normal_grid_step_strict base j Hb); lra.
Qed.

Lemma normal_grid_increasing base i j :
  0<base -> (i<j)%nat -> normal_grid_from base i<normal_grid_from base j.
Proof.
 intros Hb Hij.
 pose proof (normal_grid_step_strict base i Hb).
 pose proof (normal_grid_le base (S i) j Hb ltac:(lia)); lra.
Qed.

Lemma normal_grid_format base n :
  format64 base -> format64 (normal_grid_from base n).
Proof.
 intro Hb; induction n as [|n IH]; [exact Hb|].
 simpl; unfold successor64,format64 in *.
 exact (generic_format_succ radix2 (FLT_exp (-1074) 53) (normal_grid_from base n) IH).
Qed.

Lemma normal_grid_normal base n :
  0<base -> normal64 base -> normal64 (normal_grid_from base n).
Proof.
 intros Hb Hnormal; unfold normal64 in *.
 rewrite (Rabs_pos_eq base) in Hnormal by lra.
 rewrite (Rabs_pos_eq (normal_grid_from base n)) by (pose proof (normal_grid_positive base n Hb); lra).
 pose proof (normal_grid_base_le base n Hb); lra.
Qed.

Lemma normal_grid_adjacent_complete base n x :
  format64 base -> format64 x ->
  ~(normal_grid_from base n<x<normal_grid_from base (S n)).
Proof.
 intros Hb Hx [Hlo Hhi].
 pose proof (normal_grid_format base n Hb) as Hn.
 pose proof (succ_le_lt radix2 (FLT_exp (-1074) 53) (normal_grid_from base n) x Hn Hx Hlo) as Hsucc.
 change (successor64 (normal_grid_from base n)<=x) in Hsucc.
 rewrite normal_grid_step in Hhi; lra.
Qed.

Lemma minnormal64_positive : 0<minnormal64.
Proof. apply bpow_gt_0. Qed.
Lemma minnormal64_normal : normal64 minnormal64.
Proof. unfold normal64,minnormal64; rewrite Rabs_pos_eq; [reflexivity|left; apply bpow_gt_0]. Qed.
Lemma minnormal64_format : format64 minnormal64.
Proof.
 unfold format64,minnormal64; apply generic_format_bpow.
 unfold FLT_exp; lia.
Qed.

Lemma grid64_positive n : 0<grid64 n.
Proof. apply normal_grid_positive; apply minnormal64_positive. Qed.
Lemma grid64_normal n : normal64 (grid64 n).
Proof. apply normal_grid_normal; [apply minnormal64_positive|apply minnormal64_normal]. Qed.
Lemma grid64_format n : format64 (grid64 n).
Proof. apply normal_grid_format; apply minnormal64_format. Qed.
Lemma grid64_increasing i j : (i<j)%nat -> grid64 i<grid64 j.
Proof. apply normal_grid_increasing; apply minnormal64_positive. Qed.
Lemma grid64_le i j : (i<=j)%nat -> grid64 i<=grid64 j.
Proof. apply normal_grid_le; apply minnormal64_positive. Qed.
Lemma grid64_adjacent_complete n x : format64 x -> ~(grid64 n<x<grid64 (S n)).
Proof. apply normal_grid_adjacent_complete; apply minnormal64_format. Qed.

(* These are exactly the five structural hypotheses used by both loops. *)
Theorem normal_grid_loop_contracts base lower upper :
  0<base -> normal64 base -> format64 base ->
  (forall i, (lower<=i<=upper)%nat -> 0<normal_grid_from base i) /\
  (forall i j, (lower<=i)%nat -> (i<j)%nat -> (j<=upper)%nat ->
     normal_grid_from base i<normal_grid_from base j) /\
  (forall i, (lower<=i<=upper)%nat -> format64 (normal_grid_from base i)) /\
  (forall i, (lower<=i<=upper)%nat -> normal64 (normal_grid_from base i)) /\
  (forall i, (lower<=i)%nat -> (S i<=upper)%nat -> forall x, format64 x ->
     ~(normal_grid_from base i<x<normal_grid_from base (S i))).
Proof.
 intros Hb Hn Hf; repeat split; intros.
 - apply normal_grid_positive; assumption.
 - apply normal_grid_increasing; assumption.
 - apply normal_grid_format; assumption.
 - apply normal_grid_normal; assumption.
 - eapply normal_grid_adjacent_complete; eauto.
Qed.

Theorem grid64_loop_contracts lower upper :
  (forall i, (lower<=i<=upper)%nat -> 0<grid64 i) /\
  (forall i j, (lower<=i)%nat -> (i<j)%nat -> (j<=upper)%nat -> grid64 i<grid64 j) /\
  (forall i, (lower<=i<=upper)%nat -> format64 (grid64 i)) /\
  (forall i, (lower<=i<=upper)%nat -> normal64 (grid64 i)) /\
  (forall i, (lower<=i)%nat -> (S i<=upper)%nat -> forall x, format64 x ->
     ~(grid64 i<x<grid64 (S i))).
Proof.
 apply normal_grid_loop_contracts; [apply minnormal64_positive|apply minnormal64_normal|apply minnormal64_format].
Qed.

Lemma maxfinite64_positive : 0<maxfinite64.
Proof.
 unfold maxfinite64; pose proof (bpow_lt radix2 971 1024 ltac:(lia)); lra.
Qed.

(* FLT has no upper exponent limit. Checking the selected upper endpoint is
   sufficient to enforce the IEEE finite upper range throughout the grid. *)
Theorem normal_grid_finite_segment base upper :
  0<base -> normal64 base -> format64 base ->
  normal_grid_from base upper<=maxfinite64 ->
  forall i, (i<=upper)%nat ->
  0<normal_grid_from base i /\ normal64 (normal_grid_from base i) /\
  format64 (normal_grid_from base i) /\ normal_grid_from base i<=maxfinite64.
Proof.
 intros Hb Hn Hf Hupper i Hi; repeat split.
 - apply normal_grid_positive; assumption.
 - apply normal_grid_normal; assumption.
 - apply normal_grid_format; assumption.
 - pose proof (normal_grid_le base i upper Hb Hi); lra.
Qed.

Print Assumptions grid64_loop_contracts.
Print Assumptions normal_grid_finite_segment.
