(* Every representable target at or above a positive starting value occurs at
   a unique finite successor-grid index. This removes an assumed rank/endpoint
   relation from mathematical initialization. An efficient decoder is separate. *)
From Coq Require Import Reals Psatz Lia Arith ZArith.
Require Import Flocq.Core.Core.
From BlackBox Require Import FloatingPoint GuardFloatBridge NormalGrid.
Open Scope R_scope.
Local Instance reach_precision64 : Prec_gt_0 53. Proof. unfold Prec_gt_0; lia. Qed.
Definition minsubnormal64 : R := bpow radix2 (-1074).

Lemma minsubnormal64_positive : 0<minsubnormal64.
Proof. apply bpow_gt_0. Qed.

Lemma normal_grid_minimum_spacing base n : 0<base ->
  minsubnormal64<=normal_grid_from base (S n)-normal_grid_from base n.
Proof.
 intro Hb; pose proof (normal_grid_positive base n Hb) as Hp.
 rewrite normal_grid_step; unfold successor64; rewrite succ_eq_pos by lra.
 replace (normal_grid_from base n+ulp radix2 (FLT_exp (-1074) 53) (normal_grid_from base n)-normal_grid_from base n)
   with (ulp radix2 (FLT_exp (-1074) 53) (normal_grid_from base n)) by ring.
 rewrite ulp_neq_0 by lra.
 unfold minsubnormal64; apply bpow_le.
 unfold cexp,FLT_exp; apply Z.le_max_r.
Qed.

Lemma normal_grid_linear_growth base n : 0<base ->
  base+INR n*minsubnormal64<=normal_grid_from base n.
Proof.
 intro Hb; induction n as [|n IH].
 - simpl; lra.
 - pose proof (normal_grid_minimum_spacing base n Hb).
   rewrite S_INR; nra.
Qed.

Lemma normal_grid_eventually_above base target : 0<base ->
  exists n, target<normal_grid_from base n.
Proof.
 intro Hb.
 destruct (INR_archimed minsubnormal64 (target-base) minsubnormal64_positive) as [n Hn].
 exists n; pose proof (normal_grid_linear_growth base n Hb); lra.
Qed.

Lemma normal_grid_bounded_reaches base target n :
  0<base -> format64 base -> format64 target -> base<=target ->
  target<=normal_grid_from base n ->
  exists k, (k<=n)%nat /\ normal_grid_from base k=target.
Proof.
 intros Hb Hbase Htarget Hlower; induction n as [|n IH]; intro Hupper.
 - exists O; simpl in *; split; [lia|lra].
 - destruct (Rle_dec target (normal_grid_from base n)) as [Hprev|Hprev].
   + destruct (IH Hprev) as [k [Hk Heq]]; exists k; split; [lia|exact Heq].
   + exists (S n); split; [lia|].
     destruct Hupper as [Hlt|Heq]; [|lra].
     exfalso; apply (normal_grid_adjacent_complete base n target Hbase Htarget); split; lra.
Qed.

Theorem normal_grid_reaches_representable base target :
  0<base -> format64 base -> format64 target -> base<=target ->
  exists n, normal_grid_from base n=target.
Proof.
 intros Hb Hbase Htarget Hlower.
 destruct (normal_grid_eventually_above base target Hb) as [n Hn].
 destruct (normal_grid_bounded_reaches base target n Hb Hbase Htarget Hlower ltac:(lra)) as [k [Hk Heq]].
 exists k; exact Heq.
Qed.

Theorem normal_grid_rank_unique base target i j :
  0<base -> normal_grid_from base i=target -> normal_grid_from base j=target -> i=j.
Proof.
 intros Hb Hi Hj.
 destruct (Nat.lt_trichotomy i j) as [Hlt|[Heq|Hgt]]; [|exact Heq|].
 - pose proof (normal_grid_increasing base i j Hb Hlt); lra.
 - pose proof (normal_grid_increasing base j i Hb Hgt); lra.
Qed.

Theorem normal_grid_endpoint_rank base target :
  0<base -> normal64 base -> format64 base -> format64 target -> base<=target ->
  target<=maxfinite64 ->
  exists n, normal_grid_from base n=target /\
    (forall k, (k<=n)%nat -> 0<normal_grid_from base k /\
      normal64 (normal_grid_from base k) /\ format64 (normal_grid_from base k) /\
      normal_grid_from base k<=maxfinite64) /\
    (forall m, normal_grid_from base m=target -> m=n).
Proof.
 intros Hb Hn Hbase Htarget Hlower Hfinite.
 destruct (normal_grid_reaches_representable base target Hb Hbase Htarget Hlower) as [n Heq].
 exists n; split; [exact Heq|split].
 - apply normal_grid_finite_segment; try assumption; rewrite Heq; exact Hfinite.
 - intros m Hm; apply (normal_grid_rank_unique base target m n Hb Hm Heq).
Qed.

Corollary grid64_reaches_every_normal target :
  0<target -> normal64 target -> format64 target -> exists n, grid64 n=target.
Proof.
 intros Hp Hn Hf; unfold grid64.
 apply normal_grid_reaches_representable; auto using minnormal64_positive,minnormal64_format.
 unfold normal64 in Hn; rewrite Rabs_pos_eq in Hn by lra; exact Hn.
Qed.

Print Assumptions normal_grid_endpoint_rank.
Print Assumptions grid64_reaches_every_normal.

Lemma positive_finite_format_maxfinite target :
  0<target -> finite64 target -> format64 target -> target<=maxfinite64.
Proof.
 intros Hp Hfinite Hformat.
 unfold finite64 in Hfinite; rewrite Rabs_pos_eq in Hfinite by lra.
 assert (Htop:format64 (bpow radix2 1024)).
 { unfold format64; apply generic_format_bpow; unfold FLT_exp; lia. }
 pose proof (pred_ge_gt radix2 (FLT_exp (-1074) 53) target (bpow radix2 1024)
   Hformat Htop Hfinite) as Hpred.
 rewrite pred_bpow in Hpred.
 change (target<=bpow radix2 1024-bpow radix2 971) in Hpred.
 exact Hpred.
Qed.

Theorem normal_grid_endpoint_rank_finite base target :
  0<base -> normal64 base -> format64 base -> format64 target -> base<=target ->
  finite64 target ->
  exists n, normal_grid_from base n=target /\
    (forall k, (k<=n)%nat -> 0<normal_grid_from base k /\
      normal64 (normal_grid_from base k) /\ format64 (normal_grid_from base k) /\
      normal_grid_from base k<=maxfinite64) /\
    (forall m, normal_grid_from base m=target -> m=n).
Proof.
 intros Hb Hn Hbase Htarget Hlower Hfinite.
 apply normal_grid_endpoint_rank; try assumption.
 apply positive_finite_format_maxfinite; try assumption; lra.
Qed.

Print Assumptions normal_grid_endpoint_rank_finite.
