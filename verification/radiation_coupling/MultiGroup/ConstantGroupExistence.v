(* Finite-group constant-coefficient scalar reduction.
   This supplement proves real-arithmetic existence and uniqueness.  It does
   not claim an RN64 implementation theorem or prove the analytic properties
   of Planck band integrals; those properties are explicit hypotheses.
   Existing BlackBox sources are imported unchanged. *)
From Coq Require Import Reals Ranalysis5 Psatz Field Lia.
From BlackBox Require Import GasMap.
Open Scope R_scope.

(* Exactly n groups, indexed 0,...,n-1; in particular n=0 is permitted. *)
Fixpoint group_sum (n:nat) (f:nat -> R) : R :=
 match n with O => 0 | S k => group_sum k f + f k end.

Lemma group_sum_nonnegative n f :
 (forall g, (g<n)%nat -> 0<=f g) -> 0<=group_sum n f.
Proof.
 induction n as [|n IH]; intro H; simpl; [lra|].
 assert (Hn:0<=f n) by (apply H; lia).
 assert (Hs:0<=group_sum n f) by (apply IH; intros; apply H; lia).
 lra.
Qed.

Lemma group_sum_nondecreasing n f h :
 (forall g, (g<n)%nat -> f g<=h g) -> group_sum n f<=group_sum n h.
Proof.
 induction n as [|n IH]; intro H; simpl; [lra|].
 assert (Hn:f n<=h n) by (apply H; lia).
 assert (Hs:group_sum n f<=group_sum n h) by
  (apply IH; intros; apply H; lia).
 lra.
Qed.

Lemma group_sum_zero n f :
 (forall g, (g<n)%nat -> f g=0) -> group_sum n f=0.
Proof.
 induction n as [|n IH]; intro H; simpl; [reflexivity|].
 rewrite IH, H; try (intros; apply H; lia); try lia; ring.
Qed.

Lemma group_sum_continuous n (f:nat -> R -> R) x :
 (forall g, (g<n)%nat -> continuity_pt (f g) x) ->
 continuity_pt (fun y => group_sum n (fun g => f g y)) x.
Proof.
 induction n as [|n IH]; intro H; simpl.
 - apply continuity_pt_const; unfold constant; reflexivity.
 - apply continuity_pt_plus.
   + apply IH; intros; apply H; lia.
   + apply H; lia.
Qed.

Definition group_emission n (c:nat->R) (B:nat->R->R) x :=
 group_sum n (fun g => c g * B g x).
Definition constant_group_residual A D T n c B H x :=
 A*(gas_map A D T x-T)+group_emission n c B x-H.

Lemma group_emission_nonnegative n c B x :
 (forall g, (g<n)%nat -> 0<=c g) ->
 (forall g, (g<n)%nat -> 0<=B g x) ->
 0<=group_emission n c B x.
Proof.
 intros Hc HB; apply group_sum_nonnegative; intros g Hg.
 apply Rmult_le_pos; auto.
Qed.

Lemma group_emission_nondecreasing n c B x y :
 (forall g, (g<n)%nat -> 0<=c g) ->
 (forall g, (g<n)%nat -> B g x<=B g y) ->
 group_emission n c B x<=group_emission n c B y.
Proof.
 intros Hc HB; apply group_sum_nondecreasing; intros g Hg.
 apply Rmult_le_compat_l; auto.
Qed.

Lemma group_emission_zero n c B :
 (forall g, (g<n)%nat -> B g 0=0) -> group_emission n c B 0=0.
Proof.
 intro HB; apply group_sum_zero; intros g Hg; rewrite HB by assumption; ring.
Qed.

Lemma group_emission_continuous n c B x :
 (forall g, (g<n)%nat -> continuity_pt (B g) x) ->
 continuity_pt (group_emission n c B) x.
Proof.
 intro HB; unfold group_emission; apply group_sum_continuous; intros g Hg.
 apply continuity_pt_mult.
 - apply continuity_pt_const; unfold constant; reflexivity.
 - apply HB; exact Hg.
Qed.

Section ConstantGroups.
 Variables A D T H : R.
 Variable n : nat.
 Variable c : nat -> R.
 Variable B : nat -> R -> R.
 Hypotheses (HA:0<A) (HD:0<D) (HT:0<T) (HH:0<=H).
 Hypothesis Hc:forall g, (g<n)%nat -> 0<=c g.
 Hypothesis HBnonnegative:forall g x, (g<n)%nat -> 0<x -> 0<=B g x.
 Hypothesis HBmonotone:forall g x y, (g<n)%nat -> 0<x -> x<=y -> B g x<=B g y.
 Hypothesis HBcontinuous:forall g x, (g<n)%nat -> 0<x -> continuity_pt (B g) x.
 Hypothesis HBcontinuous0:forall g, (g<n)%nat -> continuity_pt (B g) 0.
 Hypothesis HBzero:forall g, (g<n)%nat -> B g 0=0.

 Let F := constant_group_residual A D T n c B H.
 Let S := group_emission n c B.

 Theorem constant_group_strictly_increasing x y :
  0<x -> x<y -> F x<F y.
 Proof.
  intros Hx Hxy.
  pose proof (gas_map_increasing A D T x y HA HD HT Hx Hxy) as Hgas.
  assert (Hem:S x<=S y).
  { apply group_emission_nondecreasing; [exact Hc|].
    intros g Hg; apply HBmonotone; auto; lra. }
  unfold F,constant_group_residual; unfold S in Hem; nra.
 Qed.

 Lemma constant_group_continuous x : 0<x -> continuity_pt F x.
 Proof.
  intro Hx; unfold F,constant_group_residual.
  apply continuity_pt_minus.
  - apply continuity_pt_plus.
    + apply continuity_pt_mult.
      * apply continuity_pt_const; unfold constant; reflexivity.
      * apply continuity_pt_minus.
        -- apply gas_map_continuous; assumption.
        -- apply continuity_pt_const; unfold constant; reflexivity.
    + apply group_emission_continuous; intros g Hg; apply HBcontinuous; assumption.
  - apply continuity_pt_const; unfold constant; reflexivity.
 Qed.

 (* A positive lower point is derived, rather than postulated.  At T/2 the
    gas is strictly cooler than T.  For sufficiently small positive dust
    temperature the continuous finite emission sum cannot offset that loss. *)
 Lemma constant_group_negative_lower : exists l, 0<l /\ l<=T/2 /\ F l<0.
 Proof.
  set (m := A*(T-gas_map A D T (T/2))).
  assert (Hcool:gas_map A D T (T/2)<T).
  { pose proof (gas_map_cooling_order A D T (T/2) HA HD ltac:(lra) ltac:(lra)); lra. }
  assert (Hm:0<m) by (unfold m; nra).
  assert (HC:continuity_pt S 0).
  { apply group_emission_continuous; exact HBcontinuous0. }
  assert (HZ:S 0=0) by (apply group_emission_zero; exact HBzero).
  unfold continuity_pt,continue_in,limit1_in,limit_in in HC.
  destruct (HC m Hm) as [delta [Hd Hnear]].
  set (l := Rmin (T/2) (delta/2)).
  assert (Hl:0<l) by (unfold l; apply Rmin_pos; lra).
  assert (HlT:l<=T/2) by (apply Rmin_l).
  assert (Hld:l<delta) by (pose proof (Rmin_r (T/2) (delta/2)); unfold l; lra).
  assert (HSm:S l<m).
  { assert (Hdist:dist R_met l 0<delta).
    { change (Rabs (l-0)<delta); rewrite Rabs_right; lra. }
    assert (HDx:D_x no_cond 0 l) by (unfold D_x,no_cond; split; [exact I|lra]).
    specialize (Hnear l (conj HDx Hdist)).
    change (Rabs (S l-S 0)<m) in Hnear.
    rewrite HZ, Rminus_0_r in Hnear.
    pose proof (Rle_abs (S l)); lra. }
  assert (Hgas:gas_map A D T l<=gas_map A D T (T/2)).
  { apply gas_map_nondecreasing; assumption. }
  exists l; repeat split; try assumption.
  unfold F,constant_group_residual; unfold S in HSm; unfold m in HSm; nra.
 Qed.

 (* This sharp finite upper bound also covers H=0 and all c_g=0. *)
 Definition constant_group_upper :=
  forward_map A D T (T+H/A).

 Lemma constant_group_upper_spec :
  T<=constant_group_upper /\ 0<=F constant_group_upper.
 Proof.
  assert (Hdiv:0<=H/A) by (unfold Rdiv; apply Rmult_le_pos; [exact HH|apply Rlt_le,Rinv_0_lt_compat; exact HA]).
  assert (Hhi:0<T+H/A) by lra.
  assert (Hs:0<sqrt (T+H/A)) by (apply sqrt_lt_R0; exact Hhi).
  assert (Hden:0<D*sqrt (T+H/A)) by nra.
  assert (Hfrac:0<=A*(T+H/A-T)/(D*sqrt (T+H/A)))
   by (unfold Rdiv; apply Rmult_le_pos; [nra|apply Rlt_le,Rinv_0_lt_compat; exact Hden]).
  assert (Hu:T<=constant_group_upper) by (unfold constant_group_upper,forward_map; lra).
  assert (Hgu:gas_map A D T constant_group_upper=T+H/A).
  { unfold constant_group_upper; apply gas_map_left_inverse; try assumption.
    fold constant_group_upper; lra. }
  assert (Hem:0<=S constant_group_upper).
  { apply group_emission_nonnegative; [exact Hc|].
    intros g Hg; apply HBnonnegative; auto; lra. }
  split; [exact Hu|].
  unfold F,constant_group_residual; rewrite Hgu.
  assert (He:A*(T+H/A-T)=H) by (field; lra).
  rewrite He; unfold S in Hem; lra.
 Qed.

 Theorem constant_group_positive_root_exists_unique :
  exists x, 0<x /\ F x=0 /\
   forall y, 0<y -> F y=0 -> y=x.
 Proof.
  destruct constant_group_negative_lower as [l [Hl [HlT HFl]]].
  destruct constant_group_upper_spec as [Hu HFu].
  assert (Hlu:l<constant_group_upper) by lra.
  destruct (f_interv_is_interv F l constant_group_upper 0 Hlu)
   as [x [Hx HF]].
  - split; lra.
  - intros y Hy; apply constant_group_continuous; lra.
  - exists x; split; [lra|]; split; [exact HF|].
    intros y Hy HFy.
    destruct (Rtotal_order y x) as [Hlt|[Heq|Hgt]]; [|exact Heq|].
    + pose proof (constant_group_strictly_increasing y x Hy Hlt); lra.
    + pose proof (constant_group_strictly_increasing x y ltac:(lra) Hgt); lra.
 Qed.

 (* Exact physical two-temperature equations after the finite radiation
    variables have been algebraically eliminated. *)
 Definition constant_group_physical_solution x t :=
  0<x /\ 0<t /\ collision A D T x t=0 /\ A*(t-T)+S x-H=0.

 Theorem constant_group_physical_reduction x t :
  constant_group_physical_solution x t <->
  0<x /\ F x=0 /\ t=gas_map A D T x.
 Proof.
  split.
  - intros [Hx [Ht [Hcol HE]]].
    assert (Hg:gas_map A D T x=t).
    { apply gas_map_unique; try assumption.
      apply (proj1 (collision_forward A D T x t HD Ht)); exact Hcol. }
    split; [exact Hx|]; split; [|symmetry; exact Hg].
    unfold F,constant_group_residual; rewrite Hg; exact HE.
  - intros [Hx [HF Ht]]. subst t.
    unfold constant_group_physical_solution; repeat split; try assumption.
    + exact (proj1 (gas_map_spec A D T x HA HD HT Hx)).
    + apply gas_map_collision; assumption.
 Qed.

 Theorem constant_group_positive_physical_exists_unique :
  exists x t, constant_group_physical_solution x t /\
   forall y u, constant_group_physical_solution y u -> y=x /\ u=t.
 Proof.
  destruct constant_group_positive_root_exists_unique as [x [Hx [HF HU]]].
  exists x,(gas_map A D T x); split.
  - apply constant_group_physical_reduction; repeat split; auto.
  - intros y u HY.
    apply constant_group_physical_reduction in HY.
    destruct HY as [Hy [HFy Hu]].
    assert (He:y=x) by (apply HU; assumption).
    subst y; split; auto.
 Qed.
End ConstantGroups.

Print Assumptions constant_group_strictly_increasing.
Print Assumptions constant_group_positive_root_exists_unique.
Print Assumptions constant_group_physical_reduction.
Print Assumptions constant_group_positive_physical_exists_unique.
