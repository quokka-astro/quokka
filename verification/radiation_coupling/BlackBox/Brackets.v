(* Root containment and finite safeguarded search on a ranked finite grid.
   No convergence conclusion is drawn from an iteration budget or a small step. *)
From Coq Require Import Reals Psatz Lia Arith Bool Classical_Prop.
From BlackBox Require Import Guards.
Open Scope R_scope.

Definition root_bracket (a b root:R) : Prop := a<=root<=b.
Definition nondecreasing_on (f:R->R) (a b:R) : Prop :=
  forall x y, a<=x -> x<=y -> y<=b -> f x<=f y.

Lemma increasing_negative_before_root f a b root x :
  nondecreasing_on f a b -> root_bracket a b root -> a<=x<=b ->
  f root=0 -> f x<0 -> x<root.
Proof.
 intros Hmono Hr Hx Hroot Hsign.
 destruct (Rlt_le_dec x root) as [H|H]; [exact H|].
 unfold root_bracket in Hr.
 pose proof (Hmono root x ltac:(lra) H ltac:(lra)); lra.
Qed.

Lemma increasing_positive_after_root f a b root x :
  nondecreasing_on f a b -> root_bracket a b root -> a<=x<=b ->
  f root=0 -> 0<f x -> root<x.
Proof.
 intros Hmono Hr Hx Hroot Hsign.
 destruct (Rlt_le_dec root x) as [H|H]; [exact H|].
 unfold root_bracket in Hr.
 pose proof (Hmono x root ltac:(lra) H ltac:(lra)); lra.
Qed.

Theorem lower_update_preserves_root f a b root x :
  nondecreasing_on f a b -> root_bracket a b root -> a<=x<=b ->
  f root=0 -> f x<0 -> root_bracket x b root.
Proof.
 intros Hmono Hr Hx He Hz.
 pose proof (increasing_negative_before_root f a b root x Hmono Hr Hx He Hz).
 unfold root_bracket in *; lra.
Qed.

Theorem upper_update_preserves_root f a b root x :
  nondecreasing_on f a b -> root_bracket a b root -> a<=x<=b ->
  f root=0 -> 0<f x -> root_bracket a x root.
Proof.
 intros Hmono Hr Hx He Hz.
 pose proof (increasing_positive_after_root f a b root x Hmono Hr Hx He Hz).
 unfold root_bracket in *; lra.
Qed.

(* A decreasing balance is covered by f(x)=1-S(x); an increasing balance by
   f(x)=S(x)-1. Derivative/Newton signs play no role in these updates. *)
Theorem increasing_outer_guard_preserves_lower S a b root x shat :
  nondecreasing_on (fun y=>S y-1) a b -> root_bracket a b root -> a<=x<=b ->
  S root=1 -> 0<shat -> 0<S x -> shat<1-128*binary64_u ->
  Rabs (ln shat-ln (S x))<=71*binary64_lambda -> root_bracket x b root.
Proof.
 intros Hm Hr Hx He Hp Hs Hw Herr.
 apply (lower_update_preserves_root (fun y=>S y-1) a b root x); try assumption.
 - simpl; lra.
 - simpl; pose proof (outer_lower_sign shat (S x) Hp Hs Hw Herr); lra.
Qed.

Theorem increasing_outer_guard_preserves_upper S a b root x shat :
  nondecreasing_on (fun y=>S y-1) a b -> root_bracket a b root -> a<=x<=b ->
  S root=1 -> 0<shat -> 0<S x -> 1+128*binary64_u<shat ->
  Rabs (ln shat-ln (S x))<=71*binary64_lambda -> root_bracket a x root.
Proof.
 intros Hm Hr Hx He Hp Hs Hw Herr.
 apply (upper_update_preserves_root (fun y=>S y-1) a b root x); try assumption.
 - simpl; lra.
 - simpl; pose proof (outer_upper_sign shat (S x) Hp Hs Hw Herr); lra.
Qed.

Lemma bracket_log_error a b root x :
  0<a -> root_bracket a b root -> a<=x<=b ->
  Rabs (ln (x/root))<=ln (b/a).
Proof.
 intros Ha Hr Hx; unfold root_bracket in Hr.
 assert (Hroot:0<root) by lra; assert (Hxpos:0<x) by lra.
 assert (Hb:0<b) by lra.
 unfold Rdiv.
 rewrite (ln_mult x (/root)), (ln_Rinv root) by
   (try assumption; apply Rinv_0_lt_compat; assumption).
 rewrite (ln_mult b (/a)), (ln_Rinv a) by
   (try assumption; apply Rinv_0_lt_compat; assumption).
 pose proof (guard_ln_le a x Ha ltac:(lra)).
 pose proof (guard_ln_le x b Hxpos ltac:(lra)).
 pose proof (guard_ln_le a root Ha ltac:(lra)).
 pose proof (guard_ln_le root b Hroot ltac:(lra)).
 apply Rabs_le; lra.
Qed.

Theorem rounded_width_acceptance a b root x es ed width :
  0<a -> root_bracket a b root -> a<=x<=b ->
  Rabs es<=binary64_u -> Rabs ed<=binary64_u ->
  width=((b-a)*(1+es)/a)*(1+ed) -> width<=16*binary64_u ->
  Rabs (ln (x/root))<32*binary64_lambda.
Proof.
 intros Ha Hr Hx Hes Hed Hw Htest.
 pose proof (bracket_log_error a b root x Ha Hr Hx).
 pose proof (rounded_bracket_width a b es ed width Ha ltac:(unfold root_bracket in Hr; lra) Hes Hed Hw Htest).
 lra.
Qed.

(* Grid ranks count representable values. A concrete format must provide its
   increasing rank/value correspondence. Unlike real bisection, any strictly
   interior representable trial gives strict natural-number progress. *)
Definition grid_midpoint (a b:nat) : nat := (a+(b-a)/2)%nat.
Definition safeguarded_index (a b proposal:nat) : nat :=
  if (a <? proposal)%nat && (proposal <? b)%nat then proposal
  else grid_midpoint a b.

Lemma grid_midpoint_inside a b :
  (a+1<b)%nat -> (a<grid_midpoint a b<b)%nat.
Proof.
 intro H; unfold grid_midpoint.
 assert (0<(b-a)/2)%nat by (apply Nat.div_str_pos; lia).
 assert ((b-a)/2<b-a)%nat by (apply Nat.div_lt; lia).
 lia.
Qed.

Lemma safeguarded_index_inside a b proposal :
  (a+1<b)%nat -> (a<safeguarded_index a b proposal<b)%nat.
Proof.
 intro H; unfold safeguarded_index.
 destruct ((a <? proposal)%nat && (proposal <? b)%nat) eqn:Hsafe.
 - apply andb_true_iff in Hsafe as [Ha Hb].
   apply Nat.ltb_lt in Ha; apply Nat.ltb_lt in Hb; lia.
 - apply grid_midpoint_inside; assumption.
Qed.

Lemma interior_update_decreases_rank a b trial :
  (a<trial<b)%nat ->
  (b-trial<b-a)%nat /\ (trial-a<b-a)%nat.
Proof. lia. Qed.

(* This theorem states the precise progress obligation. A rejected proposal
   cannot leave the bracket unchanged indefinitely: the safeguarded trial must
   replace an endpoint unless a stopping condition holds. *)
Theorem finite_rank_descent (rank:nat->nat) (stop:nat->Prop) :
  (forall n, ~stop n -> (rank (S n)<rank n)%nat) ->
  exists n, (n<=rank O)%nat /\ stop n.
Proof.
 intro Hp.
 assert (Hbounded:forall fuel (r:nat->nat) (P:nat->Prop),
   (r O<=fuel)%nat ->
   (forall n, ~P n -> (r (S n)<r n)%nat) ->
   exists n, (n<=fuel)%nat /\ P n).
 { intro fuel; induction fuel as [|fuel IH]; intros r P Hfuel Hstep.
   - exists O; split; [lia|].
     destruct (classic (P O)) as [H|H]; [exact H|].
     pose proof (Hstep O H); exfalso; lia.
   - destruct (classic (P O)) as [H|H].
     + exists O; split; [lia|exact H].
     + assert (Hr:(r (S O)<=fuel)%nat) by (pose proof (Hstep O H); lia).
       destruct (IH (fun n=>r (S n)) (fun n=>P (S n)) Hr) as [n [Hn Hpn]].
       * intros n Hnot; exact (Hstep (S n) Hnot).
       * exists (S n); split; [lia|exact Hpn]. }
 apply (Hbounded (rank O) rank stop); [lia|exact Hp].
Qed.

Theorem safeguarded_grid_terminates
  (lower upper proposal:nat->nat) (accepted:nat->Prop) :
  (forall n, (lower n<=upper n)%nat) ->
  (forall n, ~accepted n -> (lower n+1<upper n)%nat ->
    let trial:=safeguarded_index (lower n) (upper n) (proposal n) in
    (lower (S n)=trial /\ upper (S n)=upper n) \/
    (lower (S n)=lower n /\ upper (S n)=trial)) ->
  exists n, (n<=upper O-lower O)%nat /\
    (accepted n \/ (upper n<=lower n+1)%nat).
Proof.
 intros Horder Hupdate.
 apply (finite_rank_descent (fun n=>(upper n-lower n)%nat)
   (fun n=>accepted n \/ (upper n<=lower n+1)%nat)).
 intros n Hnot.
 assert (Hacc:~accepted n) by tauto.
 assert (Hgap:(lower n+1<upper n)%nat) by
   (destruct (Nat.le_gt_cases (upper n) (lower n+1)); [tauto|lia]).
 pose proof (safeguarded_index_inside (lower n) (upper n) (proposal n) Hgap) as Htrial.
 specialize (Hupdate n Hacc Hgap); cbn zeta in Hupdate.
 destruct Hupdate as [[Hl Hu]|[Hl Hu]]; rewrite Hl, Hu; lia.
Qed.

(* A finite increasing value map transports rank adjacency to a certificate
   that there is no representable value strictly between the endpoints. *)
Theorem adjacent_ranks_have_no_grid_value (value:nat->R) a b :
  (forall i j, (i<j)%nat -> value i<value j) ->
  (b=a+1)%nat ->
  ~exists k, value a<value k<value b.
Proof.
 intros Hinc Hb [k [Hak Hkb]].
 destruct (Nat.le_gt_cases k a) as [Hka|Hka].
 - apply Nat.lt_eq_cases in Hka; destruct Hka as [Hka|Hka].
   + pose proof (Hinc k a Hka); lra.
   + subst; lra.
 - assert (Hbk:(b<=k)%nat) by lia.
   apply Nat.lt_eq_cases in Hbk; destruct Hbk as [Hbk|Hbk].
   + pose proof (Hinc b k Hbk); lra.
   + subst; lra.
Qed.

Print Assumptions lower_update_preserves_root.
Print Assumptions rounded_width_acceptance.
Print Assumptions safeguarded_grid_terminates.
