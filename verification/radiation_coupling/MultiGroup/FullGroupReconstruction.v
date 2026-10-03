(* Exact reconstruction of the complete finite radiation vector, with the
   reduced-speed-of-light factor chi retained explicitly. *)
From Coq Require Import Reals Ranalysis5 Psatz Field Lia.
From BlackBox Require Import GasMap.
From MultiGroup Require Import ConstantGroupExistence.
Open Scope R_scope.

Lemma group_sum_ext n f k :
 (forall g, (g<n)%nat -> f g=k g) -> group_sum n f=group_sum n k.
Proof.
 induction n as [|n IH]; intro HE; simpl; [reflexivity|].
 assert (Hs:group_sum n f=group_sum n k) by (apply IH; intros; apply HE; lia).
 rewrite Hs, HE by lia; reflexivity.
Qed.

Lemma group_sum_scale_minus n f k a :
 group_sum n (fun g => a*(f g-k g))=a*(group_sum n f-group_sum n k).
Proof. induction n; simpl; [ring|rewrite IHn; ring]. Qed.

Lemma group_sum_minus n f k :
 group_sum n (fun g => f g-k g)=group_sum n f-group_sum n k.
Proof. induction n; simpl; [ring|rewrite IHn; ring]. Qed.

Lemma group_sum_scale n f a :
 group_sum n (fun g => a*f g)=a*group_sum n f.
Proof. induction n; simpl; [ring|rewrite IHn; ring]. Qed.

Definition group_energy h (alpha p r:nat->R) (B:nat->R->R) g x :=
 (r g+h*p g*B g x)/(1+h*alpha g).
Definition group_weight chi h (alpha p:nat->R) g :=
 chi*h*p g/(1+h*alpha g).
Definition group_absorption chi h (alpha r:nat->R) g :=
 chi*h*alpha g*r g/(1+h*alpha g).
Definition absorption_total n chi h alpha r :=
 group_sum n (group_absorption chi h alpha r).

Definition full_group_physical_solution A D T chi h n alpha p r B x t
 (E:nat->R) :=
 0<x /\ 0<t /\
 (forall g, (g<n)%nat -> 0<=E g) /\
 (forall g, (g<n)%nat -> E g-r g=h*(p g*B g x-alpha g*E g)) /\
 chi*group_sum n (fun g => E g-r g)=D*sqrt t*(t-x) /\
 collision A D T x t=0.

Section FullGroups.
 Variables A D T chi h:R.
 Variable n:nat.
 Variables alpha p r:nat->R.
 Variable B:nat->R->R.
 Hypotheses (HA:0<A) (HD:0<D) (HT:0<T) (Hchi:0<chi) (Hh:0<h).
 Hypotheses (Halpha:forall g, (g<n)%nat -> 0<=alpha g)
            (Hp:forall g, (g<n)%nat -> 0<=p g)
            (Hr:forall g, (g<n)%nat -> 0<=r g).
 Hypothesis HBnonnegative:forall g x, (g<n)%nat -> 0<x -> 0<=B g x.

 Let c := group_weight chi h alpha p.
 Let H := absorption_total n chi h alpha r.
 Let F := constant_group_residual A D T n c B H.
 Let Eg := group_energy h alpha p r B.

 Lemma group_den_positive g : (g<n)%nat -> 0<1+h*alpha g.
 Proof. intro Hg; pose proof (Halpha g Hg); nra. Qed.

 Lemma physical_weights_nonnegative g : (g<n)%nat -> 0<=c g.
 Proof.
  intro Hg; pose proof (Hp g Hg); pose proof (group_den_positive g Hg).
  unfold c,group_weight,Rdiv.
  apply Rmult_le_pos; [repeat apply Rmult_le_pos; lra|apply Rlt_le,Rinv_0_lt_compat; assumption].
 Qed.

 Lemma physical_absorption_nonnegative : 0<=H.
 Proof.
  unfold H,absorption_total; apply group_sum_nonnegative; intros g Hg.
  pose proof (Halpha g Hg); pose proof (Hr g Hg).
  pose proof (group_den_positive g Hg).
  unfold group_absorption,Rdiv; apply Rmult_le_pos;
   [repeat apply Rmult_le_pos; lra|apply Rlt_le,Rinv_0_lt_compat; assumption].
 Qed.

 Lemma reconstructed_group_nonnegative g x :
  (g<n)%nat -> 0<x -> 0<=Eg g x.
 Proof.
  intros Hg Hx; pose proof (Hp g Hg); pose proof (Hr g Hg).
  pose proof (HBnonnegative g x Hg Hx); pose proof (group_den_positive g Hg).
  unfold Eg,group_energy,Rdiv; apply Rmult_le_pos.
  - assert (0<=h*p g*B g x) by (repeat apply Rmult_le_pos; lra); lra.
  - apply Rlt_le,Rinv_0_lt_compat; assumption.
 Qed.

 Lemma group_equation_reconstruction g x e :
  (g<n)%nat ->
  (e-r g=h*(p g*B g x-alpha g*e) <-> e=Eg g x).
 Proof.
  intro Hg; pose proof (group_den_positive g Hg) as Hd.
  unfold Eg,group_energy; split; intro HE.
  - apply (Rmult_eq_reg_r (1+h*alpha g)); [|lra].
    field_simplify; nra.
  - apply (f_equal (fun y => y*(1+h*alpha g))) in HE.
    field_simplify in HE; nra.
 Qed.

 Lemma reconstructed_weighted_exchange g x :
  (g<n)%nat ->
  chi*(Eg g x-r g)=c g*B g x-group_absorption chi h alpha r g.
 Proof.
  intro Hg; pose proof (group_den_positive g Hg) as Hd.
  unfold Eg,group_energy,c,group_weight,group_absorption.
  field; lra.
 Qed.

 Lemma reconstructed_total_exchange x E :
  (forall g, (g<n)%nat -> E g=Eg g x) ->
  chi*group_sum n (fun g => E g-r g)=group_emission n c B x-H.
 Proof.
  intro HE; rewrite <- group_sum_scale.
  transitivity (group_sum n (fun g => c g*B g x-group_absorption chi h alpha r g)).
  - apply group_sum_ext; intros g Hg.
    rewrite HE by assumption; apply reconstructed_weighted_exchange; assumption.
  - rewrite group_sum_minus; reflexivity.
 Qed.

 Theorem full_group_scalar_bijection x t E :
  full_group_physical_solution A D T chi h n alpha p r B x t E <->
  0<x /\ F x=0 /\ t=gas_map A D T x /\
  forall g, (g<n)%nat -> E g=Eg g x.
 Proof.
  split.
  - intros [Hx [Ht [HEpos [HEgroup [Hdust Hcol]]]]].
    assert (HE:forall g, (g<n)%nat -> E g=Eg g x).
    { intros g Hg; apply (proj1 (group_equation_reconstruction g x (E g) Hg)); auto. }
    assert (Hg:gas_map A D T x=t).
    { apply gas_map_unique; try assumption.
      apply (proj1 (collision_forward A D T x t HD Ht)); assumption. }
    pose proof (reconstructed_total_exchange x E HE) as Hex.
    split; [exact Hx|]; split.
    + unfold F,constant_group_residual; rewrite Hg.
      unfold collision in Hcol; lra.
    + split; [symmetry; exact Hg|exact HE].
  - intros [Hx [HF [Ht HE]]]. subst t.
    assert (Hgaspos:0<gas_map A D T x)
     by (exact (proj1 (gas_map_spec A D T x HA HD HT Hx))).
    assert (Hcol:collision A D T x (gas_map A D T x)=0)
     by (apply gas_map_collision; assumption).
    pose proof (reconstructed_total_exchange x E HE) as Hex.
    unfold full_group_physical_solution; split; [exact Hx|]; split; [exact Hgaspos|].
    split.
    + intros g Hg; rewrite HE by assumption; apply reconstructed_group_nonnegative; assumption.
    + split.
      * intros g Hg; apply (proj2 (group_equation_reconstruction g x (E g) Hg)); auto.
      * split; [|exact Hcol].
        unfold F,constant_group_residual in HF; unfold collision in Hcol; lra.
 Qed.

 Theorem full_group_exact_energy_conservation x t E :
  full_group_physical_solution A D T chi h n alpha p r B x t E ->
  A*t+chi*group_sum n E=A*T+chi*group_sum n r.
 Proof.
  intros [_ [_ [_ [_ [Hdust Hcol]]]]].
  rewrite group_sum_minus in Hdust; unfold collision in Hcol; nra.
 Qed.

 Hypothesis HBmonotone:forall g x y, (g<n)%nat -> 0<x -> x<=y -> B g x<=B g y.
 Hypothesis HBcontinuous:forall g x, (g<n)%nat -> 0<x -> continuity_pt (B g) x.
 Hypothesis HBcontinuous0:forall g, (g<n)%nat -> continuity_pt (B g) 0.
 Hypothesis HBzero:forall g, (g<n)%nat -> B g 0=0.

 (* Radiation-vector uniqueness is componentwise over the physical finite
    index range. Values of an encoding function outside 0,...,n-1 are irrelevant. *)
 Theorem full_group_positive_solution_exists_unique :
  exists x t E, full_group_physical_solution A D T chi h n alpha p r B x t E /\
   forall y u V, full_group_physical_solution A D T chi h n alpha p r B y u V ->
    y=x /\ u=t /\ forall g, (g<n)%nat -> V g=E g.
 Proof.
  destruct (constant_group_positive_root_exists_unique A D T H n c B HA HD HT
   physical_absorption_nonnegative physical_weights_nonnegative HBnonnegative
   HBmonotone HBcontinuous HBcontinuous0 HBzero) as [x [Hx [HF HU]]].
  exists x,(gas_map A D T x),(fun g => Eg g x); split.
  - apply full_group_scalar_bijection; repeat split; auto.
  - intros y u V HY; apply full_group_scalar_bijection in HY.
    destruct HY as [Hy [HFy [Hu HE]]].
    assert (He:y=x) by (apply HU; assumption).
    subst y; split; [reflexivity|]; split; auto.
 Qed.
End FullGroups.

Print Assumptions full_group_scalar_bijection.
Print Assumptions full_group_exact_energy_conservation.
Print Assumptions full_group_positive_solution_exists_unique.
