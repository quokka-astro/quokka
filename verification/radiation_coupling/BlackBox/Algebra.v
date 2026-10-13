(* Exact algebra for the new nested scalar design. No numerical assumptions. *)
From Coq Require Import Reals Psatz Field.
Open Scope R_scope.

Definition map_elasticity K A z := (K+A*(1-z))/(K+A*(1+z)/2).

Lemma map_den_positive K A z : 0<K -> 0<A -> 0<z ->
 0<K+A*(1+z)/2.
Proof. intros; nra. Qed.

Lemma map_elasticity_bounds K A z : 0<K -> 0<A -> 0<z ->
 0<K+A*(1-z) -> 0<map_elasticity K A z<=2.
Proof.
 intros HK HA Hz Hn.
 pose proof (map_den_positive K A z HK HA Hz) as Hd.
 unfold map_elasticity; split.
 - apply Rdiv_lt_0_compat; assumption.
 - apply (Rmult_le_reg_r (K+A*(1+z)/2)); [lra|].
   field_simplify; nra.
Qed.

Lemma cooling_map_elasticity K A z : 0<K -> 0<A -> 1<=z ->
 map_elasticity K A z<=1.
Proof.
 intros HK HA Hz.
 pose proof (map_den_positive K A z HK HA ltac:(lra)) as Hd.
 unfold map_elasticity.
 apply (Rmult_le_reg_r (K+A*(1+z)/2)); [lra|].
 field_simplify; nra.
Qed.

Lemma heating_transfer_elasticity K A z : 0<K -> 0<A -> 0<z<1 ->
 1<=map_elasticity K A z/(1-z).
Proof.
 intros HK HA Hz.
 pose proof (map_den_positive K A z HK HA ltac:(lra)) as Hd.
 unfold map_elasticity.
 assert (HKz:0<K*z) by nra.
 assert (HAz:0<=A*(1-z)^2) by (apply Rmult_le_pos; [lra|apply pow2_ge_0]).
 apply (Rmult_le_reg_r ((K+A*(1+z)/2)*(1-z))); [nra|].
 field_simplify; nra.
Qed.

Lemma heating_margin w g ell mu :
 0<=w<=1 -> 1<=g -> mu<=1-ell -> mu<=4 ->
 mu<=w*(g-ell)+(1-w)*4.
Proof.
 intros Hw Hg He H4.
 assert (0<=w*(g-ell-mu)) by (apply Rmult_le_pos; lra).
 assert (0<=(1-w)*(4-mu)) by (apply Rmult_le_pos; lra).
 nra.
Qed.

Lemma cooling_margin w k ell mu :
 0<=w<=1 -> 0<=k -> mu<=4 -> mu<=4+ell ->
 mu<=4+ell*w+k.
Proof.
 intros Hw Hk H4 He.
 assert (0<=w*(4+ell-mu)) by (apply Rmult_le_pos; lra).
 assert (0<=(1-w)*(4-mu)) by (apply Rmult_le_pos; lra).
 nra.
Qed.

Lemma example_effective_slope p tau :
 -(7/2)<=p<=1/2 -> 0<=tau -> -(7/2)<=p/(1+tau)<=1/2.
Proof.
 intros Hp Ht; split.
 - apply (Rmult_le_reg_r (1+tau)); [lra|]. field_simplify; nra.
 - apply (Rmult_le_reg_r (1+tau)); [lra|]. field_simplify; nra.
Qed.

Lemma positive_fraction q b : 0<=q -> 0<b -> 0<=q/(q+b)<=1.
Proof.
 intros Hq Hb; split.
 - unfold Rdiv; apply Rmult_le_pos; [lra|left; apply Rinv_0_lt_compat; lra].
 - apply (Rmult_le_reg_r (q+b)); [lra|]. field_simplify; nra.
Qed.

Lemma radiation_slope_bound tau B r p P :
 0<tau -> 0<B -> 0<r -> 0<=P -> Rabs p<=P ->
 Rabs (4*tau*B/(r+tau*B)+
 p*(tau*(B-r)/((1+tau)*(r+tau*B)))) <= 4+P.
Proof.
 intros Ht HB Hr HP Hp.
 assert (Hd:0<r+tau*B) by nra.
 assert (HD:0<(1+tau)*(r+tau*B)) by nra.
 assert (Ha:0<=4*tau*B/(r+tau*B)<=4).
 { split; [unfold Rdiv; apply Rmult_le_pos; [nra|left; apply Rinv_0_lt_compat; nra]|].
   apply (Rmult_le_reg_r (r+tau*B)); [lra|]. field_simplify; nra. }
 assert (Hb:Rabs(tau*(B-r)/((1+tau)*(r+tau*B)))<=1).
 { apply Rabs_le; split;
   [apply (Rmult_le_reg_r ((1+tau)*(r+tau*B)))|
    apply (Rmult_le_reg_r ((1+tau)*(r+tau*B)))];
   try lra; field_simplify; nra. }
 eapply Rle_trans; [apply Rabs_triang|].
 rewrite Rabs_mult.
 rewrite Rabs_pos_eq by lra.
 pose proof (Rabs_pos (tau*(B-r)/((1+tau)*(r+tau*B)))).
 pose proof (Rabs_pos p). nra.
Qed.

Lemma inner_heating_elasticity w s : 0<=w<=1 -> 0<=s<=1 ->
 1/2<=1-w*s/2<=1.
Proof. intros; assert (0<=w*s) by nra; assert (w*s<=1) by nra; lra. Qed.
Lemma inner_weak_elasticity w s : 0<=w<=1 -> 0<=s<=1 ->
 1<=1+w*s/2<=3/2.
Proof. intros; assert (0<=w*s) by nra; assert (w*s<=1) by nra; lra. Qed.
Lemma inner_strong_elasticity w s : 0<=w<=1 -> 0<=s<=1 ->
 1<=1+w*(s+1/2)<=5/2.
Proof. intros; assert (0<=w*(s+1/2)) by nra;
 assert (w*(s+1/2)<=3/2) by nra; lra. Qed.

Print Assumptions heating_margin.
Print Assumptions radiation_slope_bound.
