(* Finite gas-map sensitivity follows from the proved inverse derivative and
   the exact collision equation, rather than a postulated Lipschitz bound. *)
From Coq Require Import Reals Psatz Field.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap Algebra LogCalculus.
Open Scope R_scope.

Definition gas_derivative A D T0 x :=
  1/forward_derivative A D T0 (gas_map A D T0 x).
Definition gas_elasticity A D T0 x :=
  x*gas_derivative A D T0 x/gas_map A D T0 x.
Definition gas_energy C A D T0 x := C*gas_map A D T0 x.

Lemma forward_inverse_elasticity A D T0 t x :
 0<A -> 0<D -> 0<T0 -> 0<t -> forward_map A D T0 t=x ->
 x*(1/forward_derivative A D T0 t)/t =
 map_elasticity (D*sqrt t) A (T0/t).
Proof.
 intros HA HD HT Ht Hforward.
 assert (Hs:0<sqrt t) by (apply sqrt_lt_R0; exact Ht).
 assert (HK:0<D*sqrt t) by nra.
 assert (Hz:0<T0/t) by (apply Rdiv_lt_0_compat; assumption).
 pose proof (map_den_positive (D*sqrt t) A (T0/t) HK HA Hz) as Hden.
 pose proof (forward_derivative_positive A D T0 t HA HD HT Ht) as Hfd.
 rewrite <- Hforward.
 unfold forward_map,forward_derivative,map_elasticity in *.
 field; nra.
Qed.

Lemma forward_map_positive_numerator A D T0 t x :
 0<D -> 0<t -> 0<x -> forward_map A D T0 t=x ->
 0<D*sqrt t+A*(1-T0/t).
Proof.
 intros HD Ht Hx Hforward.
 assert (Hs:0<sqrt t) by (apply sqrt_lt_R0; exact Ht).
 assert (Heq:D*sqrt t+A*(1-T0/t)=x*D*sqrt t/t).
 { rewrite <- Hforward; unfold forward_map; field; nra. }
 rewrite Heq; apply Rdiv_lt_0_compat; [|exact Ht].
 repeat apply Rmult_lt_0_compat; assumption.
Qed.

Lemma gas_elasticity_identity A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x ->
 gas_elasticity A D T0 x =
 map_elasticity (D*sqrt(gas_map A D T0 x)) A (T0/gas_map A D T0 x).
Proof.
 intros HA HD HT Hx.
 destruct (gas_map_spec A D T0 x HA HD HT Hx) as [Ht Hforward].
 unfold gas_elasticity,gas_derivative.
 apply forward_inverse_elasticity; assumption.
Qed.

Lemma gas_elasticity_bounds A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x -> 0<gas_elasticity A D T0 x<=2.
Proof.
 intros HA HD HT Hx.
 destruct (gas_map_spec A D T0 x HA HD HT Hx) as [Ht Hforward].
 rewrite gas_elasticity_identity by assumption.
 apply map_elasticity_bounds; try assumption.
 - apply Rmult_lt_0_compat; [exact HD|apply sqrt_lt_R0; exact Ht].
 - apply Rdiv_lt_0_compat; assumption.
 - apply (forward_map_positive_numerator A D T0 _ x); assumption.
Qed.

Lemma gas_cooling_elasticity_bounds A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x -> x<=T0 ->
 0<gas_elasticity A D T0 x<=1.
Proof.
 intros HA HD HT Hx Hcool.
 pose proof (gas_elasticity_bounds A D T0 x HA HD HT Hx) as Hb.
 split; [lra|].
 rewrite gas_elasticity_identity by assumption.
 apply cooling_map_elasticity; try assumption.
 - apply Rmult_lt_0_compat; [exact HD|apply sqrt_lt_R0].
   exact (proj1 (gas_map_spec A D T0 x HA HD HT Hx)).
 - pose proof (gas_map_between A D T0 x HA HD HT Hx) as Hbetween.
   rewrite Rmin_right,Rmax_left in Hbetween by exact Hcool.
   destruct (gas_map_spec A D T0 x HA HD HT Hx) as [Ht Hforward].
   apply (Rmult_le_reg_r (gas_map A D T0 x)); [exact Ht|].
   field_simplify; lra.
Qed.

Lemma positive_closed_interval a b t :
 0<a -> 0<b -> Rmin a b<=t<=Rmax a b -> 0<t.
Proof.
 intros Ha Hb Ht; unfold Rmin in Ht; destruct (Rle_dec a b); lra.
Qed.

Lemma gas_map_log_lipschitz A D T0 x y :
 0<A -> 0<D -> 0<T0 -> 0<x -> 0<y ->
 log_error (gas_map A D T0 y) (gas_map A D T0 x)<=2*log_error y x.
Proof.
 intros HA HD HT Hx Hy.
 apply (elasticity_upper_distance (gas_map A D T0)
         (gas_derivative A D T0) x y 2 Hx Hy ltac:(lra)).
 - intros t Ht.
   exact (proj1 (gas_map_spec A D T0 t HA HD HT
                  (positive_closed_interval x y t Hx Hy Ht))).
 - intros t Ht; apply gas_map_continuous; try assumption.
   apply (positive_closed_interval x y t Hx Hy Ht).
 - intros t Ht; apply gas_map_is_derive; try assumption.
   apply (positive_closed_interval x y t Hx Hy); lra.
 - intros t Ht.
   change (Rabs(gas_elasticity A D T0 t)<=2).
   pose proof (gas_elasticity_bounds A D T0 t HA HD HT
      (positive_closed_interval x y t Hx Hy ltac:(lra))) as Hb.
   rewrite Rabs_right by lra; lra.
Qed.

Lemma gas_map_cooling_log_lipschitz A D T0 x y :
 0<A -> 0<D -> 0<T0 -> 0<x -> 0<y -> x<=T0 -> y<=T0 ->
 log_error (gas_map A D T0 y) (gas_map A D T0 x)<=log_error y x.
Proof.
 intros HA HD HT Hx Hy Hxc Hyc.
 replace (log_error y x) with (1*log_error y x) by ring.
 apply (elasticity_upper_distance (gas_map A D T0)
         (gas_derivative A D T0) x y 1 Hx Hy ltac:(lra)).
 - intros t Ht.
   exact (proj1 (gas_map_spec A D T0 t HA HD HT
                  (positive_closed_interval x y t Hx Hy Ht))).
 - intros t Ht; apply gas_map_continuous; try assumption.
   apply (positive_closed_interval x y t Hx Hy Ht).
 - intros t Ht; apply gas_map_is_derive; try assumption.
   apply (positive_closed_interval x y t Hx Hy); lra.
 - intros t Ht.
   change (Rabs(gas_elasticity A D T0 t)<=1).
   assert (Htc:t<=T0).
   { unfold Rmax in Ht; destruct (Rle_dec x y); lra. }
   pose proof (gas_cooling_elasticity_bounds A D T0 t HA HD HT
      (positive_closed_interval x y t Hx Hy ltac:(lra)) Htc) as Hb.
   rewrite Rabs_right by lra; lra.
Qed.

Lemma gas_energy_positive C A D T0 x :
 0<C -> 0<A -> 0<D -> 0<T0 -> 0<x -> 0<gas_energy C A D T0 x.
Proof.
 intros HC HA HD HT Hx; unfold gas_energy.
 apply Rmult_lt_0_compat; [exact HC|].
 exact (proj1 (gas_map_spec A D T0 x HA HD HT Hx)).
Qed.

Lemma gas_energy_log_error C A D T0 x y :
 0<C -> 0<A -> 0<D -> 0<T0 -> 0<x -> 0<y ->
 log_error (gas_energy C A D T0 y) (gas_energy C A D T0 x)=
 log_error (gas_map A D T0 y) (gas_map A D T0 x).
Proof.
 intros HC HA HD HT Hx Hy.
 pose proof (proj1 (gas_map_spec A D T0 x HA HD HT Hx)) as Htx.
 pose proof (proj1 (gas_map_spec A D T0 y HA HD HT Hy)) as Hty.
 unfold log_error,gas_energy.
 rewrite !ln_mult by assumption.
 f_equal; ring.
Qed.

Lemma gas_energy_log_lipschitz C A D T0 x y :
 0<C -> 0<A -> 0<D -> 0<T0 -> 0<x -> 0<y ->
 log_error (gas_energy C A D T0 y) (gas_energy C A D T0 x)<=2*log_error y x.
Proof.
 intros; rewrite gas_energy_log_error by assumption.
 apply gas_map_log_lipschitz; assumption.
Qed.

Lemma gas_energy_cooling_log_lipschitz C A D T0 x y :
 0<C -> 0<A -> 0<D -> 0<T0 -> 0<x -> 0<y -> x<=T0 -> y<=T0 ->
 log_error (gas_energy C A D T0 y) (gas_energy C A D T0 x)<=log_error y x.
Proof.
 intros; rewrite gas_energy_log_error by assumption.
 apply gas_map_cooling_log_lipschitz; assumption.
Qed.
