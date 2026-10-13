(* Early equilibrium acceptance is an accuracy test for the component outputs. *)
From Coq Require Import Reals Psatz Field.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap OuterDerivatives PhysicalRoot LogCalculus Guards.
Open Scope R_scope.

Lemma log_error_between t x y : 0<x -> 0<y ->
 Rmin x y<=t<=Rmax x y -> log_error t x<=log_error y x.
Proof.
 intros Hx Hy Ht. unfold log_error.
 destruct (Rle_dec x y) as [Hxy|Hyx].
 - rewrite Rmin_left,Rmax_right in Ht by lra.
   pose proof (ln_le x t Hx ltac:(lra)).
   pose proof (ln_le t y ltac:(lra) ltac:(lra)).
   rewrite !Rabs_right by lra; lra.
 - rewrite Rmin_right,Rmax_left in Ht by lra.
   pose proof (ln_le y t Hy ltac:(lra)).
   pose proof (ln_le t x ltac:(lra) ltac:(lra)).
   rewrite !Rabs_left1 by lra; lra.
Qed.

Lemma radiation_weighted_bounds tau r B lo hi : 0<tau ->
 lo<=r<=hi -> lo<=B<=hi ->
 lo<=(r+tau*B)/(1+tau)<=hi.
Proof.
 intros Ht Hr HB. split; apply (Rmult_le_reg_r (1+tau));
   try lra; field_simplify; nra.
Qed.

Lemma emission_between a x T0 Tr : 0<a -> 0<T0 -> 0<Tr ->
 Rmin T0 Tr<=x<=Rmax T0 Tr ->
 Rmin (emission a T0) (emission a Tr)<=emission a x<=
 Rmax (emission a T0) (emission a Tr).
Proof.
 intros Ha HT Hr Hx.
 destruct (Rle_dec T0 Tr) as [HL|HG].
 - rewrite Rmin_left,Rmax_right in Hx by lra.
   assert (H1:emission a T0<=emission a x).
   { unfold emission; apply Rmult_le_compat_l; [lra|apply pow_incr; lra]. }
   assert (H2:emission a x<=emission a Tr).
   { unfold emission; apply Rmult_le_compat_l; [lra|apply pow_incr; lra]. }
   rewrite Rmin_left,Rmax_right by lra; lra.
 - rewrite Rmin_right,Rmax_left in Hx by lra.
   assert (H1:emission a Tr<=emission a x).
   { unfold emission; apply Rmult_le_compat_l; [lra|apply pow_incr; lra]. }
   assert (H2:emission a x<=emission a T0).
   { unfold emission; apply Rmult_le_compat_l; [lra|apply pow_incr; lra]. }
   rewrite Rmin_right,Rmax_left by lra; lra.
Qed.

Lemma radiation_between_initial C a r kap T0 Tr x :
 0<C -> 0<a -> 0<r -> 0<T0 -> 0<Tr -> 0<kap x ->
 emission a Tr=r -> Rmin T0 Tr<=x<=Rmax T0 Tr ->
 Rmin r (emission a T0)<=OuterDerivatives.radiation C a r kap x<=
 Rmax r (emission a T0).
Proof.
 intros HC Ha Hr HT HTr Hk HB Hx.
 pose proof (emission_between a x T0 Tr Ha HT HTr Hx) as HBx.
 rewrite HB,Rmin_comm,Rmax_comm in HBx.
 unfold OuterDerivatives.radiation; apply radiation_weighted_bounds.
 - apply optical_depth_positive; assumption.
 - split; [apply Rmin_l|apply Rmax_l].
 - exact HBx.
Qed.

Lemma gas_in_physical_interval A D T0 Tr x :
 0<A -> 0<D -> 0<T0 -> 0<Tr ->
 Rmin T0 Tr<=x<=Rmax T0 Tr ->
 Rmin T0 Tr<=gas_map A D T0 x<=Rmax T0 Tr.
Proof.
 intros HA HD HT Hr Hx.
 assert (Hxp:0<x).
 { unfold Rmin in Hx; destruct (Rle_dec T0 Tr); lra. }
 pose proof (gas_map_between A D T0 x HA HD HT Hxp) as Hg.
 unfold Rmin,Rmax in *.
 repeat destruct Rle_dec; lra.
Qed.

Lemma initial_temperature_log_identity a r T0 Tr :
 0<a -> 0<r -> 0<T0 -> 0<Tr -> emission a Tr=r ->
 log_error Tr T0=Rabs(ln(emission a T0/r))/4.
Proof.
 intros Ha Hr HT HTr HB.
 assert (HP0:0<T0^4) by (apply pow_lt; assumption).
 assert (HPr:0<Tr^4) by (apply pow_lt; assumption).
 unfold log_error. rewrite ln_div by (try exact Hr; apply emission_positive; assumption).
 unfold emission in *. rewrite <- HB at 1.
 rewrite !ln_mult by assumption. rewrite !ln_pow by assumption.
 replace (INR 4) with 4 by (simpl; ring). replace (ln a+4*ln T0-(ln a+4*ln Tr)) with
   (4*(ln T0-ln Tr)) by ring.
 rewrite Rabs_mult, (Rabs_right 4) by lra.
 rewrite Rabs_minus_sym. field.
Qed.

Lemma early_equilibrium_component_bounds A D T0 C a r kap Tr x b :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<a -> 0<r -> 0<Tr ->
 0<kap x -> emission a Tr=r ->
 Rmin T0 Tr<=x<=Rmax T0 Tr ->
 Rabs(ln(emission a T0/r))<=b ->
 log_error T0 x<=b/4 /\
 log_error (A*T0) (A*gas_map A D T0 x)<=b/4 /\
 log_error r (OuterDerivatives.radiation C a r kap x)<=b.
Proof.
 intros HA HD HT HC Ha Hr HTr Hk HB Hx Hbound.
 assert (Hxp:0<x).
 { unfold Rmin in Hx; destruct (Rle_dec T0 Tr); lra. }
 pose proof (gas_map_spec A D T0 x HA HD HT Hxp) as [Hgp HF].
 pose proof (initial_temperature_log_identity a r T0 Tr Ha Hr HT HTr HB) as Hid.
 pose proof (log_error_between x T0 Tr HT HTr Hx) as Hdx.
 pose proof (gas_in_physical_interval A D T0 Tr x HA HD HT HTr Hx) as Hgb.
 pose proof (log_error_between (gas_map A D T0 x) T0 Tr HT HTr Hgb) as Hdg.
 assert (HB0:0<emission a T0) by (apply emission_positive; assumption).
 pose proof (radiation_between_initial C a r kap T0 Tr x HC Ha Hr HT HTr Hk HB Hx) as HRb.
 pose proof (log_error_between (OuterDerivatives.radiation C a r kap x)
   r (emission a T0) Hr HB0 HRb) as HDR.
 rewrite (log_error_ratio (emission a T0) r HB0 Hr) in HDR.
 split.
 - rewrite log_error_sym; nra.
 - split.
   + unfold log_error. rewrite !ln_mult by assumption.
     replace (ln A+ln T0-(ln A+ln(gas_map A D T0 x)))
       with (ln T0-ln(gas_map A D T0 x)) by ring.
     rewrite Rabs_minus_sym. unfold log_error in Hdg,Hid; nra.
   + rewrite log_error_sym; exact (Rle_trans _ _ _ HDR Hbound).
Qed.

Lemma early_binary64_component_bounds A D T0 C a r kap Tr x zhat :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<a -> 0<r -> 0<Tr ->
 0<kap x -> emission a Tr=r -> Rmin T0 Tr<=x<=Rmax T0 Tr ->
 1-64*binary64_u<=zhat<=1+64*binary64_u ->
 Rabs(ln zhat-ln(emission a T0/r))<=5*binary64_lambda ->
 log_error T0 x<=(35/2)*binary64_lambda /\
 log_error (A*T0) (A*gas_map A D T0 x)<=(35/2)*binary64_lambda /\
 log_error r (OuterDerivatives.radiation C a r kap x)<=70*binary64_lambda.
Proof.
 intros HA HD HT HC Ha Hr HTr Hk HB Hx Hwin Herr.
 pose proof (initial_exact_window zhat (emission a T0/r) Hwin Herr) as Hz.
 pose proof (early_equilibrium_component_bounds A D T0 C a r kap Tr x
   (70*binary64_lambda) HA HD HT HC Ha Hr HTr Hk HB Hx Hz) as H.
 replace (70*binary64_lambda/4) with ((35/2)*binary64_lambda) in H by field.
 exact H.
Qed.
