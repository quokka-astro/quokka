(* Existence for the specified physical equations, independent of opacity slopes. *)
From Coq Require Import Reals Ranalysis5 Psatz Field ClassicalEpsilon.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap OuterDerivatives LogCalculus.
Open Scope R_scope.

Definition radiation_temperature a r := sqrt (sqrt (r/a)).
Definition source_energy A D T0 C chi a r kap x :=
 A*(gas_map A D T0 x-T0)+coupling C chi kap x*(emission a x-r).

Lemma radiation_temperature_spec a r : 0<a -> 0<r ->
 0<radiation_temperature a r /\ emission a (radiation_temperature a r)=r.
Proof.
 intros Ha Hr.
 assert (Hq:0<r/a) by (apply Rdiv_lt_0_compat; assumption).
 pose proof (sqrt_lt_R0 (r/a) Hq) as Hs.
 pose proof (sqrt_lt_R0 (sqrt(r/a)) Hs) as Hss.
 pose proof (sqrt_sqrt (r/a) ltac:(lra)) as HS.
 pose proof (sqrt_sqrt (sqrt(r/a)) ltac:(lra)) as HSS.
 unfold radiation_temperature,emission. split; [exact Hss|].
 replace (sqrt (sqrt (r/a))^4) with (r/a) by nra.
 field; lra.
Qed.

Lemma source_energy_continuous A D T0 C chi a r kap x :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<x -> 0<kap x ->
 continuity_pt kap x -> continuity_pt (source_energy A D T0 C chi a r kap) x.
Proof.
 intros HA HD HT HC Hchi Hx Hk Hcont.
 pose proof (gas_map_continuous A D T0 x HA HD HT Hx) as Hg.
 unfold source_energy,coupling,optical_depth,emission.
 reg; try assumption; nra.
Qed.

Lemma source_energy_heating_endpoints A D T0 C chi a r kap Tr :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 T0<=Tr -> emission a Tr=r -> 0<kap T0 -> 0<kap Tr ->
 source_energy A D T0 C chi a r kap T0<=0 /\
 0<=source_energy A D T0 C chi a r kap Tr.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hord HB Hk0 Hkr.
 pose proof (coupling_positive C chi kap T0 HC Hchi Hk0) as Hv0.
 pose proof (gas_map_between A D T0 Tr HA HD HT ltac:(lra)) as Hgt.
 rewrite Rmin_left,Rmax_right in Hgt by lra.
 pose proof (pow_incr T0 Tr 4 ltac:(lra)) as HP.
 assert (HE:emission a T0<=r).
 { rewrite <- HB. unfold emission; apply Rmult_le_compat_l; lra. }
 split.
 - unfold source_energy. rewrite gas_map_equilibrium by assumption.
   nra.
 - unfold source_energy. rewrite HB. nra.
Qed.

Lemma source_energy_cooling_endpoints A D T0 C chi a r kap Tr :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<Tr -> Tr<=T0 -> emission a Tr=r -> 0<kap T0 -> 0<kap Tr ->
 source_energy A D T0 C chi a r kap Tr<=0 /\
 0<=source_energy A D T0 C chi a r kap T0.
Proof.
 intros HA HD HT HC Hchi Ha Hr HTr Hord HB Hk0 Hkr.
 pose proof (coupling_positive C chi kap T0 HC Hchi Hk0) as Hv0.
 pose proof (gas_map_between A D T0 Tr HA HD HT HTr) as Hgt.
 rewrite Rmin_right,Rmax_left in Hgt by lra.
 pose proof (pow_incr Tr T0 4 ltac:(lra)) as HP.
 assert (HE:r<=emission a T0).
 { rewrite <- HB. unfold emission; apply Rmult_le_compat_l; lra. }
 split.
 - unfold source_energy. rewrite HB. nra.
 - unfold source_energy. rewrite gas_map_equilibrium by assumption.
   nra.
Qed.

Lemma physical_root_exists A D T0 C chi a r kap Tr :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<Tr -> emission a Tr=r ->
 (forall x, Rmin T0 Tr<=x<=Rmax T0 Tr -> 0<kap x /\ continuity_pt kap x) ->
 exists x, Rmin T0 Tr<=x<=Rmax T0 Tr /\
 source_energy A D T0 C chi a r kap x=0 /\
 collision A D T0 x (gas_map A D T0 x)=0.
Proof.
 intros HA HD HT HC Hchi Ha Hr HTr HB Hkap.
 assert (Hk0:0<kap T0).
 { apply (proj1 (Hkap T0 (conj (Rmin_l _ _) (Rmax_l _ _)))). }
 assert (Hkr:0<kap Tr).
 { apply (proj1 (Hkap Tr (conj (Rmin_r _ _) (Rmax_r _ _)))). }
 destruct (Rtotal_order T0 Tr) as [HL|[HE|HG]].
 - rewrite Rmin_left,Rmax_right in * by lra.
   destruct (source_energy_heating_endpoints A D T0 C chi a r kap Tr
     HA HD HT HC Hchi Ha Hr ltac:(lra) HB Hk0 Hkr) as [Hlo Hhi].
   destruct (f_interv_is_interv (source_energy A D T0 C chi a r kap)
     T0 Tr 0 HL ltac:(lra)) as [x [Hx HF]].
   + intros y Hy. apply source_energy_continuous; try lra; apply (Hkap y Hy).
   + exists x; repeat split; try tauto; apply gas_map_collision; lra.
 - subst Tr. exists T0. rewrite Rmin_left,Rmax_left by lra.
   split; [lra|]. split.
   + unfold source_energy. rewrite gas_map_equilibrium by assumption.
     rewrite HB; ring.
   + apply gas_map_collision; assumption.
 - rewrite Rmin_right,Rmax_left in * by lra.
   destruct (source_energy_cooling_endpoints A D T0 C chi a r kap Tr
     HA HD HT HC Hchi Ha Hr HTr ltac:(lra) HB Hk0 Hkr) as [Hlo Hhi].
   destruct (f_interv_is_interv (source_energy A D T0 C chi a r kap)
     Tr T0 0 HG ltac:(lra)) as [x [Hx HF]].
   + intros y Hy. apply source_energy_continuous; try lra; apply (Hkap y Hy).
   + exists x; repeat split; try tauto; apply gas_map_collision; lra.
Qed.

Lemma heating_balance_root_equiv A T0 C chi a r kap gas x :
 0<C -> 0<chi -> 0<kap x -> 0<r ->
 (heating_balance A T0 C chi a r kap gas x=1 <->
 A*(gas x-T0)+coupling C chi kap x*(emission a x-r)=0).
Proof.
 intros HC Hchi Hk Hr.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as Hv.
 unfold heating_balance. split; intro H.
 - apply (f_equal (fun y => y*(coupling C chi kap x*r))) in H.
   field_simplify in H; nra.
 - apply (Rmult_eq_reg_r (coupling C chi kap x*r)); [|nra].
   field_simplify; nra.
Qed.

Lemma cooling_balance_root_equiv A T0 C chi a r kap gas x :
 0<C -> 0<chi -> 0<kap x -> 0<a -> 0<x ->
 (cooling_balance A T0 C chi a r kap gas x=1 <->
 A*(gas x-T0)+coupling C chi kap x*(emission a x-r)=0).
Proof.
 intros HC Hchi Hk Ha Hx.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as Hv.
 pose proof (emission_positive a x Ha Hx) as HB.
 unfold cooling_balance. split; intro H.
 - apply (f_equal (fun y => y*(coupling C chi kap x*emission a x))) in H.
   field_simplify in H; nra.
 - apply (Rmult_eq_reg_r (coupling C chi kap x*emission a x)); [|nra].
   field_simplify; nra.
Qed.
