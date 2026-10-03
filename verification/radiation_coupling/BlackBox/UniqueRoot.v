(* Every positive physical solution lies on the selected branch. Together with
   the interval conditioning theorem this yields global physical uniqueness. *)
From Coq Require Import Reals Psatz Field ClassicalEpsilon.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra GasMap OuterDerivatives LogCalculus PhysicalRoot OuterBounds.
Open Scope R_scope.

Definition positive_physical_solution A D T0 C chi a r kap x t :=
 0<x /\ 0<t /\ 0<kap x /\
 A*(t-T0)+coupling C chi kap x*(emission a x-r)=0 /\
 collision A D T0 x t=0.

Lemma emission_strict_increasing a x y :
 0<a -> 0<x -> x<y -> emission a x<emission a y.
Proof.
 intros Ha Hx Hxy.
 assert (Hsq:0<x^2<y^2) by nra.
 assert (Hfour:x^4<y^4) by nra.
 unfold emission; apply Rmult_lt_compat_l; assumption.
Qed.

Lemma emission_injective a x y :
 0<a -> 0<x -> 0<y -> emission a x=emission a y -> x=y.
Proof.
 intros Ha Hx Hy He.
 destruct (Rtotal_order x y) as [Hlt|[Heq|Hgt]]; [|assumption|].
 - pose proof (emission_strict_increasing a x y Ha Hx Hlt); lra.
 - pose proof (emission_strict_increasing a y x Ha Hy Hgt); lra.
Qed.

Lemma positive_solution_gas_map A D T0 C chi a r kap x t :
 0<A -> 0<D -> 0<T0 ->
 positive_physical_solution A D T0 C chi a r kap x t ->
 gas_map A D T0 x=t.
Proof.
 intros HA HD HT0 [Hx [Ht [Hk [HE Hcol]]]].
 apply gas_map_unique; try assumption.
 apply (proj1 (collision_forward A D T0 x t HD Ht)); assumption.
Qed.

Lemma source_root_in_physical_interval A D T0 C chi a r kap Tr x :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<Tr -> emission a Tr=r -> 0<x -> 0<kap x ->
 source_energy A D T0 C chi a r kap x=0 ->
 Rmin T0 Tr<=x<=Rmax T0 Tr.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr HTr HB Hx Hk HE.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 unfold source_energy in HE.
 split.
 - destruct (Rle_dec (Rmin T0 Tr) x) as [Hle|Hn]; [assumption|].
   assert (Hxt:x<T0) by (pose proof (Rmin_l T0 Tr); lra).
   assert (Hxr:x<Tr) by (pose proof (Rmin_r T0 Tr); lra).
   pose proof (gas_map_cooling_order A D T0 x HA HD Hx Hxt) as Hgas.
   pose proof (emission_strict_increasing a x Tr Ha Hx Hxr) as HBx.
   rewrite HB in HBx; nra.
 - destruct (Rle_dec x (Rmax T0 Tr)) as [Hle|Hn]; [assumption|].
   assert (Htx:T0<x) by (pose proof (Rmax_l T0 Tr); lra).
   assert (Hrx:Tr<x) by (pose proof (Rmax_r T0 Tr); lra).
   pose proof (gas_map_heating_order A D T0 x HA HD HT0 Htx) as Hgas.
   pose proof (emission_strict_increasing a Tr x Ha HTr Hrx) as HBx.
   rewrite HB in HBx; nra.
Qed.

Theorem positive_solution_physical_interval A D T0 C chi a r kap Tr x t :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<Tr -> emission a Tr=r ->
 positive_physical_solution A D T0 C chi a r kap x t ->
 Rmin T0 Tr<=x<=Rmax T0 Tr /\ gas_map A D T0 x=t.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr HTr HB HS.
 pose proof (positive_solution_gas_map A D T0 C chi a r kap x t HA HD HT0 HS) as Hg.
 destruct HS as [Hx [Ht [Hk [HE Hcol]]]]. split; [|assumption].
 apply (source_root_in_physical_interval A D T0 C chi a r kap Tr x); try assumption.
 unfold source_energy; rewrite Hg; exact HE.
Qed.

Theorem positive_solution_heating_strict A D T0 C chi a r kap Tr x t :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 T0<Tr -> emission a Tr=r ->
 positive_physical_solution A D T0 C chi a r kap x t ->
 T0<t /\ t<x /\ x<Tr.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hbranch HB HS.
 assert (HTr:0<Tr) by lra.
 pose proof (positive_solution_physical_interval A D T0 C chi a r kap Tr x t
   HA HD HT0 HC Hchi Ha Hr HTr HB HS) as [Hrange Hg].
 rewrite Rmin_left,Rmax_right in Hrange by lra.
 destruct HS as [Hx [Ht [Hk [HE Hcol]]]].
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 assert (Htx:T0<x).
 { destruct (Req_dec x T0) as [Heq|Hne]; [|lra].
   subst x; rewrite gas_map_equilibrium in Hg by assumption; subst t.
   pose proof (emission_strict_increasing a T0 Tr Ha HT0 Hbranch) as HBi.
   rewrite HB in HBi; nra. }
 pose proof (gas_map_heating_order A D T0 x HA HD HT0 Htx) as Horder.
 rewrite Hg in Horder.
 repeat split; try tauto.
 destruct (Req_dec x Tr) as [Heq|Hne]; [|lra].
 subst x; rewrite HB in HE; nra.
Qed.

Theorem positive_solution_cooling_strict A D T0 C chi a r kap Tr x t :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<Tr -> Tr<T0 -> emission a Tr=r ->
 positive_physical_solution A D T0 C chi a r kap x t ->
 Tr<x /\ x<t /\ t<T0.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr HTr Hbranch HB HS.
 pose proof (positive_solution_physical_interval A D T0 C chi a r kap Tr x t
   HA HD HT0 HC Hchi Ha Hr HTr HB HS) as [Hrange Hg].
 rewrite Rmin_right,Rmax_left in Hrange by lra.
 destruct HS as [Hx [Ht [Hk [HE Hcol]]]].
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 assert (Hxt:x<T0).
 { destruct (Req_dec x T0) as [Heq|Hne]; [|lra].
   subst x; rewrite gas_map_equilibrium in Hg by assumption; subst t.
   pose proof (emission_strict_increasing a Tr T0 Ha HTr Hbranch) as HBi.
   rewrite HB in HBi; nra. }
 pose proof (gas_map_cooling_order A D T0 x HA HD Hx Hxt) as Horder.
 rewrite Hg in Horder.
 assert (Hxr:Tr<x).
 { destruct (Req_dec x Tr) as [Heq|Hne]; [|lra].
   subst x; rewrite HB in HE; nra. }
 tauto.
Qed.

Theorem positive_solution_equilibrium A D T0 C chi a r kap Tr x t :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 Tr=T0 -> emission a Tr=r ->
 positive_physical_solution A D T0 C chi a r kap x t ->
 x=T0 /\ t=T0.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Heq HB HS; subst Tr.
 pose proof (positive_solution_physical_interval A D T0 C chi a r kap T0 x t
   HA HD HT0 HC Hchi Ha Hr HT0 HB HS) as [Hrange Hg].
 rewrite Rmin_left,Rmax_left in Hrange by lra.
 assert (Hx:x=T0) by lra; subst x.
 rewrite gas_map_equilibrium in Hg by assumption; auto.
Qed.

Lemma subinterval_closed a b x y z :
 a<=x<=b -> a<=y<=b -> Rmin x y<=z<=Rmax x y -> a<=z<=b.
Proof.
 intros Hx Hy Hz; split.
 - apply (closed_interval_lower x y z a); tauto.
 - apply (closed_interval_upper x y z b); tauto.
Qed.
Lemma subinterval_open a b x y z :
 a<=x<=b -> a<=y<=b -> Rmin x y<z<Rmax x y -> a<z<b.
Proof.
 intros Hx Hy Hz; unfold Rmin,Rmax in Hz; destruct (Rle_dec x y); lra.
Qed.

Theorem heating_physical_solution_unique A D T0 C chi a r kap kp Tr mu x t y u :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 T0<=Tr -> emission a Tr=r -> 0<mu -> mu<=4 ->
 (forall z, T0<=z<=Tr -> 0<kap z /\ continuity_pt kap z) ->
 (forall z, T0<z<Tr -> is_derive kap z (kp z)) ->
 (forall z, T0<z<Tr -> mu<=1-effective_slope C kap z (kp z)) ->
 positive_physical_solution A D T0 C chi a r kap x t ->
 positive_physical_solution A D T0 C chi a r kap y u -> x=y /\ t=u.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hbranch HB Hmu H4 Hkap Hder Hmargin HSx HSy.
 pose proof (positive_solution_physical_interval A D T0 C chi a r kap Tr x t
  HA HD HT0 HC Hchi Ha Hr ltac:(lra) HB HSx) as [Hrx Hgx].
 pose proof (positive_solution_physical_interval A D T0 C chi a r kap Tr y u
  HA HD HT0 HC Hchi Ha Hr ltac:(lra) HB HSy) as [Hry Hgy].
 rewrite Rmin_left,Rmax_right in Hrx,Hry by lra.
 destruct HSx as [Hx [Ht [Hkx [HEx HCx]]]].
 destruct HSy as [Hy [Hu [Hky [HEy HCy]]]].
 assert (Hxy:y=x).
 { apply (heating_unique A D T0 C chi a r kap kp x y mu); try assumption; try tauto.
   - intros z Hz; apply (proj1 (Hkap z (subinterval_closed T0 Tr x y z Hrx Hry Hz))).
   - intros z Hz; apply (proj2 (Hkap z (subinterval_closed T0 Tr x y z Hrx Hry Hz))).
   - intros z Hz; apply Hder; apply (subinterval_open T0 Tr x y z Hrx Hry Hz).
   - intros z Hz; apply Hmargin; apply (subinterval_open T0 Tr x y z Hrx Hry Hz).
   - unfold physical_heating; apply (proj2 (heating_balance_root_equiv A T0 C chi a r kap
       (gas_map A D T0) x HC Hchi Hkx Hr)).
     rewrite Hgx; exact HEx.
   - unfold physical_heating; apply (proj2 (heating_balance_root_equiv A T0 C chi a r kap
       (gas_map A D T0) y HC Hchi Hky Hr)).
     rewrite Hgy; exact HEy. }
 subst y; split; congruence.
Qed.

Theorem cooling_physical_solution_unique A D T0 C chi a r kap kp Tr mu x t y u :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<Tr -> Tr<=T0 -> emission a Tr=r -> 0<mu -> mu<=4 ->
 (forall z, Tr<=z<=T0 -> 0<kap z /\ continuity_pt kap z) ->
 (forall z, Tr<z<T0 -> is_derive kap z (kp z)) ->
 (forall z, Tr<z<T0 -> mu<=4+effective_slope C kap z (kp z)) ->
 positive_physical_solution A D T0 C chi a r kap x t ->
 positive_physical_solution A D T0 C chi a r kap y u -> x=y /\ t=u.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr HTr Hbranch HB Hmu H4 Hkap Hder Hmargin HSx HSy.
 pose proof (positive_solution_physical_interval A D T0 C chi a r kap Tr x t
  HA HD HT0 HC Hchi Ha Hr HTr HB HSx) as [Hrx Hgx].
 pose proof (positive_solution_physical_interval A D T0 C chi a r kap Tr y u
  HA HD HT0 HC Hchi Ha Hr HTr HB HSy) as [Hry Hgy].
 rewrite Rmin_right,Rmax_left in Hrx,Hry by lra.
 destruct HSx as [Hx [Ht [Hkx [HEx HCx]]]].
 destruct HSy as [Hy [Hu [Hky [HEy HCy]]]].
 assert (Hxy:y=x).
 { apply (cooling_unique A D T0 C chi a r kap kp x y mu); try assumption; try tauto.
   - intros z Hz; apply (proj1 (Hkap z (subinterval_closed Tr T0 x y z Hrx Hry Hz))).
   - intros z Hz; apply (proj2 (Hkap z (subinterval_closed Tr T0 x y z Hrx Hry Hz))).
   - intros z Hz; apply Hder; apply (subinterval_open Tr T0 x y z Hrx Hry Hz).
   - intros z Hz; apply Hmargin; apply (subinterval_open Tr T0 x y z Hrx Hry Hz).
   - unfold physical_cooling; apply (proj2 (cooling_balance_root_equiv A T0 C chi a r kap
       (gas_map A D T0) x HC Hchi Hkx Ha Hx)).
     rewrite Hgx; exact HEx.
   - unfold physical_cooling; apply (proj2 (cooling_balance_root_equiv A T0 C chi a r kap
       (gas_map A D T0) y HC Hchi Hky Ha Hy)).
     rewrite Hgy; exact HEy. }
 subst y; split; congruence.
Qed.

Theorem heating_positive_solution_exists_unique A D T0 C chi a r kap kp Tr mu :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 T0<=Tr -> emission a Tr=r -> 0<mu -> mu<=4 ->
 (forall z, T0<=z<=Tr -> 0<kap z /\ continuity_pt kap z) ->
 (forall z, T0<z<Tr -> is_derive kap z (kp z)) ->
 (forall z, T0<z<Tr -> mu<=1-effective_slope C kap z (kp z)) ->
 exists x t, positive_physical_solution A D T0 C chi a r kap x t /\
 (forall y u, positive_physical_solution A D T0 C chi a r kap y u -> y=x /\ u=t).
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hbranch HB Hmu H4 Hkap Hder Hmargin.
 assert (Hkap_full:forall z, Rmin T0 Tr<=z<=Rmax T0 Tr ->
  0<kap z /\ continuity_pt kap z).
 { intros z Hz; apply Hkap; rewrite Rmin_left,Rmax_right in Hz by lra; exact Hz. }
 destruct (physical_root_exists A D T0 C chi a r kap Tr HA HD HT0 HC Hchi Ha Hr
   ltac:(lra) HB Hkap_full) as [x [Hx [HE Hcol]]].
 rewrite Rmin_left,Rmax_right in Hx by lra.
 exists x, (gas_map A D T0 x).
 assert (HS:positive_physical_solution A D T0 C chi a r kap x (gas_map A D T0 x)).
 { unfold positive_physical_solution; repeat split; try assumption; try lra.
   - apply (proj1 (gas_map_spec A D T0 x HA HD HT0 ltac:(lra))).
   - apply (proj1 (Hkap x Hx)). }
 split; [assumption|].
 intros y u Hy.
 apply (heating_physical_solution_unique A D T0 C chi a r kap kp Tr mu y u x
  (gas_map A D T0 x)); assumption.
Qed.

Theorem cooling_positive_solution_exists_unique A D T0 C chi a r kap kp Tr mu :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<Tr -> Tr<=T0 -> emission a Tr=r -> 0<mu -> mu<=4 ->
 (forall z, Tr<=z<=T0 -> 0<kap z /\ continuity_pt kap z) ->
 (forall z, Tr<z<T0 -> is_derive kap z (kp z)) ->
 (forall z, Tr<z<T0 -> mu<=4+effective_slope C kap z (kp z)) ->
 exists x t, positive_physical_solution A D T0 C chi a r kap x t /\
 (forall y u, positive_physical_solution A D T0 C chi a r kap y u -> y=x /\ u=t).
Proof.
 intros HA HD HT0 HC Hchi Ha Hr HTr Hbranch HB Hmu H4 Hkap Hder Hmargin.
 assert (Hkap_full:forall z, Rmin T0 Tr<=z<=Rmax T0 Tr ->
  0<kap z /\ continuity_pt kap z).
 { intros z Hz; apply Hkap; rewrite Rmin_right,Rmax_left in Hz by lra; exact Hz. }
 destruct (physical_root_exists A D T0 C chi a r kap Tr HA HD HT0 HC Hchi Ha Hr
   HTr HB Hkap_full) as [x [Hx [HE Hcol]]].
 rewrite Rmin_right,Rmax_left in Hx by lra.
 exists x, (gas_map A D T0 x).
 assert (HS:positive_physical_solution A D T0 C chi a r kap x (gas_map A D T0 x)).
 { unfold positive_physical_solution; repeat split; try assumption; try lra.
   - apply (proj1 (gas_map_spec A D T0 x HA HD HT0 ltac:(lra))).
   - apply (proj1 (Hkap x Hx)). }
 split; [assumption|].
 intros y u Hy.
 apply (cooling_physical_solution_unique A D T0 C chi a r kap kp Tr mu y u x
  (gas_map A D T0 x)); assumption.
Qed.

Print Assumptions heating_positive_solution_exists_unique.
Print Assumptions cooling_positive_solution_exists_unique.

(* A mathematical reference root, selected only from existence. This is not an
   executable inversion algorithm and its definition imposes no opacity slope
   condition. Uniqueness above identifies it whenever conditioning holds. *)
Definition exact_dust A D T0 C chi a r kap :=
 epsilon (inhabits T0) (fun x=>
   Rmin T0 (radiation_temperature a r)<=x<=Rmax T0 (radiation_temperature a r) /\
   source_energy A D T0 C chi a r kap x=0).
Definition exact_gas A D T0 C chi a r kap :=
 gas_map A D T0 (exact_dust A D T0 C chi a r kap).

Theorem exact_dust_spec A D T0 C chi a r kap :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 (forall x, Rmin T0 (radiation_temperature a r)<=x<=Rmax T0 (radiation_temperature a r) ->
   0<kap x /\ continuity_pt kap x) ->
 0<exact_dust A D T0 C chi a r kap /\
 Rmin T0 (radiation_temperature a r)<=exact_dust A D T0 C chi a r kap<=
 Rmax T0 (radiation_temperature a r) /\
 source_energy A D T0 C chi a r kap (exact_dust A D T0 C chi a r kap)=0 /\
 collision A D T0 (exact_dust A D T0 C chi a r kap)
   (exact_gas A D T0 C chi a r kap)=0.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hkap.
 destruct (radiation_temperature_spec a r Ha Hr) as [HTr HB].
 assert (Hchosen:
 Rmin T0 (radiation_temperature a r)<=exact_dust A D T0 C chi a r kap<=
 Rmax T0 (radiation_temperature a r) /\
 source_energy A D T0 C chi a r kap (exact_dust A D T0 C chi a r kap)=0).
 { unfold exact_dust; apply epsilon_spec.
   destruct (physical_root_exists A D T0 C chi a r kap (radiation_temperature a r)
     HA HD HT0 HC Hchi Ha Hr HTr HB Hkap) as [x [Hx [HE Hcol]]].
   exists x; tauto. }
 assert (Hx:0<exact_dust A D T0 C chi a r kap).
 { unfold Rmin in Hchosen; destruct (Rle_dec T0 (radiation_temperature a r)); lra. }
 split; [assumption|]. split; [tauto|]. split; [tauto|].
 unfold exact_gas; apply gas_map_collision; assumption.
Qed.

Theorem exact_physical_solution A D T0 C chi a r kap :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 (forall x, Rmin T0 (radiation_temperature a r)<=x<=Rmax T0 (radiation_temperature a r) ->
   0<kap x /\ continuity_pt kap x) ->
 positive_physical_solution A D T0 C chi a r kap
  (exact_dust A D T0 C chi a r kap) (exact_gas A D T0 C chi a r kap).
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hkap.
 destruct (exact_dust_spec A D T0 C chi a r kap HA HD HT0 HC Hchi Ha Hr Hkap)
   as [Hx [Hrange [HE Hcol]]].
 unfold positive_physical_solution; repeat split; try assumption.
 - unfold exact_gas; apply (proj1 (gas_map_spec A D T0 _ HA HD HT0 Hx)).
 - apply (proj1 (Hkap _ Hrange)).
Qed.

Print Assumptions exact_dust_spec.
Print Assumptions exact_physical_solution.
