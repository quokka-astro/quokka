(* Exact collision map. The inverse is specified by a unique positive root;
   all derivative facts below are derived from this equation. *)
From Coq Require Import Reals Ranalysis5 Psatz Field ClassicalEpsilon.
From Coquelicot Require Import Coquelicot.
Open Scope R_scope.

Definition collision A D T0 x t := A*(t-T0)+D*sqrt t*(t-x).
Definition forward_map A D T0 t := t+A*(t-T0)/(D*sqrt t).
Definition forward_derivative A D T0 t :=
  1+A/(2*D*sqrt t)*(1+T0/t).
Definition gas_map A D T0 x :=
  epsilon (inhabits T0) (fun t => 0<t /\ forward_map A D T0 t=x).

Lemma collision_continuous A D T0 x t : 0<t ->
 continuity_pt (collision A D T0 x) t.
Proof.
 intro Ht. unfold collision. reg; lra.
Qed.

Lemma collision_forward A D T0 x t :
  0<D -> 0<t -> (collision A D T0 x t=0 <-> forward_map A D T0 t=x).
Proof.
 intros HD Ht. pose proof (sqrt_lt_R0 t Ht) as Hs.
 unfold collision,forward_map.
 assert (HDs:D*sqrt t<>0) by nra.
 split; intro H.
 - apply (Rmult_eq_reg_r (D*sqrt t)); [|exact HDs]. field_simplify; nra.
 - apply (f_equal (fun y => y*(D*sqrt t))) in H.
   field_simplify in H; nra.
Qed.

Lemma collision_root_exists A D T0 x :
  0<A -> 0<D -> 0<T0 -> 0<x ->
  exists t, Rmin T0 x<=t<=Rmax T0 x /\ collision A D T0 x t=0.
Proof.
 intros HA HD HT Hx.
 assert (HS:0<sqrt T0) by (apply sqrt_lt_R0; assumption).
 assert (HDS:0<D*sqrt T0) by (apply Rmult_lt_0_compat; assumption).
 assert (C0:collision A D T0 x T0=D*sqrt T0*(T0-x))
   by (unfold collision; ring).
 assert (Cx:collision A D T0 x x=A*(x-T0))
   by (unfold collision; ring).
 destruct (Rtotal_order T0 x) as [HL|[HE|HG]].
 - destruct (f_interv_is_interv (collision A D T0 x) T0 x 0 HL) as [t [Hb Hc]].
   + rewrite C0,Cx; split; nra.
   + intros t Hb; apply collision_continuous; lra.
   + exists t. rewrite Rmin_left,Rmax_right by lra. auto.
 - subst x. exists T0. rewrite Rmin_left,Rmax_left by lra.
   split; [lra|]. rewrite C0; ring.
 - destruct (f_interv_is_interv (collision A D T0 x) x T0 0 HG) as [t [Hb Hc]].
   + rewrite C0,Cx; split; nra.
   + intros t Hb; apply collision_continuous; lra.
   + exists t. rewrite Rmin_right,Rmax_left by lra. auto.
Qed.

Lemma gas_map_spec A D T0 x :
  0<A -> 0<D -> 0<T0 -> 0<x ->
  0<gas_map A D T0 x /\ forward_map A D T0 (gas_map A D T0 x)=x.
Proof.
 intros HA HD HT Hx. unfold gas_map. apply epsilon_spec.
 destruct (collision_root_exists A D T0 x HA HD HT Hx) as [t [Hb Hc]].
 assert (Ht:0<t).
 { unfold Rmin in Hb. destruct (Rle_dec T0 x); lra. }
 exists t; split; [exact Ht|].
 apply (proj1 (collision_forward A D T0 x t HD Ht)); exact Hc.
Qed.

Lemma forward_derivative_positive A D T0 t :
  0<A -> 0<D -> 0<T0 -> 0<t -> 1<forward_derivative A D T0 t.
Proof.
 intros HA HD HT Ht. unfold forward_derivative.
 assert (Hs:0<sqrt t) by (apply sqrt_lt_R0; assumption).
 assert (Ha:0<A/(2*D*sqrt t)) by (apply Rdiv_lt_0_compat; nra).
 assert (Hz:0<T0/t) by (apply Rdiv_lt_0_compat; assumption).
 nra.
Qed.

Lemma forward_is_derive A D T0 t : 0<D -> 0<t ->
 is_derive (forward_map A D T0) t (forward_derivative A D T0 t).
Proof.
 intros HD Ht.
 assert (Hs:0<sqrt t) by (apply sqrt_lt_R0; assumption).
 assert (Hs2:sqrt t * sqrt t=t) by (apply sqrt_sqrt; lra).
 unfold forward_map,forward_derivative.
 evar (df:R).
 assert (Hder:is_derive (fun y => y + A*(y-T0)/(D*sqrt y)) t df).
 { unfold df. auto_derive; [repeat split; nra|reflexivity]. }
 replace (1 + A / (2 * D * sqrt t) * (1 + T0 / t)) with df; [exact Hder|].
 unfold df. remember (sqrt t) as s in *.
 replace t with (s*s) by nra. field; nra.
Qed.

Lemma forward_continuous A D T0 t : 0<D -> 0<t ->
 continuity_pt (forward_map A D T0) t.
Proof.
 intros HD Ht. apply continuity_pt_filterlim.
 apply (ex_derive_continuous (forward_map A D T0) t). exists (forward_derivative A D T0 t).
 apply forward_is_derive; assumption.
Qed.

Lemma forward_increasing A D T0 a b :
 0<A -> 0<D -> 0<T0 -> 0<a -> a<b ->
 forward_map A D T0 a < forward_map A D T0 b.
Proof.
 intros HA HD HT Ha Hab.
 destruct (MVT_gen (forward_map A D T0) a b (forward_derivative A D T0))
   as [c [Hc HE]].
 - intros t Hb. apply forward_is_derive; [assumption|].
   rewrite Rmin_left,Rmax_right in Hb by lra; lra.
 - intros t Hb. apply forward_continuous; [assumption|].
   rewrite Rmin_left,Rmax_right in Hb by lra; lra.
 - rewrite Rmin_left,Rmax_right in Hc by lra.
   pose proof (forward_derivative_positive A D T0 c HA HD HT ltac:(lra)).
   nra.
Qed.

Lemma forward_injective A D T0 a b :
 0<A -> 0<D -> 0<T0 -> 0<a -> 0<b ->
 forward_map A D T0 a=forward_map A D T0 b -> a=b.
Proof.
 intros HA HD HT Ha Hb HE.
 destruct (Rtotal_order a b) as [HL|[HEq|HG]]; [|exact HEq|].
 - pose proof (forward_increasing A D T0 a b HA HD HT Ha HL); lra.
 - pose proof (forward_increasing A D T0 b a HA HD HT Hb HG); lra.
Qed.

Lemma gas_map_unique A D T0 x t :
 0<A -> 0<D -> 0<T0 -> 0<x -> 0<t ->
 forward_map A D T0 t=x -> gas_map A D T0 x=t.
Proof.
 intros HA HD HT Hx Ht Hf.
 destruct (gas_map_spec A D T0 x HA HD HT Hx) as [Hg HF].
 eapply (forward_injective A D T0); eauto. congruence.
Qed.

Lemma gas_map_between A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x ->
 Rmin T0 x <= gas_map A D T0 x <= Rmax T0 x.
Proof.
 intros HA HD HT Hx.
 destruct (collision_root_exists A D T0 x HA HD HT Hx) as [t [Hb Hc]].
 assert (Ht:0<t).
 { unfold Rmin in Hb. destruct (Rle_dec T0 x); lra. }
 rewrite (gas_map_unique A D T0 x t HA HD HT Hx Ht); [exact Hb|].
 apply (proj1 (collision_forward A D T0 x t HD Ht)); exact Hc.
Qed.

Lemma gas_map_equilibrium A D T0 : 0<A -> 0<D -> 0<T0 ->
 gas_map A D T0 T0=T0.
Proof.
 intros HA HD HT. pose proof (gas_map_between A D T0 T0 HA HD HT HT).
 rewrite Rmin_left,Rmax_left in H by lra; lra.
Qed.

Lemma gas_map_increasing A D T0 a b :
 0<A -> 0<D -> 0<T0 -> 0<a -> a<b ->
 gas_map A D T0 a < gas_map A D T0 b.
Proof.
 intros HA HD HT Ha Hab.
 destruct (gas_map_spec A D T0 a HA HD HT Ha) as [Hga Hfa].
 destruct (gas_map_spec A D T0 b HA HD HT ltac:(lra)) as [Hgb Hfb].
 destruct (Rtotal_order (gas_map A D T0 a) (gas_map A D T0 b))
   as [HL|[HE|HG]]; [exact HL| |].
 - rewrite HE in Hfa; lra.
 - pose proof (forward_increasing A D T0 _ _ HA HD HT Hgb HG); lra.
Qed.

Lemma gas_map_nondecreasing A D T0 a b :
 0<A -> 0<D -> 0<T0 -> 0<a -> a<=b ->
 gas_map A D T0 a <= gas_map A D T0 b.
Proof.
 intros HA HD HT Ha [HL|HE].
 - apply Rlt_le,gas_map_increasing; assumption.
 - subst b; lra.
Qed.

Lemma gas_map_left_inverse A D T0 t :
 0<A -> 0<D -> 0<T0 -> 0<t -> 0<forward_map A D T0 t ->
 gas_map A D T0 (forward_map A D T0 t)=t.
Proof. intros. apply gas_map_unique; auto. Qed.

Lemma gas_map_continuous A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x -> continuity_pt (gas_map A D T0) x.
Proof.
 intros HA HD HT Hx.
 set (lb:=gas_map A D T0 (x/2)).
 set (ub:=gas_map A D T0 (2*x)).
 assert (Hlb:0<lb /\ forward_map A D T0 lb=x/2).
 { unfold lb; apply gas_map_spec; lra. }
 assert (Hub:0<ub /\ forward_map A D T0 ub=2*x).
 { unfold ub; apply gas_map_spec; lra. }
 assert (HL:lb<ub).
 { unfold lb,ub; apply gas_map_increasing; lra. }
 eapply (continuity_pt_recip_prelim (forward_map A D T0) (gas_map A D T0) lb ub HL).
 - intros a b Hal Hab Hbu. apply forward_increasing; lra.
 - intros t Hb. unfold comp,id.
   apply gas_map_left_inverse; try lra.
   destruct (Rle_lt_or_eq_dec lb t ltac:(lra)) as [HLt|HEt].
   + pose proof (forward_increasing A D T0 lb t HA HD HT ltac:(lra) HLt); lra.
   + subst t; lra.
 - intros t Hb. apply forward_continuous; lra.
 - lra.
Qed.

Lemma gas_map_is_derive A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x ->
 is_derive (gas_map A D T0) x
   (1/forward_derivative A D T0 (gas_map A D T0 x)).
Proof.
 intros HA HD HT Hx.
 assert (Hpos:forall y,0<y -> 0<gas_map A D T0 y).
 { intros y Hy. exact (proj1 (gas_map_spec A D T0 y HA HD HT Hy)). }
 assert (Prf:forall t, gas_map A D T0 (x/2)<=t<=gas_map A D T0 (2*x) ->
   derivable_pt (forward_map A D T0) t).
 { intros t Ht. apply ex_derive_Reals_0.
   exists (forward_derivative A D T0 t). apply forward_is_derive; [exact HD|].
   pose proof (Hpos (x/2) ltac:(lra)); lra. }
 assert (Hrange:gas_map A D T0 (x/2)<=gas_map A D T0 x<=gas_map A D T0 (2*x)).
 { split; apply gas_map_nondecreasing; lra. }
 pose proof (derivable_pt_lim_recip_interv (forward_map A D T0)
   (gas_map A D T0) (x/2) (2*x) x Prf
   (gas_map_continuous A D T0 x HA HD HT Hx)
   ltac:(lra) ltac:(lra) Hrange) as HI.
 assert (Hdf:derive_pt (forward_map A D T0) (gas_map A D T0 x)
   (Prf (gas_map A D T0 x) Hrange)=forward_derivative A D T0 (gas_map A D T0 x)).
 { apply derive_pt_eq,is_derive_Reals,forward_is_derive; auto. }
 rewrite Hdf in HI. apply is_derive_Reals,HI.
 - intros y Hy. unfold comp,id.
   exact (proj2 (gas_map_spec A D T0 y HA HD HT ltac:(lra))).
 - pose proof (forward_derivative_positive A D T0 (gas_map A D T0 x)
     HA HD HT (Hpos x Hx)); lra.
Qed.

Print Assumptions gas_map_is_derive.

Lemma gas_map_collision A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x ->
 collision A D T0 x (gas_map A D T0 x)=0.
Proof.
 intros HA HD HT Hx.
 destruct (gas_map_spec A D T0 x HA HD HT Hx) as [Ht HF].
 apply (proj2 (collision_forward A D T0 x _ HD Ht)); exact HF.
Qed.

Lemma gas_map_heating_order A D T0 x :
 0<A -> 0<D -> 0<T0 -> T0<x ->
 T0<gas_map A D T0 x<x.
Proof.
 intros HA HD HT Hx.
 assert (Hg:T0<gas_map A D T0 x).
 { pose proof (gas_map_increasing A D T0 T0 x HA HD HT HT Hx) as HH.
   rewrite (gas_map_equilibrium A D T0 HA HD HT) in HH; exact HH. }
 split; [exact Hg|].
 pose proof (gas_map_collision A D T0 x HA HD HT ltac:(lra)) as Hc.
 pose proof (sqrt_lt_R0 (gas_map A D T0 x) ltac:(lra)) as Hs.
 assert (Hp:0<D*sqrt (gas_map A D T0 x)) by nra.
 unfold collision in Hc; nra.
Qed.

Lemma gas_map_cooling_order A D T0 x :
 0<A -> 0<D -> 0<x -> x<T0 ->
 x<gas_map A D T0 x<T0.
Proof.
 intros HA HD Hx HT.
 assert (Hg:gas_map A D T0 x<T0).
 { pose proof (gas_map_increasing A D T0 x T0 HA HD ltac:(lra) Hx HT) as HH.
   rewrite (gas_map_equilibrium A D T0 HA HD ltac:(lra)) in HH; exact HH. }
 split; [|exact Hg].
 pose proof (gas_map_collision A D T0 x HA HD ltac:(lra) Hx) as Hc.
 pose proof (proj1 (gas_map_spec A D T0 x HA HD ltac:(lra) Hx)) as Hgp.
 pose proof (sqrt_lt_R0 (gas_map A D T0 x) Hgp) as Hs.
 assert (Hp:0<D*sqrt (gas_map A D T0 x)) by nra.
 unfold collision in Hc; nra.
Qed.
