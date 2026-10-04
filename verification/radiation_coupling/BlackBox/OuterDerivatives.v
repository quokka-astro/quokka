(* Actual derivatives for arbitrary differentiable opacity and the gas map.
   All statements are pointwise: a bracket theorem must instantiate them at
   every point of its interval. No derivative or inverse slope is postulated. *)
From Coq Require Import Reals Psatz Field.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra GasMap.
Open Scope R_scope.

Definition optical_depth (C:R) (kap:R->R) x := C*kap x.
Definition emission (a x:R) := a*x^4.
Definition coupling (C chi:R) (kap:R->R) x :=
 chi * optical_depth C kap x / (1+optical_depth C kap x).
Definition heating_balance (A T0 C chi a r:R) (kap gas:R->R) x :=
 (A*(gas x-T0)+coupling C chi kap x*emission a x)/
 (coupling C chi kap x*r).
Definition cooling_balance (A T0 C chi a r:R) (kap gas:R->R) x :=
 (A*(T0-gas x)+coupling C chi kap x*r)/
 (coupling C chi kap x*emission a x).
Definition radiation (C a r:R) (kap:R->R) x :=
 (r+optical_depth C kap x*emission a x)/(1+optical_depth C kap x).
Definition opacity_slope (kap:R->R) (x kp:R) := x*kp/kap x.
Definition effective_slope (C:R) (kap:R->R) (x kp:R) :=
 opacity_slope kap x kp/(1+optical_depth C kap x).
Definition elasticity (f:R->R) x df := x*df/f x.

Lemma optical_depth_positive C kap x : 0<C -> 0<kap x ->
 0<optical_depth C kap x.
Proof. unfold optical_depth; intros; nra. Qed.
Lemma emission_positive a x : 0<a -> 0<x -> 0<emission a x.
Proof. unfold emission; intros; apply Rmult_lt_0_compat; [assumption|apply pow_lt; assumption]. Qed.
Lemma coupling_positive C chi kap x : 0<C -> 0<chi -> 0<kap x ->
 0<coupling C chi kap x.
Proof. unfold coupling, optical_depth; intros HC Hchi Hk; apply Rdiv_lt_0_compat.
 - apply Rmult_lt_0_compat; [assumption|nra].
 - nra.
Qed.

Lemma optical_depth_is_derive C kap x kp :
 is_derive kap x kp -> is_derive (optical_depth C kap) x (C*kp).
Proof.
 intro H; unfold optical_depth; auto_derive.
 - exists kp; exact H.
 - rewrite (is_derive_unique (fun y:R=>kap y) x kp H); ring.
Qed.
Lemma emission_is_derive a x :
 is_derive (emission a) x (4*a*x^3).
Proof. unfold emission; auto_derive; [trivial|ring]. Qed.
Lemma coupling_is_derive C chi kap x kp :
 is_derive kap x kp -> 1+optical_depth C kap x<>0 ->
 is_derive (coupling C chi kap) x
 (chi*C*kp/(1+optical_depth C kap x)^2).
Proof.
 intros H Hden; unfold coupling, optical_depth in *; auto_derive.
 - repeat split; try assumption; exists kp; exact H.
 - rewrite (is_derive_unique (fun y:R=>kap y) x kp H); field; exact Hden.
Qed.

Lemma coupling_elasticity C chi kap x kp :
 0<C -> 0<chi -> 0<kap x ->
 elasticity (coupling C chi kap) x
 (chi*C*kp/(1+optical_depth C kap x)^2) = effective_slope C kap x kp.
Proof.
 intros; unfold elasticity, coupling, effective_slope, opacity_slope, optical_depth.
 field; repeat split; nra.
Qed.

(* Raw quotient derivatives, still valid when the transfer vanishes. *)
Definition heating_derivative (A T0 C chi a r:R) (kap gas:R->R) x kp tp :=
 let V:=coupling C chi kap x in let B:=emission a x in
 let Vp:=chi*C*kp/(1+optical_depth C kap x)^2 in
 ((A*tp+Vp*B+V*(4*a*x^3))*(V*r)-
  (A*(gas x-T0)+V*B)*(Vp*r))/(V*r)^2.
Definition cooling_derivative (A T0 C chi a r:R) (kap gas:R->R) x kp tp :=
 let V:=coupling C chi kap x in let B:=emission a x in
 let Vp:=chi*C*kp/(1+optical_depth C kap x)^2 in
 ((-A*tp+Vp*r)*(V*B)-
  (A*(T0-gas x)+V*r)*(Vp*B+V*(4*a*x^3)))/(V*B)^2.
Definition radiation_derivative (C a r:R) (kap:R->R) x kp :=
 let tau:=optical_depth C kap x in let B:=emission a x in
 ((C*kp*B+tau*(4*a*x^3))*(1+tau)-(r+tau*B)*(C*kp))/(1+tau)^2.

Lemma heating_balance_is_derive A T0 C chi a r kap gas x kp tp :
 is_derive kap x kp -> is_derive gas x tp ->
 0<C -> 0<chi -> 0<kap x -> 0<r ->
 is_derive (heating_balance A T0 C chi a r kap gas) x
 (heating_derivative A T0 C chi a r kap gas x kp tp).
Proof.
 intros Hkap Hgas HC Hchi Hkap0 Hr.
 pose proof (coupling_is_derive C chi kap x kp Hkap ltac:(unfold optical_depth; nra)) as HV.
 pose proof (emission_is_derive a x) as HB.
 pose proof (coupling_positive C chi kap x HC Hchi Hkap0) as HV0.
 unfold heating_balance, heating_derivative; cbv zeta.
 apply (is_derive_div (fun y=>A*(gas y-T0)+coupling C chi kap y*emission a y)
   (fun y=>coupling C chi kap y*r) x
   (A*tp+chi*C*kp/(1+optical_depth C kap x)^2*emission a x+coupling C chi kap x*(4*a*x^3))
   (chi*C*kp/(1+optical_depth C kap x)^2*r)).
 - replace (A*tp+chi*C*kp/(1+optical_depth C kap x)^2*emission a x+
   coupling C chi kap x*(4*a*x^3)) with
   (A*tp+(chi*C*kp/(1+optical_depth C kap x)^2*emission a x+
   coupling C chi kap x*(4*a*x^3))) by ring.
   apply (@is_derive_plus R_AbsRing R_NormedModule).
   + auto_derive; [exists tp; exact Hgas|rewrite (is_derive_unique (fun y:R=>gas y) x tp Hgas); ring].
   + apply Derive.is_derive_mult; assumption.
 - replace (chi*C*kp/(1+optical_depth C kap x)^2*r) with
   (chi*C*kp/(1+optical_depth C kap x)^2*r+coupling C chi kap x*0) by ring.
   apply Derive.is_derive_mult; [exact HV|apply (@is_derive_const R_AbsRing R_NormedModule)].
 - nra.
Qed.

Lemma cooling_balance_is_derive A T0 C chi a r kap gas x kp tp :
 is_derive kap x kp -> is_derive gas x tp ->
 0<C -> 0<chi -> 0<kap x -> 0<a -> 0<x ->
 is_derive (cooling_balance A T0 C chi a r kap gas) x
 (cooling_derivative A T0 C chi a r kap gas x kp tp).
Proof.
 intros Hkap Hgas HC Hchi Hkap0 Ha Hx.
 pose proof (coupling_is_derive C chi kap x kp Hkap ltac:(unfold optical_depth; nra)) as HV.
 pose proof (emission_is_derive a x) as HB.
 pose proof (coupling_positive C chi kap x HC Hchi Hkap0) as HV0.
 pose proof (emission_positive a x Ha Hx) as HB0.
 unfold cooling_balance, cooling_derivative; cbv zeta.
 apply (is_derive_div (fun y=>A*(T0-gas y)+coupling C chi kap y*r)
   (fun y=>coupling C chi kap y*emission a y) x
   (-A*tp+chi*C*kp/(1+optical_depth C kap x)^2*r)
   (chi*C*kp/(1+optical_depth C kap x)^2*emission a x+coupling C chi kap x*(4*a*x^3))).
 - apply (@is_derive_plus R_AbsRing R_NormedModule).
   + auto_derive; [exists tp; exact Hgas|rewrite (is_derive_unique (fun y:R=>gas y) x tp Hgas); ring].
   + replace (chi*C*kp/(1+optical_depth C kap x)^2*r) with
     (chi*C*kp/(1+optical_depth C kap x)^2*r+coupling C chi kap x*0) by ring.
     apply Derive.is_derive_mult; [exact HV|apply (@is_derive_const R_AbsRing R_NormedModule)].
 - apply Derive.is_derive_mult; assumption.
 - nra.
Qed.

Lemma radiation_is_derive C a r kap x kp :
 is_derive kap x kp -> 0<C -> 0<kap x ->
 is_derive (radiation C a r kap) x (radiation_derivative C a r kap x kp).
Proof.
 intros Hkap HC Hkap0.
 pose proof (optical_depth_is_derive C kap x kp Hkap) as HT.
 pose proof (emission_is_derive a x) as HB.
 unfold radiation, radiation_derivative; cbv zeta.
 apply (is_derive_div (fun y=>r+optical_depth C kap y*emission a y)
   (fun y=>1+optical_depth C kap y) x
   (C*kp*emission a x+optical_depth C kap x*(4*a*x^3)) (C*kp)).
 - replace (C*kp*emission a x+optical_depth C kap x*(4*a*x^3)) with
   (0+(C*kp*emission a x+optical_depth C kap x*(4*a*x^3))) by ring.
   apply (@is_derive_plus R_AbsRing R_NormedModule); [apply (@is_derive_const R_AbsRing R_NormedModule)|apply Derive.is_derive_mult; assumption].
 - replace (C*kp) with (0+C*kp) by ring.
   apply (@is_derive_plus R_AbsRing R_NormedModule); [apply (@is_derive_const R_AbsRing R_NormedModule)|exact HT].
 - unfold optical_depth; nra.
Qed.

Lemma heating_elasticity_algebra q qp V Vp B Bp r x :
 V<>0 -> r<>0 -> q+V*B<>0 -> x*Bp=4*B ->
 x*(((qp+Vp*B+V*Bp)*(V*r)-(q+V*B)*(Vp*r))/(V*r)^2)/
 ((q+V*B)/(V*r)) =
 (x*qp-(x*Vp/V)*q+4*V*B)/(q+V*B).
Proof.
 intros HV Hr Hsum HB.
 replace (4*V*B) with (V*(4*B)) by ring.
 rewrite <- HB.
 field; auto.
Qed.

Lemma cooling_elasticity_algebra q qp V Vp B Bp r x :
 V<>0 -> B<>0 -> q+V*r<>0 -> x*Bp=4*B ->
 -(x*(((qp+Vp*r)*(V*B)-(q+V*r)*(Vp*B+V*Bp))/(V*B)^2)/
 ((q+V*r)/(V*B))) =
 4+(x*Vp/V)*q/(q+V*r)-x*qp/(q+V*r).
Proof.
 intros HV HB Hsum Hpow.
 replace 4 with (x*Bp/B) by (rewrite Hpow; field; exact HB).
 field; auto.
Qed.

Lemma heating_elasticity_formula A T0 C chi a r kap gas x kp tp :
 0<C -> 0<chi -> 0<kap x -> 0<r ->
 A*(gas x-T0)+coupling C chi kap x*emission a x<>0 ->
 elasticity (heating_balance A T0 C chi a r kap gas) x
 (heating_derivative A T0 C chi a r kap gas x kp tp) =
 (A*x*tp-effective_slope C kap x kp*(A*(gas x-T0))+
  4*coupling C chi kap x*emission a x)/
 (A*(gas x-T0)+coupling C chi kap x*emission a x).
Proof.
 intros HC Hchi Hk Hr Hsum.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 pose proof (coupling_elasticity C chi kap x kp HC Hchi Hk) as Hell.
 unfold elasticity in Hell.
 unfold elasticity, heating_balance, heating_derivative; cbv zeta.
 rewrite heating_elasticity_algebra; try lra.
 - rewrite Hell; f_equal; ring.
 - unfold emission; ring.
Qed.

Lemma cooling_elasticity_formula A T0 C chi a r kap gas x kp tp :
 0<C -> 0<chi -> 0<kap x -> 0<a -> 0<x ->
 A*(T0-gas x)+coupling C chi kap x*r<>0 ->
 -elasticity (cooling_balance A T0 C chi a r kap gas) x
 (cooling_derivative A T0 C chi a r kap gas x kp tp) =
 4+effective_slope C kap x kp*(A*(T0-gas x))/
 (A*(T0-gas x)+coupling C chi kap x*r)+
 A*x*tp/(A*(T0-gas x)+coupling C chi kap x*r).
Proof.
 intros HC Hchi Hk Ha Hx Hsum.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 pose proof (emission_positive a x Ha Hx) as HB.
 pose proof (coupling_elasticity C chi kap x kp HC Hchi Hk) as Hell.
 unfold elasticity in Hell.
 unfold elasticity, cooling_balance, cooling_derivative; cbv zeta.
 rewrite cooling_elasticity_algebra; try lra.
 - rewrite Hell; field; exact Hsum.
 - unfold emission; ring.
Qed.

Lemma radiation_elasticity_formula C a r kap x kp :
 0<C -> 0<kap x -> 0<a -> 0<x -> 0<r ->
 elasticity (radiation C a r kap) x (radiation_derivative C a r kap x kp) =
 4*optical_depth C kap x*emission a x/(r+optical_depth C kap x*emission a x)+
 opacity_slope kap x kp *
 (optical_depth C kap x*(emission a x-r)/
 ((1+optical_depth C kap x)*(r+optical_depth C kap x*emission a x))).
Proof.
 intros HC Hk Ha Hx Hr.
 pose proof (optical_depth_positive C kap x HC Hk) as Htau.
 pose proof (emission_positive a x Ha Hx) as HB.
 assert (Hsum: r+optical_depth C kap x*emission a x<>0) by nra.
 unfold elasticity, radiation, radiation_derivative; cbv zeta.
 unfold opacity_slope.
 unfold optical_depth, emission in *.
 field; repeat split; nra.
Qed.

Lemma heating_margin_without_transfer_division q z v ell mu :
 0<=q -> q<=z -> 0<v -> mu<=1-ell -> mu<=4 ->
 mu <= (z-ell*q+4*v)/(q+v).
Proof.
 intros Hq Hz Hv Hmu H4.
 assert (H1:0<=q*(1-ell-mu)) by (apply Rmult_le_pos; lra).
 assert (H2:0<=v*(4-mu)) by (apply Rmult_le_pos; lra).
 apply (Rmult_le_reg_r (q+v)); [lra|].
 field_simplify; nra.
Qed.

(* The derivative supplied by GasMap is an inverse of a strictly positive
   primitive derivative. The following facts are proved from that formula and
   the physical equation, not assumed as an inverse condition number. *)
Definition physical_gas_derivative A D T0 t :=
 1/(1+A/(2*D*sqrt t)*(1+T0/t)).

Lemma physical_gas_derivative_positive A D T0 t :
 0<A -> 0<D -> 0<T0 -> 0<t ->
 0<physical_gas_derivative A D T0 t.
Proof.
 intros HA HD HT0 Ht.
 pose proof (sqrt_lt_R0 t Ht) as Hsqrt.
 assert (0<A/(2*D*sqrt t)) by (apply Rdiv_lt_0_compat; nra).
 assert (0<T0/t) by (apply Rdiv_lt_0_compat; assumption).
 unfold physical_gas_derivative; apply Rdiv_lt_0_compat; nra.
Qed.

Lemma physical_gas_elasticity A D T0 t x :
 0<A -> 0<D -> 0<T0 -> 0<t ->
 x=t+A*(t-T0)/(D*sqrt t) ->
 x*physical_gas_derivative A D T0 t/t =
 map_elasticity (D*sqrt t) A (T0/t).
Proof.
 intros HA HD HT0 Ht Hmap.
 pose proof (sqrt_lt_R0 t Ht) as Hsqrt.
 assert (Hratio:0<T0/t) by (apply Rdiv_lt_0_compat; assumption).
 assert (HK:0<D*sqrt t) by nra.
 assert (Hden:0<D*sqrt t+A*(1+T0/t)/2) by nra.
 rewrite Hmap.
 unfold physical_gas_derivative, map_elasticity.
 field; repeat split; nra.
Qed.

Lemma heating_map_elasticity_lower K A z :
 0<K -> 0<A -> 0<z<=1 -> 1-z<=map_elasticity K A z.
Proof.
 intros HK HA Hz.
 pose proof (map_den_positive K A z HK HA ltac:(lra)) as Hd.
 assert (HKz:0<=K*z) by nra.
 assert (HAz:0<=A*(1-z)^2) by (apply Rmult_le_pos; [lra|apply pow2_ge_0]).
 unfold map_elasticity.
 apply (Rmult_le_reg_r (K+A*(1+z)/2)); [lra|].
 field_simplify; nra.
Qed.

Lemma heating_transfer_derivative_lower A D T0 t x :
 0<A -> 0<D -> 0<T0 -> T0<=t ->
 x=t+A*(t-T0)/(D*sqrt t) ->
 A*(t-T0)<=A*x*physical_gas_derivative A D T0 t.
Proof.
 intros HA HD HT0 Hheat Hmap.
 assert (Ht:0<t) by lra.
 pose proof (sqrt_lt_R0 t Ht) as Hsqrt.
 assert (Hz:0<T0/t<=1).
 { split; [apply Rdiv_lt_0_compat; assumption|].
   apply (Rmult_le_reg_r t); [lra|]; field_simplify; nra. }
 pose proof (heating_map_elasticity_lower (D*sqrt t) A (T0/t) ltac:(nra) HA Hz) as Hel.
 rewrite <- (physical_gas_elasticity A D T0 t x HA HD HT0 Ht Hmap) in Hel.
 assert (Hprod: (1-T0/t)*t <= (x*physical_gas_derivative A D T0 t/t)*t).
 { apply Rmult_le_compat_r; lra. }
 field_simplify in Hprod; nra.
Qed.

Lemma heating_balance_positive A T0 C chi a r kap gas x :
 0<A -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<x -> 0<kap x ->
 T0<=gas x -> 0<heating_balance A T0 C chi a r kap gas x.
Proof.
 intros HA HC Hchi Ha Hr Hx Hk Hheat.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 pose proof (emission_positive a x Ha Hx) as HB.
 unfold heating_balance; apply Rdiv_lt_0_compat; nra.
Qed.
Lemma cooling_balance_positive A T0 C chi a r kap gas x :
 0<A -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<x -> 0<kap x ->
 gas x<=T0 -> 0<cooling_balance A T0 C chi a r kap gas x.
Proof.
 intros HA HC Hchi Ha Hr Hx Hk Hcool.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 pose proof (emission_positive a x Ha Hx) as HB.
 unfold cooling_balance; apply Rdiv_lt_0_compat; nra.
Qed.
Lemma radiation_positive C a r kap x :
 0<C -> 0<a -> 0<r -> 0<x -> 0<kap x -> 0<radiation C a r kap x.
Proof.
 intros HC Ha Hr Hx Hk.
 pose proof (optical_depth_positive C kap x HC Hk) as Htau.
 pose proof (emission_positive a x Ha Hx) as HB.
 unfold radiation; apply Rdiv_lt_0_compat; nra.
Qed.

Theorem heating_conditioning A D T0 C chi a r kap gas x kp :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<x ->
 0<kap x -> T0<=gas x ->
 x=gas x+A*(gas x-T0)/(D*sqrt (gas x)) ->
 forall mu, mu<=1-effective_slope C kap x kp -> mu<=4 ->
 mu<=elasticity (heating_balance A T0 C chi a r kap gas) x
 (heating_derivative A T0 C chi a r kap gas x kp
 (physical_gas_derivative A D T0 (gas x))).
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hx Hk Hheat Hmap mu Hmu H4.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 pose proof (emission_positive a x Ha Hx) as HB.
 assert (Hq:0<=A*(gas x-T0)) by nra.
 pose proof (heating_transfer_derivative_lower A D T0 (gas x) x
  HA HD HT0 Hheat Hmap) as Hq'.
 rewrite heating_elasticity_formula; try assumption; try nra.
 replace (4*coupling C chi kap x*emission a x) with
 (4*(coupling C chi kap x*emission a x)) by ring.
 apply heating_margin_without_transfer_division; assumption || nra.
Qed.

Theorem cooling_conditioning A D T0 C chi a r kap gas x kp :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<x ->
 0<kap x -> 0<gas x -> gas x<=T0 ->
 forall mu, mu<=4 -> mu<=4+effective_slope C kap x kp ->
 mu<= -elasticity (cooling_balance A T0 C chi a r kap gas) x
 (cooling_derivative A T0 C chi a r kap gas x kp
 (physical_gas_derivative A D T0 (gas x))).
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hx Hk Ht Hcool mu H4 Hmu.
 pose proof (coupling_positive C chi kap x HC Hchi Hk) as HV.
 pose proof (emission_positive a x Ha Hx) as HB.
 assert (Hq:0<=A*(T0-gas x)) by nra.
 pose proof (physical_gas_derivative_positive A D T0 (gas x) HA HD HT0 Ht) as Htp.
 rewrite cooling_elasticity_formula; try assumption; try nra.
 replace (effective_slope C kap x kp*(A*(T0-gas x))/
 (A*(T0-gas x)+coupling C chi kap x*r)) with
 (effective_slope C kap x kp*((A*(T0-gas x))/
 (A*(T0-gas x)+coupling C chi kap x*r))) by (unfold Rdiv; ring).
 apply cooling_margin; try assumption.
 - apply positive_fraction; nra.
 - left; apply Rdiv_lt_0_compat; [apply Rmult_lt_0_compat; nra|nra].
Qed.

Theorem radiation_conditioning C a r kap x kp P :
 0<C -> 0<a -> 0<r -> 0<x -> 0<kap x ->
 0<=P -> Rabs (opacity_slope kap x kp)<=P ->
 Rabs (elasticity (radiation C a r kap) x
 (radiation_derivative C a r kap x kp))<=4+P.
Proof.
 intros HC Ha Hr Hx Hk HP Hp.
 rewrite radiation_elasticity_formula; try assumption.
 apply radiation_slope_bound; try assumption.
 - apply optical_depth_positive; assumption.
 - apply emission_positive; assumption.
Qed.

(* This bridge exposes genuine derivatives in log-temperature, suitable for a
   mean-value theorem. It needs only a positive value at the point; is_derive
   supplies local continuity and the required chain rules. *)
Lemma log_temperature_is_derive (f:R->R) x df :
 0<x -> 0<f x -> is_derive f x df ->
 is_derive (fun y=>ln (f (exp y))) (ln x) (elasticity f x df).
Proof.
 intros Hx Hf Hder.
 assert (Hln:is_derive (fun y=>ln (f y)) x (df/f x)).
 { unfold Rdiv.
   apply (@is_derive_comp R_AbsRing R_NormedModule ln f x (/f x) df).
   - apply is_derive_ln; exact Hf.
   - exact Hder. }
 unfold elasticity, Rdiv.
 replace (x*df*/f x) with (x*(df*/f x)) by ring.
 apply (@is_derive_comp R_AbsRing R_NormedModule (fun y=>ln (f y)) exp
 (ln x) (df*/f x) x).
 - rewrite exp_ln by assumption; exact Hln.
 - replace x with (exp (ln x)) at 2 by (rewrite exp_ln; assumption || reflexivity).
   apply is_derive_exp.
Qed.

Lemma physical_gas_elasticity_bounds A D T0 t x :
 0<A -> 0<D -> 0<T0 -> 0<t -> 0<x ->
 x=t+A*(t-T0)/(D*sqrt t) ->
 0<x*physical_gas_derivative A D T0 t/t<=2.
Proof.
 intros HA HD HT0 Ht Hx Hmap.
 pose proof (sqrt_lt_R0 t Ht) as Hsqrt.
 rewrite (physical_gas_elasticity A D T0 t x HA HD HT0 Ht Hmap).
 apply map_elasticity_bounds; try assumption; try nra.
 - apply Rdiv_lt_0_compat; assumption.
 - replace (D*sqrt t+A*(1-T0/t)) with (D*sqrt t*x/t).
   + apply Rdiv_lt_0_compat; [apply Rmult_lt_0_compat; nra|assumption].
   + rewrite Hmap; field; repeat split; nra.
Qed.
Lemma physical_cooling_gas_elasticity_bound A D T0 t x :
 0<A -> 0<D -> 0<T0 -> 0<t -> t<=T0 ->
 x=t+A*(t-T0)/(D*sqrt t) ->
 x*physical_gas_derivative A D T0 t/t<=1.
Proof.
 intros HA HD HT0 Ht Hcool Hmap.
 pose proof (sqrt_lt_R0 t Ht) as Hsqrt.
 rewrite (physical_gas_elasticity A D T0 t x HA HD HT0 Ht Hmap).
 apply cooling_map_elasticity; try assumption; try nra.
 apply (Rmult_le_reg_r t); [lra|]; field_simplify; nra.
Qed.

Definition exact_heating (A D T0 C chi a r:R) (kap:R->R) :=
 heating_balance A T0 C chi a r kap (gas_map A D T0).
Definition exact_cooling (A D T0 C chi a r:R) (kap:R->R) :=
 cooling_balance A T0 C chi a r kap (gas_map A D T0).

(* End-to-end certificates. Only the opacity's primitive derivative remains an
   assumption; gas-map differentiability and conditioning are established. *)
Theorem exact_heating_log_derivative A D T0 C chi a r kap x kp mu :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 T0<=x -> 0<kap x -> is_derive kap x kp ->
 mu<=1-effective_slope C kap x kp -> mu<=4 ->
 exists s, is_derive (fun y=>ln (exact_heating A D T0 C chi a r kap (exp y)))
  (ln x) s /\ mu<=s.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hheat Hk Hkap Hmu H4.
 assert (Hx:0<x) by lra.
 destruct (gas_map_spec A D T0 x HA HD HT0 Hx) as [Ht Hmap].
 pose proof (gas_map_between A D T0 x HA HD HT0 Hx) as Hbetween.
 rewrite Rmin_left,Rmax_right in Hbetween by lra.
 exists (elasticity (exact_heating A D T0 C chi a r kap) x
 (heating_derivative A T0 C chi a r kap (gas_map A D T0) x kp
 (physical_gas_derivative A D T0 (gas_map A D T0 x)))).
 split.
 - apply log_temperature_is_derive; [assumption| |].
   + unfold exact_heating; apply heating_balance_positive; tauto.
   + unfold exact_heating; apply heating_balance_is_derive; try assumption.
     unfold physical_gas_derivative; change (is_derive (gas_map A D T0) x
       (1/forward_derivative A D T0 (gas_map A D T0 x))).
     apply gas_map_is_derive; assumption.
 - unfold exact_heating; apply heating_conditioning; try assumption; try tauto.
   symmetry; exact Hmap.
Qed.

Theorem exact_cooling_log_derivative A D T0 C chi a r kap x kp mu :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> x<=T0 -> 0<kap x -> is_derive kap x kp ->
 mu<=4 -> mu<=4+effective_slope C kap x kp ->
 exists s, is_derive (fun y=>ln (exact_cooling A D T0 C chi a r kap (exp y)))
  (ln x) s /\ mu<= -s.
Proof.
 intros HA HD HT0 HC Hchi Ha Hr Hx Hcool Hk Hkap H4 Hmu.
 destruct (gas_map_spec A D T0 x HA HD HT0 Hx) as [Ht Hmap].
 pose proof (gas_map_between A D T0 x HA HD HT0 Hx) as Hbetween.
 rewrite Rmin_right,Rmax_left in Hbetween by lra.
 exists (elasticity (exact_cooling A D T0 C chi a r kap) x
 (cooling_derivative A T0 C chi a r kap (gas_map A D T0) x kp
 (physical_gas_derivative A D T0 (gas_map A D T0 x)))).
 split.
 - apply log_temperature_is_derive; [assumption| |].
   + unfold exact_cooling; apply cooling_balance_positive; tauto.
   + unfold exact_cooling; apply cooling_balance_is_derive; try assumption.
     unfold physical_gas_derivative; change (is_derive (gas_map A D T0) x
       (1/forward_derivative A D T0 (gas_map A D T0 x))).
     apply gas_map_is_derive; assumption.
 - unfold exact_cooling; apply cooling_conditioning; tauto.
Qed.

Theorem radiation_log_derivative C a r kap x kp P :
 0<C -> 0<a -> 0<r -> 0<x -> 0<kap x -> is_derive kap x kp ->
 0<=P -> Rabs (opacity_slope kap x kp)<=P ->
 exists s, is_derive (fun y=>ln (radiation C a r kap (exp y))) (ln x) s /\
 Rabs s<=4+P.
Proof.
 intros HC Ha Hr Hx Hk Hkap HP Hp.
 exists (elasticity (radiation C a r kap) x (radiation_derivative C a r kap x kp)).
 split.
 - apply log_temperature_is_derive; try assumption.
   + apply radiation_positive; assumption.
   + apply radiation_is_derive; assumption.
 - apply radiation_conditioning; assumption.
Qed.

Theorem gas_map_log_derivative A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x ->
 exists m, is_derive (fun y=>ln (gas_map A D T0 (exp y))) (ln x) m /\
 0<m<=2 /\ (x<=T0 -> m<=1).
Proof.
 intros HA HD HT0 Hx.
 destruct (gas_map_spec A D T0 x HA HD HT0 Hx) as [Ht Hmap].
 pose proof (gas_map_between A D T0 x HA HD HT0 Hx) as Hbetween.
 exists (elasticity (gas_map A D T0) x
 (physical_gas_derivative A D T0 (gas_map A D T0 x))).
 split.
 - apply log_temperature_is_derive; try assumption.
   unfold physical_gas_derivative; change (is_derive (gas_map A D T0) x
     (1/forward_derivative A D T0 (gas_map A D T0 x))).
   apply gas_map_is_derive; assumption.
 - split.
   + unfold elasticity; apply physical_gas_elasticity_bounds; try assumption.
     symmetry; exact Hmap.
   + intro Hcool; unfold elasticity; apply physical_cooling_gas_elasticity_bound; try assumption.
     * rewrite Rmin_right,Rmax_left in Hbetween by lra; tauto.
     * symmetry; exact Hmap.
Qed.

Print Assumptions exact_heating_log_derivative.
Print Assumptions exact_cooling_log_derivative.
Print Assumptions radiation_log_derivative.
Print Assumptions gas_map_log_derivative.

Lemma opacity_log_derivative kap x kp :
 0<x -> 0<kap x -> is_derive kap x kp ->
 is_derive (fun y=>ln (kap (exp y))) (ln x) (opacity_slope kap x kp).
Proof. intros; apply log_temperature_is_derive; assumption. Qed.

Lemma coupling_log_derivative C chi kap x kp :
 0<C -> 0<chi -> 0<x -> 0<kap x -> is_derive kap x kp ->
 is_derive (fun y=>ln (coupling C chi kap (exp y))) (ln x)
 (effective_slope C kap x kp).
Proof.
 intros HC Hchi Hx Hk Hder.
 rewrite <- (coupling_elasticity C chi kap x kp HC Hchi Hk).
 apply log_temperature_is_derive; try assumption.
 - apply coupling_positive; assumption.
 - apply coupling_is_derive; try assumption; unfold optical_depth; nra.
Qed.

Lemma effective_slope_interval p tau lo hi :
 lo<=p<=hi -> 0<=tau ->
 Rmin lo 0<=p/(1+tau)<=Rmax hi 0.
Proof.
 intros Hp Htau.
 pose proof (Rmin_l lo 0) as Hll.
 pose proof (Rmin_r lo 0) as Hl0.
 pose proof (Rmax_l hi 0) as Hhh.
 pose proof (Rmax_r hi 0) as Hh0.
 assert (Hlp:Rmin lo 0*tau<=0) by nra.
 assert (Hhp:0<=Rmax hi 0*tau) by nra.
 split; apply (Rmult_le_reg_r (1+tau)); try lra; field_simplify; nra.
Qed.

Lemma heating_weighted_elasticity q z v ell :
 q<>0 -> q+v<>0 ->
 (z-ell*q+4*v)/(q+v) =
 (q/(q+v))*(z/q-ell)+(1-q/(q+v))*4.
Proof. intros; field; auto. Qed.
