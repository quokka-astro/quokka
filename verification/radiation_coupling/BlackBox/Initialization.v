(* Concrete ordinary-precision bracket initialization. The informal phrase
   "a few outward representable steps" is made precise here by one rounded
   multiplication by 1 +/- 8u after division and two correctly-rounded roots.
   No interval or arbitrary-precision operation occurs in this graph. *)
From Coq Require Import Reals Psatz Field List.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap FloatingPoint Guards GuardFloatBridge PhysicalRoot
  OuterDerivatives UniqueRoot LogCalculus EarlyEquilibrium.
Import ListNotations.
Open Scope R_scope.

Definition eval_initial_ratio rnd a T r := rnd(eval_B rnd a T/r).
Definition initial_nodes rnd a T r := B_nodes rnd a T ++ [eval_B rnd a T/r].
Definition eval_radiation_temperature rnd a r := rnd(sqrt(rnd(sqrt(rnd(r/a))))).
Definition radiation_temperature_nodes rnd a r :=
 [r/a; sqrt(rnd(r/a)); sqrt(rnd(sqrt(rnd(r/a))))].
Definition outward_endpoint rnd (heating:bool) a r :=
 let h:=eval_radiation_temperature rnd a r in
 rnd(h*(if heating then 1+8*binary64_u else 1-8*binary64_u)).
Definition outward_nodes rnd (heating:bool) a r := radiation_temperature_nodes rnd a r ++
 [eval_radiation_temperature rnd a r*
   (if heating then 1+8*binary64_u else 1-8*binary64_u)].

Lemma radiation_temperature_graph rnd l a r : 0<=l -> 0<a -> 0<r ->
 RoundNodes rnd l (radiation_temperature_nodes rnd a r) ->
 LogBound ((7/4)*l) (eval_radiation_temperature rnd a r) (radiation_temperature a r).
Proof.
 intros Hl Ha Hr N.
 pose proof (N (r/a) ltac:(simpl; auto)) as H1.
 pose proof (N (sqrt(rnd(r/a))) ltac:(simpl; auto)) as H2.
 pose proof (N (sqrt(rnd(sqrt(rnd(r/a))))) ltac:(simpl; auto)) as H3.
 pose proof (rounded_sqrt_budget _ _ _ _ _ H1 H2) as HS1.
 pose proof (rounded_sqrt_budget _ _ _ _ _ HS1 H3) as HS2.
 unfold eval_radiation_temperature,radiation_temperature.
 eapply LogBound_mono; [|exact HS2]; lra.
Qed.

Lemma initial_ratio_RN64_budget choice a T r : 0<a -> 0<T -> 0<r ->
 List.Forall (fun z => 0<z /\ normal64 z) (initial_nodes (RN64 choice) a T r) ->
 LogBound (5*binary64_lambda) (eval_initial_ratio (RN64 choice) a T r)
   (emission a T/r).
Proof.
 intros Ha HT Hr N.
 apply (RN64_nodes choice) in N. rewrite <- guard_lambda64 in N.
 unfold initial_nodes in N; apply RoundNodes_app in N as [NB ND].
 pose proof (initial_ratio_graph_budget (RN64 choice) binary64_lambda a T r
  ltac:(pose proof binary64_lambda_positive; lra) Ha HT Hr NB
  (ND _ ltac:(simpl; auto))) as H.
 unfold eval_initial_ratio; rewrite exact_B_quartic in H; exact H.
Qed.

Lemma outward_log_margins :
 4*binary64_lambda<=ln(1+8*binary64_u) /\
 ln(1-8*binary64_u)<= -4*binary64_lambda /\ 0<1-8*binary64_u.
Proof.
 pose proof binary64_u_bounds as Hu.
 pose proof binary64_lambda_bounds as Hl.
 assert (Hsmall:binary64_lambda<=2*binary64_u).
 { assert (binary64_u/(1-binary64_u)<=2*binary64_u).
   { apply (Rmult_le_reg_r (1-binary64_u)); [lra|].
     field_simplify; nra. }
   lra. }
 pose proof (guard_ln_lower (1+8*binary64_u) ltac:(lra)) as Hplus.
 pose proof (guard_ln_upper (1-8*binary64_u) ltac:(lra)) as Hminus.
 assert (Hplus':4*(binary64_u/(1-binary64_u))<=
   ((1+8*binary64_u)-1)/(1+8*binary64_u)).
 { apply (Rmult_le_reg_r ((1-binary64_u)*(1+8*binary64_u))); [nra|].
   field_simplify; nra. }
 repeat split; lra.
Qed.

Lemma outward_endpoint_encloses rnd heating a r : 0<a -> 0<r ->
 RoundNodes rnd binary64_lambda (outward_nodes rnd heating a r) ->
 0<outward_endpoint rnd heating a r /\
 (if heating then radiation_temperature a r<outward_endpoint rnd heating a r
  else outward_endpoint rnd heating a r<radiation_temperature a r).
Proof.
 intros Ha Hr N.
 unfold outward_nodes in N; apply RoundNodes_app in N as [NT NO].
 pose proof (radiation_temperature_graph rnd binary64_lambda a r
   ltac:(pose proof binary64_lambda_positive; lra) Ha Hr NT) as Htemp.
 destruct Htemp as [Hhat [HTr Hlog]].
 pose proof (outward_log_margins) as [Hp [Hm Hmp]].
 pose proof binary64_u_bounds as Hu.
 pose proof binary64_lambda_positive as Hl.
 unfold outward_endpoint.
 destruct heating.
 - pose proof (NO _ ltac:(simpl; auto)) as [He [Hex Hround]]. split; [exact He|].
   rewrite <- (exp_ln (radiation_temperature a r) HTr) at 1.
   rewrite <- (exp_ln (rnd(eval_radiation_temperature rnd a r*(1+8*binary64_u))) He) at 1.
   apply exp_increasing.
   rewrite ln_mult in Hround by lra.
   pose proof (guard_abs_bounds _ _ Hlog).
   pose proof (guard_abs_bounds _ _ Hround). lra.
 - pose proof (NO _ ltac:(simpl; auto)) as [He [Hex Hround]]. split; [exact He|].
   rewrite <- (exp_ln (radiation_temperature a r) HTr) at 1.
   rewrite <- (exp_ln (rnd(eval_radiation_temperature rnd a r*(1-8*binary64_u))) He) at 1.
   apply exp_increasing.
   rewrite ln_mult in Hround by lra.
   pose proof (guard_abs_bounds _ _ Hlog).
   pose proof (guard_abs_bounds _ _ Hround). lra.
Qed.

Lemma initial_heating_order a T r zhat : 0<a -> 0<T -> 0<r ->
 LogBound (5*binary64_lambda) zhat (emission a T/r) ->
 zhat<1-64*binary64_u -> T<radiation_temperature a r.
Proof.
 intros Ha HT Hr [Hzh [Hz HE]] Hw.
 pose proof (initial_lower_sign zhat (emission a T/r) Hzh Hz Hw HE) as Hsign.
 destruct (radiation_temperature_spec a r Ha Hr) as [HTr HB].
 assert (Hb:emission a T<r).
 { apply (Rmult_lt_reg_r (/r)); [apply Rinv_0_lt_compat; assumption|].
   unfold Rdiv in Hsign; replace (r*/r) with 1 by (field; lra); exact Hsign. }
 destruct (Rtotal_order T (radiation_temperature a r)) as [Hlt|[He|Hgt]]; [exact Hlt| |].
 - rewrite He,HB in Hb; lra.
 - pose proof (emission_strict_increasing a (radiation_temperature a r) T Ha HTr Hgt); lra.
Qed.

Lemma initial_cooling_order a T r zhat : 0<a -> 0<T -> 0<r ->
 LogBound (5*binary64_lambda) zhat (emission a T/r) ->
 1+64*binary64_u<zhat -> radiation_temperature a r<T.
Proof.
 intros Ha HT Hr [Hzh [Hz HE]] Hw.
 pose proof (initial_upper_sign zhat (emission a T/r) Hzh Hz Hw HE) as Hsign.
 destruct (radiation_temperature_spec a r Ha Hr) as [HTr HB].
 assert (Hb:r<emission a T).
 { apply (Rmult_lt_reg_r (/r)); [apply Rinv_0_lt_compat; assumption|].
   unfold Rdiv in Hsign; replace (r*/r) with 1 by (field; lra); exact Hsign. }
 destruct (Rtotal_order (radiation_temperature a r) T) as [Hlt|[He|Hgt]]; [exact Hlt| |].
 - rewrite <- He,HB in Hb; lra.
 - pose proof (emission_strict_increasing a T (radiation_temperature a r) Ha HT Hgt); lra.
Qed.

Lemma initial_ratio_finite64_budget choice a T r : 0<a -> 0<T -> 0<r ->
 FiniteNormalNodes choice (initial_nodes (RN64 choice) a T r) ->
 LogBound (5*binary64_lambda) (eval_initial_ratio (RN64 choice) a T r)
   (emission a T/r).
Proof.
 intros Ha HT Hr N.
 apply finite_nodes_rounded_normal in N; apply RN64_rounded_nodes in N.
 rewrite <- guard_lambda64 in N.
 unfold initial_nodes in N; apply RoundNodes_app in N as [NB ND].
 pose proof (initial_ratio_graph_budget (RN64 choice) binary64_lambda a T r
  ltac:(pose proof binary64_lambda_positive; lra) Ha HT Hr NB
  (ND _ ltac:(simpl; auto))) as H.
 unfold eval_initial_ratio; rewrite exact_B_quartic in H; exact H.
Qed.

Lemma outward_endpoint_finite64_encloses choice heating a r : 0<a -> 0<r ->
 FiniteNormalNodes choice (outward_nodes (RN64 choice) heating a r) ->
 0<outward_endpoint (RN64 choice) heating a r /\
 (if heating then radiation_temperature a r<outward_endpoint (RN64 choice) heating a r
  else outward_endpoint (RN64 choice) heating a r<radiation_temperature a r).
Proof.
 intros Ha Hr N.
 apply outward_endpoint_encloses; try assumption.
 apply finite_nodes_rounded_normal in N; apply RN64_rounded_nodes in N.
 rewrite guard_lambda64; exact N.
Qed.

Section InitialBrackets.
Variables A D T C chi a r : R.
Variable kap : R->R.
Hypotheses (HA:0<A) (HD:0<D) (HT:0<T) (HC:0<C) (Hchi:0<chi) (Ha:0<a) (Hr:0<r).
Hypothesis Hkap : forall x,
 Rmin T (radiation_temperature a r)<=x<=Rmax T (radiation_temperature a r) ->
 0<kap x /\ continuity_pt kap x.

Theorem heating_initial_bracket choice :
 FiniteNormalNodes choice (initial_nodes (RN64 choice) a T r) ->
 FiniteNormalNodes choice (outward_nodes (RN64 choice) true a r) ->
 eval_initial_ratio (RN64 choice) a T r<1-64*binary64_u ->
 T<outward_endpoint (RN64 choice) true a r /\
 T<=exact_dust A D T C chi a r kap<=outward_endpoint (RN64 choice) true a r.
Proof.
 intros NI NO Hwin.
 pose proof (initial_ratio_finite64_budget choice a T r Ha HT Hr NI) as Hrat.
 pose proof (initial_heating_order a T r _ Ha HT Hr Hrat Hwin) as Hbranch.
 pose proof (outward_endpoint_finite64_encloses choice true a r Ha Hr NO) as Hout.
 destruct (exact_dust_spec A D T C chi a r kap HA HD HT HC Hchi Ha Hr Hkap)
   as [Hx [Hb _]].
 rewrite Rmin_left,Rmax_right in Hb by lra. simpl in Hout; split; lra.
Qed.

Theorem cooling_initial_bracket choice :
 FiniteNormalNodes choice (initial_nodes (RN64 choice) a T r) ->
 FiniteNormalNodes choice (outward_nodes (RN64 choice) false a r) ->
 1+64*binary64_u<eval_initial_ratio (RN64 choice) a T r ->
 0<outward_endpoint (RN64 choice) false a r /\
 outward_endpoint (RN64 choice) false a r<T /\
 outward_endpoint (RN64 choice) false a r<=exact_dust A D T C chi a r kap<=T.
Proof.
 intros NI NO Hwin.
 pose proof (initial_ratio_finite64_budget choice a T r Ha HT Hr NI) as Hrat.
 pose proof (initial_cooling_order a T r _ Ha HT Hr Hrat Hwin) as Hbranch.
 pose proof (outward_endpoint_finite64_encloses choice false a r Ha Hr NO) as Hout.
 destruct (exact_dust_spec A D T C chi a r kap HA HD HT HC Hchi Ha Hr Hkap)
   as [Hx [Hb _]].
 rewrite Rmin_right,Rmax_left in Hb by lra. simpl in Hout; repeat split; lra.
Qed.

Theorem initial_early_finite64_accuracy choice :
 FiniteNormalNodes choice (initial_nodes (RN64 choice) a T r) ->
 FiniteNormalNodes choice [A*T] ->
 1-64*binary64_u<=eval_initial_ratio (RN64 choice) a T r<=1+64*binary64_u ->
 log_error T (exact_dust A D T C chi a r kap)<=(35/2)*binary64_lambda /\
 log_error (RN64 choice (A*T)) (A*exact_gas A D T C chi a r kap)<=(37/2)*binary64_lambda /\
 log_error r (OuterDerivatives.radiation C a r kap (exact_dust A D T C chi a r kap))<=70*binary64_lambda.
Proof.
 intros NI NG Hwin.
 pose proof (initial_ratio_finite64_budget choice a T r Ha HT Hr NI) as [Hz [Hzex Hrat]].
 destruct (exact_dust_spec A D T C chi a r kap HA HD HT HC Hchi Ha Hr Hkap)
   as [Hx [Hb [HE Hcol]]].
 destruct (radiation_temperature_spec a r Ha Hr) as [HTr Hrad].
 pose proof (early_binary64_component_bounds A D T C a r kap
   (radiation_temperature a r) (exact_dust A D T C chi a r kap)
   (eval_initial_ratio (RN64 choice) a T r) HA HD HT HC Ha Hr HTr
   (proj1 (Hkap _ Hb)) Hrad Hb Hwin Hrat) as [Hdust [Hgas HR]].
 apply finite_nodes_rounded_normal in NG; apply RN64_rounded_nodes in NG.
 rewrite <- guard_lambda64 in NG.
 pose proof (NG (A*T) ltac:(simpl; auto)) as [HEp [HEref Herr]].
 split; [exact Hdust|]. split; [|exact HR].
 unfold exact_gas.
 pose proof (log_error_triangle (RN64 choice (A*T)) (A*T)
   (A*gas_map A D T (exact_dust A D T C chi a r kap))) as Htri.
 change (log_error (RN64 choice (A*T)) (A*T)<=binary64_lambda) in Herr.
 lra.
Qed.
End InitialBrackets.

Print Assumptions heating_initial_bracket.
Print Assumptions cooling_initial_bracket.
Print Assumptions initial_early_finite64_accuracy.
