(* Actual outer functions: finite inverse conditioning, uniqueness and output
   accuracy. No finite Lipschitz or inverse-condition estimate is assumed. *)
From Coq Require Import Reals Psatz Field.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import OuterDerivatives GasMap GasBounds LogCalculus.
Open Scope R_scope.

Definition physical_heating A D T0 C chi a r kap :=
 heating_balance A T0 C chi a r kap (gas_map A D T0).
Definition physical_cooling A D T0 C chi a r kap :=
 cooling_balance A T0 C chi a r kap (gas_map A D T0).
Definition physical_heating_derivative A D T0 C chi a r kap kp x :=
 heating_derivative A T0 C chi a r kap (gas_map A D T0) x (kp x)
  (gas_derivative A D T0 x).
Definition physical_cooling_derivative A D T0 C chi a r kap kp x :=
 cooling_derivative A T0 C chi a r kap (gas_map A D T0) x (kp x)
  (gas_derivative A D T0 x).

Lemma heating_balance_continuous A T0 C chi a r kap gas x :
 0<C -> 0<chi -> 0<r -> 0<kap x ->
 continuity_pt kap x -> continuity_pt gas x ->
 continuity_pt (heating_balance A T0 C chi a r kap gas) x.
Proof.
 intros HC Hchi Hr Hk Hkap Hgas.
 unfold heating_balance,coupling,optical_depth,emission.
 reg; try assumption; try nra.
 apply Rmult_integral_contrapositive_currified; [|lra].
 assert (Hp:0<chi*(C*kap x)/(1+C*kap x)) by
   (apply Rdiv_lt_0_compat; [repeat apply Rmult_lt_0_compat; assumption|nra]); lra.
Qed.

Lemma cooling_balance_continuous A T0 C chi a r kap gas x :
 0<C -> 0<chi -> 0<a -> 0<x -> 0<kap x ->
 continuity_pt kap x -> continuity_pt gas x ->
 continuity_pt (cooling_balance A T0 C chi a r kap gas) x.
Proof.
 intros HC Hchi Ha Hx Hk Hkap Hgas.
 unfold cooling_balance,coupling,optical_depth,emission.
 reg; try assumption; try nra.
 apply Rmult_integral_contrapositive_currified.
 - assert (Hp:0<chi*(C*kap x)/(1+C*kap x)) by
   (apply Rdiv_lt_0_compat; [repeat apply Rmult_lt_0_compat; assumption|nra]); lra.
 - apply Rmult_integral_contrapositive_currified; [lra|].
   apply pow_nonzero; lra.
Qed.

Lemma radiation_continuous C a r kap x :
 0<C -> 0<kap x -> continuity_pt kap x ->
 continuity_pt (radiation C a r kap) x.
Proof.
 intros HC Hk Hkap; unfold radiation,optical_depth,emission.
 reg; assumption || nra.
Qed.

Lemma closed_interval_lower a b t L : L<=a -> L<=b ->
 Rmin a b<=t<=Rmax a b -> L<=t.
Proof. intros; unfold Rmin in *; destruct (Rle_dec a b); lra. Qed.
Lemma closed_interval_upper a b t U : a<=U -> b<=U ->
 Rmin a b<=t<=Rmax a b -> t<=U.
Proof. intros; unfold Rmax in *; destruct (Rle_dec a b); lra. Qed.

Lemma heating_gas_branch A D T0 x :
 0<A -> 0<D -> 0<T0 -> T0<=x -> T0<=gas_map A D T0 x.
Proof.
 intros HA HD HT Hx.
 pose proof (gas_map_between A D T0 x HA HD HT ltac:(lra)) as H.
 rewrite Rmin_left,Rmax_right in H by exact Hx; lra.
Qed.
Lemma cooling_gas_branch A D T0 x :
 0<A -> 0<D -> 0<T0 -> 0<x -> x<=T0 -> gas_map A D T0 x<=T0.
Proof.
 intros HA HD HT Hx Hcool.
 pose proof (gas_map_between A D T0 x HA HD HT Hx) as H.
 rewrite Rmin_right,Rmax_left in H by exact Hcool; lra.
Qed.

Theorem heating_inverse_log_bound A D T0 C chi a r kap kp x y mu :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<y -> T0<=x -> T0<=y -> 0<mu -> mu<=4 ->
 (forall t, Rmin x y<=t<=Rmax x y -> 0<kap t) ->
 (forall t, Rmin x y<=t<=Rmax x y -> continuity_pt kap t) ->
 (forall t, Rmin x y<t<Rmax x y -> is_derive kap t (kp t)) ->
 (forall t, Rmin x y<t<Rmax x y -> mu<=1-effective_slope C kap t (kp t)) ->
 log_error y x <=
 log_error (physical_heating A D T0 C chi a r kap y)
           (physical_heating A D T0 C chi a r kap x)/mu.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hy Hbx Hby Hmu H4 Hkap Hcont Hder Hmargin.
 apply (elasticity_lower_distance (physical_heating A D T0 C chi a r kap)
   (physical_heating_derivative A D T0 C chi a r kap kp) x y mu Hx Hy Hmu).
 - intros t Ht; apply heating_balance_positive; try assumption.
   + apply (positive_closed_interval x y t Hx Hy Ht).
   + apply Hkap; exact Ht.
   + apply heating_gas_branch; try assumption.
     apply (closed_interval_lower x y t T0 Hbx Hby Ht).
 - intros t Ht; apply heating_balance_continuous; try assumption.
   + apply Hkap; exact Ht.
   + apply Hcont; exact Ht.
   + apply gas_map_continuous; try assumption.
     apply (positive_closed_interval x y t Hx Hy Ht).
 - intros t Ht; apply heating_balance_is_derive; try assumption.
   + apply Hder; exact Ht.
   + apply gas_map_is_derive; try assumption.
     apply (positive_closed_interval x y t Hx Hy); lra.
   + apply Hkap; lra.
 - intros t Ht.
   change (mu<=elasticity (heating_balance A T0 C chi a r kap (gas_map A D T0)) t
      (heating_derivative A T0 C chi a r kap (gas_map A D T0) t (kp t)
       (physical_gas_derivative A D T0 (gas_map A D T0 t)))).
   apply heating_conditioning; try assumption.
   + apply (positive_closed_interval x y t Hx Hy); lra.
   + apply Hkap; lra.
   + apply heating_gas_branch; try assumption.
     apply (closed_interval_lower x y t T0 Hbx Hby); lra.
   + symmetry; apply (proj2 (gas_map_spec A D T0 t HA HD HT
       (positive_closed_interval x y t Hx Hy ltac:(lra)))).
   + apply Hmargin; exact Ht.
Qed.

Theorem cooling_inverse_log_bound A D T0 C chi a r kap kp x y mu :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<y -> x<=T0 -> y<=T0 -> 0<mu -> mu<=4 ->
 (forall t, Rmin x y<=t<=Rmax x y -> 0<kap t) ->
 (forall t, Rmin x y<=t<=Rmax x y -> continuity_pt kap t) ->
 (forall t, Rmin x y<t<Rmax x y -> is_derive kap t (kp t)) ->
 (forall t, Rmin x y<t<Rmax x y -> mu<=4+effective_slope C kap t (kp t)) ->
 log_error y x <=
 log_error (physical_cooling A D T0 C chi a r kap y)
           (physical_cooling A D T0 C chi a r kap x)/mu.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hy Hbx Hby Hmu H4 Hkap Hcont Hder Hmargin.
 apply (elasticity_negative_lower_distance (physical_cooling A D T0 C chi a r kap)
   (physical_cooling_derivative A D T0 C chi a r kap kp) x y mu Hx Hy Hmu).
 - intros t Ht; apply cooling_balance_positive; try assumption.
   + apply (positive_closed_interval x y t Hx Hy Ht).
   + apply Hkap; exact Ht.
   + apply cooling_gas_branch; try assumption.
     * apply (positive_closed_interval x y t Hx Hy Ht).
     * apply (closed_interval_upper x y t T0 Hbx Hby Ht).
 - intros t Ht; apply cooling_balance_continuous; try assumption.
   + apply (positive_closed_interval x y t Hx Hy Ht).
   + apply Hkap; exact Ht.
   + apply Hcont; exact Ht.
   + apply gas_map_continuous; try assumption.
     apply (positive_closed_interval x y t Hx Hy Ht).
 - intros t Ht; apply cooling_balance_is_derive; try assumption.
   + apply Hder; exact Ht.
   + apply gas_map_is_derive; try assumption.
     apply (positive_closed_interval x y t Hx Hy); lra.
   + apply Hkap; lra.
   + apply (positive_closed_interval x y t Hx Hy); lra.
 - intros t Ht.
   change (mu<= -elasticity (cooling_balance A T0 C chi a r kap (gas_map A D T0)) t
      (cooling_derivative A T0 C chi a r kap (gas_map A D T0) t (kp t)
       (physical_gas_derivative A D T0 (gas_map A D T0 t)))).
   apply cooling_conditioning; try assumption.
   + apply (positive_closed_interval x y t Hx Hy); lra.
   + apply Hkap; lra.
   + apply (proj1 (gas_map_spec A D T0 t HA HD HT
       (positive_closed_interval x y t Hx Hy ltac:(lra)))).
   + apply cooling_gas_branch; try assumption.
     * apply (positive_closed_interval x y t Hx Hy); lra.
     * apply (closed_interval_upper x y t T0 Hbx Hby); lra.
   + apply Hmargin; exact Ht.
Qed.

Theorem radiation_log_lipschitz C a r kap kp x y P :
 0<C -> 0<a -> 0<r -> 0<x -> 0<y -> 0<=P ->
 (forall t, Rmin x y<=t<=Rmax x y -> 0<kap t) ->
 (forall t, Rmin x y<=t<=Rmax x y -> continuity_pt kap t) ->
 (forall t, Rmin x y<t<Rmax x y -> is_derive kap t (kp t)) ->
 (forall t, Rmin x y<t<Rmax x y -> Rabs(opacity_slope kap t (kp t))<=P) ->
 log_error (radiation C a r kap y) (radiation C a r kap x)<=
 (4+P)*log_error y x.
Proof.
 intros HC Ha Hr Hx Hy HP Hkap Hcont Hder Hslope.
 apply (elasticity_upper_distance (radiation C a r kap)
   (fun t=>radiation_derivative C a r kap t (kp t)) x y (4+P) Hx Hy ltac:(lra)).
 - intros t Ht; apply radiation_positive; try assumption.
   + apply (positive_closed_interval x y t Hx Hy Ht).
   + apply Hkap; exact Ht.
 - intros t Ht; apply radiation_continuous; try assumption.
   + apply Hkap; exact Ht.
   + apply Hcont; exact Ht.
 - intros t Ht; apply radiation_is_derive; try assumption.
   + apply Hder; exact Ht.
   + apply Hkap; lra.
 - intros t Ht; apply radiation_conditioning; try assumption.
   + apply (positive_closed_interval x y t Hx Hy); lra.
   + apply Hkap; lra.
   + apply Hslope; exact Ht.
Qed.

Lemma log_error_zero_eq x y : 0<x -> 0<y -> log_error x y<=0 -> x=y.
Proof.
 intros Hx Hy H.
 unfold log_error in H.
 assert (Hz:Rabs(ln x-ln y)=0) by (pose proof (Rabs_pos (ln x-ln y)); lra).
 apply Rabs_eq_0 in Hz.
 rewrite <- (exp_ln x Hx), <- (exp_ln y Hy); f_equal; lra.
Qed.

Theorem heating_unique A D T0 C chi a r kap kp x y mu :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<y -> T0<=x -> T0<=y -> 0<mu -> mu<=4 ->
 (forall t, Rmin x y<=t<=Rmax x y -> 0<kap t) ->
 (forall t, Rmin x y<=t<=Rmax x y -> continuity_pt kap t) ->
 (forall t, Rmin x y<t<Rmax x y -> is_derive kap t (kp t)) ->
 (forall t, Rmin x y<t<Rmax x y -> mu<=1-effective_slope C kap t (kp t)) ->
 physical_heating A D T0 C chi a r kap x=1 ->
 physical_heating A D T0 C chi a r kap y=1 -> y=x.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hy Hbx Hby Hmu H4 Hkap Hcont Hder Hmargin Hrx Hry.
 pose proof (heating_inverse_log_bound A D T0 C chi a r kap kp x y mu
  HA HD HT HC Hchi Ha Hr Hx Hy Hbx Hby Hmu H4 Hkap Hcont Hder Hmargin) as H.
 rewrite Hrx,Hry in H; unfold log_error at 2 in H.
 rewrite Rminus_diag_eq, Rabs_R0 in H by reflexivity.
 replace (0/mu) with 0 in H by (field; lra).
 apply log_error_zero_eq; assumption.
Qed.

Theorem cooling_unique A D T0 C chi a r kap kp x y mu :
 0<A -> 0<D -> 0<T0 -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<x -> 0<y -> x<=T0 -> y<=T0 -> 0<mu -> mu<=4 ->
 (forall t, Rmin x y<=t<=Rmax x y -> 0<kap t) ->
 (forall t, Rmin x y<=t<=Rmax x y -> continuity_pt kap t) ->
 (forall t, Rmin x y<t<Rmax x y -> is_derive kap t (kp t)) ->
 (forall t, Rmin x y<t<Rmax x y -> mu<=4+effective_slope C kap t (kp t)) ->
 physical_cooling A D T0 C chi a r kap x=1 ->
 physical_cooling A D T0 C chi a r kap y=1 -> y=x.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hy Hbx Hby Hmu H4 Hkap Hcont Hder Hmargin Hrx Hry.
 pose proof (cooling_inverse_log_bound A D T0 C chi a r kap kp x y mu
  HA HD HT HC Hchi Ha Hr Hx Hy Hbx Hby Hmu H4 Hkap Hcont Hder Hmargin) as H.
 rewrite Hrx,Hry in H; unfold log_error at 2 in H.
 rewrite Rminus_diag_eq, Rabs_R0 in H by reflexivity.
 replace (0/mu) with 0 in H by (field; lra).
 apply log_error_zero_eq; assumption.
Qed.

Lemma inverse_residual_budget x root S mu shat beta rho :
 0<mu -> S root=1 ->
 log_error x root<=log_error (S x) (S root)/mu ->
 Rabs(shat-ln(S x))<=beta -> Rabs shat<=rho ->
 log_error x root<=(rho+beta)/mu.
Proof.
 intros Hmu Hroot Hinv Herr Haccept.
 eapply Rle_trans; [exact Hinv|].
 unfold log_error; rewrite Hroot,ln_1,Rminus_0_r.
 unfold Rdiv; apply Rmult_le_compat_r.
 - left; apply Rinv_0_lt_compat; exact Hmu.
 - apply (residual_with_evaluator_error shat (ln(S x)) beta rho Herr Haccept).
Qed.

Section GeneralOutputAccuracy.
 Variables A D T0 C chi a r cg : R.
 Variables kap kp : R -> R.
 Variables root trial : R.
 Hypotheses (HA:0<A) (HD:0<D) (HT:0<T0) (HC:0<C)
  (Hchi:0<chi) (Ha:0<a) (Hr:0<r) (Hcg:0<cg)
  (Hroot:0<root) (Htrial:0<trial).
 Hypothesis Hkap:forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t.
 Hypothesis Hcont:forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t.
 Hypothesis Hder:forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t).

 Theorem heating_residual_log_bound mu shat beta rho :
  T0<=root -> T0<=trial -> 0<mu -> mu<=4 ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    mu<=1-effective_slope C kap t (kp t)) ->
  physical_heating A D T0 C chi a r kap root=1 ->
  Rabs(shat-ln(physical_heating A D T0 C chi a r kap trial))<=beta ->
  Rabs shat<=rho -> log_error trial root<=(rho+beta)/mu.
 Proof.
  intros Hbr Hbt Hmu H4 Hmargin Hbalance Herr Haccept.
  eapply inverse_residual_budget; try eassumption.
  apply (heating_inverse_log_bound A D T0 C chi a r kap kp root trial mu);
   assumption.
 Qed.

 Theorem cooling_residual_log_bound mu shat beta rho :
  root<=T0 -> trial<=T0 -> 0<mu -> mu<=4 ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    mu<=4+effective_slope C kap t (kp t)) ->
  physical_cooling A D T0 C chi a r kap root=1 ->
  Rabs(shat-ln(physical_cooling A D T0 C chi a r kap trial))<=beta ->
  Rabs shat<=rho -> log_error trial root<=(rho+beta)/mu.
 Proof.
  intros Hbr Hbt Hmu H4 Hmargin Hbalance Herr Haccept.
  eapply inverse_residual_budget; try eassumption.
  apply (cooling_inverse_log_bound A D T0 C chi a r kap kp root trial mu);
   assumption.
 Qed.

 Theorem heating_output_from_coordinate P bx gh rh eg er :
  0<=P ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    Rabs(opacity_slope kap t (kp t))<=P) ->
  log_error trial root<=bx ->
  log_error gh (gas_energy cg A D T0 trial)<=eg ->
  log_error rh (radiation C a r kap trial)<=er ->
  log_error gh (gas_energy cg A D T0 root)<=2*bx+eg /\
  log_error rh (radiation C a r kap root)<=(4+P)*bx+er.
 Proof.
  intros HP Hslope Hcoord Hgas Hrad.
  pose proof (gas_energy_log_lipschitz cg A D T0 root trial
   Hcg HA HD HT Hroot Htrial) as HG.
  pose proof (radiation_log_lipschitz C a r kap kp root trial P
   HC Ha Hr Hroot Htrial HP Hkap Hcont Hder Hslope) as HR.
  pose proof (log_error_triangle gh (gas_energy cg A D T0 trial)
   (gas_energy cg A D T0 root)) as HGtri.
  pose proof (log_error_triangle rh (radiation C a r kap trial)
   (radiation C a r kap root)) as HRtri.
  split; nra.
 Qed.

 Theorem cooling_output_from_coordinate P bx gh rh eg er :
  root<=T0 -> trial<=T0 -> 0<=P ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    Rabs(opacity_slope kap t (kp t))<=P) ->
  log_error trial root<=bx ->
  log_error gh (gas_energy cg A D T0 trial)<=eg ->
  log_error rh (radiation C a r kap trial)<=er ->
  log_error gh (gas_energy cg A D T0 root)<=bx+eg /\
  log_error rh (radiation C a r kap root)<=(4+P)*bx+er.
 Proof.
  intros Hbr Hbt HP Hslope Hcoord Hgas Hrad.
  pose proof (gas_energy_cooling_log_lipschitz cg A D T0 root trial
   Hcg HA HD HT Hroot Htrial Hbr Hbt) as HG.
  pose proof (radiation_log_lipschitz C a r kap kp root trial P
   HC Ha Hr Hroot Htrial HP Hkap Hcont Hder Hslope) as HR.
  pose proof (log_error_triangle gh (gas_energy cg A D T0 trial)
   (gas_energy cg A D T0 root)) as HGtri.
  pose proof (log_error_triangle rh (radiation C a r kap trial)
   (radiation C a r kap root)) as HRtri.
  split; nra.
 Qed.

 Theorem heating_residual_outputs mu shat beta rho P gh rh et lam er :
  T0<=root -> T0<=trial -> 0<mu -> mu<=4 -> 0<=P ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    mu<=1-effective_slope C kap t (kp t)) ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    Rabs(opacity_slope kap t (kp t))<=P) ->
  physical_heating A D T0 C chi a r kap root=1 ->
  Rabs(shat-ln(physical_heating A D T0 C chi a r kap trial))<=beta ->
  Rabs shat<=rho ->
  log_error gh (gas_energy cg A D T0 trial)<=et+lam ->
  log_error rh (radiation C a r kap trial)<=er ->
  log_error trial root<=(rho+beta)/mu /\
  log_error gh (gas_energy cg A D T0 root)<=2*((rho+beta)/mu)+et+lam /\
  log_error rh (radiation C a r kap root)<=(4+P)*((rho+beta)/mu)+er.
 Proof.
  intros Hbr Hbt Hmu H4 HP Hmargin Hslope Hbalance Herr Haccept Hgas Hrad.
  pose proof (heating_residual_log_bound mu shat beta rho Hbr Hbt
   Hmu H4 Hmargin Hbalance Herr Haccept) as Hcoord.
  pose proof (heating_output_from_coordinate P ((rho+beta)/mu) gh rh (et+lam) er
   HP Hslope Hcoord Hgas Hrad) as Hout.
  split; [exact Hcoord|]; destruct Hout; split; lra.
 Qed.

 Theorem cooling_residual_outputs mu shat beta rho P gh rh et lam er :
  root<=T0 -> trial<=T0 -> 0<mu -> mu<=4 -> 0<=P ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    mu<=4+effective_slope C kap t (kp t)) ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    Rabs(opacity_slope kap t (kp t))<=P) ->
  physical_cooling A D T0 C chi a r kap root=1 ->
  Rabs(shat-ln(physical_cooling A D T0 C chi a r kap trial))<=beta ->
  Rabs shat<=rho ->
  log_error gh (gas_energy cg A D T0 trial)<=et+lam ->
  log_error rh (radiation C a r kap trial)<=er ->
  log_error trial root<=(rho+beta)/mu /\
  log_error gh (gas_energy cg A D T0 root)<=(rho+beta)/mu+et+lam /\
  log_error rh (radiation C a r kap root)<=(4+P)*((rho+beta)/mu)+er.
 Proof.
  intros Hbr Hbt Hmu H4 HP Hmargin Hslope Hbalance Herr Haccept Hgas Hrad.
  pose proof (cooling_residual_log_bound mu shat beta rho Hbr Hbt
   Hmu H4 Hmargin Hbalance Herr Haccept) as Hcoord.
  pose proof (cooling_output_from_coordinate P ((rho+beta)/mu) gh rh (et+lam) er
   Hbr Hbt HP Hslope Hcoord Hgas Hrad) as Hout.
  split; [exact Hcoord|]; destruct Hout; split; lra.
 Qed.

 Theorem bracket_outputs lb ub P gh rh et lam er :
  0<lb -> lb<=trial<=ub -> lb<=root<=ub -> 0<=P ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    Rabs(opacity_slope kap t (kp t))<=P) ->
  log_error gh (gas_energy cg A D T0 trial)<=et+lam ->
  log_error rh (radiation C a r kap trial)<=er ->
  log_error trial root<=ln(ub/lb) /\
  log_error gh (gas_energy cg A D T0 root)<=2*ln(ub/lb)+et+lam /\
  log_error rh (radiation C a r kap root)<=(4+P)*ln(ub/lb)+er.
 Proof.
  intros Hlb Hbt Hbr HP Hslope Hgas Hrad.
  pose proof (log_bracket_width lb ub trial root Hlb Hbt Hbr) as Hcoord.
  pose proof (heating_output_from_coordinate P (ln(ub/lb)) gh rh (et+lam) er
   HP Hslope Hcoord Hgas Hrad) as Hout.
  split; [exact Hcoord|]; destruct Hout; split; lra.
 Qed.

 Theorem cooling_bracket_outputs lb ub P gh rh et lam er :
  root<=T0 -> trial<=T0 ->
  0<lb -> lb<=trial<=ub -> lb<=root<=ub -> 0<=P ->
  (forall t, Rmin root trial<t<Rmax root trial ->
    Rabs(opacity_slope kap t (kp t))<=P) ->
  log_error gh (gas_energy cg A D T0 trial)<=et+lam ->
  log_error rh (radiation C a r kap trial)<=er ->
  log_error trial root<=ln(ub/lb) /\
  log_error gh (gas_energy cg A D T0 root)<=ln(ub/lb)+et+lam /\
  log_error rh (radiation C a r kap root)<=(4+P)*ln(ub/lb)+er.
 Proof.
  intros Hcr Hct Hlb Hbt Hbr HP Hslope Hgas Hrad.
  pose proof (log_bracket_width lb ub trial root Hlb Hbt Hbr) as Hcoord.
  pose proof (cooling_output_from_coordinate P (ln(ub/lb)) gh rh (et+lam) er
   Hcr Hct HP Hslope Hcoord Hgas Hrad) as Hout.
  split; [exact Hcoord|]; destruct Hout; split; lra.
 Qed.
End GeneralOutputAccuracy.
