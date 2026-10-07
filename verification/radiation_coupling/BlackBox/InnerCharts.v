(* The three collision charts: actual derivatives and finite recovery bounds. *)
From Coq Require Import Reals Psatz Field.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import LogCalculus GasMap.
Open Scope R_scope.

Definition Hh (T A D d z : R) := (z+A*z/(D*sqrt(T+z)))/d.
Definition Hw (T A D d z : R) := (z+A*z/(D*sqrt(T-z)))/d.
Definition Hs (T A D x z : R) := (x+A*(T-z)/(D*sqrt z))/z.
Definition Hh' (T A D d z : R) :=
 (1 + A/(D*sqrt(T+z))*(1-z/(2*(T+z))))/d.
Definition Hw' (T A D d z : R) :=
 (1 + A/(D*sqrt(T-z))*(1+z/(2*(T-z))))/d.
Definition Hs' (T A D x z : R) :=
 -(x+A*(T-z)/(D*sqrt z)*(3/2+z/(T-z)))/(z*z).

Lemma Hh_derivative T A D d z :
 0<T+z -> D<>0 -> d<>0 -> is_derive (Hh T A D d) z (Hh' T A D d z).
Proof.
 intros Ht HD Hd.
 pose proof (sqrt_lt_R0 (T+z) Ht) as Hsqrt.
 pose proof (Rsqr_sqrt (T+z) ltac:(lra)) as Hss.
 unfold Rsqr in Hss.
 unfold Hh, Hh'.
 auto_derive.
 - repeat split; try assumption; nra.
 - remember (sqrt(T+z)) as s in *.
   assert (HT : T=s*s-z) by nra. rewrite HT.
   field; repeat split; nra.
Qed.

Lemma Hw_derivative T A D d z :
 0<T-z -> D<>0 -> d<>0 -> is_derive (Hw T A D d) z (Hw' T A D d z).
Proof.
 intros Ht HD Hd.
 pose proof (sqrt_lt_R0 (T-z) Ht) as Hsqrt.
 pose proof (Rsqr_sqrt (T-z) ltac:(lra)) as Hss.
 unfold Rsqr in Hss.
 unfold Hw, Hw'. auto_derive; unfold Rminus in *.
 - repeat split; try assumption; nra.
 - remember (sqrt(T + -z)) as s in *.
   assert (HT : T=s*s+z) by nra. rewrite HT.
   field; repeat split; nra.
Qed.

Lemma Hs_derivative T A D x z :
 0<z -> z<T -> D<>0 -> is_derive (Hs T A D x) z (Hs' T A D x z).
Proof.
 intros Hz HzT HD.
 pose proof (sqrt_lt_R0 z Hz) as Hsqrt.
 pose proof (Rsqr_sqrt z ltac:(lra)) as Hss.
 unfold Rsqr in Hss.
 unfold Hs, Hs'. auto_derive; unfold Rminus in *.
 - repeat split; try assumption; nra.
 - remember (sqrt z) as s in *.
   assert (Hzs : z=s*s) by nra. rewrite Hzs.
   field; repeat split; nra.
Qed.

Lemma Hh_positive T A D d z :
 0<T -> 0<A -> 0<D -> 0<d -> 0<z -> 0<Hh T A D d z.
Proof.
 intros HT HA HD Hd Hz. unfold Hh.
 apply Rdiv_lt_0_compat; [|assumption].
 assert (0<sqrt(T+z)) by (apply sqrt_lt_R0; lra).
 assert (0<A*z/(D*sqrt(T+z))) by (apply Rdiv_lt_0_compat; nra). lra.
Qed.
Lemma Hw_positive T A D d z :
 0<A -> 0<D -> 0<d -> 0<z<T -> 0<Hw T A D d z.
Proof.
 intros HA HD Hd Hz. unfold Hw.
 apply Rdiv_lt_0_compat; [|assumption].
 assert (0<sqrt(T-z)) by (apply sqrt_lt_R0; lra).
 assert (0<A*z/(D*sqrt(T-z))) by (apply Rdiv_lt_0_compat; nra). lra.
Qed.
Lemma Hs_positive T A D x z :
 0<A -> 0<D -> 0<x -> 0<z<T -> 0<Hs T A D x z.
Proof.
 intros HA HD Hx Hz. unfold Hs.
 apply Rdiv_lt_0_compat; [|lra].
 assert (0<sqrt z) by (apply sqrt_lt_R0; lra).
 assert (0<A*(T-z)/(D*sqrt z)) by (apply Rdiv_lt_0_compat; nra). lra.
Qed.

Lemma inner_ratio_unit a b : 0<=a -> 0<b -> a<=b -> 0<=a/b<=1.
Proof.
 intros Ha Hb Hab. split.
 - apply Rdiv_le_0_compat; lra.
 - apply (Rmult_le_reg_r b); [lra|]. field_simplify; nra.
Qed.

Lemma Hh_elasticity T A D d z :
 0<T -> 0<A -> 0<D -> 0<d -> 0<z ->
 1/2 <= z*Hh' T A D d z/Hh T A D d z <= 1.
Proof.
 intros HT HA HD Hd Hz.
 pose proof (sqrt_lt_R0 (T+z) ltac:(lra)) as HS.
 assert (HDS:0<D*sqrt(T+z)) by nra.
 assert (HZDS:0<z*(D*sqrt(T+z))) by (apply Rmult_lt_0_compat; lra).
 set (k:=A/(D*sqrt(T+z))).
 assert (Hk:0<k) by (unfold k; apply Rdiv_lt_0_compat; nra).
 assert (HE:z*Hh' T A D d z/Hh T A D d z =
  1-(k/(1+k))*(z/(T+z))/2).
 { unfold Hh', Hh, k. field; repeat split; nra. }
 rewrite HE.
 pose proof (inner_ratio_unit k (1+k) ltac:(lra) ltac:(lra) ltac:(lra)) as Hw.
 pose proof (inner_ratio_unit z (T+z) ltac:(lra) ltac:(lra) ltac:(lra)) as Hsqrt.
 assert (0<=(k/(1+k))*(z/(T+z))) by nra.
 assert ((k/(1+k))*(z/(T+z))<=1) by nra. lra.
Qed.

Lemma Hw_elasticity T A D d z :
 0<T -> 0<A -> 0<D -> 0<d -> 0<z<=T/2 ->
 1 <= z*Hw' T A D d z/Hw T A D d z <= 3/2.
Proof.
 intros HT HA HD Hd Hz.
 pose proof (sqrt_lt_R0 (T-z) ltac:(lra)) as HS.
 assert (HDS:0<D*sqrt(T-z)) by nra.
 assert (HZDS:0<z*(D*sqrt(T-z))) by (apply Rmult_lt_0_compat; lra).
 set (k:=A/(D*sqrt(T-z))).
 assert (Hk:0<k) by (unfold k; apply Rdiv_lt_0_compat; nra).
 assert (HE:z*Hw' T A D d z/Hw T A D d z =
  1+(k/(1+k))*(z/(T-z))/2).
 { unfold Hw', Hw, k. field; repeat split; nra. }
 rewrite HE.
 pose proof (inner_ratio_unit k (1+k) ltac:(lra) ltac:(lra) ltac:(lra)) as Hw.
 pose proof (inner_ratio_unit z (T-z) ltac:(lra) ltac:(lra) ltac:(lra)) as Hsqrt.
 assert (0<=(k/(1+k))*(z/(T-z))) by nra.
 assert ((k/(1+k))*(z/(T-z))<=1) by nra. lra.
Qed.

Lemma Hs_elasticity T A D x z :
 0<T -> 0<A -> 0<D -> 0<x -> 0<z<=T/2 ->
 1 <= -(z*Hs' T A D x z/Hs T A D x z) <= 5/2.
Proof.
 intros HT HA HD Hx Hz.
 pose proof (sqrt_lt_R0 z ltac:(lra)) as HS.
 assert (HDS:0<D*sqrt z) by nra.
 assert (HXDS:0<x*(D*sqrt z)) by (apply Rmult_lt_0_compat; lra).
 set (k:=A*(T-z)/(D*sqrt z)).
 assert (Hk:0<k) by (unfold k; apply Rdiv_lt_0_compat; nra).
 assert (HE:-(z*Hs' T A D x z/Hs T A D x z) =
  1+(k/(x+k))*(z/(T-z)+1/2)).
 { unfold Hs', Hs, k. field; repeat split; nra. }
 rewrite HE.
 pose proof (inner_ratio_unit k (x+k) ltac:(lra) ltac:(lra) ltac:(lra)) as Hw.
 pose proof (inner_ratio_unit z (T-z) ltac:(lra) ltac:(lra) ltac:(lra)) as Hsqrt.
 assert (0<=(k/(x+k))*(z/(T-z)+1/2)) by nra.
 assert ((k/(x+k))*(z/(T-z)+1/2)<=3/2) by nra. lra.
Qed.

Lemma inner_between_positive x y t :
 0<x -> 0<y -> Rmin x y<=t<=Rmax x y -> 0<t.
Proof. unfold Rmin, Rmax; destruct (Rle_dec x y); intros; lra. Qed.
Lemma inner_between_upper x y t U :
 x<=U -> y<=U -> Rmin x y<=t<=Rmax x y -> t<=U.
Proof. unfold Rmin, Rmax; destruct (Rle_dec x y); intros; lra. Qed.
Lemma inner_between_strict_upper x y t U :
 x<U -> y<U -> Rmin x y<=t<=Rmax x y -> t<U.
Proof. unfold Rmin, Rmax; destruct (Rle_dec x y); intros; lra. Qed.

Lemma heating_coordinate_error T A D d z0 z :
 0<T -> 0<A -> 0<D -> 0<d -> 0<z0 -> 0<z -> Hh T A D d z0=1 ->
 log_error z z0 <= 2*Rabs(ln(Hh T A D d z)).
Proof.
 intros HT HA HD Hd Hz0 Hz Hroot.
 pose proof (elasticity_lower_distance (Hh T A D d) (Hh' T A D d)
 z0 z (1/2) Hz0 Hz ltac:(lra)) as H.
 assert (Hp:forall t, Rmin z0 z<=t<=Rmax z0 z -> 0<t).
 { intros t Ht; exact (inner_between_positive z0 z t Hz0 Hz Ht). }
 assert (Hpos:forall t, Rmin z0 z<=t<=Rmax z0 z ->0<Hh T A D d t).
 { intros; apply Hh_positive; auto. }
 assert (Hder:forall t, Rmin z0 z<=t<=Rmax z0 z ->
 is_derive (Hh T A D d) t (Hh' T A D d t)).
 { intros t Ht. apply Hh_derivative; try lra. pose proof (Hp t Ht); lra. }
 specialize (H Hpos ltac:(intros; eapply is_derive_continuity_pt; eauto)
 ltac:(intros; apply Hder; lra)
 ltac:(intros t Ht; apply (proj1 (Hh_elasticity T A D d t HT HA HD Hd (Hp t ltac:(lra)))))).
 unfold log_error in H at 2. rewrite Hroot, ln_1, Rminus_0_r in H.
 eapply Rle_trans; [exact H|]. right; field.
Qed.

Lemma weak_coordinate_error T A D d z0 z :
 0<T -> 0<A -> 0<D -> 0<d -> 0<z0<=T/2 -> 0<z<=T/2 -> Hw T A D d z0=1 ->
 log_error z z0 <= Rabs(ln(Hw T A D d z)).
Proof.
 intros HT HA HD Hd Hz0 Hz Hroot.
 assert (Hp:forall t, Rmin z0 z<=t<=Rmax z0 z -> 0<t<=T/2).
 { intros t Ht; split.
   - apply (inner_between_positive z0 z t); lra.
   - apply (inner_between_upper z0 z t); lra. }
 pose proof (elasticity_lower_distance (Hw T A D d) (Hw' T A D d)
 z0 z 1 ltac:(lra) ltac:(lra) ltac:(lra)) as H.
 assert (Hpos:forall t, Rmin z0 z<=t<=Rmax z0 z ->0<Hw T A D d t).
 { intros t Ht; apply Hw_positive; auto; pose proof (Hp t Ht); lra. }
 assert (Hder:forall t, Rmin z0 z<=t<=Rmax z0 z ->
 is_derive (Hw T A D d) t (Hw' T A D d t)).
 { intros t Ht. apply Hw_derivative; try lra. pose proof (Hp t Ht); lra. }
 specialize (H Hpos ltac:(intros; eapply is_derive_continuity_pt; eauto)
 ltac:(intros; apply Hder; lra)
 ltac:(intros t Ht; apply (proj1 (Hw_elasticity T A D d t HT HA HD Hd (Hp t ltac:(lra)))))).
 unfold log_error in H at 2. rewrite Hroot, ln_1, Rminus_0_r, Rdiv_1 in H. exact H.
Qed.

Lemma positive_scale_log_error A z z0 : 0<A -> 0<z -> 0<z0 ->
 log_error (A*z) (A*z0)=log_error z z0.
Proof.
 intros; unfold log_error; rewrite !ln_mult by assumption.
 f_equal; ring.
Qed.

Lemma heating_recovery_lipschitz T z z0 :
 0<T -> 0<z -> 0<z0 -> log_error (T+z) (T+z0)<=log_error z z0.
Proof.
 intros HT Hz Hz0.
 assert (Hp:forall t, Rmin z0 z<=t<=Rmax z0 z -> 0<t).
 { intros t Ht; exact (inner_between_positive z0 z t Hz0 Hz Ht). }
 pose proof (elasticity_upper_distance (fun t => T+t) (fun _ => 1)
 z0 z 1 Hz0 Hz ltac:(lra)) as H.
 assert (Hder:forall t,is_derive (fun t => T+t) t 1).
 { intro; auto_derive; easy. }
 specialize (H ltac:(intros t Ht; pose proof (Hp t Ht); lra)
 ltac:(intros; eapply is_derive_continuity_pt; apply Hder)
 ltac:(intros; apply Hder)).
 assert (HB:forall t, Rmin z0 z<t<Rmax z0 z -> Rabs(t*1/(T+t))<=1).
 { intros t Ht. pose proof (Hp t ltac:(lra)) as Htp.
   rewrite Rmult_1_r. pose proof (inner_ratio_unit t (T+t) ltac:(lra) ltac:(lra) ltac:(lra)).
   rewrite Rabs_pos_eq by lra. lra. }
 specialize (H HB); simpl in H. lra.
Qed.

(* A larger interval gives the factor two needed when a boundary acceptance
   lies on the opposite side of the exact root. *)
Lemma difference_recovery_lipschitz T z z0 M :
 0<T -> 0<z<T -> 0<z0<T -> 0<=M ->
 z<=M*(T-z) -> z0<=M*(T-z0) ->
 log_error (T-z) (T-z0)<=M*log_error z z0.
Proof.
 intros HT Hz Hz0 HM Huz Huz0.
 assert (Hp:forall t, Rmin z0 z<=t<=Rmax z0 z -> 0<t<T).
 { intros t Ht; split.
   - apply (inner_between_positive z0 z t); lra.
   - apply (inner_between_strict_upper z0 z t); lra. }
 assert (Hub:forall t, Rmin z0 z<=t<=Rmax z0 z -> t<=M*(T-t)).
 { unfold Rmin,Rmax; destruct (Rle_dec z0 z); intros t Ht; nra. }
 pose proof (elasticity_upper_distance (fun t => T-t) (fun _ => -1)
 z0 z M ltac:(lra) ltac:(lra) HM) as H.
 assert (Hder:forall t,is_derive (fun t => T-t) t (-1)).
 { intro; auto_derive; easy. }
 specialize (H ltac:(intros t Ht; pose proof (Hp t Ht); lra)
 ltac:(intros; eapply is_derive_continuity_pt; apply Hder)
 ltac:(intros; apply Hder)).
 apply H. intros t Ht. pose proof (Hp t ltac:(lra)) as Htp.
 pose proof (Hub t ltac:(lra)) as Hubt.
 replace (t * -1/(T-t)) with (-(t/(T-t))) by (field; lra).
 rewrite Rabs_Ropp, Rabs_pos_eq by (apply Rdiv_le_0_compat; lra).
 apply (Rmult_le_reg_r (T-t)); [lra|].
 replace (t/(T-t)*(T-t)) with t by (field; lra). lra.
Qed.

Lemma weak_recovery_lipschitz T z z0 :
 0<T -> 0<z<=T/2 -> 0<z0<=T/2 ->
 log_error (T-z) (T-z0)<=log_error z z0.
Proof.
 intros. replace (log_error z z0) with (1*log_error z z0) by ring.
 apply difference_recovery_lipschitz; lra.
Qed.
Lemma boundary_recovery_lipschitz T z z0 :
 0<T -> 0<z<=2*T/3 -> 0<z0<=2*T/3 ->
 log_error (T-z) (T-z0)<=2*log_error z z0.
Proof. intros; apply difference_recovery_lipschitz; lra. Qed.

(* This lower bound holds beyond the strong chart. It justifies acceptance
   at the shared boundary even if the root is on the weak side. *)
Lemma Hs_global_lower_elasticity T A D x z :
 0<A -> 0<D -> 0<x -> 0<z<T ->
 1 <= -(z*Hs' T A D x z/Hs T A D x z).
Proof.
 intros HA HD Hx Hz.
 pose proof (sqrt_lt_R0 z ltac:(lra)) as HS.
 assert (HDS:0<D*sqrt z) by nra.
 assert (HXDS:0<x*(D*sqrt z)) by (apply Rmult_lt_0_compat; lra).
 set (k:=A*(T-z)/(D*sqrt z)).
 assert (Hk:0<k) by (unfold k; apply Rdiv_lt_0_compat; nra).
 assert (HE:-(z*Hs' T A D x z/Hs T A D x z) =
  1+(k/(x+k))*(z/(T-z)+1/2)).
 { unfold Hs', Hs, k. field; repeat split; nra. }
 rewrite HE.
 assert (0<k/(x+k)) by (apply Rdiv_lt_0_compat; lra).
 assert (0<z/(T-z)) by (apply Rdiv_lt_0_compat; lra). nra.
Qed.

Lemma strong_coordinate_error T A D x z0 z :
 0<A -> 0<D -> 0<x -> 0<z0<T -> 0<z<T -> Hs T A D x z0=1 ->
 log_error z z0 <= Rabs(ln(Hs T A D x z)).
Proof.
 intros HA HD Hx Hz0 Hz Hroot.
 assert (Hp:forall t, Rmin z0 z<=t<=Rmax z0 z -> 0<t<T).
 { intros t Ht; split.
   - apply (inner_between_positive z0 z t); lra.
   - apply (inner_between_strict_upper z0 z t); lra. }
 pose proof (elasticity_negative_lower_distance (Hs T A D x) (Hs' T A D x)
 z0 z 1 ltac:(lra) ltac:(lra) ltac:(lra)) as H.
 assert (Hpos:forall t, Rmin z0 z<=t<=Rmax z0 z ->0<Hs T A D x t).
 { intros t Ht; apply Hs_positive; auto. }
 assert (Hder:forall t, Rmin z0 z<=t<=Rmax z0 z ->
 is_derive (Hs T A D x) t (Hs' T A D x t)).
 { intros t Ht. apply Hs_derivative; pose proof (Hp t Ht); lra. }
 specialize (H Hpos ltac:(intros; eapply is_derive_continuity_pt; eauto)
 ltac:(intros; apply Hder; lra)
 ltac:(intros t Ht; apply Hs_global_lower_elasticity; auto; apply Hp; lra)).
 unfold log_error in H at 2. rewrite Hroot, ln_1, Rminus_0_r, Rdiv_1 in H. exact H.
Qed.

Lemma small_boundary_displacement T z b :
 0<T -> 0<z -> log_error z (T/2)<=b -> b<=ln(4/3) -> z<=2*T/3.
Proof.
 intros HT Hz Herr Hb.
 assert (Hh:0<T/2) by lra.
 assert (Hr:0<z/(T/2)) by (apply Rdiv_lt_0_compat; lra).
 rewrite log_error_ratio in Herr by assumption.
 assert (Hln:ln(z/(T/2))<=ln(4/3)).
 { pose proof (Rle_abs (ln(z/(T/2)))); lra. }
 pose proof (exp_le_mono _ _ Hln) as Hexp.
 rewrite !exp_ln in Hexp by lra.
 apply (Rmult_le_compat_r (T/2)) in Hexp; [|lra].
 replace (z/(T/2)*(T/2)) with z in Hexp by (field; lra). nra.
Qed.

Lemma cooling_boundary_exact_outputs T A D x t b :
 0<T -> 0<A -> 0<D -> 0<x -> 0<t<T -> Hs T A D x t=1 ->
 Rabs(ln(Hs T A D x (T/2)))<=b -> b<=ln(4/3) ->
 log_error (T/2) t<=b /\
 log_error (A*(T-T/2)) (A*(T-t))<=2*b.
Proof.
 intros HT HA HD Hx Ht Hroot Hr Hb.
 pose proof (strong_coordinate_error T A D x t (T/2) HA HD Hx Ht ltac:(lra) Hroot) as He.
 assert (HE:log_error (T/2) t<=b) by lra.
 assert (Htbound:t<=2*T/3).
 { apply (small_boundary_displacement T t b); try lra.
   rewrite log_error_sym; exact HE. }
 split; [exact HE|].
 rewrite positive_scale_log_error by lra.
 pose proof (boundary_recovery_lipschitz T (T/2) t HT ltac:(lra) ltac:(lra)). lra.
Qed.

(* The following contracts separate real-coordinate accuracy from the at most
   two-lambda error of stable floating-point output recovery. *)
Lemma inner_rounded_outputs t q that qhat tstar qstar L e :
 log_error that t<=e -> log_error qhat q<=e ->
 log_error t tstar<=L -> log_error q qstar<=L ->
 log_error that tstar<=L+e /\ log_error qhat qstar<=L+e.
Proof.
 intros Ht Hq Htt Hqq.
 pose proof (log_error_triangle that t tstar).
 pose proof (log_error_triangle qhat q qstar). split; lra.
Qed.

Lemma heating_residual_outputs T A D d z0 z lam that qhat :
 0<T -> 0<A -> 0<D -> 0<d -> 0<z0 -> 0<z -> 0<=lam ->
 Hh T A D d z0=1 -> Rabs(ln(Hh T A D d z))<=25*lam ->
 log_error that (T+z)<=2*lam -> log_error qhat (A*z)<=2*lam ->
 log_error that (T+z0)<=52*lam /\ log_error qhat (A*z0)<=52*lam.
Proof.
 intros HT HA HD Hd Hz0 Hz Hl Hroot Hres Hrt Hrq.
 pose proof (heating_coordinate_error T A D d z0 z HT HA HD Hd Hz0 Hz Hroot) as He.
 pose proof (heating_recovery_lipschitz T z z0 HT Hz Hz0) as Ht.
 pose proof (positive_scale_log_error A z z0 HA Hz Hz0) as Hq.
 pose proof (inner_rounded_outputs (T+z) (A*z) that qhat (T+z0) (A*z0)
 (50*lam) (2*lam) Hrt Hrq ltac:(lra) ltac:(lra)).
 split; lra.
Qed.

Lemma weak_residual_outputs T A D d z0 z lam that qhat :
 0<T -> 0<A -> 0<D -> 0<d -> 0<z0<=T/2 -> 0<z<=T/2 -> 0<=lam ->
 Hw T A D d z0=1 -> Rabs(ln(Hw T A D d z))<=25*lam ->
 log_error that (T-z)<=2*lam -> log_error qhat (A*z)<=2*lam ->
 log_error that (T-z0)<=52*lam /\ log_error qhat (A*z0)<=52*lam.
Proof.
 intros HT HA HD Hd Hz0 Hz Hl Hroot Hres Hrt Hrq.
 pose proof (weak_coordinate_error T A D d z0 z HT HA HD Hd Hz0 Hz Hroot) as He.
 pose proof (weak_recovery_lipschitz T z z0 HT Hz Hz0) as Ht.
 pose proof (positive_scale_log_error A z z0 HA ltac:(lra) ltac:(lra)) as Hq.
 pose proof (inner_rounded_outputs (T-z) (A*z) that qhat (T-z0) (A*z0)
 (50*lam) (2*lam) Hrt Hrq ltac:(lra) ltac:(lra)).
 split; lra.
Qed.

Lemma strong_residual_outputs T A D x z0 z lam that qhat :
 0<T -> 0<A -> 0<D -> 0<x -> 0<z0<=T/2 -> 0<z<=T/2 -> 0<=lam ->
 Hs T A D x z0=1 -> Rabs(ln(Hs T A D x z))<=25*lam ->
 log_error that z<=2*lam -> log_error qhat (A*(T-z))<=2*lam ->
 log_error that z0<=52*lam /\ log_error qhat (A*(T-z0))<=52*lam.
Proof.
 intros HT HA HD Hx Hz0 Hz Hl Hroot Hres Hrt Hrq.
 pose proof (strong_coordinate_error T A D x z0 z HA HD Hx ltac:(lra) ltac:(lra) Hroot) as He.
 pose proof (weak_recovery_lipschitz T z z0 HT Hz Hz0) as Hq0.
 pose proof (positive_scale_log_error A (T-z) (T-z0) HA ltac:(lra) ltac:(lra)) as Hq.
 pose proof (inner_rounded_outputs z (A*(T-z)) that qhat z0 (A*(T-z0))
 (50*lam) (2*lam) Hrt Hrq ltac:(lra) ltac:(lra)).
 split; lra.
Qed.

Lemma cooling_boundary_residual_outputs T A D x t lam that qhat :
 0<T -> 0<A -> 0<D -> 0<x -> 0<t<T -> 0<=lam ->
 Hs T A D x t=1 -> Rabs(ln(Hs T A D x (T/2)))<=25*lam ->
 25*lam<=ln(4/3) ->
 log_error that (T/2)<=2*lam -> log_error qhat (A*(T-T/2))<=2*lam ->
 log_error that t<=52*lam /\ log_error qhat (A*(T-t))<=52*lam.
Proof.
 intros HT HA HD Hx Ht Hl Hroot Hres Hsmall Hrt Hrq.
 pose proof (cooling_boundary_exact_outputs T A D x t (25*lam) HT HA HD Hx Ht Hroot Hres Hsmall) as He.
 pose proof (inner_rounded_outputs (T/2) (A*(T-T/2)) that qhat t (A*(T-t))
 (50*lam) (2*lam) Hrt Hrq ltac:(lra) ltac:(lra)).
 split; lra.
Qed.

Lemma heating_bracket_outputs T A lo hi z0 z lam that qhat :
 0<T -> 0<A -> 0<lo -> lo<=z0<=hi -> lo<=z<=hi -> 0<=lam ->
 ln(hi/lo)<=8*lam ->
 log_error that (T+z)<=2*lam -> log_error qhat (A*z)<=2*lam ->
 log_error that (T+z0)<=52*lam /\ log_error qhat (A*z0)<=52*lam.
Proof.
 intros HT HA Hl Hz0 Hz Hlam Hwidth Hrt Hrq.
 pose proof (log_bracket_width lo hi z z0 Hl Hz Hz0) as He.
 pose proof (heating_recovery_lipschitz T z z0 HT ltac:(lra) ltac:(lra)) as Ht.
 pose proof (positive_scale_log_error A z z0 HA ltac:(lra) ltac:(lra)) as Hq.
 pose proof (inner_rounded_outputs (T+z) (A*z) that qhat (T+z0) (A*z0)
 (50*lam) (2*lam) Hrt Hrq ltac:(lra) ltac:(lra)). split; lra.
Qed.

Lemma weak_bracket_outputs T A lo hi z0 z lam that qhat :
 0<T -> 0<A -> 0<lo -> lo<=z0<=hi -> lo<=z<=hi -> hi<=T/2 -> 0<=lam ->
 ln(hi/lo)<=8*lam ->
 log_error that (T-z)<=2*lam -> log_error qhat (A*z)<=2*lam ->
 log_error that (T-z0)<=52*lam /\ log_error qhat (A*z0)<=52*lam.
Proof.
 intros HT HA Hl Hz0 Hz Hhi Hlam Hwidth Hrt Hrq.
 pose proof (log_bracket_width lo hi z z0 Hl Hz Hz0) as He.
 pose proof (weak_recovery_lipschitz T z z0 HT ltac:(lra) ltac:(lra)) as Ht.
 pose proof (positive_scale_log_error A z z0 HA ltac:(lra) ltac:(lra)) as Hq.
 pose proof (inner_rounded_outputs (T-z) (A*z) that qhat (T-z0) (A*z0)
 (50*lam) (2*lam) Hrt Hrq ltac:(lra) ltac:(lra)). split; lra.
Qed.

Lemma strong_bracket_outputs T A lo hi z0 z lam that qhat :
 0<T -> 0<A -> 0<lo -> lo<=z0<=hi -> lo<=z<=hi -> hi<=T/2 -> 0<=lam ->
 ln(hi/lo)<=8*lam ->
 log_error that z<=2*lam -> log_error qhat (A*(T-z))<=2*lam ->
 log_error that z0<=52*lam /\ log_error qhat (A*(T-z0))<=52*lam.
Proof.
 intros HT HA Hl Hz0 Hz Hhi Hlam Hwidth Hrt Hrq.
 pose proof (log_bracket_width lo hi z z0 Hl Hz Hz0) as He.
 pose proof (weak_recovery_lipschitz T z z0 HT ltac:(lra) ltac:(lra)) as Hq0.
 pose proof (positive_scale_log_error A (T-z) (T-z0) HA ltac:(lra) ltac:(lra)) as Hq.
 pose proof (inner_rounded_outputs z (A*(T-z)) that qhat z0 (A*(T-z0))
 (50*lam) (2*lam) Hrt Hrq ltac:(lra) ltac:(lra)). split; lra.
Qed.

(* Root identities connect these chart theorems to the specified exact gas map;
   the chart root is not an unconnected assumed numerical answer. *)
Lemma gas_map_heating_chart A D T x :
 0<A -> 0<D -> 0<T -> T<x ->
 Hh T A D (x-T) (gas_map A D T x-T)=1.
Proof.
 intros HA HD HT Hx.
 destruct (gas_map_spec A D T x HA HD HT ltac:(lra)) as [Hg HF].
 unfold forward_map in HF. unfold Hh.
 replace (T+(gas_map A D T x-T)) with (gas_map A D T x) by ring.
 replace (gas_map A D T x-T + A*(gas_map A D T x-T)/(D*sqrt(gas_map A D T x)))
 with (x-T) by lra. field; lra.
Qed.

Lemma gas_map_weak_chart A D T x :
 0<A -> 0<D -> 0<x -> x<T ->
 Hw T A D (T-x) (T-gas_map A D T x)=1.
Proof.
 intros HA HD Hx HT.
 destruct (gas_map_spec A D T x HA HD ltac:(lra) Hx) as [Hg HF].
 unfold forward_map in HF. unfold Hw.
 replace (T-(T-gas_map A D T x)) with (gas_map A D T x) by ring.
 assert (HE:T-gas_map A D T x + A*(T-gas_map A D T x)/(D*sqrt(gas_map A D T x))=T-x).
 { unfold Rdiv in *; nra. }
 rewrite HE. field; lra.
Qed.

Lemma gas_map_strong_chart A D T x :
 0<A -> 0<D -> 0<T -> 0<x ->
 Hs T A D x (gas_map A D T x)=1.
Proof.
 intros HA HD HT Hx.
 destruct (gas_map_spec A D T x HA HD HT Hx) as [Hg HF].
 unfold forward_map in HF. unfold Hs.
 assert (HE:x+A*(T-gas_map A D T x)/(D*sqrt(gas_map A D T x))=gas_map A D T x).
 { unfold Rdiv in *; nra. }
 rewrite HE. field; lra.
Qed.

Print Assumptions Hh_derivative.
Print Assumptions Hw_elasticity.
Print Assumptions heating_residual_outputs.
Print Assumptions cooling_boundary_residual_outputs.
Print Assumptions strong_bracket_outputs.
Print Assumptions gas_map_strong_chart.

Corollary gas_map_heating_residual_outputs A D T x z lam that qhat :
 0<A -> 0<D -> 0<T -> T<x -> 0<z -> 0<=lam ->
 Rabs(ln(Hh T A D (x-T) z))<=25*lam ->
 log_error that (T+z)<=2*lam -> log_error qhat (A*z)<=2*lam ->
 log_error that (gas_map A D T x)<=52*lam /\
 log_error qhat (A*(gas_map A D T x-T))<=52*lam.
Proof.
 intros HA HD HT Hx Hz Hl Hres Hrt Hrq.
 pose proof (gas_map_heating_order A D T x HA HD HT Hx) as Hg.
 pose proof (gas_map_heating_chart A D T x HA HD HT Hx) as Hroot.
 pose proof (heating_residual_outputs T A D (x-T) (gas_map A D T x-T) z lam that qhat
 HT HA HD ltac:(lra) ltac:(lra) Hz Hl Hroot Hres Hrt Hrq) as H.
 replace (T+(gas_map A D T x-T)) with (gas_map A D T x) in H by ring. exact H.
Qed.

Corollary gas_map_weak_residual_outputs A D T x z lam that qhat :
 0<A -> 0<D -> 0<x -> x<T -> T/2<=gas_map A D T x -> 0<z<=T/2 -> 0<=lam ->
 Rabs(ln(Hw T A D (T-x) z))<=25*lam ->
 log_error that (T-z)<=2*lam -> log_error qhat (A*z)<=2*lam ->
 log_error that (gas_map A D T x)<=52*lam /\
 log_error qhat (A*(T-gas_map A D T x))<=52*lam.
Proof.
 intros HA HD Hx HT Hweak Hz Hl Hres Hrt Hrq.
 pose proof (gas_map_cooling_order A D T x HA HD Hx HT) as Hg.
 pose proof (gas_map_weak_chart A D T x HA HD Hx HT) as Hroot.
 pose proof (weak_residual_outputs T A D (T-x) (T-gas_map A D T x) z lam that qhat
 ltac:(lra) HA HD ltac:(lra) ltac:(lra) Hz Hl Hroot Hres Hrt Hrq) as H.
 replace (T-(T-gas_map A D T x)) with (gas_map A D T x) in H by ring. exact H.
Qed.

Corollary gas_map_strong_residual_outputs A D T x z lam that qhat :
 0<A -> 0<D -> 0<x -> x<T -> gas_map A D T x<=T/2 -> 0<z<=T/2 -> 0<=lam ->
 Rabs(ln(Hs T A D x z))<=25*lam ->
 log_error that z<=2*lam -> log_error qhat (A*(T-z))<=2*lam ->
 log_error that (gas_map A D T x)<=52*lam /\
 log_error qhat (A*(T-gas_map A D T x))<=52*lam.
Proof.
 intros HA HD Hx HT Hstrong Hz Hl Hres Hrt Hrq.
 pose proof (gas_map_cooling_order A D T x HA HD Hx HT) as Hg.
 pose proof (gas_map_strong_chart A D T x HA HD ltac:(lra) Hx) as Hroot.
 apply (strong_residual_outputs T A D x (gas_map A D T x) z lam that qhat);
 assumption || lra.
Qed.

Corollary gas_map_boundary_residual_outputs A D T x lam that qhat :
 0<A -> 0<D -> 0<x -> x<T -> 0<=lam ->
 Rabs(ln(Hs T A D x (T/2)))<=25*lam -> 25*lam<=ln(4/3) ->
 log_error that (T/2)<=2*lam -> log_error qhat (A*(T-T/2))<=2*lam ->
 log_error that (gas_map A D T x)<=52*lam /\
 log_error qhat (A*(T-gas_map A D T x))<=52*lam.
Proof.
 intros HA HD Hx HT Hl Hres Hsmall Hrt Hrq.
 pose proof (gas_map_cooling_order A D T x HA HD Hx HT) as Hg.
 pose proof (gas_map_strong_chart A D T x HA HD ltac:(lra) Hx) as Hroot.
 apply (cooling_boundary_residual_outputs T A D x (gas_map A D T x) lam that qhat);
 assumption || lra.
Qed.
