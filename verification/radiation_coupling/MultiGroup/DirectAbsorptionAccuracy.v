(* The emission-free branch uses an explicitly charged positive RN graph. *)
From Coq Require Import Reals Psatz Field List.
From BlackBox Require Import GasMap LogCalculus FloatingPoint Guards GuardFloatBridge.
From Coquelicot Require Import Coquelicot.
From MultiGroup Require Import ConstantGroupExistence.
Import ListNotations.
Open Scope R_scope.

Definition da_t T H A := T+H/A.
Definition da_x T H A D := da_t T H A+H/(D*sqrt(da_t T H A)).
Definition da_t_hat (rnd:R->R) T Hh A := rnd(T+rnd(Hh/A)).
Definition da_root_hat (rnd:R->R) T Hh A D :=
 rnd(da_t_hat rnd T Hh A + rnd(Hh/rnd(D*rnd(sqrt(da_t_hat rnd T Hh A))))).
Definition da_nodes (rnd:R->R) T Hh A D :=
 [Hh/A; T+rnd(Hh/A); sqrt(da_t_hat rnd T Hh A);
 D*rnd(sqrt(da_t_hat rnd T Hh A));
 Hh/rnd(D*rnd(sqrt(da_t_hat rnd T Hh A)));
 da_t_hat rnd T Hh A + rnd(Hh/rnd(D*rnd(sqrt(da_t_hat rnd T Hh A))))].

Theorem direct_absorption_graph_budget rnd l e T Hh H A D :
 0<=l -> 0<=e -> 0<T -> 0<A -> 0<D -> LogBound e Hh H ->
 RoundNodes rnd l (da_nodes rnd T Hh A D) ->
 LogBound (e+2*l) (da_t_hat rnd T Hh A) (da_t T H A) /\
 LogBound (3*e/2+5*l) (da_root_hat rnd T Hh A D) (da_x T H A D).
Proof.
 intros Hl He HT HA HD HE N.
 pose proof (LogBound_exact T HT) as ET.
 pose proof (LogBound_exact A HA) as EA.
 pose proof (LogBound_exact D HD) as ED.
 unfold RoundNodes,da_nodes in N.
 pose proof (rounded_div_budget rnd l e 0 Hh A H A HE EA
  (N (Hh/A) ltac:(simpl; tauto))) as Hdiv.
 pose proof (rounded_add_budget rnd l 0 (e+0+l) T (rnd(Hh/A)) T (H/A) ET Hdiv
  (N (T+rnd(Hh/A)) ltac:(simpl; tauto))) as Ht.
 assert (HTout:LogBound (e+2*l) (da_t_hat rnd T Hh A) (da_t T H A)).
 { unfold da_t_hat,da_t; eapply LogBound_mono; [|exact Ht].
   rewrite Rmax_right by lra; lra. }
 pose proof (rounded_sqrt_budget rnd l (e+2*l) (da_t_hat rnd T Hh A) (da_t T H A)
  HTout (N (sqrt(da_t_hat rnd T Hh A)) ltac:(simpl; tauto))) as Hs.
 pose proof (rounded_mult_budget rnd l 0 ((e+2*l)/2+l) D
  (rnd(sqrt(da_t_hat rnd T Hh A))) D (sqrt(da_t T H A)) ED Hs
  (N (D*rnd(sqrt(da_t_hat rnd T Hh A))) ltac:(simpl; tauto))) as Hd.
 pose proof (rounded_div_budget rnd l e (0+((e+2*l)/2+l)+l) Hh
  (rnd(D*rnd(sqrt(da_t_hat rnd T Hh A)))) H (D*sqrt(da_t T H A)) HE Hd
  (N (Hh/rnd(D*rnd(sqrt(da_t_hat rnd T Hh A)))) ltac:(simpl; tauto))) as Hq.
 pose proof (rounded_add_budget rnd l (e+2*l) (e+(0+((e+2*l)/2+l)+l)+l)
  (da_t_hat rnd T Hh A) (rnd(Hh/rnd(D*rnd(sqrt(da_t_hat rnd T Hh A)))))
  (da_t T H A) (H/(D*sqrt(da_t T H A))) HTout Hq
  (N (da_t_hat rnd T Hh A + rnd(Hh/rnd(D*rnd(sqrt(da_t_hat rnd T Hh A)))))
  ltac:(simpl; tauto))) as Hx.
 split; [exact HTout|].
 unfold da_root_hat,da_x; eapply LogBound_mono; [|exact Hx].
 rewrite Rmax_right by lra; lra.
Qed.

Theorem direct_absorption_RN64_budget choice e T Hh H A D :
 0<=e -> 0<T -> 0<A -> 0<D -> LogBound e Hh H ->
 FiniteNormalNodes choice (da_nodes (RN64 choice) T Hh A D) ->
 LogBound (e+2*binary64_lambda) (da_t_hat (RN64 choice) T Hh A) (da_t T H A) /\
 LogBound (3*e/2+5*binary64_lambda) (da_root_hat (RN64 choice) T Hh A D) (da_x T H A D).
Proof.
 intros He HT HA HD HH HN.
 rewrite guard_lambda64.
 apply direct_absorption_graph_budget; try assumption.
 - rewrite <-guard_lambda64; pose proof binary64_lambda_positive; lra.
 - apply RN64_rounded_nodes; apply finite_nodes_rounded_normal; exact HN.
Qed.

Lemma direct_absorption_exact_root A D T H n c B :
 0<A -> 0<D -> 0<T -> 0<=H ->
 (forall g x, (g<n)%nat -> c g*B g x=0) ->
 0<da_x T H A D /\
 gas_map A D T (da_x T H A D)=da_t T H A /\
 constant_group_residual A D T n c B H (da_x T H A D)=0.
Proof.
 intros HA HD HT HH HM.
 assert (HHdiv:0<=H/A) by (apply Rdiv_le_0_compat; assumption).
 assert (Ht:0<da_t T H A) by (unfold da_t; lra).
 assert (Hsqrt:0<sqrt(da_t T H A)) by (apply sqrt_lt_R0; exact Ht).
 assert (Hextra:0<=H/(D*sqrt(da_t T H A))) by (apply Rdiv_le_0_compat; nra).
 assert (Hx:0<da_x T H A D) by (unfold da_x; lra).
 assert (Hforward:da_x T H A D=forward_map A D T (da_t T H A)).
 { unfold da_x,forward_map,da_t; field; repeat split; try lra.
   apply Rgt_not_eq; apply sqrt_lt_R0; lra. }
 assert (Hgas:gas_map A D T (da_x T H A D)=da_t T H A).
 { rewrite Hforward; apply gas_map_left_inverse; try assumption; rewrite <-Hforward; exact Hx. }
 split; [exact Hx|]; split; [exact Hgas|].
 unfold constant_group_residual; rewrite Hgas.
 assert (Hz:group_emission n c B (da_x T H A D)=0).
 { unfold group_emission; apply group_sum_zero; intros; apply HM; assumption. }
 rewrite Hz; unfold da_t; field; lra.
Qed.

Theorem direct_absorption_finite_accuracy64 choice e T Hh H A D :
 0<=e -> 0<T -> 0<A -> 0<D -> LogBound e Hh H ->
 FiniteNormalNodes choice (da_nodes (RN64 choice) T Hh A D) ->
 FiniteNormalNodes choice [A*da_t_hat (RN64 choice) T Hh A] ->
 Rabs(da_root_hat (RN64 choice) T Hh A D/da_x T H A D-1)<=
  exp(3*e/2+5*binary64_lambda)-1 /\
 Rabs(RN64 choice (A*da_t_hat (RN64 choice) T Hh A)/(A*da_t T H A)-1)<=
  exp(e+3*binary64_lambda)-1.
Proof.
 intros He HT HA HD HH HN HNg.
 destruct (direct_absorption_RN64_budget choice e T Hh H A D He HT HA HD HH HN) as [Htemp Hroot].
 pose proof (RN64_rounded_nodes choice _ (finite_nodes_rounded_normal choice _ HNg)) as NG.
 pose proof (gas_energy_graph_budget (RN64 choice) lambda64 (e+2*binary64_lambda)
  A (da_t_hat (RN64 choice) T Hh A) (da_t T H A) HA Htemp
  (NG _ ltac:(simpl; auto))) as Hgas.
 rewrite <-guard_lambda64 in Hgas.
 replace (e+2*binary64_lambda+binary64_lambda) with (e+3*binary64_lambda) in Hgas by ring.
 destruct Hroot as [HXh [HX Hxe]].
 destruct Hgas as [HUh [HU Hue]].
 split; apply log_error_relative; assumption.
Qed.

Theorem direct_zero_absorption A D T n c B :
 0<A -> 0<D -> 0<T ->
 (forall g x, (g<n)%nat -> c g*B g x=0) ->
 gas_map A D T T=T /\ constant_group_residual A D T n c B 0 T=0.
Proof.
 intros HA HD HT HM.
 rewrite gas_map_equilibrium by assumption.
 split; [reflexivity|].
 unfold constant_group_residual; rewrite gas_map_equilibrium by assumption.
 assert (Hz:group_emission n c B T=0).
 { unfold group_emission; apply group_sum_zero; intros; apply HM; assumption. }
 rewrite Hz; ring.
Qed.

Print Assumptions direct_absorption_exact_root.
Print Assumptions direct_absorption_finite_accuracy64.
