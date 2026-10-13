(* Concrete main-solve accuracy, assembled from accepted primitive RN64
   graphs, actual inner results, and MVT-derived physical sensitivities. *)
From Coq Require Import Reals Psatz Field List ZArith.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import Algebra LogCalculus GasMap GasBounds OuterDerivatives
 OuterBounds FloatingPoint Guards GuardFloatBridge WidthSafety InnerEvaluator.
Import ListNotations.
Open Scope R_scope.

Definition branch_balance (heating:bool) A D T C chi a r kap :=
 if heating then physical_heating A D T C chi a r kap
 else physical_cooling A D T C chi a r kap.
Definition branch_region (heating:bool) T root trial :=
 if heating then T<=root /\ T<=trial else root<=T /\ trial<=T.
Definition outer_window64 s := 1-128*binary64_u<=s<=1+128*binary64_u.
Definition returned_gas choice cg th := RN64 choice (cg*th).
Definition returned_radiation choice r C khat a x :=
 eval_radiation (RN64 choice) r (eval_tau (RN64 choice) C (khat x))
  (eval_B (RN64 choice) a x).

Inductive OuterAccepted64 choice heating qh C khat chi a r root trial : Prop :=
 | OuterResidual64 :
   FiniteNormalNodes choice (outer_nodes (RN64 choice) heating qh C (khat trial) chi a trial r) ->
   outer_window64 (eval_outer (RN64 choice) heating qh C (khat trial) chi a trial r) ->
   OuterAccepted64 choice heating qh C khat chi a r root trial
 | OuterBracket64 : forall lb ub,
   0<lb -> lb<ub -> lb<=root<=ub -> lb<=trial<=ub ->
   SafeDifference64 lb ub -> normal64 (RN64 choice (ub-lb)/lb) ->
   (finite64 (RN64 choice (ub-lb)) /\
    FiniteNormalNodes choice [RN64 choice (ub-lb)/lb]) ->
   RN64 choice (RN64 choice (ub-lb)/lb)<=16*binary64_u ->
   OuterAccepted64 choice heating qh C khat chi a r root trial.

Lemma binary64_exp_budget k K :
 0<=k -> 0<=K ->
 k*(1+K*binary64_u)<=K*(1-binary64_u) ->
 exp(k*binary64_lambda)-1<=K*binary64_u.
Proof.
 intros Hk HK Hroom.
 pose proof binary64_u_bounds as [Hu Hus].
 pose proof binary64_lambda_bounds as [Hl Hlu].
 assert (HKu:0<=K*binary64_u) by (apply Rmult_le_pos; lra).
 assert (Hfrac:k*(binary64_u/(1-binary64_u))<=
                 K*binary64_u/(1+K*binary64_u)).
 { apply (Rmult_le_reg_r ((1-binary64_u)*(1+K*binary64_u))); [nra|].
   field_simplify; nra. }
 pose proof (guard_ln_lower (1+K*binary64_u) ltac:(nra)) as Hln.
 replace ((1+K*binary64_u-1)/(1+K*binary64_u))
    with (K*binary64_u/(1+K*binary64_u)) in Hln by (field; nra).
 assert (Hbound:k*binary64_lambda<=ln(1+K*binary64_u)) by nra.
 pose proof (exp_le_mono _ _ Hbound) as Hexp.
 rewrite exp_ln in Hexp by nra; lra.
Qed.

Lemma binary64_relative_400 x y : 0<x -> 0<y ->
 log_error x y<=400*binary64_lambda -> Rabs(x/y-1)<=512*binary64_u.
Proof.
 intros Hx Hy H.
 eapply Rle_trans; [apply (log_error_relative x y _ Hx Hy H)|].
 apply (binary64_exp_budget 400 512); try lra.
 unfold binary64_u; field_simplify; lra.
Qed.
Lemma binary64_relative_853 x y : 0<x -> 0<y ->
 log_error x y<=853*binary64_lambda -> Rabs(x/y-1)<=1024*binary64_u.
Proof.
 intros Hx Hy H.
 eapply Rle_trans; [apply (log_error_relative x y _ Hx Hy H)|].
 apply (binary64_exp_budget 853 1024); try lra.
 unfold binary64_u; field_simplify; lra.
Qed.
Lemma binary64_relative_3017 x y : 0<x -> 0<y ->
 log_error x y<=3017*binary64_lambda -> Rabs(x/y-1)<=4096*binary64_u.
Proof.
 intros Hx Hy H.
 eapply Rle_trans; [apply (log_error_relative x y _ Hx Hy H)|].
 apply (binary64_exp_budget 3017 4096); try lra.
 unfold binary64_u; field_simplify; lra.
Qed.

Lemma concrete_opacity_margins C kap x kp :
 0<C -> 0<kap x -> -(7/2)<=opacity_slope kap x kp<=1/2 ->
 1/2<=1-effective_slope C kap x kp /\
 1/2<=4+effective_slope C kap x kp /\
 Rabs(opacity_slope kap x kp)<=7/2.
Proof.
 intros HC Hkap Hp.
 pose proof (example_effective_slope (opacity_slope kap x kp)
  (optical_depth C kap x) Hp
  ltac:(pose proof (optical_depth_positive C kap x HC Hkap); lra)) as He.
 unfold effective_slope; split; [lra|]; split; [lra|].
 apply Rabs_le; lra.
Qed.

Lemma concrete_outer_beta :
 outer_beta (52*binary64_lambda) (8*binary64_lambda) lambda64 =
 71*binary64_lambda.
Proof.
 rewrite <-guard_lambda64; unfold outer_beta.
 rewrite Rmax_left by (pose proof binary64_lambda_positive; lra); ring.
Qed.

Lemma physical_coupling_identity C chi kap x :
 chi*saturate(C*kap x)=coupling C chi kap x.
Proof. unfold saturate,coupling,optical_depth,Rdiv; ring. Qed.

Lemma exact_outer_branch (heating:bool) A D T C chi a r kap x :
 0<A -> 0<D -> 0<T -> 0<x ->
 (if heating then T<=x else x<=T) ->
 exact_outer heating (A*Rabs(gas_map A D T x-T))
  (chi*saturate(C*kap x)) (a*x^4) r =
 branch_balance heating A D T C chi a r kap x.
Proof.
 intros HA HD HT Hx Hregion.
 rewrite physical_coupling_identity.
 destruct heating; unfold exact_outer,branch_balance,physical_heating,physical_cooling,
  OuterDerivatives.heating_balance,OuterDerivatives.cooling_balance,emission.
 - pose proof (heating_gas_branch A D T x HA HD HT Hregion).
   rewrite Rabs_right by lra; reflexivity.
 - pose proof (cooling_gas_branch A D T x HA HD HT Hx Hregion).
   rewrite Rabs_left1 by lra.
   replace (-(gas_map A D T x-T)) with (T-gas_map A D T x) by ring.
   reflexivity.
Qed.

Lemma branch_inverse_concrete (heating:bool) A D T C chi a r kap kp root trial :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<root -> 0<trial -> branch_region heating T root trial ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -(7/2)<=opacity_slope kap t (kp t)<=1/2) ->
 log_error trial root <=
 log_error (branch_balance heating A D T C chi a r kap trial)
  (branch_balance heating A D T C chi a r kap root)/(1/2).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion Hkap Hcont Hder Hslope.
 destruct heating; simpl in Hregion |- *; destruct Hregion as [Hbr Hbt].
 - apply (heating_inverse_log_bound A D T C chi a r kap kp root trial (1/2));
    try assumption; try lra.
   intros t Ht; apply (proj1 (concrete_opacity_margins C kap t (kp t)
     HC (Hkap t ltac:(lra)) (Hslope t Ht))).
 - apply (cooling_inverse_log_bound A D T C chi a r kap kp root trial (1/2));
    try assumption; try lra.
   intros t Ht; apply (proj1 (proj2 (concrete_opacity_margins C kap t (kp t)
     HC (Hkap t ltac:(lra)) (Hslope t Ht)))).
Qed.

Lemma inner_outer_graph_concrete choice (heating:bool) A D T C chi a r kap khat x th qh :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<x ->
 (if heating then T<=x else x<=T) ->
 PlanckValueContract kap khat (8*binary64_lambda) ->
 InnerResult choice A D T x th qh ->
 FiniteNormalNodes choice (outer_nodes (RN64 choice) heating qh C (khat x) chi a x r) ->
 LogBound (71*binary64_lambda)
  (eval_outer (RN64 choice) heating qh C (khat x) chi a x r)
  (branch_balance heating A D T C chi a r kap x).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hregion HK HI N.
 destruct (inner_result_accuracy choice A D T x th qh HA HD HT Hx HI)
  as [Ht [[Heq Hqzero]|Hq]].
 - subst qh.
   pose proof (planck_outer_zero_finite64_budget choice (52*binary64_lambda)
    (8*binary64_lambda) kap khat heating C chi a x r HC Hchi Ha Hx Hr HK N) as H.
   rewrite concrete_outer_beta in H.
   assert (Hq:A*Rabs(gas_map A D T x-T)=0).
   { rewrite Heq,gas_map_equilibrium by assumption.
     rewrite Rminus_diag_eq,Rabs_R0 by reflexivity; ring. }
   assert (HE:exact_outer heating 0 (chi*saturate(C*kap x)) (a*x^4) r=
     branch_balance heating A D T C chi a r kap x).
   { rewrite <-Hq; apply exact_outer_branch; assumption. }
   rewrite HE in H; exact H.
 - pose proof (planck_outer_finite64_budget choice (52*binary64_lambda)
    (8*binary64_lambda) kap khat heating qh (A*Rabs(gas_map A D T x-T))
    C chi a x r HC Hchi Ha Hx Hr HK Hq N) as H.
   rewrite concrete_outer_beta,exact_outer_branch in H by assumption; exact H.
Qed.

Lemma returned_gas_concrete choice cg A D T x th qh :
 0<cg -> 0<A -> 0<D -> 0<T -> 0<x ->
 InnerResult choice A D T x th qh ->
 FiniteNormalNodes choice [cg*th] ->
 LogBound (53*binary64_lambda) (returned_gas choice cg th) (gas_energy cg A D T x).
Proof.
 intros Hcg HA HD HT Hx HI N.
 pose proof (proj1 (inner_result_accuracy choice A D T x th qh HA HD HT Hx HI)) as Ht.
 pose proof (RN64_rounded_nodes choice [cg*th] (finite_nodes_rounded_normal choice _ N)) as Hnode.
 pose proof (gas_energy_graph_budget (RN64 choice) lambda64 (52*binary64_lambda)
  cg th (gas_map A D T x) Hcg Ht (Hnode (cg*th) ltac:(simpl; auto))) as H.
 rewrite <-guard_lambda64 in H.
 replace (52*binary64_lambda+binary64_lambda) with (53*binary64_lambda) in H by ring.
 exact H.
Qed.

Lemma returned_radiation_concrete choice C a r kap khat x :
 0<C -> 0<a -> 0<r -> 0<x ->
 PlanckValueContract kap khat (8*binary64_lambda) ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat x) a x) ->
 LogBound (17*binary64_lambda) (returned_radiation choice r C khat a x)
  (OuterDerivatives.radiation C a r kap x).
Proof.
 intros HC Ha Hr Hx HK N.
 pose proof (planck_radiation_finite64_budget choice (8*binary64_lambda) kap khat
  C a x r HC Ha Hx Hr HK N) as H.
 rewrite <-guard_lambda64 in H.
 replace (8*binary64_lambda+9*binary64_lambda) with (17*binary64_lambda) in H by ring.
 exact H.
Qed.

Lemma accepted_coordinate_400 choice heating A D T C chi a r kap kp khat root trial th qh :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<root -> 0<trial -> branch_region heating T root trial ->
 PlanckValueContract kap khat (8*binary64_lambda) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -(7/2)<=opacity_slope kap t (kp t)<=1/2) ->
 branch_balance heating A D T C chi a r kap root=1 ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r root trial ->
 log_error trial root<=400*binary64_lambda.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion HK Hcont Hder Hslope Hbalance HI Hacc.
 assert (Hkap:forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t).
 { intros t Ht; exact (proj1 (proj2 (HK t
     (positive_closed_interval root trial t Hroot Htrial Ht)))). }
 destruct Hacc as [N W|lb ub Hlb Hlub Hbr Hbt Hns Hnd Nwidth W].
 - assert (Hreg:if heating then T<=trial else trial<=T).
   { destruct heating; simpl in *; tauto. }
   pose proof (inner_outer_graph_concrete choice heating A D T C chi a r kap khat
    trial th qh HA HD HT HC Hchi Ha Hr Htrial Hreg HK HI N) as Heval.
   pose proof (outer_exact_window _ _ W (proj2(proj2 Heval))) as Hres.
   pose proof (branch_inverse_concrete heating A D T C chi a r kap kp root trial
    HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion Hkap Hcont Hder Hslope) as Hinv.
   rewrite Hbalance in Hinv; unfold log_error at 2 in Hinv.
   rewrite ln_1,Rminus_0_r in Hinv.
   replace (Rabs(ln(branch_balance heating A D T C chi a r kap trial))/(1/2))
    with (2*Rabs(ln(branch_balance heating A D T C chi a r kap trial))) in Hinv by field.
   lra.
 - pose proof (rounded_outer_width_safe_RN64 choice lb ub Hlb Hlub Hns Hnd W) as Hwidth.
   pose proof (log_bracket_width lb ub trial root Hlb Hbt Hbr) as Hcoord.
   pose proof binary64_lambda_positive; lra.
Qed.

Record MainAccuracy64 (trial root gh gs rh rs:R) : Prop := {
 main_gas_positive : 0<gh;
 main_radiation_positive : 0<rh;
 main_dust_log_error : log_error trial root<=400*binary64_lambda;
 main_gas_log_error : log_error gh gs<=853*binary64_lambda;
 main_radiation_log_error : log_error rh rs<=3017*binary64_lambda;
 main_dust_relative_error : Rabs(trial/root-1)<=512*binary64_u;
 main_gas_relative_error : Rabs(gh/gs-1)<=1024*binary64_u;
 main_radiation_relative_error : Rabs(rh/rs-1)<=4096*binary64_u
}.

(* This end-to-end theorem assumes primitive graph/range conditions, the
   accepted inner algorithm result, and an actual stopping guard. It does not
   assume any aggregate evaluator, conditioning, or finite output error bound. *)
Theorem binary64_main_accuracy choice heating A D T C chi a r cg kap kp khat root trial th qh :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<cg ->
 0<root -> 0<trial -> branch_region heating T root trial ->
 PlanckValueContract kap khat (8*binary64_lambda) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -(7/2)<=opacity_slope kap t (kp t)<=1/2) ->
 branch_balance heating A D T C chi a r kap root=1 ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r root trial ->
 FiniteNormalNodes choice [cg*th] ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat trial) a trial) ->
 MainAccuracy64 trial root
  (returned_gas choice cg th) (gas_energy cg A D T root)
  (returned_radiation choice r C khat a trial) (OuterDerivatives.radiation C a r kap root).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hcg Hroot Htrial Hregion HK Hcont Hder Hslope
   Hbalance HI Haccept NG NR.
 pose proof (accepted_coordinate_400 choice heating A D T C chi a r kap kp khat
   root trial th qh HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion HK
   Hcont Hder Hslope Hbalance HI Haccept) as Hcoord.
 pose proof (returned_gas_concrete choice cg A D T trial th qh
   Hcg HA HD HT Htrial HI NG) as HG.
 pose proof (returned_radiation_concrete choice C a r kap khat trial
   HC Ha Hr Htrial HK NR) as HR.
 assert (Hkap:forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t).
 { intros t Ht; exact (proj1 (proj2 (HK t
     (positive_closed_interval root trial t Hroot Htrial Ht)))). }
 assert (Habs:forall t, Rmin root trial<t<Rmax root trial ->
   Rabs(opacity_slope kap t (kp t))<=7/2).
 { intros t Ht; apply (proj2 (proj2 (concrete_opacity_margins C kap t (kp t)
     HC (Hkap t ltac:(lra)) (Hslope t Ht)))). }
 pose proof (heating_output_from_coordinate A D T C a r cg kap kp root trial
   HA HD HT HC Ha Hr Hcg Hroot Htrial Hkap Hcont Hder
   (7/2) (400*binary64_lambda) (returned_gas choice cg th)
   (returned_radiation choice r C khat a trial) (53*binary64_lambda) (17*binary64_lambda)
   ltac:(lra) Habs Hcoord (proj2(proj2 HG)) (proj2(proj2 HR))) as [Hgas Hrad].
 assert (Hgas':log_error (returned_gas choice cg th) (gas_energy cg A D T root)
  <=853*binary64_lambda) by lra.
 assert (Hrad':log_error (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap root)<=3017*binary64_lambda) by lra.
 assert (HGroot:0<gas_energy cg A D T root) by
   (apply gas_energy_positive; assumption).
 assert (HRroot:0<OuterDerivatives.radiation C a r kap root).
 { apply radiation_positive; try assumption; exact (proj1(proj2(HK root Hroot))). }
 constructor.
 - exact (proj1 HG).
 - exact (proj1 HR).
 - exact Hcoord.
 - exact Hgas'.
 - exact Hrad'.
 - apply binary64_relative_400; assumption.
 - apply binary64_relative_853; [exact (proj1 HG)|assumption|exact Hgas'].
 - apply binary64_relative_3017; [exact (proj1 HR)|assumption|exact Hrad'].
Qed.

Definition binary64_heating_accuracy (choice:Z->bool) := binary64_main_accuracy choice true.
Definition binary64_cooling_accuracy (choice:Z->bool) := binary64_main_accuracy choice false.

Print Assumptions binary64_main_accuracy.

(* The evaluator is called only at the returned trial. These local variants
   do not require an opacity evaluator contract away from that point. Constant
   auxiliary functions instantiate the primitive graph theorem, whose value
   depends only on its argument at this one point. *)
Lemma inner_outer_graph_concrete_local choice (heating:bool) A D T C chi a r kap khat x th qh :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<x ->
 (if heating then T<=x else x<=T) ->
 LogBound (8*binary64_lambda) (khat x) (kap x) ->
 InnerResult choice A D T x th qh ->
 FiniteNormalNodes choice (outer_nodes (RN64 choice) heating qh C (khat x) chi a x r) ->
 LogBound (71*binary64_lambda)
  (eval_outer (RN64 choice) heating qh C (khat x) chi a x r)
  (branch_balance heating A D T C chi a r kap x).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hx Hregion HK HI N.
 assert (Hconst:PlanckValueContract (fun _=>kap x) (fun _=>khat x) (8*binary64_lambda)).
 { intros y Hy; exact HK. }
 pose proof (inner_outer_graph_concrete choice heating A D T C chi a r
  (fun _=>kap x) (fun _=>khat x) x th qh
  HA HD HT HC Hchi Ha Hr Hx Hregion Hconst HI N) as H.
 destruct heating; exact H.
Qed.

Lemma returned_radiation_concrete_local choice C a r kap khat x :
 0<C -> 0<a -> 0<r -> 0<x ->
 LogBound (8*binary64_lambda) (khat x) (kap x) ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat x) a x) ->
 LogBound (17*binary64_lambda) (returned_radiation choice r C khat a x)
  (OuterDerivatives.radiation C a r kap x).
Proof.
 intros HC Ha Hr Hx HK N.
 assert (Hconst:PlanckValueContract (fun _=>kap x) (fun _=>khat x) (8*binary64_lambda)).
 { intros y Hy; exact HK. }
 exact (returned_radiation_concrete choice C a r (fun _=>kap x) (fun _=>khat x)
  x HC Ha Hr Hx Hconst N).
Qed.

Lemma accepted_coordinate_400_local choice heating A D T C chi a r kap kp khat root trial th qh :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r ->
 0<root -> 0<trial -> branch_region heating T root trial ->
 LogBound (8*binary64_lambda) (khat trial) (kap trial) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -(7/2)<=opacity_slope kap t (kp t)<=1/2) ->
 branch_balance heating A D T C chi a r kap root=1 ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r root trial ->
 log_error trial root<=400*binary64_lambda.
Proof.
 intros HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion HK Hkap Hcont Hder Hslope Hbalance HI Hacc.
 destruct Hacc as [N W|lb ub Hlb Hlub Hbr Hbt Hns Hnd Nwidth W].
 - assert (Hreg:if heating then T<=trial else trial<=T).
   { destruct heating; simpl in *; tauto. }
   pose proof (inner_outer_graph_concrete_local choice heating A D T C chi a r kap khat
    trial th qh HA HD HT HC Hchi Ha Hr Htrial Hreg HK HI N) as Heval.
   pose proof (outer_exact_window _ _ W (proj2(proj2 Heval))) as Hres.
   pose proof (branch_inverse_concrete heating A D T C chi a r kap kp root trial
    HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion Hkap Hcont Hder Hslope) as Hinv.
   rewrite Hbalance in Hinv; unfold log_error at 2 in Hinv.
   rewrite ln_1,Rminus_0_r in Hinv.
   replace (Rabs(ln(branch_balance heating A D T C chi a r kap trial))/(1/2))
    with (2*Rabs(ln(branch_balance heating A D T C chi a r kap trial))) in Hinv by field.
   lra.
 - pose proof (rounded_outer_width_safe_RN64 choice lb ub Hlb Hlub Hns Hnd W) as Hwidth.
   pose proof (log_bracket_width lb ub trial root Hlb Hbt Hbr) as Hcoord.
   pose proof binary64_lambda_positive; lra.
Qed.

Theorem binary64_main_accuracy_local choice heating A D T C chi a r cg kap kp khat root trial th qh :
 0<A -> 0<D -> 0<T -> 0<C -> 0<chi -> 0<a -> 0<r -> 0<cg ->
 0<root -> 0<trial -> branch_region heating T root trial ->
 LogBound (8*binary64_lambda) (khat trial) (kap trial) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<kap t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt kap t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive kap t (kp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial ->
   -(7/2)<=opacity_slope kap t (kp t)<=1/2) ->
 branch_balance heating A D T C chi a r kap root=1 ->
 InnerResult choice A D T trial th qh ->
 OuterAccepted64 choice heating qh C khat chi a r root trial ->
 FiniteNormalNodes choice [cg*th] ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat trial) a trial) ->
 MainAccuracy64 trial root
  (returned_gas choice cg th) (gas_energy cg A D T root)
  (returned_radiation choice r C khat a trial) (OuterDerivatives.radiation C a r kap root).
Proof.
 intros HA HD HT HC Hchi Ha Hr Hcg Hroot Htrial Hregion HK Hkap Hcont Hder Hslope
   Hbalance HI Haccept NG NR.
 pose proof (accepted_coordinate_400_local choice heating A D T C chi a r kap kp khat
   root trial th qh HA HD HT HC Hchi Ha Hr Hroot Htrial Hregion HK Hkap
   Hcont Hder Hslope Hbalance HI Haccept) as Hcoord.
 pose proof (returned_gas_concrete choice cg A D T trial th qh
   Hcg HA HD HT Htrial HI NG) as HG.
 pose proof (returned_radiation_concrete_local choice C a r kap khat trial
   HC Ha Hr Htrial HK NR) as HR.
 assert (Habs:forall t, Rmin root trial<t<Rmax root trial ->
   Rabs(opacity_slope kap t (kp t))<=7/2).
 { intros t Ht; apply (proj2 (proj2 (concrete_opacity_margins C kap t (kp t)
     HC (Hkap t ltac:(lra)) (Hslope t Ht)))). }
 pose proof (heating_output_from_coordinate A D T C a r cg kap kp root trial
   HA HD HT HC Ha Hr Hcg Hroot Htrial Hkap Hcont Hder
   (7/2) (400*binary64_lambda) (returned_gas choice cg th)
   (returned_radiation choice r C khat a trial) (53*binary64_lambda) (17*binary64_lambda)
   ltac:(lra) Habs Hcoord (proj2(proj2 HG)) (proj2(proj2 HR))) as [Hgas Hrad].
 assert (Hgas':log_error (returned_gas choice cg th) (gas_energy cg A D T root)
  <=853*binary64_lambda) by lra.
 assert (Hrad':log_error (returned_radiation choice r C khat a trial)
  (OuterDerivatives.radiation C a r kap root)<=3017*binary64_lambda) by lra.
 assert (HGroot:0<gas_energy cg A D T root) by
   (apply gas_energy_positive; assumption).
 assert (HRroot:0<OuterDerivatives.radiation C a r kap root).
 { apply radiation_positive; try assumption.
   apply Hkap; split; [apply Rmin_l|apply Rmax_l]. }
 constructor.
 - exact (proj1 HG).
 - exact (proj1 HR).
 - exact Hcoord.
 - exact Hgas'.
 - exact Hrad'.
 - apply binary64_relative_400; assumption.
 - apply binary64_relative_853; [exact (proj1 HG)|assumption|exact Hgas'].
 - apply binary64_relative_3017; [exact (proj1 HR)|assumption|exact Hrad'].
Qed.

Definition binary64_heating_accuracy_local (choice:Z->bool) := binary64_main_accuracy_local choice true.
Definition binary64_cooling_accuracy_local (choice:Z->bool) := binary64_main_accuracy_local choice false.
Print Assumptions binary64_main_accuracy_local.
