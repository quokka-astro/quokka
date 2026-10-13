(* Direct pointwise RN64 graph -> physical varying-coefficient balance ->
   derived inverse bound. The evaluator conclusion is discharged here. *)
From Coq Require Import Reals Psatz Field List Lia.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap GasBounds LogCalculus OuterBounds FloatingPoint
 Guards GuardFloatBridge InnerEvaluator Binary64Accuracy.
From MultiGroup Require Import ConstantGroupExistence FullGroupReconstruction ConstantGroupAccuracy
 MultigroupGraph ComponentAccuracy AcceptedMultigroupAccuracy VariableGroupAccuracy VariableGroupOutputs.
Import ListNotations.
Open Scope R_scope.

Definition variable_eval_budget heating ea ep eb tM tH :=
 mg_outer_budget heating (52*binary64_lambda)
  (ea+ep+eb+(6+INR(tree_depth tM))*binary64_lambda)
  (ea+(5+INR(tree_depth tH))*binary64_lambda) binary64_lambda.

Lemma variable_tree_M n chi h a p B x t : CoversGroups n t ->
 tree_value (fun g=>variable_M_term chi h (a g) (p g) (B g) x) t=
 variable_M_sum n chi h a p B x.
Proof.
 intro HC; rewrite (tree_value_finite_sum _ _ _ HC),finite_sum_group_sum; reflexivity.
Qed.
Lemma variable_tree_H n chi h a r x t : CoversGroups n t ->
 tree_value (fun g=>variable_H_term chi h (a g) (r g) x) t=
 variable_H_sum n chi h a r x.
Proof.
 intro HC; rewrite (tree_value_finite_sum _ _ _ HC),finite_sum_group_sum; reflexivity.
Qed.

Lemma variable_physical_graph_identity heating A D T chi h n a p r B x tM tH :
 0<A -> 0<D -> 0<T -> 0<x -> thermal_branch heating T x ->
 CoversGroups n tM -> CoversGroups n tH ->
 mg_outer_exact heating (mg_q A D T x)
 (tree_value (fun g=>variable_M_term chi h (a g) (p g) (B g) x) tM)
 (tree_value (fun g=>variable_H_term chi h (a g) (r g) x) tH)=
 variable_balance heating A D T (variable_M_sum n chi h a p B) (variable_H_sum n chi h a r) x.
Proof.
 intros HA HD HT HX Hbranch HCM HCH.
 rewrite (variable_tree_M _ _ _ _ _ _ _ _ HCM), (variable_tree_H _ _ _ _ _ _ _ HCH).
 destruct heating; simpl in *; unfold mg_outer_exact,mg_q,variable_heat,variable_cool,variable_cool_den.
 - pose proof (heating_gas_branch A D T x HA HD HT Hbranch).
   rewrite Rabs_right by lra; reflexivity.
 - pose proof (cooling_gas_branch A D T x HA HD HT HX Hbranch).
   rewrite Rabs_left1 by lra.
   replace (-(gas_map A D T x-T)) with (T-gas_map A D T x) by ring; reflexivity.
Qed.

Theorem variable_formed_evaluator64 choice heating A D T chi h n a p r B ah ph bh
 trial th qh tM tH ea ep eb :
 0<A -> 0<D -> 0<T -> 0<chi -> 0<h -> 0<trial -> 0<=ea ->
 (forall g, (g<n)%nat -> 0<=r g) ->
 (forall g, (g<n)%nat -> ZLogBound ea (ah g) (a g trial)) ->
 (forall g, (g<n)%nat -> ZLogBound ep (ph g) (p g trial)) ->
 (forall g, (g<n)%nat -> ZLogBound eb (bh g) (B g trial)) ->
 thermal_branch heating T trial ->
 variable_region heating A D T (variable_H_sum n chi h a r) trial ->
 0<variable_M_sum n chi h a p B trial ->
 CoversGroups n tM -> CoversGroups n tH ->
 InnerResult choice A D T trial th qh ->
 ZFiniteNormalNodes choice (mg_all_nodes (RN64 choice) heating chi h qh ah ph bh r tM tH) ->
 LogBound (variable_eval_budget heating ea ep eb tM tH)
 (mg_total_eval (RN64 choice) heating chi h qh ah ph bh r tM tH)
 (variable_balance heating A D T (variable_M_sum n chi h a p B) (variable_H_sum n chi h a r) trial).
Proof.
 intros HA HD HT Hchi Hh HX Hea Hr Ha Hp HB Hbr Hreg HM HCM HCH HI HN.
 pose proof (inner_transfer_zbudget choice A D T trial th qh HA HD HT HX HI) as HQ.
 assert (Hac:forall g, In g (tree_leaves tM++tree_leaves tH) -> ZLogBound ea (ah g) (a g trial)).
 { intros g Hg; apply Ha; apply in_app_or in Hg; destruct Hg as [Hg|Hg].
   - exact (CoversGroups_member n tM g HCM Hg).
   - exact (CoversGroups_member n tH g HCH Hg). }
 assert (Hpc:forall g, In g (tree_leaves tM) -> ZLogBound ep (ph g) (p g trial)).
 { intros g Hg; apply Hp; exact (CoversGroups_member n tM g HCM Hg). }
 assert (Hbc:forall g, In g (tree_leaves tM) -> ZLogBound eb (bh g) (B g trial)).
 { intros g Hg; apply HB; exact (CoversGroups_member n tM g HCM Hg). }
 assert (Hrc:forall g, In g (tree_leaves tH) -> 0<=r g).
 { intros g Hg; apply Hr; exact (CoversGroups_member n tH g HCH Hg). }
 assert (Hposit:if heating then
  0<mg_q A D T trial+tree_value (fun g=>variable_M_term chi h (a g) (p g) (B g) trial) tM /\
  0<tree_value (fun g=>variable_H_term chi h (a g) (r g) trial) tH else
  0<tree_value (fun g=>variable_M_term chi h (a g) (p g) (B g) trial) tM /\
  0<mg_q A D T trial+tree_value (fun g=>variable_H_term chi h (a g) (r g) trial) tH).
 { rewrite (variable_tree_M _ _ _ _ _ _ _ _ HCM), (variable_tree_H _ _ _ _ _ _ _ HCH).
   destruct heating; simpl in *.
   - unfold mg_q; pose proof (Rabs_pos (gas_map A D T trial-T)); nra.
   - unfold mg_q; pose proof (cooling_gas_branch A D T trial HA HD HT HX Hbr).
     rewrite Rabs_left1 by lra; unfold variable_cool_den in Hreg; split; nra. }
 pose proof (variable_pointwise_RN64_budget choice heating (52*binary64_lambda) ea ep eb
  chi h qh (mg_q A D T trial) ah ph bh a p B r tM tH trial
  Hea Hchi Hh HQ Hac Hpc Hbc Hrc Hposit HN) as HE.
 rewrite (variable_physical_graph_identity heating A D T chi h n a p r B trial tM tH
  HA HD HT HX Hbr HCM HCH) in HE.
 rewrite <-guard_lambda64 in HE; exact HE.
Qed.

Theorem variable_opacity_RN64_coordinate_accuracy choice heating A D T chi h n
 a ap p pp r B Bp ah ph bh root trial th qh tM tH ea ep eb m v :
 0<A -> 0<D -> 0<T -> 0<chi -> 0<h -> 0<root -> 0<trial -> 0<=ea ->
 0<m -> 0<=v -> 0<variable_margin m v ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<variable_M_sum n chi h a p B s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> variable_region heating A D T (variable_H_sum n chi h a r) s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> 0<a g s /\ 0<p g s /\ 0<B g s /\ 0<=r g) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (a g) s (ap g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (p g) s (pp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (B g) s (Bp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> m<=variable_emission_slope h (a g) (ap g) (p g) (pp g) (B g) (Bp g) s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> variable_absorption_slope h (a g) (ap g) s<=v) ->
 variable_residual A D T (variable_M_sum n chi h a p B) (variable_H_sum n chi h a r) root=0 ->
 (forall g, (g<n)%nat -> ZLogBound ea (ah g) (a g trial)) ->
 (forall g, (g<n)%nat -> ZLogBound ep (ph g) (p g trial)) ->
 (forall g, (g<n)%nat -> ZLogBound eb (bh g) (B g trial)) ->
 thermal_branch heating T trial -> CoversGroups n tM -> CoversGroups n tH ->
 InnerResult choice A D T trial th qh ->
 ZFiniteNormalNodes choice (mg_all_nodes (RN64 choice) heating chi h qh ah ph bh r tM tH) ->
 1-128*binary64_u<=mg_total_eval (RN64 choice) heating chi h qh ah ph bh r tM tH<=1+128*binary64_u ->
 let b:=(variable_eval_budget heating ea ep eb tM tH+129*binary64_lambda)/variable_margin m v in
 log_error trial root<=b /\ Rabs(trial/root-1)<=exp b-1.
Proof.
 intros HA HD HT Hchi Hh HR HX Hea Hm Hv Hdelta HM Hreg Hpos HDa HDp HDB Hmu Hnu HF
  Ha Hp HB Hbr HCM HCH HI HN HW b.
 assert (Htrial:Rmin root trial<=trial<=Rmax root trial) by (split; [apply Rmin_r|apply Rmax_r]).
 assert (Hr:forall g, (g<n)%nat -> 0<=r g).
 { intros g Hg; pose proof (Hpos g trial Hg Htrial); tauto. }
 pose proof (variable_formed_evaluator64 choice heating A D T chi h n a p r B ah ph bh
  trial th qh tM tH ea ep eb HA HD HT Hchi Hh HX Hea Hr Ha Hp HB Hbr
  (Hreg trial Htrial) (HM trial Htrial) HCM HCH HI HN) as HE.
 assert (HC:log_error trial root<=b).
 { unfold b; apply (finite_variable_group_residual_coordinate64 heating A D T n chi h
   a ap p pp B Bp r m v root trial
   (mg_total_eval (RN64 choice) heating chi h qh ah ph bh r tM tH)
   (variable_eval_budget heating ea ep eb tM tH)); assumption. }
 split; [exact HC|]; apply dust_finite_accuracy; assumption.
Qed.

Print Assumptions variable_formed_evaluator64.
Print Assumptions variable_opacity_RN64_coordinate_accuracy.

Lemma variable_eight_lambda_depth10_budget heating tM tH :
 (tree_depth tM<=10)%nat -> (tree_depth tH<=10)%nat ->
 variable_eval_budget heating (8*binary64_lambda) (8*binary64_lambda) (8*binary64_lambda) tM tH
 <=94*binary64_lambda.
Proof.
 intros HM HH; apply le_INR in HM,HH; simpl in HM,HH.
 pose proof (pos_INR (tree_depth tM)); pose proof (pos_INR (tree_depth tH)).
 pose proof binary64_lambda_positive.
 unfold variable_eval_budget; destruct heating; unfold mg_outer_budget;
 rewrite Rmax_left by nra; nra.
Qed.

Lemma variable_lower_sign94 rhat r :
 LogBound (94*binary64_lambda) rhat r -> rhat<1-128*binary64_u -> r<1.
Proof.
 intros [Hhat [Hr Herr]] Hwin.
 apply (guard_lower_sign 128 94 rhat r); try assumption;
 unfold binary64_u; field_simplify; lra.
Qed.
Lemma variable_upper_sign94 rhat r :
 LogBound (94*binary64_lambda) rhat r -> 1+128*binary64_u<rhat -> 1<r.
Proof.
 intros [Hhat [Hr Herr]] Hwin.
 apply (guard_upper_sign 128 94 rhat r); try assumption;
 unfold binary64_u; field_simplify; lra.
Qed.
Theorem variable_guard_total94 rhat r :
 LogBound (94*binary64_lambda) rhat r ->
 (rhat<1-128*binary64_u /\ r<1) \/
 (1-128*binary64_u<=rhat<=1+128*binary64_u) \/
 (1+128*binary64_u<rhat /\ 1<r).
Proof.
 intro HE; destruct (Rlt_dec rhat (1-128*binary64_u)) as [HL|HL].
 - left; split; [exact HL|apply variable_lower_sign94 with rhat; assumption].
 - destruct (Rlt_dec (1+128*binary64_u) rhat) as [HU|HU].
   + right; right; split; [exact HU|apply variable_upper_sign94 with rhat; assumption].
   + right; left; lra.
Qed.

Lemma variable_residual_budget223 delta eta :
 0<delta -> eta<=94*binary64_lambda ->
 (eta+129*binary64_lambda)/delta<=223*binary64_lambda/delta.
Proof.
 intros HD HE; unfold Rdiv; apply Rmult_le_compat_r;
 [apply Rlt_le,Rinv_0_lt_compat; assumption|lra].
Qed.

Print Assumptions variable_eight_lambda_depth10_budget.
Print Assumptions variable_guard_total94.
