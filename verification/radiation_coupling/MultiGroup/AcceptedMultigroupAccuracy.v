(* Composition of the proved inverse, primitive graph, and component bounds.
   Numerical driver, band evaluator, and nonzero-node range contracts remain
   explicit. No conclusion is postulated as a hypothesis. *)
From Coq Require Import Reals Psatz Field List Lia.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap GasBounds LogCalculus OuterBounds FloatingPoint
 Guards GuardFloatBridge InnerEvaluator Binary64Accuracy.
From MultiGroup Require Import ConstantGroupExistence FullGroupReconstruction
 ConstantGroupAccuracy MultigroupGraph ComponentAccuracy.
Import ListNotations.
Open Scope R_scope.

Definition mg_q A D T x := A*Rabs(gas_map A D T x-T).

Lemma inner_transfer_zbudget choice A D T x th qh :
 0<A -> 0<D -> 0<T -> 0<x -> InnerResult choice A D T x th qh ->
 ZLogBound (52*binary64_lambda) qh (mg_q A D T x).
Proof.
 intros HA HD HT Hx HI.
 destruct (inner_result_accuracy choice A D T x th qh HA HD HT Hx HI)
  as [_ [[HxT Hzero]|Hq]].
 - left; split; [exact Hzero|].
   unfold mg_q; rewrite HxT,gas_map_equilibrium by assumption.
   rewrite Rminus_diag_eq,Rabs_R0 by reflexivity; ring.
 - right; exact Hq.
Qed.

Lemma concrete_group_output14 choice h alpha p r B bh :
 0<h -> 0<=alpha -> 0<=p -> 0<=r ->
 ZLogBound (8*binary64_lambda) bh B ->
 ZFiniteNormalNodes choice (E_nodes (RN64 choice) h alpha p bh r) ->
 ZLogBound (14*binary64_lambda) (g_E (RN64 choice) h alpha p bh r)
 ((r+h*p*B)/(1+h*alpha)).
Proof.
 intros Hh Ha Hp Hr HB HN.
 pose proof (group_E_RN64_budget choice 0 0 (8*binary64_lambda) 0
  h alpha alpha p p bh B r r ltac:(lra) Hh
  (ZLogBound_exact 0 alpha ltac:(lra) Ha)
  (ZLogBound_exact 0 p ltac:(lra) Hp) HB
  (ZLogBound_exact 0 r ltac:(lra) Hr) HN) as HE.
 rewrite <-guard_lambda64 in HE.
 rewrite Rmax_right in HE by (pose proof binary64_lambda_positive; lra).
 replace (0+8*binary64_lambda+2*binary64_lambda+0+4*binary64_lambda)
  with (14*binary64_lambda) in HE by ring.
 exact HE.
Qed.

Lemma concrete_outer_budget78 heating dM dH :
 0<=dM -> dM<=10 -> 0<=dH -> dH<=10 ->
 mg_outer_budget heating (52*binary64_lambda)
  ((14+dM)*binary64_lambda) ((5+dH)*binary64_lambda) binary64_lambda
 <=78*binary64_lambda.
Proof.
 intros HM HMb HH HHb.
 pose proof binary64_lambda_positive as Hl.
 destruct heating; unfold mg_outer_budget;
 rewrite Rmax_left by nra; nra.
Qed.

Theorem accepted_residual_components64 choice heating A D T H n c B Bp root trial th qh rhat :
 0<A -> 0<D -> 0<T -> 0<root -> 0<trial ->
 (forall g, (g<n)%nat -> 0<=c g) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> 0<group_emission n c B s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial -> mg_region heating A D T H s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (B g) s (Bp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> B g s<=s*Bp g s) ->
 constant_group_residual A D T n c B H root=0 ->
 LogBound (78*binary64_lambda) rhat (mg_balance heating A D T H (group_emission n c B) trial) ->
 1-128*binary64_u<=rhat<=1+128*binary64_u ->
 InnerResult choice A D T trial th qh -> FiniteNormalNodes choice [A*th] ->
 log_error trial root<=207*binary64_lambda /\
 Rabs(trial/root-1)<=256*binary64_u /\
 LogBound (467*binary64_lambda) (returned_gas choice A th) (gas_energy A A D T root) /\
 Rabs(returned_gas choice A th/gas_energy A A D T root-1)<=512*binary64_u.
Proof.
 intros HA HD HT HR HX Hc HM Hreg HDer Hmargin HF HE HW HI HN.
 pose proof (scalar_root_residual_coordinate64 heating A D T H n c B Bp root trial rhat
  (78*binary64_lambda) HA HD HT HR HX Hc HM Hreg HDer Hmargin HF HE HW) as HC.
 replace (78*binary64_lambda+129*binary64_lambda) with (207*binary64_lambda) in HC by ring.
 pose proof (gas_finite_accuracy64 choice A A D T root trial th qh (207*binary64_lambda)
  HA HA HD HT HR HX HI HN HC) as [HG _].
 replace (53*binary64_lambda+2*(207*binary64_lambda)) with (467*binary64_lambda) in HG by ring.
 repeat split; try exact HC; try exact (proj1 HG); try exact (proj1(proj2 HG)); try exact (proj2(proj2 HG)).
 - apply component_relative_207; repeat split; assumption.
 - apply component_relative_467; exact HG.
Qed.

Theorem accepted_band_residual64 choice h alpha p r B Bp bh root trial L :
 0<h -> 0<=alpha -> 0<=p -> 0<=r -> (0<r \/ 0<p) ->
 0<root -> 0<trial -> 0<=L ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> 0<B t) ->
 (forall t, Rmin root trial<=t<=Rmax root trial -> continuity_pt B t) ->
 (forall t, Rmin root trial<t<Rmax root trial -> is_derive B t (Bp t)) ->
 (forall t, Rmin root trial<t<Rmax root trial -> Rabs(t*Bp t/B t)<=L) ->
 log_error trial root<=207*binary64_lambda ->
 ZLogBound (8*binary64_lambda) bh (B trial) ->
 ZFiniteNormalNodes choice (E_nodes (RN64 choice) h alpha p bh r) ->
 LogBound ((14+207*L)*binary64_lambda)
 (g_E (RN64 choice) h alpha p bh r) (band_energy h alpha p r B root) /\
 Rabs(g_E (RN64 choice) h alpha p bh r/band_energy h alpha p r B root-1)
 <=exp((14+207*L)*binary64_lambda)-1.
Proof.
 intros Hh Ha Hp Hr Hactive HR HX HL HB HC HD HLip HXerr HBand HN.
 pose proof (concrete_group_output14 choice h alpha p r (B trial) bh Hh Ha Hp Hr HBand HN) as HE.
 assert (Hpos:0<band_energy h alpha p r B trial).
 { apply band_energy_positive; try assumption; apply HB; split; [apply Rmin_r|apply Rmax_r]. }
 apply ZLogBound_positive in HE; [|exact Hpos].
 replace ((14+207*L)*binary64_lambda) with
  (14*binary64_lambda+L*(207*binary64_lambda)) by ring.
 apply (radiation_band_finite_accuracy h alpha p r B Bp root trial
  (g_E (RN64 choice) h alpha p bh r) (14*binary64_lambda) L (207*binary64_lambda)); assumption.
Qed.

Lemma finite_sum_group_sum n f : finite_sum n f=group_sum n f.
Proof. induction n; simpl; congruence. Qed.

Lemma tree_exact_emission n t chi h alpha p B x :
 CoversGroups n t ->
 tree_value (exact_M chi h alpha p (fun g=>B g x)) t=
 group_emission n (group_weight chi h alpha p) B x.
Proof.
 intro HC; rewrite (tree_value_finite_sum _ _ _ HC), finite_sum_group_sum.
 unfold group_emission; apply group_sum_ext; intros.
 unfold exact_M,group_weight,Rdiv; ring.
Qed.

Lemma tree_exact_absorption n t chi h alpha r :
 CoversGroups n t ->
 tree_value (exact_H chi h alpha r) t=absorption_total n chi h alpha r.
Proof.
 intro HC; rewrite (tree_value_finite_sum _ _ _ HC), finite_sum_group_sum.
 unfold absorption_total; apply group_sum_ext; intros.
 unfold exact_H,group_absorption,saturate,Rdiv; ring.
Qed.

Definition thermal_branch (heating:bool) T x := if heating then T<=x else x<=T.

Lemma physical_outer_graph_identity heating A D T chi h n alpha p r B x tM tH :
 0<A -> 0<D -> 0<T -> 0<x -> thermal_branch heating T x ->
 CoversGroups n tM -> CoversGroups n tH ->
 mg_outer_exact heating (mg_q A D T x)
 (tree_value (exact_M chi h alpha p (fun g=>B g x)) tM)
 (tree_value (exact_H chi h alpha r) tH)=
 mg_balance heating A D T (absorption_total n chi h alpha r)
  (group_emission n (group_weight chi h alpha p) B) x.
Proof.
 intros HA HD HT HX Hbranch HCM HCH.
 rewrite (tree_exact_emission _ _ _ _ _ _ _ _ HCM), (tree_exact_absorption _ _ _ _ _ _ HCH).
 destruct heating; simpl in *; unfold mg_outer_exact,mg_q,mg_heat,mg_cool,mg_cool_den.
 - pose proof (heating_gas_branch A D T x HA HD HT Hbranch).
   rewrite Rabs_right by lra; reflexivity.
 - pose proof (cooling_gas_branch A D T x HA HD HT HX Hbranch).
   rewrite Rabs_left1 by lra.
   replace (-(gas_map A D T x-T)) with (T-gas_map A D T x) by ring; reflexivity.
Qed.

Theorem concrete_formed_outer78 choice heating A D T chi h n alpha p r B bh
 trial th qh tM tH :
 0<A -> 0<D -> 0<T -> 0<chi -> 0<h -> 0<trial ->
 (forall g, (g<n)%nat -> 0<=alpha g) ->
 (forall g, (g<n)%nat -> 0<=p g) ->
 (forall g, (g<n)%nat -> 0<=r g) ->
 (forall g, (g<n)%nat -> ZLogBound (8*binary64_lambda) (bh g) (B g trial)) ->
 thermal_branch heating T trial ->
 mg_region heating A D T (absorption_total n chi h alpha r) trial ->
 0<group_emission n (group_weight chi h alpha p) B trial ->
 CoversGroups n tM -> CoversGroups n tH ->
 (tree_depth tM<=10)%nat -> (tree_depth tH<=10)%nat ->
 InnerResult choice A D T trial th qh ->
 ZFiniteNormalNodes choice (mg_all_nodes (RN64 choice) heating chi h qh alpha p bh r tM tH) ->
 LogBound (78*binary64_lambda)
 (mg_total_eval (RN64 choice) heating chi h qh alpha p bh r tM tH)
 (mg_balance heating A D T (absorption_total n chi h alpha r)
  (group_emission n (group_weight chi h alpha p) B) trial).
Proof.
 intros HA HD HT Hchi Hh HX Ha Hp Hr HB Hbr Hreg HM HCM HCH HdM HdH HI HN.
 pose proof (inner_transfer_zbudget choice A D T trial th qh HA HD HT HX HI) as HQ.
 assert (Hac:forall g, In g (tree_leaves tM++tree_leaves tH) -> ZLogBound 0 (alpha g) (alpha g)).
 { intros g Hg; apply ZLogBound_exact; [lra|apply Ha].
   apply in_app_or in Hg; destruct Hg as [Hg|Hg].
   - exact (CoversGroups_member n tM g HCM Hg).
   - exact (CoversGroups_member n tH g HCH Hg). }
 assert (Hpc:forall g, In g (tree_leaves tM) -> ZLogBound 0 (p g) (p g)).
 { intros g Hg; apply ZLogBound_exact; [lra|apply Hp]; exact (CoversGroups_member n tM g HCM Hg). }
 assert (Hbc:forall g, In g (tree_leaves tM) -> ZLogBound (8*binary64_lambda) (bh g) (B g trial)).
 { intros g Hg; apply HB; exact (CoversGroups_member n tM g HCM Hg). }
 assert (Hrc:forall g, In g (tree_leaves tH) -> 0<=r g).
 { intros g Hg; apply Hr; exact (CoversGroups_member n tH g HCH Hg). }
 assert (Hposit:if heating then
  0<mg_q A D T trial+tree_value (exact_M chi h alpha p (fun g=>B g trial)) tM /\
  0<tree_value (exact_H chi h alpha r) tH else
  0<tree_value (exact_M chi h alpha p (fun g=>B g trial)) tM /\
  0<mg_q A D T trial+tree_value (exact_H chi h alpha r) tH).
 { rewrite (tree_exact_emission _ _ _ _ _ _ _ _ HCM), (tree_exact_absorption _ _ _ _ _ _ HCH).
   destruct heating; simpl in *.
   - unfold mg_q; pose proof (Rabs_pos (gas_map A D T trial-T)); nra.
   - unfold mg_q; pose proof (cooling_gas_branch A D T trial HA HD HT HX Hbr).
     rewrite Rabs_left1 by lra; unfold mg_cool_den in Hreg; split; nra. }
 pose proof (multigroup_formed_outer_RN64_budget choice heating (52*binary64_lambda) 0 0
  (8*binary64_lambda) chi h qh (mg_q A D T trial) alpha alpha p p bh
  (fun g=>B g trial) r tM tH ltac:(lra) Hchi Hh HQ Hac Hpc Hbc Hrc Hposit HN) as HE.
 rewrite (physical_outer_graph_identity heating A D T chi h n alpha p r B trial tM tH
  HA HD HT HX Hbr HCM HCH) in HE.
 rewrite <-guard_lambda64 in HE.
 replace (0+0+8*binary64_lambda+(6+INR(tree_depth tM))*binary64_lambda)
  with ((14+INR(tree_depth tM))*binary64_lambda) in HE by ring.
 replace (0+(5+INR(tree_depth tH))*binary64_lambda)
  with ((5+INR(tree_depth tH))*binary64_lambda) in HE by ring.
 eapply LogBound_mono; [|exact HE].
 apply concrete_outer_budget78; try apply pos_INR;
  apply le_INR in HdM,HdH; simpl in *; lra.
Qed.

Theorem constant_opacity_multigroup_residual_accuracy64 choice heating A D T chi h n
 alpha p r B Bp bh L root trial th qh tM tH :
 0<A -> 0<D -> 0<T -> 0<chi -> 0<h -> 0<root -> 0<trial ->
 (forall g, (g<n)%nat -> 0<=alpha g) ->
 (forall g, (g<n)%nat -> 0<=p g) ->
 (forall g, (g<n)%nat -> 0<=r g) ->
 (forall g, (g<n)%nat -> ZLogBound (8*binary64_lambda) (bh g) (B g trial)) ->
 thermal_branch heating T trial ->
 (forall s, Rmin root trial<=s<=Rmax root trial ->
   mg_region heating A D T (absorption_total n chi h alpha r) s) ->
 (forall s, Rmin root trial<=s<=Rmax root trial ->
   0<group_emission n (group_weight chi h alpha p) B s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (B g) s (Bp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> B g s<=s*Bp g s) ->
 (forall g, (g<n)%nat -> 0<=L g) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> 0<B g s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> Rabs(s*Bp g s/B g s)<=L g) ->
 constant_group_residual A D T n (group_weight chi h alpha p) B
  (absorption_total n chi h alpha r) root=0 ->
 CoversGroups n tM -> CoversGroups n tH ->
 (tree_depth tM<=10)%nat -> (tree_depth tH<=10)%nat ->
 InnerResult choice A D T trial th qh -> FiniteNormalNodes choice [A*th] ->
 ZFiniteNormalNodes choice (mg_all_nodes (RN64 choice) heating chi h qh alpha p bh r tM tH) ->
 (forall g, (g<n)%nat -> ZFiniteNormalNodes choice (E_nodes (RN64 choice) h (alpha g) (p g) (bh g) (r g))) ->
 1-128*binary64_u<=mg_total_eval (RN64 choice) heating chi h qh alpha p bh r tM tH<=1+128*binary64_u ->
 log_error trial root<=207*binary64_lambda /\
 Rabs(trial/root-1)<=256*binary64_u /\
 Rabs(returned_gas choice A th/gas_energy A A D T root-1)<=512*binary64_u /\
 (forall g, (g<n)%nat -> (0<r g \/ 0<p g) ->
  Rabs(g_E (RN64 choice) h (alpha g) (p g) (bh g) (r g)/
   band_energy h (alpha g) (p g) (r g) (B g) root-1)
  <=exp((14+207*L g)*binary64_lambda)-1).
Proof.
 intros HA HD HT Hchi Hh HR HX Ha Hp Hr HB Hbr Hreg HM HDer Hmargin HL HBpos HLip HF
  HCM HCH HdM HdH HI HNg HNo HNe HW.
 assert (Htrial:Rmin root trial<=trial<=Rmax root trial) by (split; [apply Rmin_r|apply Rmax_r]).
 pose proof (concrete_formed_outer78 choice heating A D T chi h n alpha p r B bh trial th qh tM tH
  HA HD HT Hchi Hh HX Ha Hp Hr HB Hbr (Hreg trial Htrial) (HM trial Htrial)
  HCM HCH HdM HdH HI HNo) as HOuter.
 assert (Hc:forall g, (g<n)%nat -> 0<=group_weight chi h alpha p g).
 { intros g Hg; unfold group_weight; apply Rdiv_le_0_compat.
   - repeat apply Rmult_le_pos; try lra; apply Hp; assumption.
   - pose proof (Ha g Hg); nra. }
 destruct (accepted_residual_components64 choice heating A D T (absorption_total n chi h alpha r)
  n (group_weight chi h alpha p) B Bp root trial th qh
  (mg_total_eval (RN64 choice) heating chi h qh alpha p bh r tM tH)
  HA HD HT HR HX Hc HM Hreg HDer Hmargin HF HOuter HW HI HNg)
  as [HXerr [HXrel [HGlog HGrel]]].
 split; [exact HXerr|]; split; [exact HXrel|]; split; [exact HGrel|].
 intros g Hg Hactive.
 apply (proj2 (accepted_band_residual64 choice h (alpha g) (p g) (r g) (B g) (Bp g)
  (bh g) root trial (L g) Hh (Ha g Hg) (Hp g Hg) (Hr g Hg) Hactive HR HX (HL g Hg)
  (fun s Hs=>HBpos g s Hg Hs) ltac:(intros s Hs; apply (is_derive_continuity_pt _ _ (Bp g s)); apply HDer; assumption)
  ltac:(intros; apply HDer; try assumption; lra) (fun s Hs=>HLip g s Hg Hs) HXerr (HB g Hg) (HNe g Hg))).
Qed.

Print Assumptions concrete_formed_outer78.
Print Assumptions constant_opacity_multigroup_residual_accuracy64.

Theorem constant_opacity_multigroup_bracket_accuracy64 choice A D T h n alpha p r B Bp bh L
 root trial th qh lo hi :
 0<A -> 0<D -> 0<T -> 0<h ->
 (forall g, (g<n)%nat -> 0<=alpha g) ->
 (forall g, (g<n)%nat -> 0<=p g) ->
 (forall g, (g<n)%nat -> 0<=r g) ->
 (forall g, (g<n)%nat -> 0<=L g) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> 0<B g s) ->
 (forall g s, (g<n)%nat -> Rmin root trial<=s<=Rmax root trial -> is_derive (B g) s (Bp g s)) ->
 (forall g s, (g<n)%nat -> Rmin root trial<s<Rmax root trial -> Rabs(s*Bp g s/B g s)<=L g) ->
 (forall g, (g<n)%nat -> ZLogBound (8*binary64_lambda) (bh g) (B g trial)) ->
 (forall g, (g<n)%nat -> ZFiniteNormalNodes choice (E_nodes (RN64 choice) h (alpha g) (p g) (bh g) (r g))) ->
 InnerResult choice A D T trial th qh -> FiniteNormalNodes choice [A*th] ->
 0<lo -> lo<hi -> lo<=root<=hi -> lo<=trial<=hi ->
 WidthSafety.SafeDifference64 lo hi -> normal64 (RN64 choice (hi-lo)/lo) ->
 RN64 choice (RN64 choice (hi-lo)/lo)<=16*binary64_u ->
 log_error trial root<=32*binary64_lambda /\
 Rabs(trial/root-1)<=64*binary64_u /\
 Rabs(returned_gas choice A th/gas_energy A A D T root-1)<=128*binary64_u /\
 (forall g, (g<n)%nat -> (0<r g \/ 0<p g) ->
  Rabs(g_E (RN64 choice) h (alpha g) (p g) (bh g) (r g)/
   band_energy h (alpha g) (p g) (r g) (B g) root-1)
  <=exp((14+32*L g)*binary64_lambda)-1).
Proof.
 intros HA HD HT Hh Ha Hp Hr HL HB HDer HLip HE HN HI HNg Hlo Hlh HR HX Hdiff Hnormal Hwidth.
 assert (HRpos:0<root) by lra.
 assert (HXpos:0<trial) by lra.
 pose proof (certified_bracket_coordinate64 choice lo hi root trial Hlo Hlh HR HX Hdiff Hnormal Hwidth) as HC.
 pose proof (gas_finite_accuracy64 choice A A D T root trial th qh (32*binary64_lambda)
  HA HA HD HT HRpos HXpos HI HNg HC) as [HG _].
 replace (53*binary64_lambda+2*(32*binary64_lambda)) with (117*binary64_lambda) in HG by ring.
 split; [exact HC|]; split.
 - apply component_relative_32; repeat split; assumption.
 - split; [apply component_relative_117; exact HG|].
   intros g Hg Hactive.
   pose proof (concrete_group_output14 choice h (alpha g) (p g) (r g) (B g trial) (bh g)
    Hh (Ha g Hg) (Hp g Hg) (Hr g Hg) (HE g Hg) (HN g Hg)) as HGraph.
   assert (Hpos:0<band_energy h (alpha g) (p g) (r g) (B g) trial).
   { apply band_energy_positive; try assumption; try (apply Ha || apply Hp || apply Hr); try assumption.
     apply HB; [exact Hg|split; [apply Rmin_r|apply Rmax_r]]. }
   apply ZLogBound_positive in HGraph; [|exact Hpos].
   pose proof (radiation_band_finite_accuracy h (alpha g) (p g) (r g) (B g) (Bp g)
    root trial (g_E (RN64 choice) h (alpha g) (p g) (bh g) (r g)) (14*binary64_lambda)
    (L g) (32*binary64_lambda) Hh (Ha g Hg) (Hp g Hg) (Hr g Hg) Hactive HRpos HXpos
    (HL g Hg) (fun s Hs=>HB g s Hg Hs) ltac:(intros s Hs; apply (is_derive_continuity_pt _ _ (Bp g s)); apply HDer; assumption)
    ltac:(intros; apply HDer; try assumption; lra) (fun s Hs=>HLip g s Hg Hs) HC HGraph) as [_ Hrel].
   replace ((14+32*L g)*binary64_lambda) with (14*binary64_lambda+L g*(32*binary64_lambda)) by ring.
   exact Hrel.
Qed.

Print Assumptions constant_opacity_multigroup_bracket_accuracy64.
