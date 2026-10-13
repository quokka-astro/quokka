(* Positive multigroup operation graphs. Structural zeros are exact zeros;
   positive values use logarithmic error. Every rounded node is explicit. *)
From Coq Require Import Reals Psatz Field Lia List.
From Flocq.Core Require Import Core.
From BlackBox Require Import FloatingPoint.
Import ListNotations.
Open Scope R_scope.

Definition ZLogBound (e h x:R) : Prop := (h=0 /\ x=0) \/ LogBound e h x.
Lemma ZLogBound_zero e : ZLogBound e 0 0.
Proof. left; auto. Qed.
Lemma ZLogBound_exact e x : 0<=e -> 0<=x -> ZLogBound e x x.
Proof.
 intros He Hx; destruct (Req_dec x 0) as [->|Hnz]; [apply ZLogBound_zero|].
 right; apply (LogBound_mono 0 e); [exact He|apply LogBound_exact; lra].
Qed.
Lemma ZLogBound_mono e E h x : e<=E -> ZLogBound e h x -> ZLogBound E h x.
Proof. intros HE [Hz|Hp]; [left; exact Hz|right; eapply LogBound_mono; eauto]. Qed.
Lemma ZLogBound_nonnegative e h x : ZLogBound e h x -> 0<=h /\ 0<=x.
Proof. intros [[-> ->]|[Hh [Hx He]]]; lra. Qed.
Lemma ZLogBound_positive e h x : 0<x -> ZLogBound e h x -> LogBound e h x.
Proof. intros Hx [[_ Hz]|H]; [lra|assumption]. Qed.
Lemma ZLogBound_trans e d h y x : ZLogBound e h y -> ZLogBound d y x -> ZLogBound (e+d) h x.
Proof.
 intros [[Hh Hy]|H1] [[Hy' Hx]|H2].
 - left; auto.
 - destruct H2 as [Hp _]; lra.
 - destruct H1 as [_ [Hp _]]; lra.
 - right; eapply LogBound_trans; eauto.
Qed.
Lemma ZLogBound_mult e d h k x y : ZLogBound e h x -> ZLogBound d k y -> ZLogBound (e+d) (h*k) (x*y).
Proof.
 intros [[-> ->]|H1] [[-> ->]|H2]; try (left; split; ring).
 right; now apply LogBound_mult.
Qed.
Lemma ZLogBound_div e d h k x y : ZLogBound e h x -> LogBound d k y -> ZLogBound (e+d) (h/k) (x/y).
Proof.
 intros [[-> ->]|H1] H2; [left; split; unfold Rdiv; ring|right; now apply LogBound_div].
Qed.
Lemma ZLogBound_add_same e h k x y : ZLogBound e h x -> ZLogBound e k y -> ZLogBound e (h+k) (x+y).
Proof.
 intros [[-> ->]|H1] [[-> ->]|H2].
 - rewrite Rplus_0_l; apply ZLogBound_zero.
 - rewrite !Rplus_0_l; right; exact H2.
 - rewrite !Rplus_0_r; right; exact H1.
 - right; now apply LogBound_add_same.
Qed.
Lemma ZLogBound_add e d h k x y : ZLogBound e h x -> ZLogBound d k y -> ZLogBound (Rmax e d) (h+k) (x+y).
Proof.
 intros H1 H2; apply ZLogBound_add_same.
 - eapply ZLogBound_mono; [apply Rmax_l|exact H1].
 - eapply ZLogBound_mono; [apply Rmax_r|exact H2].
Qed.
Lemma ZLogBound_saturate e h x : ZLogBound e h x -> ZLogBound e (saturate h) (saturate x).
Proof.
 intros [[-> ->]|H]; [left; split; unfold saturate,Rdiv; ring|right; now apply LogBound_saturate].
Qed.
Definition ZRoundNodes (rnd:R->R) l (nodes:list R) :=
 forall z, In z nodes -> ZLogBound l (rnd z) z.
Lemma ZRoundNodes_app rnd l xs ys : ZRoundNodes rnd l (xs++ys) <-> ZRoundNodes rnd l xs /\ ZRoundNodes rnd l ys.
Proof.
 unfold ZRoundNodes; split.
 - intro H; split; intros z Hz; apply H; apply in_or_app; auto.
 - intros [Hx Hy] z Hz; apply in_app_or in Hz; destruct Hz; auto.
Qed.
Lemma RN64_zero choice : RN64 choice 0=0.
Proof. unfold RN64; apply round_0; apply valid_rnd_N. Qed.
(* An executable range obligation: each node is exactly zero, or its actual
   rounded output is positive, normal and finite. This excludes nonzero
   underflow and overflow, without excluding legitimate structural zeros. *)
Definition ZFiniteNormalNodes choice nodes :=
 Forall (fun z => z=0 \/ (0<RN64 choice z /\ normal64 (RN64 choice z) /\ finite64 (RN64 choice z))) nodes.
Lemma RN64_znodes choice nodes : ZFiniteNormalNodes choice nodes -> ZRoundNodes (RN64 choice) lambda64 nodes.
Proof.
 intros H z Hz; apply Forall_forall with (x:=z) in H; auto.
 destruct H as [->|[Hp [Hn Hf]]].
 - rewrite RN64_zero; apply ZLogBound_zero.
 - right; now apply RN64_log_rounded_normal.
Qed.
Lemma lambda64_nonnegative : 0<=lambda64.
Proof. unfold lambda64; apply log_unit_nonneg; pose proof u64_range; lra. Qed.
Lemma zrounded_mult_budget rnd l e d h k x y :
 ZLogBound e h x -> ZLogBound d k y -> ZLogBound l (rnd(h*k)) (h*k) ->
 ZLogBound (e+d+l) (rnd(h*k)) (x*y).
Proof.
 intros H1 H2 H3; pose proof (ZLogBound_trans _ _ _ _ _ H3 (ZLogBound_mult _ _ _ _ _ _ H1 H2)) as H.
 eapply ZLogBound_mono; [|exact H]; lra.
Qed.
Lemma zrounded_div_budget rnd l e d h k x y :
 ZLogBound e h x -> LogBound d k y -> ZLogBound l (rnd(h/k)) (h/k) ->
 ZLogBound (e+d+l) (rnd(h/k)) (x/y).
Proof.
 intros H1 H2 H3; pose proof (ZLogBound_trans _ _ _ _ _ H3 (ZLogBound_div _ _ _ _ _ _ H1 H2)) as H.
 eapply ZLogBound_mono; [|exact H]; lra.
Qed.
Lemma zrounded_add_budget rnd l e d h k x y :
 ZLogBound e h x -> ZLogBound d k y -> ZLogBound l (rnd(h+k)) (h+k) ->
 ZLogBound (Rmax e d+l) (rnd(h+k)) (x+y).
Proof.
 intros H1 H2 H3; pose proof (ZLogBound_trans _ _ _ _ _ H3 (ZLogBound_add _ _ _ _ _ _ H1 H2)) as H.
 eapply ZLogBound_mono; [|exact H]; lra.
Qed.

(* Arbitrary binary reduction topology, with an optional exact empty sum. *)
Inductive SumTree : Type := Empty | Leaf (g:nat) | Join (left right:SumTree).
Fixpoint tree_depth (t:SumTree) : nat :=
 match t with Empty | Leaf _ => O | Join a b => S (Nat.max (tree_depth a) (tree_depth b)) end.
Fixpoint tree_value (f:nat->R) (t:SumTree) : R :=
 match t with Empty => 0 | Leaf g => f g | Join a b => tree_value f a+tree_value f b end.
Fixpoint tree_eval (rnd:R->R) (f:nat->R) (t:SumTree) : R :=
 match t with Empty => 0 | Leaf g => f g | Join a b => rnd(tree_eval rnd f a+tree_eval rnd f b) end.
Fixpoint tree_nodes (rnd:R->R) (f:nat->R) (t:SumTree) : list R :=
 match t with Empty | Leaf _ => [] | Join a b =>
 (tree_eval rnd f a+tree_eval rnd f b)::(tree_nodes rnd f a++tree_nodes rnd f b) end.
Fixpoint tree_leaves (t:SumTree) : list nat :=
 match t with Empty => [] | Leaf g => [g] | Join a b => tree_leaves a++tree_leaves b end.

Theorem positive_tree_graph_budget rnd l e fh f t :
 0<=l -> (forall g, In g (tree_leaves t) -> ZLogBound e (fh g) (f g)) ->
 ZRoundNodes rnd l (tree_nodes rnd fh t) ->
 ZLogBound (e+INR(tree_depth t)*l) (tree_eval rnd fh t) (tree_value f t).
Proof.
 intros Hl; induction t as [|g|a IHa b IHb]; intros HF HN; cbn [tree_depth tree_leaves tree_nodes tree_eval tree_value] in *.
 - apply ZLogBound_zero.
 - cbn [INR]; replace (e+0*l) with e by ring; apply HF; simpl; auto.
 - assert (HFa:forall g, In g (tree_leaves a) -> ZLogBound e (fh g) (f g)).
   { intros g Hg; apply HF; apply in_or_app; auto. }
   assert (HFb:forall g, In g (tree_leaves b) -> ZLogBound e (fh g) (f g)).
   { intros g Hg; apply HF; apply in_or_app; auto. }
   assert (HNa:ZRoundNodes rnd l (tree_nodes rnd fh a)).
   { intros z Hz; apply HN; simpl; right; apply in_or_app; auto. }
   assert (HNb:ZRoundNodes rnd l (tree_nodes rnd fh b)).
   { intros z Hz; apply HN; simpl; right; apply in_or_app; auto. }
   pose proof (IHa HFa HNa) as HA; pose proof (IHb HFb HNb) as HB.
   pose proof (HN _ ltac:(simpl; auto)) as HR.
   pose proof (zrounded_add_budget _ _ _ _ _ _ _ _ HA HB HR) as HS.
   eapply ZLogBound_mono; [|exact HS].
   rewrite S_INR.
   assert (Hda:INR(tree_depth a)<=INR(Nat.max (tree_depth a) (tree_depth b))) by (apply le_INR; apply Nat.le_max_l).
   assert (Hdb:INR(tree_depth b)<=INR(Nat.max (tree_depth a) (tree_depth b))) by (apply le_INR; apply Nat.le_max_r).
   assert (Hm:Rmax (e+INR(tree_depth a)*l) (e+INR(tree_depth b)*l)<=e+INR(Nat.max (tree_depth a) (tree_depth b))*l).
   { apply Rmax_lub; nra. } nra.
Qed.
Theorem positive_tree_RN64_budget choice e fh f t :
 (forall g, In g (tree_leaves t) -> ZLogBound e (fh g) (f g)) ->
 ZFiniteNormalNodes choice (tree_nodes (RN64 choice) fh t) ->
 ZLogBound (e+INR(tree_depth t)*lambda64) (tree_eval (RN64 choice) fh t) (tree_value f t).
Proof. intros; apply positive_tree_graph_budget; auto using lambda64_nonnegative, RN64_znodes. Qed.

Fixpoint sequential_tree (n:nat) : SumTree :=
 match n with O=>Empty | S k=>Join (sequential_tree k) (Leaf k) end.
Lemma sequential_tree_depth n : tree_depth (sequential_tree n)=n.
Proof. induction n; simpl; auto; rewrite IHn, Nat.max_0_r; reflexivity. Qed.
Lemma sequential_tree_leaves n g : In g (tree_leaves (sequential_tree n)) <-> (g<n)%nat.
Proof.
 induction n; simpl; [lia|]. rewrite in_app_iff, IHn; simpl; lia.
Qed.
Corollary positive_sequential_RN64_budget choice e fh f n :
 (forall g, (g<n)%nat -> ZLogBound e (fh g) (f g)) ->
 ZFiniteNormalNodes choice (tree_nodes (RN64 choice) fh (sequential_tree n)) ->
 ZLogBound (e+INR n*lambda64) (tree_eval (RN64 choice) fh (sequential_tree n)) (tree_value f (sequential_tree n)).
Proof.
 intros HF HN.
 assert (HL:forall g, In g (tree_leaves (sequential_tree n)) -> ZLogBound e (fh g) (f g)).
 { intros g Hg; apply HF; now apply sequential_tree_leaves. }
 pose proof (positive_tree_RN64_budget choice e fh f (sequential_tree n) HL HN) as H.
 now rewrite sequential_tree_depth in H.
Qed.

(* A cancellation-free outer ratio. Both physical totals may contain zero
   leaves, and q may vanish; only the actual denominator and numerator need
   to be strictly positive. Their positivity is explicit, never inferred
   from a fake logarithmic contract on zero. *)
Theorem multigroup_ratio_graph_budget rnd l eq eL eR qh q Lh L Rh R :
 ZLogBound eq qh q -> ZLogBound eL Lh L -> ZLogBound eR Rh R ->
 0<q+L -> 0<R -> ZRoundNodes rnd l (balance_nodes rnd qh Lh Rh) ->
 LogBound (Rmax eq eL+eR+2*l) (eval_balance rnd qh Lh Rh) ((q+L)/R).
Proof.
 intros Hq HL HR Hnum Hden N; unfold ZRoundNodes,balance_nodes in N.
 pose proof (N (qh+Lh) ltac:(simpl; auto)) as Hn.
 pose proof (N (rnd(qh+Lh)/Rh) ltac:(simpl; auto)) as Hd.
 pose proof (zrounded_add_budget _ _ _ _ _ _ _ _ Hq HL Hn) as HS.
 pose proof (ZLogBound_positive _ _ _ Hden HR) as HD.
 pose proof (zrounded_div_budget _ _ _ _ _ _ _ _ HS HD Hd) as Hout.
 apply ZLogBound_positive; [apply Rdiv_lt_0_compat; assumption|].
 unfold eval_balance; eapply ZLogBound_mono; [|exact Hout]; lra.
Qed.

Theorem multigroup_tree_outer_RN64_budget choice eq eM eH qh q mh m hh h tM tH :
 ZLogBound eq qh q ->
 (forall g, In g (tree_leaves tM) -> ZLogBound eM (mh g) (m g)) ->
 (forall g, In g (tree_leaves tH) -> ZLogBound eH (hh g) (h g)) ->
 0<q+tree_value m tM -> 0<tree_value h tH ->
 ZFiniteNormalNodes choice
 (tree_nodes (RN64 choice) mh tM ++ tree_nodes (RN64 choice) hh tH ++
  balance_nodes (RN64 choice) qh (tree_eval (RN64 choice) mh tM) (tree_eval (RN64 choice) hh tH)) ->
 LogBound (Rmax eq (eM+INR(tree_depth tM)*lambda64)+
  (eH+INR(tree_depth tH)*lambda64)+2*lambda64)
 (eval_balance (RN64 choice) qh (tree_eval (RN64 choice) mh tM) (tree_eval (RN64 choice) hh tH))
 ((q+tree_value m tM)/tree_value h tH).
Proof.
 intros Hq HM HH HP HD HN; apply RN64_znodes in HN.
 apply ZRoundNodes_app in HN as [NM HN]; apply ZRoundNodes_app in HN as [NH NO].
 pose proof (positive_tree_graph_budget _ _ _ _ _ _ lambda64_nonnegative HM NM) as SM.
 pose proof (positive_tree_graph_budget _ _ _ _ _ _ lambda64_nonnegative HH NH) as SH.
 eapply multigroup_ratio_graph_budget; eauto.
Qed.

(* Opacity and Planck evaluations are explicit leaf contracts. The following
   theorems expand all subsequent arithmetic; no group-total contract is
   assumed. Absorption a, emission p and B may independently vanish. *)
Definition g_tau (rnd:R->R) h ah := rnd(h*ah).
Definition g_den (rnd:R->R) h ah := rnd(1+g_tau rnd h ah).
Definition g_hp (rnd:R->R) h ph := rnd(h*ph).
Definition g_w (rnd:R->R) h ah ph := rnd(g_hp rnd h ph/g_den rnd h ah).
Definition g_c (rnd:R->R) chi h ah ph := rnd(chi*g_w rnd h ah ph).
Definition g_M (rnd:R->R) chi h ah ph bh := rnd(g_c rnd chi h ah ph*bh).
Definition g_E (rnd:R->R) h ah ph bh rh :=
 rnd(rnd(rh+rnd(g_hp rnd h ph*bh))/g_den rnd h ah).
Definition g_H (rnd:R->R) chi h ah rh :=
 rnd(rnd(chi*rnd(g_tau rnd h ah/g_den rnd h ah))*rh).
Definition denominator_nodes (rnd:R->R) h ah := [h*ah; 1+g_tau rnd h ah].
Definition M_nodes (rnd:R->R) chi h ah ph bh :=
 denominator_nodes rnd h ah ++ [h*ph; g_hp rnd h ph/g_den rnd h ah;
 chi*g_w rnd h ah ph; g_c rnd chi h ah ph*bh].
Definition E_nodes (rnd:R->R) h ah ph bh rh :=
 denominator_nodes rnd h ah ++ [h*ph; g_hp rnd h ph*bh;
 rh+rnd(g_hp rnd h ph*bh); rnd(rh+rnd(g_hp rnd h ph*bh))/g_den rnd h ah].
Definition H_nodes (rnd:R->R) chi h ah rh :=
 denominator_nodes rnd h ah ++ [g_tau rnd h ah/g_den rnd h ah;
 chi*rnd(g_tau rnd h ah/g_den rnd h ah);
 rnd(chi*rnd(g_tau rnd h ah/g_den rnd h ah))*rh].

Lemma denominator_graph_budget rnd l ea h ah a :
 0<=l -> 0<=ea -> 0<h -> ZLogBound ea ah a ->
 ZRoundNodes rnd l (denominator_nodes rnd h ah) ->
 ZLogBound (ea+l) (g_tau rnd h ah) (h*a) /\
 LogBound (ea+2*l) (g_den rnd h ah) (1+h*a).
Proof.
 intros Hl He Hh Ha N; unfold ZRoundNodes,denominator_nodes in N.
 pose proof (N (h*ah) ltac:(simpl; auto)) as Nt.
 pose proof (zrounded_mult_budget _ _ _ _ _ _ _ _ (ZLogBound_exact 0 h ltac:(lra) ltac:(lra)) Ha Nt) as HT.
 assert (Ht:ZLogBound (ea+l) (g_tau rnd h ah) (h*a)).
 { unfold g_tau; eapply ZLogBound_mono; [|exact HT]; lra. }
 pose proof (N (1+g_tau rnd h ah) ltac:(simpl; auto)) as Nd.
 pose proof (zrounded_add_budget _ _ _ _ _ _ _ _ (ZLogBound_exact 0 1 ltac:(lra) ltac:(lra)) Ht Nd) as HD.
 split; [exact Ht|].
 pose proof (ZLogBound_nonnegative _ _ _ Ha) as Ha0.
 apply ZLogBound_positive; [nra|].
 unfold g_den; eapply ZLogBound_mono; [|exact HD].
 assert (Hm:Rmax 0 (ea+l)<=ea+l) by (apply Rmax_lub; lra); lra.
Qed.

Theorem group_M_graph_budget rnd l ea ep eb chi h ah a ph p bh B :
 0<=l -> 0<=ea -> 0<chi -> 0<h ->
 ZLogBound ea ah a -> ZLogBound ep ph p -> ZLogBound eb bh B ->
 ZRoundNodes rnd l (M_nodes rnd chi h ah ph bh) ->
 ZLogBound (ea+ep+eb+6*l) (g_M rnd chi h ah ph bh) (chi*(h*p/(1+h*a))*B).
Proof.
 intros Hl He Hchi Hh Ha Hp HB N; unfold M_nodes in N.
 apply ZRoundNodes_app in N as [ND N].
 destruct (denominator_graph_budget _ _ _ _ _ _ Hl He Hh Ha ND) as [HT HD].
 pose proof (N (h*ph) ltac:(simpl; auto)) as Np.
 pose proof (N (g_hp rnd h ph/g_den rnd h ah) ltac:(simpl; auto)) as Nw.
 pose proof (N (chi*g_w rnd h ah ph) ltac:(simpl; auto)) as Nc.
 pose proof (N (g_c rnd chi h ah ph*bh) ltac:(simpl; auto)) as Nm.
 pose proof (zrounded_mult_budget _ _ _ _ _ _ _ _ (ZLogBound_exact 0 h ltac:(lra) ltac:(lra)) Hp Np) as HP.
 change (ZLogBound (0+ep+l) (g_hp rnd h ph) (h*p)) in HP.
 pose proof (zrounded_div_budget _ _ _ _ _ _ _ _ HP HD Nw) as HW.
 change (ZLogBound (0+ep+l+(ea+2*l)+l) (g_w rnd h ah ph) (h*p/(1+h*a))) in HW.
 pose proof (zrounded_mult_budget _ _ _ _ _ _ _ _ (ZLogBound_exact 0 chi ltac:(lra) ltac:(lra)) HW Nc) as HC.
 change (ZLogBound (0+(0+ep+l+(ea+2*l)+l)+l) (g_c rnd chi h ah ph) (chi*(h*p/(1+h*a)))) in HC.
 pose proof (zrounded_mult_budget _ _ _ _ _ _ _ _ HC HB Nm) as HM.
 unfold g_M; eapply ZLogBound_mono; [|exact HM]; lra.
Qed.

Theorem group_E_graph_budget rnd l ea ep eb er h ah a ph p bh B rh r :
 0<=l -> 0<=ea -> 0<h ->
 ZLogBound ea ah a -> ZLogBound ep ph p -> ZLogBound eb bh B -> ZLogBound er rh r ->
 ZRoundNodes rnd l (E_nodes rnd h ah ph bh rh) ->
 ZLogBound (Rmax er (ep+eb+2*l)+ea+4*l) (g_E rnd h ah ph bh rh) ((r+(h*p)*B)/(1+h*a)).
Proof.
 intros Hl He Hh Ha Hp HB Hr N; unfold E_nodes in N.
 apply ZRoundNodes_app in N as [ND N].
 destruct (denominator_graph_budget _ _ _ _ _ _ Hl He Hh Ha ND) as [HT HD].
 pose proof (N (h*ph) ltac:(simpl; auto)) as Np.
 pose proof (N (g_hp rnd h ph*bh) ltac:(simpl; auto)) as Nb.
 pose proof (N (rh+rnd(g_hp rnd h ph*bh)) ltac:(simpl; auto)) as Nn.
 pose proof (N (rnd(rh+rnd(g_hp rnd h ph*bh))/g_den rnd h ah) ltac:(simpl; auto)) as Ne.
 pose proof (zrounded_mult_budget _ _ _ _ _ _ _ _ (ZLogBound_exact 0 h ltac:(lra) ltac:(lra)) Hp Np) as HP.
 change (ZLogBound (0+ep+l) (g_hp rnd h ph) (h*p)) in HP.
 pose proof (zrounded_mult_budget _ _ _ _ _ _ _ _ HP HB Nb) as HBP.
 assert (Hbp:ZLogBound (ep+eb+2*l) (rnd(g_hp rnd h ph*bh)) ((h*p)*B)).
 { eapply ZLogBound_mono; [|exact HBP]; lra. }
 pose proof (zrounded_add_budget _ _ _ _ _ _ _ _ Hr Hbp Nn) as Hnum.
 pose proof (zrounded_div_budget _ _ _ _ _ _ _ _ Hnum HD Ne) as HE.
 unfold g_E; eapply ZLogBound_mono; [|exact HE]; lra.
Qed.

(* The same rounded tau is used in the numerator and in 1+tau. Factoring
   through saturate preserves this correlation and charges opacity once. *)
Theorem group_H_graph_budget rnd l ea er chi h ah a rh r :
 0<=l -> 0<=ea -> 0<chi -> 0<h ->
 ZLogBound ea ah a -> ZLogBound er rh r ->
 ZRoundNodes rnd l (H_nodes rnd chi h ah rh) ->
 ZLogBound (ea+er+5*l) (g_H rnd chi h ah rh) (chi*saturate(h*a)*r).
Proof.
 intros Hl He Hchi Hh Ha Hr N; unfold H_nodes in N.
 apply ZRoundNodes_app in N as [ND N].
 destruct (denominator_graph_budget _ _ _ _ _ _ Hl He Hh Ha ND) as [HT HD].
 pose proof (ZLogBound_nonnegative _ _ _ HT) as Htpos.
 pose proof (ND (1+g_tau rnd h ah) ltac:(unfold denominator_nodes; simpl; auto)) as Nden.
 assert (Hden:LogBound l (g_den rnd h ah) (1+g_tau rnd h ah)).
 { unfold g_den; apply ZLogBound_positive; [lra|exact Nden]. }
 pose proof (N (g_tau rnd h ah/g_den rnd h ah) ltac:(simpl; auto)) as Nf.
 pose proof (N (chi*rnd(g_tau rnd h ah/g_den rnd h ah)) ltac:(simpl; auto)) as Nc.
 pose proof (N (rnd(chi*rnd(g_tau rnd h ah/g_den rnd h ah))*rh) ltac:(simpl; auto)) as Nh.
 pose proof (zrounded_div_budget _ _ _ _ _ _ _ _ (ZLogBound_exact 0 (g_tau rnd h ah) ltac:(lra) ltac:(lra)) Hden Nf) as Hf.
 pose proof (ZLogBound_saturate _ _ _ HT) as Hs.
 unfold saturate at 1 in Hs.
 pose proof (ZLogBound_trans _ _ _ _ _ Hf Hs) as HF.
 pose proof (zrounded_mult_budget _ _ _ _ _ _ _ _ (ZLogBound_exact 0 chi ltac:(lra) ltac:(lra)) HF Nc) as HC.
 pose proof (zrounded_mult_budget _ _ _ _ _ _ _ _ HC Hr Nh) as HH.
 unfold g_H; eapply ZLogBound_mono; [|exact HH]; lra.
Qed.

Theorem group_M_RN64_budget choice ea ep eb chi h ah a ph p bh B :
 0<=ea -> 0<chi -> 0<h ->
 ZLogBound ea ah a -> ZLogBound ep ph p -> ZLogBound eb bh B ->
 ZFiniteNormalNodes choice (M_nodes (RN64 choice) chi h ah ph bh) ->
 ZLogBound (ea+ep+eb+6*lambda64) (g_M (RN64 choice) chi h ah ph bh) (chi*(h*p/(1+h*a))*B).
Proof. intros; apply group_M_graph_budget; auto using lambda64_nonnegative, RN64_znodes. Qed.
Theorem group_E_RN64_budget choice ea ep eb er h ah a ph p bh B rh r :
 0<=ea -> 0<h ->
 ZLogBound ea ah a -> ZLogBound ep ph p -> ZLogBound eb bh B -> ZLogBound er rh r ->
 ZFiniteNormalNodes choice (E_nodes (RN64 choice) h ah ph bh rh) ->
 ZLogBound (Rmax er (ep+eb+2*lambda64)+ea+4*lambda64) (g_E (RN64 choice) h ah ph bh rh) ((r+(h*p)*B)/(1+h*a)).
Proof. intros; apply group_E_graph_budget; auto using lambda64_nonnegative, RN64_znodes. Qed.
Theorem group_H_RN64_budget choice ea er chi h ah a rh r :
 0<=ea -> 0<chi -> 0<h -> ZLogBound ea ah a -> ZLogBound er rh r ->
 ZFiniteNormalNodes choice (H_nodes (RN64 choice) chi h ah rh) ->
 ZLogBound (ea+er+5*lambda64) (g_H (RN64 choice) chi h ah rh) (chi*saturate(h*a)*r).
Proof. intros; apply group_H_graph_budget; auto using lambda64_nonnegative, RN64_znodes. Qed.

Lemma ZRoundNodes_flat_map rnd l (F:nat->list R) gs g :
 ZRoundNodes rnd l (flat_map F gs) -> In g gs -> ZRoundNodes rnd l (F g).
Proof.
 intros H Hg z Hz; apply H; apply in_flat_map; exists g; auto.
Qed.
Definition exact_M chi h (a p B:nat->R) g := chi*(h*p g/(1+h*a g))*B g.
Definition exact_H chi h (a r:nat->R) g := chi*saturate(h*a g)*r g.
Definition formed_M (rnd:R->R) chi h (ah ph bh:nat->R) g := g_M rnd chi h (ah g) (ph g) (bh g).
Definition formed_H (rnd:R->R) chi h (ah r:nat->R) g := g_H rnd chi h (ah g) (r g).
Definition eval_reverse_balance (rnd:R->R) qh Lh Rh := rnd(Rh/rnd(qh+Lh)).
Definition reverse_balance_nodes (rnd:R->R) qh Lh Rh := [qh+Lh; Rh/rnd(qh+Lh)].
Theorem multigroup_reverse_ratio_graph_budget rnd l eq eL eR qh q Lh L Rh R :
 ZLogBound eq qh q -> ZLogBound eL Lh L -> ZLogBound eR Rh R ->
 0<q+L -> 0<R -> ZRoundNodes rnd l (reverse_balance_nodes rnd qh Lh Rh) ->
 LogBound (eR+Rmax eq eL+2*l) (eval_reverse_balance rnd qh Lh Rh) (R/(q+L)).
Proof.
 intros Hq HL HR Hsum Hnum N; unfold ZRoundNodes,reverse_balance_nodes in N.
 pose proof (N (qh+Lh) ltac:(simpl; auto)) as Hn.
 pose proof (N (Rh/rnd(qh+Lh)) ltac:(simpl; auto)) as Hd.
 pose proof (zrounded_add_budget _ _ _ _ _ _ _ _ Hq HL Hn) as HS.
 pose proof (ZLogBound_positive _ _ _ Hsum HS) as HD.
 pose proof (zrounded_div_budget _ _ _ _ _ _ _ _ HR HD Hd) as Hout.
 apply ZLogBound_positive; [apply Rdiv_lt_0_compat; assumption|].
 unfold eval_reverse_balance; eapply ZLogBound_mono; [|exact Hout]; lra.
Qed.

Definition mg_outer_exact (heating:bool) q M H := if heating then (q+M)/H else M/(q+H).
Definition mg_outer_eval (rnd:R->R) (heating:bool) qh Mh Hh :=
 if heating then eval_balance rnd qh Mh Hh else eval_reverse_balance rnd qh Hh Mh.
Definition mg_outer_budget (heating:bool) eq eM eH l :=
 if heating then Rmax eq eM+eH+2*l else eM+Rmax eq eH+2*l.
Definition mg_all_nodes (rnd:R->R) (heating:bool) chi h qh (ah ph bh r:nat->R) tM tH :=
 let mh:=formed_M rnd chi h ah ph bh in
 let hh:=formed_H rnd chi h ah r in
 flat_map (fun g => M_nodes rnd chi h (ah g) (ph g) (bh g)) (tree_leaves tM) ++
 flat_map (fun g => H_nodes rnd chi h (ah g) (r g)) (tree_leaves tH) ++
 tree_nodes rnd mh tM ++ tree_nodes rnd hh tH ++
 (if heating then balance_nodes rnd qh (tree_eval rnd mh tM) (tree_eval rnd hh tH)
 else reverse_balance_nodes rnd qh (tree_eval rnd hh tH) (tree_eval rnd mh tM)).
Definition mg_total_eval (rnd:R->R) (heating:bool) chi h qh (ah ph bh r:nat->R) tM tH :=
 mg_outer_eval rnd heating qh (tree_eval rnd (formed_M rnd chi h ah ph bh) tM)
  (tree_eval rnd (formed_H rnd chi h ah r) tH).

(* Full evaluator graph theorem: the only approximation assumptions are
   scalar opacity/Planck/inner-q leaf contracts. Totals and the final ratio
   are computed by the displayed RN64 operation graph. *)
Theorem multigroup_formed_outer_RN64_budget choice (heating:bool) eq ea ep eb chi h qh q
 (ah a ph p bh B r:nat->R) tM tH :
 0<=ea -> 0<chi -> 0<h -> ZLogBound eq qh q ->
 (forall g, In g (tree_leaves tM++tree_leaves tH) -> ZLogBound ea (ah g) (a g)) ->
 (forall g, In g (tree_leaves tM) -> ZLogBound ep (ph g) (p g)) ->
 (forall g, In g (tree_leaves tM) -> ZLogBound eb (bh g) (B g)) ->
 (forall g, In g (tree_leaves tH) -> 0<=r g) ->
 (if heating then 0<q+tree_value (exact_M chi h a p B) tM /\ 0<tree_value (exact_H chi h a r) tH
 else 0<tree_value (exact_M chi h a p B) tM /\ 0<q+tree_value (exact_H chi h a r) tH) ->
 ZFiniteNormalNodes choice (mg_all_nodes (RN64 choice) heating chi h qh ah ph bh r tM tH) ->
 LogBound
 (mg_outer_budget heating eq (ea+ep+eb+(6+INR(tree_depth tM))*lambda64)
  (ea+(5+INR(tree_depth tH))*lambda64) lambda64)
 (mg_total_eval (RN64 choice) heating chi h qh ah ph bh r tM tH)
 (mg_outer_exact heating q (tree_value (exact_M chi h a p B) tM) (tree_value (exact_H chi h a r) tH)).
Proof.
 intros He Hchi Hh Hq Ha Hp HB Hr Hpos HN.
 apply RN64_znodes in HN; unfold mg_all_nodes in HN.
 apply ZRoundNodes_app in HN as [NMF HN].
 apply ZRoundNodes_app in HN as [NHF HN].
 apply ZRoundNodes_app in HN as [NMT HN].
 apply ZRoundNodes_app in HN as [NHT NO].
 assert (HM:forall g, In g (tree_leaves tM) ->
  ZLogBound (ea+ep+eb+6*lambda64) (formed_M (RN64 choice) chi h ah ph bh g) (exact_M chi h a p B g)).
 { intros g Hg; unfold formed_M,exact_M; apply group_M_graph_budget; auto using lambda64_nonnegative.
   - apply Ha; apply in_or_app; auto.
   - exact (ZRoundNodes_flat_map _ _ _ _ g NMF Hg). }
 assert (HH:forall g, In g (tree_leaves tH) ->
  ZLogBound (ea+5*lambda64) (formed_H (RN64 choice) chi h ah r g) (exact_H chi h a r g)).
 { intros g Hg; unfold formed_H,exact_H.
   replace (ea+5*lambda64) with (ea+0+5*lambda64) by ring.
   apply group_H_graph_budget; auto using lambda64_nonnegative.
   - apply Ha; apply in_or_app; auto.
   - apply ZLogBound_exact; [lra|auto].
   - exact (ZRoundNodes_flat_map _ _ _ _ g NHF Hg). }
 pose proof (positive_tree_graph_budget _ _ _ _ _ _ lambda64_nonnegative HM NMT) as SM.
 pose proof (positive_tree_graph_budget _ _ _ _ _ _ lambda64_nonnegative HH NHT) as SH.
 replace (ea+ep+eb+6*lambda64+INR(tree_depth tM)*lambda64)
  with (ea+ep+eb+(6+INR(tree_depth tM))*lambda64) in SM by ring.
 replace (ea+5*lambda64+INR(tree_depth tH)*lambda64)
  with (ea+(5+INR(tree_depth tH))*lambda64) in SH by ring.
 destruct heating; destruct Hpos as [Hnum Hden];
 unfold mg_outer_budget,mg_total_eval,mg_outer_eval,mg_outer_exact.
 - eapply multigroup_ratio_graph_budget; eauto.
 - eapply multigroup_reverse_ratio_graph_budget; eauto.
Qed.

(* n groups, straightforward left-to-right accumulation beginning at zero.
   The n*lambda64 term is conservative: the initial zero addition is charged.
   An implementation may instead provide a smaller tree_depth certificate. *)
Corollary multigroup_sequential_outer_RN64_budget choice (heating:bool) eq ea ep eb chi h qh q
 (ah a ph p bh B r:nat->R) n :
 0<=ea -> 0<chi -> 0<h -> ZLogBound eq qh q ->
 (forall g, (g<n)%nat -> ZLogBound ea (ah g) (a g)) ->
 (forall g, (g<n)%nat -> ZLogBound ep (ph g) (p g)) ->
 (forall g, (g<n)%nat -> ZLogBound eb (bh g) (B g)) ->
 (forall g, (g<n)%nat -> 0<=r g) ->
 (if heating then 0<q+tree_value (exact_M chi h a p B) (sequential_tree n) /\ 0<tree_value (exact_H chi h a r) (sequential_tree n)
 else 0<tree_value (exact_M chi h a p B) (sequential_tree n) /\ 0<q+tree_value (exact_H chi h a r) (sequential_tree n)) ->
 ZFiniteNormalNodes choice (mg_all_nodes (RN64 choice) heating chi h qh ah ph bh r (sequential_tree n) (sequential_tree n)) ->
 LogBound
 (mg_outer_budget heating eq (ea+ep+eb+(6+INR n)*lambda64)
  (ea+(5+INR n)*lambda64) lambda64)
 (mg_total_eval (RN64 choice) heating chi h qh ah ph bh r (sequential_tree n) (sequential_tree n))
 (mg_outer_exact heating q (tree_value (exact_M chi h a p B) (sequential_tree n)) (tree_value (exact_H chi h a r) (sequential_tree n))).
Proof.
 intros He Hchi Hh Hq Ha Hp HB Hr Hpos HN.
 assert (Hal:forall g, In g (tree_leaves (sequential_tree n)++tree_leaves (sequential_tree n)) -> ZLogBound ea (ah g) (a g)).
 { intros g Hg; apply in_app_or in Hg; destruct Hg; apply Ha; now apply sequential_tree_leaves. }
 assert (Hpl:forall g, In g (tree_leaves (sequential_tree n)) -> ZLogBound ep (ph g) (p g))
 by (intros; apply Hp; now apply sequential_tree_leaves).
 assert (Hbl:forall g, In g (tree_leaves (sequential_tree n)) -> ZLogBound eb (bh g) (B g))
 by (intros; apply HB; now apply sequential_tree_leaves).
 assert (Hrl:forall g, In g (tree_leaves (sequential_tree n)) -> 0<=r g)
 by (intros; apply Hr; now apply sequential_tree_leaves).
 pose proof (multigroup_formed_outer_RN64_budget choice (heating:bool) eq ea ep eb chi h qh q ah a ph p bh B r
  (sequential_tree n) (sequential_tree n) He Hchi Hh Hq Hal Hpl Hbl Hrl Hpos HN) as H.
 now rewrite !sequential_tree_depth in H.
Qed.

Print Assumptions positive_tree_RN64_budget.
Print Assumptions group_M_RN64_budget.
Print Assumptions group_E_RN64_budget.
Print Assumptions group_H_RN64_budget.
Print Assumptions multigroup_formed_outer_RN64_budget.
Print Assumptions multigroup_sequential_outer_RN64_budget.

(* Topology/coverage certificate. Permutation enforces each group exactly
   once, so reduction-tree accuracy cannot silently duplicate/drop bands. *)
From Coq Require Import Sorting.Permutation.
Fixpoint leaf_sum (f:nat->R) (gs:list nat) : R :=
 match gs with []=>0 | g::tail=>f g+leaf_sum f tail end.
Fixpoint finite_sum (n:nat) (f:nat->R) : R :=
 match n with O=>0 | S k=>finite_sum k f+f k end.
Lemma leaf_sum_app f xs ys : leaf_sum f (xs++ys)=leaf_sum f xs+leaf_sum f ys.
Proof. induction xs; simpl; [ring|rewrite IHxs; ring]. Qed.
Lemma tree_value_leaves f t : tree_value f t=leaf_sum f (tree_leaves t).
Proof. induction t; simpl; [reflexivity|ring|rewrite leaf_sum_app, <- IHt1, <- IHt2; reflexivity]. Qed.
Lemma leaf_sum_permutation f xs ys : Permutation xs ys -> leaf_sum f xs=leaf_sum f ys.
Proof. intro H; induction H; simpl; try congruence; try lra. Qed.
Lemma leaf_sum_seq f n : leaf_sum f (seq 0 n)=finite_sum n f.
Proof.
 induction n; [reflexivity|]. rewrite seq_S, leaf_sum_app, IHn; simpl; ring.
Qed.
Definition CoversGroups (n:nat) (t:SumTree) := Permutation (tree_leaves t) (seq 0 n).
Theorem tree_value_finite_sum f t n : CoversGroups n t -> tree_value f t=finite_sum n f.
Proof.
 intro H; rewrite tree_value_leaves, <- leaf_sum_seq; now apply leaf_sum_permutation.
Qed.
Lemma CoversGroups_member n t g : CoversGroups n t -> In g (tree_leaves t) -> (g<n)%nat.
Proof.
 intros H Hg; pose proof (Permutation_in _ H Hg) as Hseq; apply in_seq in Hseq; lia.
Qed.
Lemma sequential_tree_leaves_seq n : tree_leaves (sequential_tree n)=seq 0 n.
Proof. induction n; [reflexivity|]. simpl tree_leaves; simpl sequential_tree; simpl tree_leaves; rewrite IHn, seq_S; reflexivity. Qed.
Lemma sequential_tree_covers n : CoversGroups n (sequential_tree n).
Proof. unfold CoversGroups; rewrite sequential_tree_leaves_seq; apply Permutation_refl. Qed.

Print Assumptions tree_value_finite_sum.

(* A concrete balanced topology: 2^k consecutive leaves and exactly k
   charged additions along every leaf-to-root path. *)
Fixpoint balanced_tree (k start:nat) : SumTree :=
 match k with O=>Leaf start | S j=>Join (balanced_tree j start) (balanced_tree j (start+2^j)) end.
Lemma balanced_tree_depth k start : tree_depth (balanced_tree k start)=k.
Proof. revert start; induction k; intro start; simpl; [reflexivity|rewrite !IHk, Nat.max_id; reflexivity]. Qed.
Lemma balanced_tree_leaves k start : tree_leaves (balanced_tree k start)=seq start (2^k).
Proof.
 revert start; induction k; intro start; simpl; [reflexivity|].
 rewrite !IHk, <- seq_app.
 replace (2^k+(2^k+0))%nat with (2^k+2^k)%nat by lia; reflexivity.
Qed.
Lemma balanced_tree_covers k : CoversGroups (2^k) (balanced_tree k 0).
Proof. unfold CoversGroups; rewrite balanced_tree_leaves; apply Permutation_refl. Qed.

Lemma mg_outer_budget_mono heating eq eM eH EM EH l : eM<=EM -> eH<=EH ->
 mg_outer_budget heating eq eM eH l<=mg_outer_budget heating eq EM EH l.
Proof.
 intros HM HH; unfold mg_outer_budget; destruct heating.
 - assert (Rmax eq eM<=Rmax eq EM).
   { apply Rmax_lub; [apply Rmax_l|eapply Rle_trans; [exact HM|apply Rmax_r]]. } lra.
 - assert (Rmax eq eH<=Rmax eq EH).
   { apply Rmax_lub; [apply Rmax_l|eapply Rle_trans; [exact HH|apply Rmax_r]]. } lra.
Qed.
Lemma mg_exact_opacity_depth10_budget heating l dM dH :
 0<=l -> (dM<=10)%nat -> (dH<=10)%nat ->
 mg_outer_budget heating (52*l) ((14+INR dM)*l) ((5+INR dH)*l) l
 <= (if heating then 69 else 78)*l.
Proof.
 intros Hl HM HH.
 assert (HMr:INR dM<=10) by (replace 10 with (INR 10) by (simpl; ring); now apply le_INR).
 assert (HHr:INR dH<=10) by (replace 10 with (INR 10) by (simpl; ring); now apply le_INR).
 pose proof (mg_outer_budget_mono heating (52*l) ((14+INR dM)*l) ((5+INR dH)*l) (24*l) (15*l) l ltac:(nra) ltac:(nra)) as H.
 unfold mg_outer_budget in *; destruct heating;
 assert (H24:Rmax (52*l) (24*l)<=52*l) by (apply Rmax_lub; lra);
 assert (H15:Rmax (52*l) (15*l)<=52*l) by (apply Rmax_lub; lra); lra.
Qed.
Print Assumptions balanced_tree_covers.
Print Assumptions mg_exact_opacity_depth10_budget.
