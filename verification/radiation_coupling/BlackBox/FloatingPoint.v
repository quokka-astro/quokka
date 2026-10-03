(* Operation-graph error analysis. Every rounded node is charged explicitly.
   Stored coefficients and the trial argument are exact real inputs. *)
From Coq Require Import Reals Psatz Field ZArith List.
Require Import Flocq.Core.Core Flocq.Prop.Relative.
Open Scope R_scope.

Definition LogBound (e h x : R) : Prop :=
  0 < h /\ 0 < x /\ Rabs (ln h - ln x) <= e.

Lemma fp_ln_le x y : 0 < x -> x <= y -> ln x <= ln y.
Proof. intros Hx H; destruct (Rle_lt_or_eq_dec _ _ H); [left; apply ln_increasing; assumption|subst; right; reflexivity]. Qed.
Lemma fp_exp_le x y : x <= y -> exp x <= exp y.
Proof. intros H; destruct (Rle_lt_or_eq_dec _ _ H); [left; apply exp_increasing; assumption|subst; right; reflexivity]. Qed.
Lemma fp_ln_div x y : 0 < x -> 0 < y -> ln (x/y)=ln x-ln y.
Proof. intros; unfold Rdiv; rewrite ln_mult, ln_Rinv; auto using Rinv_0_lt_compat. Qed.
Lemma fp_ln_sqrt x : 0 < x -> ln (sqrt x)=ln x/2.
Proof.
 intro H; pose proof (sqrt_lt_R0 x H) as Hs.
 pose proof (sqrt_def x ltac:(lra)) as He.
 pose proof (ln_mult (sqrt x) (sqrt x) Hs Hs) as Hl.
 rewrite He in Hl; lra.
Qed.
Lemma LogBound_exact x : 0<x -> LogBound 0 x x.
Proof. intro H; repeat split; auto; rewrite Rminus_diag_eq,Rabs_R0; lra. Qed.
Lemma LogBound_nonneg e h x : LogBound e h x -> 0<=e.
Proof. intros (_&_&H); pose proof (Rabs_pos (ln h-ln x)); lra. Qed.
Lemma LogBound_mono e E h x : e<=E -> LogBound e h x -> LogBound E h x.
Proof. intros H (Hp&Hx&He); repeat split; auto; lra. Qed.
Lemma LogBound_sym e h x : LogBound e h x -> LogBound e x h.
Proof. intros (Hp&Hx&He); repeat split; auto. replace (ln x-ln h) with (-(ln h-ln x)) by ring; now rewrite Rabs_Ropp. Qed.
Lemma LogBound_trans e d h y x : LogBound e h y -> LogBound d y x -> LogBound (e+d) h x.
Proof.
 intros (Hh&Hy&H1) (_&Hx&H2); repeat split; auto.
 replace (ln h-ln x) with ((ln h-ln y)+(ln y-ln x)) by ring.
 eapply Rle_trans; [apply Rabs_triang|lra].
Qed.
Lemma LogBound_mult e d h k x y : LogBound e h x -> LogBound d k y -> LogBound (e+d) (h*k) (x*y).
Proof.
 intros (Hh&Hx&H1) (Hk&Hy&H2); repeat split; try nra.
 rewrite !ln_mult by assumption.
 replace (ln h+ln k-(ln x+ln y)) with ((ln h-ln x)+(ln k-ln y)) by ring.
 eapply Rle_trans; [apply Rabs_triang|lra].
Qed.
Lemma LogBound_div e d h k x y : LogBound e h x -> LogBound d k y -> LogBound (e+d) (h/k) (x/y).
Proof.
 intros (Hh&Hx&H1) (Hk&Hy&H2); repeat split; try (apply Rdiv_lt_0_compat; assumption).
 rewrite !fp_ln_div by assumption.
 replace (ln h-ln k-(ln x-ln y)) with ((ln h-ln x)+ -(ln k-ln y)) by ring.
 eapply Rle_trans; [apply Rabs_triang|rewrite Rabs_Ropp; lra].
Qed.
Lemma LogBound_sqrt e h x : LogBound e h x -> LogBound (e/2) (sqrt h) (sqrt x).
Proof.
 intros (Hh&Hx&H); repeat split; try (apply sqrt_lt_R0; assumption).
 rewrite !fp_ln_sqrt by assumption.
 replace (ln h/2-ln x/2) with ((ln h-ln x)/2) by field.
 unfold Rdiv; rewrite Rabs_mult. replace (Rabs (/2)) with (/2) by (symmetry; apply Rabs_pos_eq; lra). lra.
Qed.

Lemma LogBound_exp_upper e h x : LogBound e h x -> h <= exp e*x.
Proof.
 intros (Hh&Hx&H); apply Rabs_le_inv in H.
 pose proof (fp_exp_le (ln h) (e+ln x) ltac:(lra)) as Hexp.
 rewrite exp_plus, !exp_ln in Hexp by assumption; exact Hexp.
Qed.
Lemma LogBound_exp_lower e h x : LogBound e h x -> x <= exp e*h.
Proof. intro H; apply LogBound_exp_upper; now apply LogBound_sym. Qed.
Lemma LogBound_from_exp e h x : 0<h -> 0<x -> h<=exp e*x -> x<=exp e*h -> LogBound e h x.
Proof.
 intros Hh Hx H1 H2; repeat split; auto; apply Rabs_le.
 pose proof (fp_ln_le h (exp e*x) Hh H1) as L1.
 pose proof (fp_ln_le x (exp e*h) Hx H2) as L2.
 rewrite !ln_mult, !ln_exp in L1,L2 by (auto using exp_pos); lra.
Qed.
Lemma LogBound_add_same e h k x y : LogBound e h x -> LogBound e k y -> LogBound e (h+k) (x+y).
Proof.
 intros H1 H2; pose proof (LogBound_exp_upper _ _ _ H1); pose proof (LogBound_exp_upper _ _ _ H2).
 pose proof (LogBound_exp_lower _ _ _ H1); pose proof (LogBound_exp_lower _ _ _ H2).
 destruct H1 as (Hh&Hx&_); destruct H2 as (Hk&Hy&_).
 apply LogBound_from_exp; nra.
Qed.
Lemma LogBound_add e d h k x y : LogBound e h x -> LogBound d k y -> LogBound (Rmax e d) (h+k) (x+y).
Proof. intros; apply LogBound_add_same; [eapply LogBound_mono; [apply Rmax_l|eassumption]|eapply LogBound_mono; [apply Rmax_r|eassumption]]. Qed.

Definition log_unit (u : R) := -ln(1-u).
Lemma log_unit_nonneg u : 0<=u<1 -> 0<=log_unit u.
Proof. intros H; unfold log_unit; pose proof (fp_ln_le (1-u) 1 ltac:(lra) ltac:(lra)); rewrite ln_1 in H0; lra. Qed.
Lemma relative_round_log u h x : 0<=u<1 -> 0<x -> Rabs(h-x)<=u*x -> LogBound (log_unit u) h x.
Proof.
 intros Hu Hx He; apply Rabs_le_inv in He.
 assert (Hh:0<h) by nra.
 assert (Hlo:(1-u)*x<=h) by nra.
 assert (Hhi:h<=x/(1-u)).
 { apply (Rmult_le_reg_r (1-u)); [lra|].
   field_simplify; nra. }
 repeat split; auto; apply Rabs_le.
 pose proof (fp_ln_le ((1-u)*x) h ltac:(nra) Hlo) as L.
 pose proof (fp_ln_le h (x/(1-u)) Hh Hhi) as U.
 rewrite ln_mult in L by lra; rewrite fp_ln_div in U by lra.
 unfold log_unit; lra.
Qed.

(* Ordinary radix-2 round-to-nearest in the binary64 gradual-underflow format.
   Overflow must separately be excluded in the executable implementation. *)
Definition u64 : R := bpow radix2 (-53).
Definition lambda64 : R := log_unit u64.
Definition RN64 (choice : Z->bool) (x:R) : R :=
  round radix2 (FLT_exp (-1074) 53) (Znearest choice) x.
Definition normal64 (x:R) : Prop := bpow radix2 (-1022) <= Rabs x.
Lemma u64_range : 0<u64<1.
Proof. unfold u64; split; [apply bpow_gt_0|change (bpow radix2 (-53)<bpow radix2 0); apply bpow_lt; lia]. Qed.
Lemma RN64_relative choice x : normal64 x -> Rabs(RN64 choice x-x)<=u64*Rabs x.
Proof.
 intro H; unfold RN64,u64.
 pose proof (relative_error_N_FLT radix2 (-1074) 53 ltac:(lia) choice x H) as E.
 change (Rabs (round radix2 (FLT_exp (-1074) 53) (Znearest choice) x-x) <= /2*bpow radix2 (-52)*Rabs x) in E.
 replace (/2*bpow radix2 (-52)) with (bpow radix2 (-53)) in E.
 - exact E.
 - replace (-53)%Z with ((-1)+(-52))%Z by lia; rewrite bpow_plus; reflexivity.
Qed.
Lemma RN64_log choice x : 0<x -> normal64 x -> LogBound lambda64 (RN64 choice x) x.
Proof.
 intros Hx Hn; unfold lambda64; apply relative_round_log; [pose proof u64_range; lra|assumption|].
 pose proof (RN64_relative choice x Hn); rewrite (Rabs_pos_eq x) in H by lra; exact H.
Qed.

Print Assumptions relative_round_log.
Print Assumptions RN64_log.

Definition saturate t := t/(1+t).
Lemma saturate_pos t : 0<t -> 0<saturate t.
Proof. unfold saturate; intros; apply Rdiv_lt_0_compat; lra. Qed.
Lemma saturate_scaled_upper K h x : 1<=K -> 0<h -> 0<x -> h<=K*x ->
 saturate h<=K*saturate x.
Proof.
 intros HK Hh Hx H; unfold saturate.
 assert (0<=h*x*(K-1)) by (repeat apply Rmult_le_pos; lra).
 apply (Rmult_le_reg_r ((1+h)*(1+x))); [nra|].
 field_simplify; nra.
Qed.
Lemma LogBound_saturate e h x : LogBound e h x -> LogBound e (saturate h) (saturate x).
Proof.
 intro H; pose proof (LogBound_nonneg _ _ _ H) as He.
 assert (HK:1<=exp e) by (replace 1 with (exp 0) by apply exp_0; apply fp_exp_le; assumption).
 pose proof (LogBound_exp_upper _ _ _ H) as U; pose proof (LogBound_exp_lower _ _ _ H) as L.
 destruct H as (Hh&Hx&_); apply LogBound_from_exp; auto using saturate_pos, saturate_scaled_upper.
Qed.
Definition radiation r t B := (r+t*B)/(1+t).
Lemma radiation_pos r t B : 0<r -> 0<t -> 0<B -> 0<radiation r t B.
Proof. unfold radiation; intros; apply Rdiv_lt_0_compat; nra. Qed.
Lemma radiation_tau_scaled_upper K r h x B : 1<=K -> 0<r -> 0<h -> 0<x -> 0<B -> h<=K*x -> x<=K*h -> radiation r h B<=K*radiation r x B.
Proof.
 intros HK Hr Hh Hx HB HU HL; unfold radiation.
 apply (Rmult_le_reg_r ((1+h)*(1+x))); [nra|].
 field_simplify; try lra.
 assert (0<=r*(K-1)) by nra.
 assert (0<=r*(K*h-x)) by nra.
 assert (0<=B*(K*x-h)) by nra.
 assert (0<=h*x*B*(K-1)) by (repeat apply Rmult_le_pos; lra).
 nra.
Qed.
Lemma LogBound_radiation_tau e r h x B : 0<r -> 0<B -> LogBound e h x -> LogBound e (radiation r h B) (radiation r x B).
Proof.
 intros Hr HB H; pose proof (LogBound_nonneg _ _ _ H) as He.
 assert (HK:1<=exp e) by (replace 1 with (exp 0) by apply exp_0; apply fp_exp_le; assumption).
 pose proof (LogBound_exp_upper _ _ _ H) as U; pose proof (LogBound_exp_lower _ _ _ H) as L.
 destruct H as (Hh&Hx&_); apply LogBound_from_exp;
 auto using radiation_pos, radiation_tau_scaled_upper.
Qed.

(* A graph contract is only a list of primitive rounded-node contracts, never
   an assumed accuracy statement for an entire evaluator. *)
Definition RoundNodes (rnd:R->R) (l:R) (nodes:list R) : Prop :=
 forall z, List.In z nodes -> LogBound l (rnd z) z.
Lemma RN64_nodes choice nodes :
 List.Forall (fun z => 0<z /\ normal64 z) nodes ->
 RoundNodes (RN64 choice) lambda64 nodes.
Proof.
 intros H z Hz; apply List.Forall_forall with (x:=z) in H; auto.
 destruct H; apply RN64_log; assumption.
Qed.

Definition eval_B (rnd:R->R) a x :=
 let xx:=rnd(x*x) in let xxxx:=rnd(xx*xx) in rnd(a*xxxx).
Definition exact_B a x := a*((x*x)*(x*x)).
Definition B_nodes (rnd:R->R) a x :=
 let xx:=rnd(x*x) in let xxxx:=rnd(xx*xx) in (x*x)::(xx*xx)::(a*xxxx)::nil.
Lemma B_graph_budget rnd l a x : 0<=l -> 0<a -> 0<x ->
 RoundNodes rnd l (B_nodes rnd a x) -> LogBound (4*l) (eval_B rnd a x) (exact_B a x).
Proof.
 intros Hl Ha Hx R; unfold RoundNodes,B_nodes in R.
 pose proof (R (x*x) ltac:(simpl; tauto)) as H1.
 pose proof (R (rnd(x*x)*rnd(x*x)) ltac:(simpl; tauto)) as H2.
 pose proof (R (a*rnd(rnd(x*x)*rnd(x*x))) ltac:(simpl; tauto)) as H3.
 pose proof (LogBound_mult _ _ _ _ _ _ H1 H1) as Hsq.
 pose proof (LogBound_trans _ _ _ _ _ H2 Hsq) as H4.
 pose proof (LogBound_mult _ _ _ _ _ _ (LogBound_exact a Ha) H4) as Hmul.
 pose proof (LogBound_trans _ _ _ _ _ H3 Hmul) as H5.
 unfold eval_B,exact_B; eapply LogBound_mono; [|exact H5]; lra.
Qed.
Lemma initial_ratio_graph_budget rnd l a x r : 0<=l -> 0<a -> 0<x -> 0<r ->
 RoundNodes rnd l (B_nodes rnd a x) -> LogBound l (rnd(eval_B rnd a x/r)) (eval_B rnd a x/r) ->
 LogBound (5*l) (rnd(eval_B rnd a x/r)) (exact_B a x/r).
Proof.
 intros Hl Ha Hx Hr Hn Hd.
 pose proof (B_graph_budget rnd l a x Hl Ha Hx Hn) as HB.
 pose proof (LogBound_div _ _ _ _ _ _ HB (LogBound_exact r Hr)) as Hq.
 pose proof (LogBound_trans _ _ _ _ _ Hd Hq) as H; eapply LogBound_mono; [|exact H]; lra.
Qed.

Definition eval_tau (rnd:R->R) C kh := rnd(C*kh).
Definition eval_f (rnd:R->R) th := rnd(th/rnd(1+th)).
Definition eval_V (rnd:R->R) chi th := rnd(chi*eval_f rnd th).
Definition V_nodes (rnd:R->R) C kh chi :=
 let th:=eval_tau rnd C kh in
 (C*kh)::(1+th)::(th/rnd(1+th))::(chi*eval_f rnd th)::nil.
Lemma tau_graph_budget rnd l ek C kh k : 0<C -> LogBound ek kh k ->
 LogBound l (eval_tau rnd C kh) (C*kh) -> LogBound (ek+l) (eval_tau rnd C kh) (C*k).
Proof.
 intros HC Hk Hround.
 pose proof (LogBound_mult _ _ _ _ _ _ (LogBound_exact C HC) Hk) as Hm.
 pose proof (LogBound_trans _ _ _ _ _ Hround Hm) as H.
 eapply LogBound_mono; [|exact H]; lra.
Qed.
Lemma f_graph_arithmetic rnd l th : 0<=l -> 0<th ->
 LogBound l (rnd(1+th)) (1+th) ->
 LogBound l (eval_f rnd th) (th/rnd(1+th)) ->
 LogBound (2*l) (eval_f rnd th) (saturate th).
Proof.
 intros Hl Ht Hadd Hdiv.
 pose proof (LogBound_div _ _ _ _ _ _ (LogBound_exact th Ht) Hadd) as Hq.
 pose proof (LogBound_trans _ _ _ _ _ Hdiv Hq) as H.
 unfold saturate; eapply LogBound_mono; [|exact H]; lra.
Qed.
Lemma V_graph_budget rnd l ek C kh k chi : 0<=l -> 0<C -> 0<chi -> LogBound ek kh k ->
 RoundNodes rnd l (V_nodes rnd C kh chi) ->
 LogBound (ek+4*l) (eval_V rnd chi (eval_tau rnd C kh)) (chi*saturate(C*k)).
Proof.
 intros Hl HC Hc Hk R; unfold RoundNodes,V_nodes in R.
 pose proof (R (C*kh) ltac:(simpl; tauto)) as Htau.
 pose proof (tau_graph_budget rnd l ek C kh k HC Hk Htau) as HT.
 pose proof (R (1+eval_tau rnd C kh) ltac:(simpl; tauto)) as Ha.
 pose proof (R (eval_tau rnd C kh/rnd(1+eval_tau rnd C kh)) ltac:(simpl; tauto)) as Hd.
 pose proof (R (chi*eval_f rnd (eval_tau rnd C kh)) ltac:(simpl; tauto)) as Hv.
 pose proof (f_graph_arithmetic rnd l (eval_tau rnd C kh) Hl (proj1 HT) Ha Hd) as Hf.
 pose proof (LogBound_saturate _ _ _ HT) as Hs.
 pose proof (LogBound_trans _ _ _ _ _ Hf Hs) as HF.
 pose proof (LogBound_mult _ _ _ _ _ _ (LogBound_exact chi Hc) HF) as Hm.
 pose proof (LogBound_trans _ _ _ _ _ Hv Hm) as HV.
 unfold eval_V; eapply LogBound_mono; [|exact HV]; lra.
Qed.

Lemma rounded_mult_budget rnd l e d h k x y :
 LogBound e h x -> LogBound d k y -> LogBound l (rnd(h*k)) (h*k) ->
 LogBound (e+d+l) (rnd(h*k)) (x*y).
Proof. intros H1 H2 H3; pose proof (LogBound_trans _ _ _ _ _ H3 (LogBound_mult _ _ _ _ _ _ H1 H2)) as H; eapply LogBound_mono; [|exact H]; lra. Qed.
Lemma rounded_div_budget rnd l e d h k x y :
 LogBound e h x -> LogBound d k y -> LogBound l (rnd(h/k)) (h/k) ->
 LogBound (e+d+l) (rnd(h/k)) (x/y).
Proof. intros H1 H2 H3; pose proof (LogBound_trans _ _ _ _ _ H3 (LogBound_div _ _ _ _ _ _ H1 H2)) as H; eapply LogBound_mono; [|exact H]; lra. Qed.
Lemma rounded_add_budget rnd l e d h k x y :
 LogBound e h x -> LogBound d k y -> LogBound l (rnd(h+k)) (h+k) ->
 LogBound (Rmax e d+l) (rnd(h+k)) (x+y).
Proof. intros H1 H2 H3; pose proof (LogBound_trans _ _ _ _ _ H3 (LogBound_add _ _ _ _ _ _ H1 H2)) as H; eapply LogBound_mono; [|exact H]; lra. Qed.
Lemma rounded_sqrt_budget rnd l e h x :
 LogBound e h x -> LogBound l (rnd(sqrt h)) (sqrt h) ->
 LogBound (e/2+l) (rnd(sqrt h)) (sqrt x).
Proof. intros H1 H2; pose proof (LogBound_trans _ _ _ _ _ H2 (LogBound_sqrt _ _ _ H1)) as H; eapply LogBound_mono; [|exact H]; lra. Qed.

Definition eval_balance (rnd:R->R) qh Lh Rh := rnd(rnd(qh+Lh)/Rh).
Definition balance_nodes (rnd:R->R) qh Lh Rh := (qh+Lh)::(rnd(qh+Lh)/Rh)::nil.
Lemma balance_graph_budget rnd l eq e qh q Lh L Rh R :
 LogBound eq qh q -> LogBound e Lh L -> LogBound e Rh R ->
 RoundNodes rnd l (balance_nodes rnd qh Lh Rh) ->
 LogBound (Rmax eq e+e+2*l) (eval_balance rnd qh Lh Rh) ((q+L)/R).
Proof.
 intros Hq HL HR N; unfold RoundNodes,balance_nodes in N.
 pose proof (N (qh+Lh) ltac:(simpl; tauto)) as Hn.
 pose proof (N (rnd(qh+Lh)/Rh) ltac:(simpl; tauto)) as Hd.
 pose proof (rounded_add_budget _ _ _ _ _ _ _ _ Hq HL Hn) as Hsum.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ Hsum HR Hd) as Hdiv.
 unfold eval_balance; eapply LogBound_mono; [|exact Hdiv]; lra.
Qed.

Definition outer_beta et ek l := Rmax et (ek+9*l)+ek+11*l.
Lemma outer_balance_graph_budget rnd l et ek qh q vh v bh b r :
 0<=l -> 0<r -> LogBound et qh q -> LogBound (ek+4*l) vh v -> LogBound (4*l) bh b ->
 LogBound l (rnd(vh*bh)) (vh*bh) -> LogBound l (rnd(vh*r)) (vh*r) ->
 RoundNodes rnd l (balance_nodes rnd qh (rnd(vh*bh)) (rnd(vh*r))) ->
 RoundNodes rnd l (balance_nodes rnd qh (rnd(vh*r)) (rnd(vh*bh))) ->
 LogBound (outer_beta et ek l) (eval_balance rnd qh (rnd(vh*bh)) (rnd(vh*r))) ((q+v*b)/(v*r)) /\
 LogBound (outer_beta et ek l) (eval_balance rnd qh (rnd(vh*r)) (rnd(vh*bh))) ((q+v*r)/(v*b)).
Proof.
 intros Hl Hr Hq HV HB Hvb Hvr Nh Nc.
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ HV HB Hvb) as Hb.
 assert (Hbb:LogBound (ek+9*l) (rnd(vh*bh)) (v*b)) by (eapply LogBound_mono; [|exact Hb]; lra).
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ HV (LogBound_exact r Hr) Hvr) as Hr'.
 assert (Hrr:LogBound (ek+9*l) (rnd(vh*r)) (v*r)) by (eapply LogBound_mono; [|exact Hr']; lra).
 pose proof (balance_graph_budget _ _ _ _ _ _ _ _ _ _ Hq Hbb Hrr Nh) as Hh.
 pose proof (balance_graph_budget _ _ _ _ _ _ _ _ _ _ Hq Hrr Hbb Nc) as Hc.
 unfold outer_beta; split; eapply LogBound_mono; [|eassumption| |eassumption]; lra.
Qed.

Definition eval_radiation_low (rnd:R->R) r th bh := rnd(rnd(r+rnd(th*bh))/rnd(1+th)).
Definition radiation_low_nodes (rnd:R->R) r th bh :=
 (th*bh)::(r+rnd(th*bh))::(1+th)::(rnd(r+rnd(th*bh))/rnd(1+th))::nil.
Lemma radiation_low_arithmetic rnd l r th bh B : 0<=l -> 0<r -> 0<th -> LogBound (4*l) bh B ->
 RoundNodes rnd l (radiation_low_nodes rnd r th bh) ->
 LogBound (8*l) (eval_radiation_low rnd r th bh) (radiation r th B).
Proof.
 intros Hl Hr Ht HB N; unfold RoundNodes,radiation_low_nodes in N.
 pose proof (N (th*bh) ltac:(simpl; tauto)) as Hp.
 pose proof (N (r+rnd(th*bh)) ltac:(simpl; tauto)) as Hn.
 pose proof (N (1+th) ltac:(simpl; tauto)) as Hd.
 pose proof (N (rnd(r+rnd(th*bh))/rnd(1+th)) ltac:(simpl; tauto)) as Hout.
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ (LogBound_exact th Ht) HB Hp) as Hprod.
 pose proof (rounded_add_budget _ _ _ _ _ _ _ _ (LogBound_exact r Hr) Hprod Hn) as Hnum.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ Hnum Hd Hout) as Hfinal.
 unfold eval_radiation_low,radiation; eapply LogBound_mono; [|exact Hfinal].
 assert (Rmax 0 (0+4*l+l)<=5*l) by (apply Rmax_lub; lra); lra.
Qed.
Definition eval_radiation_high (rnd:R->R) r th bh := rnd(rnd(rnd(r/th)+bh)/rnd(1+rnd(1/th))).
Definition radiation_high_nodes (rnd:R->R) r th bh :=
 (r/th)::(1/th)::(rnd(r/th)+bh)::(1+rnd(1/th))::(rnd(rnd(r/th)+bh)/rnd(1+rnd(1/th)))::nil.
Lemma radiation_high_identity r t B : 0<t -> (r/t+B)/(1+1/t)=radiation r t B.
Proof. intro H; unfold radiation; field; split; lra. Qed.
Lemma radiation_high_arithmetic rnd l r th bh B : 0<=l -> 0<r -> 0<th -> LogBound (4*l) bh B ->
 RoundNodes rnd l (radiation_high_nodes rnd r th bh) ->
 LogBound (8*l) (eval_radiation_high rnd r th bh) (radiation r th B).
Proof.
 intros Hl Hr Ht HB N; unfold RoundNodes,radiation_high_nodes in N.
 pose proof (N (r/th) ltac:(simpl; tauto)) as Hri.
 pose proof (N (1/th) ltac:(simpl; tauto)) as Hti.
 pose proof (N (rnd(r/th)+bh) ltac:(simpl; tauto)) as Hn.
 pose proof (N (1+rnd(1/th)) ltac:(simpl; tauto)) as Hd.
 pose proof (N (rnd(rnd(r/th)+bh)/rnd(1+rnd(1/th))) ltac:(simpl; tauto)) as Hout.
 pose proof (rounded_add_budget _ _ _ _ _ _ _ _ Hri HB Hn) as Hnum.
 pose proof (rounded_add_budget _ _ _ _ _ _ _ _ (LogBound_exact 1 ltac:(lra)) Hti Hd) as Hden.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ Hnum Hden Hout) as Hfinal.
 rewrite radiation_high_identity in Hfinal by assumption.
 unfold eval_radiation_high; eapply LogBound_mono; [|exact Hfinal].
 assert (Rmax l (4*l)<=4*l) by (apply Rmax_lub; lra).
 assert (Rmax 0 l<=l) by (apply Rmax_lub; lra); lra.
Qed.

Lemma radiation_positive_graph_budget rnd l ek r th tau bh B :
 0<=l -> 0<r -> LogBound (ek+l) th tau -> LogBound (4*l) bh B ->
 RoundNodes rnd l (radiation_low_nodes rnd r th bh) ->
 RoundNodes rnd l (radiation_high_nodes rnd r th bh) ->
 LogBound (ek+9*l) (eval_radiation_low rnd r th bh) (radiation r tau B) /\
 LogBound (ek+9*l) (eval_radiation_high rnd r th bh) (radiation r tau B).
Proof.
 intros Hl Hr HT HB Nl Nh.
 pose proof (LogBound_radiation_tau _ r _ _ B Hr (proj1(proj2 HB)) HT) as Htau.
 pose proof (radiation_low_arithmetic _ _ _ _ _ _ Hl Hr (proj1 HT) HB Nl) as Hlow.
 pose proof (radiation_high_arithmetic _ _ _ _ _ _ Hl Hr (proj1 HT) HB Nh) as Hhigh.
 pose proof (LogBound_trans _ _ _ _ _ Hlow Htau) as Hl'.
 pose proof (LogBound_trans _ _ _ _ _ Hhigh Htau) as Hh'.
 split; eapply LogBound_mono; [|eassumption| |eassumption]; lra.
Qed.

Print Assumptions V_graph_budget.
Print Assumptions outer_balance_graph_budget.
Print Assumptions radiation_positive_graph_budget.

(* Opacity is genuinely black-box: the evaluator must supply this explicit
   value contract. No floating-point opacity implementation is inferred from
   smoothness, derivatives, or the collision solve. *)
Definition PlanckValueContract (kappa khat:R->R) (ek:R) :=
 forall x, 0<x -> LogBound ek (khat x) (kappa x).

Definition eval_inner_increment (rnd:R->R) A D v th dh :=
 let qh:=rnd(A*v) in let sh:=rnd(sqrt th) in
 let dhs:=rnd(D*sh) in let gh:=rnd(qh/dhs) in rnd(rnd(v+gh)/dh).
Definition inner_increment_nodes (rnd:R->R) A D v th dh :=
 let qh:=rnd(A*v) in let sh:=rnd(sqrt th) in
 let dhs:=rnd(D*sh) in let gh:=rnd(qh/dhs) in
 (A*v)::(sqrt th)::(D*sh)::(qh/dhs)::(v+gh)::(rnd(v+gh)/dh)::nil.
Lemma inner_increment_graph_budget rnd l A D v th t dh d :
 0<=l -> 0<A -> 0<D -> 0<v -> LogBound l th t -> LogBound l dh d ->
 RoundNodes rnd l (inner_increment_nodes rnd A D v th dh) ->
 LogBound (8*l) (eval_inner_increment rnd A D v th dh) ((v+A*v/(D*sqrt t))/d).
Proof.
 intros Hl HA HD Hv HT Hd N; unfold RoundNodes,inner_increment_nodes in N.
 pose proof (N (A*v) ltac:(simpl; tauto)) as Hq.
 pose proof (N (sqrt th) ltac:(simpl; tauto)) as Hs.
 pose proof (N (D*rnd(sqrt th)) ltac:(simpl; tauto)) as Hds.
 pose proof (N (rnd(A*v)/rnd(D*rnd(sqrt th))) ltac:(simpl; tauto)) as Hg.
 pose proof (N (v+rnd(rnd(A*v)/rnd(D*rnd(sqrt th)))) ltac:(simpl; tauto)) as Hn.
 pose proof (N (rnd(v+rnd(rnd(A*v)/rnd(D*rnd(sqrt th))))/dh) ltac:(simpl; tauto)) as Ho.
 pose proof (rounded_sqrt_budget _ _ _ _ _ HT Hs) as HS.
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ (LogBound_exact D HD) HS Hds) as HDS.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ Hq HDS Hg) as HG.
 pose proof (rounded_add_budget _ _ _ _ _ _ _ _ (LogBound_exact v Hv) HG Hn) as HN.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ HN Hd Ho) as HO.
 unfold eval_inner_increment; eapply LogBound_mono; [|exact HO].
 assert (Rmax 0 (l+(0+(l/2+l)+l)+l)<=9*l/2) by (apply Rmax_lub; lra); lra.
Qed.

Definition eval_inner_strong (rnd:R->R) A D T0 x t :=
 let vh:=rnd(T0-t) in let qh:=rnd(A*vh) in let sh:=rnd(sqrt t) in
 let dhs:=rnd(D*sh) in let gh:=rnd(qh/dhs) in rnd(rnd(x+gh)/t).
Definition inner_strong_nodes (rnd:R->R) A D T0 x t :=
 let vh:=rnd(T0-t) in let qh:=rnd(A*vh) in let sh:=rnd(sqrt t) in
 let dhs:=rnd(D*sh) in let gh:=rnd(qh/dhs) in
 (T0-t)::(A*vh)::(sqrt t)::(D*sh)::(qh/dhs)::(x+gh)::(rnd(x+gh)/t)::nil.
Lemma inner_strong_graph_budget rnd l A D T0 x t :
 0<=l -> 0<A -> 0<D -> 0<x -> 0<t -> t<T0 ->
 RoundNodes rnd l (inner_strong_nodes rnd A D T0 x t) ->
 LogBound (8*l) (eval_inner_strong rnd A D T0 x t) ((x+A*(T0-t)/(D*sqrt t))/t).
Proof.
 intros Hl HA HD Hx Ht HT N; unfold RoundNodes,inner_strong_nodes in N.
 pose proof (N (T0-t) ltac:(simpl; tauto)) as Hv.
 pose proof (N (A*rnd(T0-t)) ltac:(simpl; tauto)) as Hq.
 pose proof (N (sqrt t) ltac:(simpl; tauto)) as Hs.
 pose proof (N (D*rnd(sqrt t)) ltac:(simpl; tauto)) as Hds.
 pose proof (N (rnd(A*rnd(T0-t))/rnd(D*rnd(sqrt t))) ltac:(simpl; tauto)) as Hg.
 pose proof (N (x+rnd(rnd(A*rnd(T0-t))/rnd(D*rnd(sqrt t)))) ltac:(simpl; tauto)) as Hn.
 pose proof (N (rnd(x+rnd(rnd(A*rnd(T0-t))/rnd(D*rnd(sqrt t))))/t) ltac:(simpl; tauto)) as Ho.
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ (LogBound_exact A HA) Hv Hq) as HQ.
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ (LogBound_exact D HD) Hs Hds) as HDS.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ HQ HDS Hg) as HG.
 pose proof (rounded_add_budget _ _ _ _ _ _ _ _ (LogBound_exact x Hx) HG Hn) as HN.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ HN (LogBound_exact t Ht) Ho) as HO.
 unfold eval_inner_strong; eapply LogBound_mono; [|exact HO].
 assert (Rmax 0 (0+l+l+(0+l+l)+l)<=5*l) by (apply Rmax_lub; lra); lra.
Qed.

Lemma heating_inner_graph_budget rnd l A D T0 x delta :
 0<=l -> 0<A -> 0<D -> 0<delta ->
 LogBound l (rnd(T0+delta)) (T0+delta) -> LogBound l (rnd(x-T0)) (x-T0) ->
 RoundNodes rnd l (inner_increment_nodes rnd A D delta (rnd(T0+delta)) (rnd(x-T0))) ->
 LogBound (8*l) (eval_inner_increment rnd A D delta (rnd(T0+delta)) (rnd(x-T0)))
 ((delta+A*delta/(D*sqrt(T0+delta)))/(x-T0)).
Proof. intros; apply inner_increment_graph_budget; assumption. Qed.
Lemma weak_cooling_inner_graph_budget rnd l A D T0 x eta :
 0<=l -> 0<A -> 0<D -> 0<eta ->
 LogBound l (rnd(T0-eta)) (T0-eta) -> LogBound l (rnd(T0-x)) (T0-x) ->
 RoundNodes rnd l (inner_increment_nodes rnd A D eta (rnd(T0-eta)) (rnd(T0-x))) ->
 LogBound (8*l) (eval_inner_increment rnd A D eta (rnd(T0-eta)) (rnd(T0-x)))
 ((eta+A*eta/(D*sqrt(T0-eta)))/(T0-x)).
Proof. intros; apply inner_increment_graph_budget; assumption. Qed.

Lemma gas_energy_graph_budget rnd l et A th t : 0<A -> LogBound et th t ->
 LogBound l (rnd(A*th)) (A*th) -> LogBound (et+l) (rnd(A*th)) (A*t).
Proof.
 intros HA HT Hround; pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ (LogBound_exact A HA) HT Hround) as H.
 eapply LogBound_mono; [|exact H]; lra.
Qed.

Print Assumptions inner_increment_graph_budget.
Print Assumptions inner_strong_graph_budget.

Lemma exact_B_quartic a x : exact_B a x=a*x^4.
Proof. unfold exact_B; ring. Qed.
Lemma RoundNodes_app rnd l xs ys : RoundNodes rnd l (xs++ys) <-> RoundNodes rnd l xs /\ RoundNodes rnd l ys.
Proof.
 unfold RoundNodes; split.
 - intro H; split; intros z Hz; apply H; apply in_or_app; auto.
 - intros [Hx Hy] z Hz; apply in_app_or in Hz; destruct Hz; auto.
Qed.
Definition outer_core (rnd:R->R) (heating:bool) qh vh bh r :=
 if heating then eval_balance rnd qh (rnd(vh*bh)) (rnd(vh*r))
 else eval_balance rnd qh (rnd(vh*r)) (rnd(vh*bh)).
Definition outer_core_nodes (rnd:R->R) (heating:bool) qh vh bh r :=
 (vh*bh)::(vh*r)::
 (if heating then balance_nodes rnd qh (rnd(vh*bh)) (rnd(vh*r))
  else balance_nodes rnd qh (rnd(vh*r)) (rnd(vh*bh))).
Definition exact_outer (heating:bool) q v b r :=
 if heating then (q+v*b)/(v*r) else (q+v*r)/(v*b).
Lemma outer_core_graph_budget rnd l et ek heating qh q vh v bh b r :
 0<=l -> 0<r -> LogBound et qh q -> LogBound (ek+4*l) vh v -> LogBound (4*l) bh b ->
 RoundNodes rnd l (outer_core_nodes rnd heating qh vh bh r) ->
 LogBound (outer_beta et ek l) (outer_core rnd heating qh vh bh r) (exact_outer heating q v b r).
Proof.
 intros Hl Hr Hq HV HB N; unfold RoundNodes,outer_core_nodes in N.
 pose proof (N (vh*bh) ltac:(simpl; tauto)) as Hvb.
 pose proof (N (vh*r) ltac:(simpl; tauto)) as Hvr.
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ HV HB Hvb) as Hb.
 assert (Hbb:LogBound (ek+9*l) (rnd(vh*bh)) (v*b)) by (eapply LogBound_mono; [|exact Hb]; lra).
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ HV (LogBound_exact r Hr) Hvr) as Hr'.
 assert (Hrr:LogBound (ek+9*l) (rnd(vh*r)) (v*r)) by (eapply LogBound_mono; [|exact Hr']; lra).
 destruct heating; unfold outer_core,exact_outer,outer_beta.
 - assert (Nb:RoundNodes rnd l (balance_nodes rnd qh (rnd(vh*bh)) (rnd(vh*r)))).
   { intros z Hz; apply N; simpl; right; right; exact Hz. }
   pose proof (balance_graph_budget _ _ _ _ _ _ _ _ _ _ Hq Hbb Hrr Nb) as H.
   eapply LogBound_mono; [|exact H]; lra.
 - assert (Nb:RoundNodes rnd l (balance_nodes rnd qh (rnd(vh*r)) (rnd(vh*bh)))).
   { intros z Hz; apply N; simpl; right; right; exact Hz. }
   pose proof (balance_graph_budget _ _ _ _ _ _ _ _ _ _ Hq Hrr Hbb Nb) as H.
   eapply LogBound_mono; [|exact H]; lra.
Qed.

Definition eval_outer (rnd:R->R) heating qh C kh chi a x r :=
 let vh:=eval_V rnd chi (eval_tau rnd C kh) in
 let bh:=eval_B rnd a x in outer_core rnd heating qh vh bh r.
Definition outer_nodes (rnd:R->R) heating qh C kh chi a x r :=
 B_nodes rnd a x ++ V_nodes rnd C kh chi ++
 outer_core_nodes rnd heating qh (eval_V rnd chi (eval_tau rnd C kh)) (eval_B rnd a x) r.
Theorem planck_outer_graph_budget rnd l et ek kappa khat heating qh q C chi a x r :
 0<=l -> 0<C -> 0<chi -> 0<a -> 0<x -> 0<r ->
 PlanckValueContract kappa khat ek -> LogBound et qh q ->
 RoundNodes rnd l (outer_nodes rnd heating qh C (khat x) chi a x r) ->
 LogBound (outer_beta et ek l) (eval_outer rnd heating qh C (khat x) chi a x r)
 (exact_outer heating q (chi*saturate(C*kappa x)) (a*x^4) r).
Proof.
 intros Hl HC Hc Ha Hx Hr HK Hq N; unfold outer_nodes in N.
 apply RoundNodes_app in N as [NB N]; apply RoundNodes_app in N as [NV NO].
 pose proof (B_graph_budget rnd l a x Hl Ha Hx NB) as HB.
 pose proof (V_graph_budget rnd l ek C (khat x) (kappa x) chi Hl HC Hc (HK x Hx) NV) as HV.
 rewrite exact_B_quartic in HB; unfold eval_outer.
 now apply outer_core_graph_budget.
Qed.
Theorem planck_outer_RN64_budget choice et ek kappa khat heating qh q C chi a x r :
 0<C -> 0<chi -> 0<a -> 0<x -> 0<r ->
 PlanckValueContract kappa khat ek -> LogBound et qh q ->
 Forall (fun z => 0<z /\ normal64 z)
 (outer_nodes (RN64 choice) heating qh C (khat x) chi a x r) ->
 LogBound (outer_beta et ek lambda64)
 (eval_outer (RN64 choice) heating qh C (khat x) chi a x r)
 (exact_outer heating q (chi*saturate(C*kappa x)) (a*x^4) r).
Proof.
 intros HC Hc Ha Hx Hr HK Hq N; apply planck_outer_graph_budget; auto.
 - unfold lambda64; apply log_unit_nonneg; pose proof u64_range; lra.
 - now apply RN64_nodes.
Qed.

Definition eval_radiation (rnd:R->R) r th bh :=
 if Rle_dec th 1 then eval_radiation_low rnd r th bh else eval_radiation_high rnd r th bh.
Definition radiation_nodes (rnd:R->R) r th bh :=
 if Rle_dec th 1 then radiation_low_nodes rnd r th bh else radiation_high_nodes rnd r th bh.
Lemma radiation_selected_graph_budget rnd l ek r th tau bh B :
 0<=l -> 0<r -> LogBound (ek+l) th tau -> LogBound (4*l) bh B ->
 RoundNodes rnd l (radiation_nodes rnd r th bh) ->
 LogBound (ek+9*l) (eval_radiation rnd r th bh) (radiation r tau B).
Proof.
 intros Hl Hr HT HB N.
 pose proof (LogBound_radiation_tau _ r _ _ B Hr (proj1(proj2 HB)) HT) as Htau.
 unfold eval_radiation,radiation_nodes in *; destruct (Rle_dec th 1).
 - pose proof (radiation_low_arithmetic _ _ _ _ _ _ Hl Hr (proj1 HT) HB N) as Hlow.
   pose proof (LogBound_trans _ _ _ _ _ Hlow Htau) as H.
   eapply LogBound_mono; [|exact H]; lra.
 - pose proof (radiation_high_arithmetic _ _ _ _ _ _ Hl Hr (proj1 HT) HB N) as Hhigh.
   pose proof (LogBound_trans _ _ _ _ _ Hhigh Htau) as H.
   eapply LogBound_mono; [|exact H]; lra.
Qed.
Definition radiation_output_nodes (rnd:R->R) r C kh a x :=
 B_nodes rnd a x ++ (C*kh)::radiation_nodes rnd r (eval_tau rnd C kh) (eval_B rnd a x).
Theorem planck_radiation_graph_budget rnd l ek kappa khat C a x r :
 0<=l -> 0<C -> 0<a -> 0<x -> 0<r -> PlanckValueContract kappa khat ek ->
 RoundNodes rnd l (radiation_output_nodes rnd r C (khat x) a x) ->
 LogBound (ek+9*l)
 (eval_radiation rnd r (eval_tau rnd C (khat x)) (eval_B rnd a x))
 (radiation r (C*kappa x) (a*x^4)).
Proof.
 intros Hl HC Ha Hx Hr HK N; unfold radiation_output_nodes in N.
 apply RoundNodes_app in N as [NB NR].
 pose proof (B_graph_budget rnd l a x Hl Ha Hx NB) as HB.
 assert (HT:LogBound l (eval_tau rnd C (khat x)) (C*khat x)) by (apply NR; simpl; auto).
 pose proof (tau_graph_budget rnd l ek C (khat x) (kappa x) HC (HK x Hx) HT) as HT'.
 assert (RN:RoundNodes rnd l (radiation_nodes rnd r (eval_tau rnd C (khat x)) (eval_B rnd a x))) by (intros z Hz; apply NR; simpl; auto).
 rewrite exact_B_quartic in HB; now apply radiation_selected_graph_budget.
Qed.
Theorem planck_radiation_RN64_budget choice ek kappa khat C a x r :
 0<C -> 0<a -> 0<x -> 0<r -> PlanckValueContract kappa khat ek ->
 Forall (fun z => 0<z /\ normal64 z)
 (radiation_output_nodes (RN64 choice) r C (khat x) a x) ->
 LogBound (ek+9*lambda64)
 (eval_radiation (RN64 choice) r (eval_tau (RN64 choice) C (khat x)) (eval_B (RN64 choice) a x))
 (radiation r (C*kappa x) (a*x^4)).
Proof.
 intros HC Ha Hx Hr HK N; apply planck_radiation_graph_budget; auto.
 - unfold lambda64; apply log_unit_nonneg; pose proof u64_range; lra.
 - now apply RN64_nodes.
Qed.

Print Assumptions planck_outer_RN64_budget.
Print Assumptions planck_radiation_RN64_budget.

(* At the smallest normal binade, a normal rounded result does not imply the
   usual error <= u*|exact|. Charge error relative to the rounded result instead;
   reversing the relative-round lemma still gives exactly the same log bound. *)
Local Instance fp_prec64 : Prec_gt_0 53. Proof. unfold Prec_gt_0; lia. Qed.
Lemma RN64_relative_to_result choice x : normal64 (RN64 choice x) ->
 Rabs(RN64 choice x-x)<=u64*Rabs(RN64 choice x).
Proof.
 intro Hn.
 pose proof (error_le_half_ulp_round radix2 (FLT_exp (-1074) 53) choice x) as He.
 pose proof (ulp_FLT_le radix2 (-1074) 53 (RN64 choice x) Hn) as Hu.
 change (ulp radix2 (FLT_exp (-1074) 53) (RN64 choice x)<=Rabs(RN64 choice x)*bpow radix2 (-52)) in Hu.
 replace (bpow radix2 (-52)) with (2*u64) in Hu.
 - unfold RN64 in *; nra.
 - unfold u64; replace (-52)%Z with (1+(-53))%Z by lia; rewrite bpow_plus; reflexivity.
Qed.
Lemma RN64_log_rounded_normal choice x :
 0<RN64 choice x -> normal64 (RN64 choice x) -> LogBound lambda64 (RN64 choice x) x.
Proof.
 intros Hp Hn; apply LogBound_sym; unfold lambda64; apply relative_round_log.
 - pose proof u64_range; lra.
 - exact Hp.
 - pose proof (RN64_relative_to_result choice x Hn) as H.
   rewrite (Rabs_pos_eq (RN64 choice x)) in H by lra.
   replace (x-RN64 choice x) with (-(RN64 choice x-x)) by ring.
   now rewrite Rabs_Ropp.
Qed.
Definition RoundedNormalNodes choice nodes :=
 Forall (fun z => 0<RN64 choice z /\ normal64 (RN64 choice z)) nodes.
Lemma RN64_rounded_nodes choice nodes : RoundedNormalNodes choice nodes -> RoundNodes (RN64 choice) lambda64 nodes.
Proof.
 intros H z Hz; unfold RoundedNormalNodes in H; apply Forall_forall with (x:=z) in H; auto.
 destruct H; apply RN64_log_rounded_normal; assumption.
Qed.
Theorem planck_outer_rounded_normal_budget choice et ek kappa khat heating qh q C chi a x r :
 0<C -> 0<chi -> 0<a -> 0<x -> 0<r ->
 PlanckValueContract kappa khat ek -> LogBound et qh q ->
 RoundedNormalNodes choice (outer_nodes (RN64 choice) heating qh C (khat x) chi a x r) ->
 LogBound (outer_beta et ek lambda64)
 (eval_outer (RN64 choice) heating qh C (khat x) chi a x r)
 (exact_outer heating q (chi*saturate(C*kappa x)) (a*x^4) r).
Proof.
 intros HC Hc Ha Hx Hr HK Hq N; apply planck_outer_graph_budget; auto.
 - unfold lambda64; apply log_unit_nonneg; pose proof u64_range; lra.
 - now apply RN64_rounded_nodes.
Qed.
Theorem planck_radiation_rounded_normal_budget choice ek kappa khat C a x r :
 0<C -> 0<a -> 0<x -> 0<r -> PlanckValueContract kappa khat ek ->
 RoundedNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat x) a x) ->
 LogBound (ek+9*lambda64)
 (eval_radiation (RN64 choice) r (eval_tau (RN64 choice) C (khat x)) (eval_B (RN64 choice) a x))
 (radiation r (C*kappa x) (a*x^4)).
Proof.
 intros HC Ha Hx Hr HK N; apply planck_radiation_graph_budget; auto.
 - unfold lambda64; apply log_unit_nonneg; pose proof u64_range; lra.
 - now apply RN64_rounded_nodes.
Qed.

Print Assumptions planck_outer_rounded_normal_budget.
Print Assumptions planck_radiation_rounded_normal_budget.

(* An explicit upper-range predicate distinguishes the real FLT rounding model
   from finite IEEE binary64. Nodes with RN64 magnitude >= 2^1024 are excluded. *)
Definition finite64 (x:R) : Prop := Rabs x < bpow radix2 1024.
Definition FiniteNormalNodes choice nodes :=
 Forall (fun z => 0<RN64 choice z /\ normal64 (RN64 choice z) /\ finite64 (RN64 choice z)) nodes.
Lemma finite_nodes_rounded_normal choice nodes : FiniteNormalNodes choice nodes -> RoundedNormalNodes choice nodes.
Proof.
 intro H; unfold FiniteNormalNodes,RoundedNormalNodes in *.
 apply Forall_forall; intros x Hx; apply Forall_forall with (x:=x) in H; tauto.
Qed.
Theorem planck_outer_finite64_budget choice et ek kappa khat heating qh q C chi a x r :
 0<C -> 0<chi -> 0<a -> 0<x -> 0<r ->
 PlanckValueContract kappa khat ek -> LogBound et qh q ->
 FiniteNormalNodes choice (outer_nodes (RN64 choice) heating qh C (khat x) chi a x r) ->
 LogBound (outer_beta et ek lambda64)
 (eval_outer (RN64 choice) heating qh C (khat x) chi a x r)
 (exact_outer heating q (chi*saturate(C*kappa x)) (a*x^4) r).
Proof. intros; apply planck_outer_rounded_normal_budget; auto using finite_nodes_rounded_normal. Qed.
Theorem planck_radiation_finite64_budget choice ek kappa khat C a x r :
 0<C -> 0<a -> 0<x -> 0<r -> PlanckValueContract kappa khat ek ->
 FiniteNormalNodes choice (radiation_output_nodes (RN64 choice) r C (khat x) a x) ->
 LogBound (ek+9*lambda64)
 (eval_radiation (RN64 choice) r (eval_tau (RN64 choice) C (khat x)) (eval_B (RN64 choice) a x))
 (radiation r (C*kappa x) (a*x^4)).
Proof. intros; apply planck_radiation_rounded_normal_budget; auto using finite_nodes_rounded_normal. Qed.

Lemma balance_zero_graph_budget rnd l e Lh L Rh R :
 LogBound e Lh L -> LogBound e Rh R ->
 RoundNodes rnd l (balance_nodes rnd 0 Lh Rh) ->
 LogBound (2*e+2*l) (eval_balance rnd 0 Lh Rh) ((0+L)/R).
Proof.
 intros HL HR N; unfold RoundNodes,balance_nodes in N.
 pose proof (N (0+Lh) ltac:(simpl; tauto)) as Hn.
 pose proof (N (rnd(0+Lh)/Rh) ltac:(simpl; tauto)) as Hd.
 rewrite Rplus_0_l in Hn.
 pose proof (LogBound_trans _ _ _ _ _ Hn HL) as Hsum.
 rewrite Rplus_0_l in Hd.
 pose proof (rounded_div_budget _ _ _ _ _ _ _ _ Hsum HR Hd) as Hdiv.
 unfold eval_balance; rewrite !Rplus_0_l; eapply LogBound_mono; [|exact Hdiv]; lra.
Qed.
Lemma outer_core_zero_graph_budget rnd l et ek heating vh v bh b r :
 0<=l -> 0<r -> LogBound (ek+4*l) vh v -> LogBound (4*l) bh b ->
 RoundNodes rnd l (outer_core_nodes rnd heating 0 vh bh r) ->
 LogBound (outer_beta et ek l) (outer_core rnd heating 0 vh bh r) (exact_outer heating 0 v b r).
Proof.
 intros Hl Hr HV HB N; unfold RoundNodes,outer_core_nodes in N.
 pose proof (N (vh*bh) ltac:(simpl; tauto)) as Hvb.
 pose proof (N (vh*r) ltac:(simpl; tauto)) as Hvr.
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ HV HB Hvb) as Hb.
 assert (Hbb:LogBound (ek+9*l) (rnd(vh*bh)) (v*b)) by (eapply LogBound_mono; [|exact Hb]; lra).
 pose proof (rounded_mult_budget _ _ _ _ _ _ _ _ HV (LogBound_exact r Hr) Hvr) as Hr'.
 assert (Hrr:LogBound (ek+9*l) (rnd(vh*r)) (v*r)) by (eapply LogBound_mono; [|exact Hr']; lra).
 pose proof (Rmax_r et (ek+9*l)) as Hmax.
 destruct heating; unfold outer_core,exact_outer,outer_beta.
 - assert (Nb:RoundNodes rnd l (balance_nodes rnd 0 (rnd(vh*bh)) (rnd(vh*r)))).
   { intros z Hz; apply N; simpl; right; right; exact Hz. }
   pose proof (balance_zero_graph_budget _ _ _ _ _ _ _ Hbb Hrr Nb) as H.
   eapply LogBound_mono; [|exact H]; lra.
 - assert (Nb:RoundNodes rnd l (balance_nodes rnd 0 (rnd(vh*r)) (rnd(vh*bh)))).
   { intros z Hz; apply N; simpl; right; right; exact Hz. }
   pose proof (balance_zero_graph_budget _ _ _ _ _ _ _ Hrr Hbb Nb) as H.
   eapply LogBound_mono; [|exact H]; lra.
Qed.
Theorem planck_outer_zero_graph_budget rnd l et ek kappa khat heating C chi a x r :
 0<=l -> 0<C -> 0<chi -> 0<a -> 0<x -> 0<r ->
 PlanckValueContract kappa khat ek ->
 RoundNodes rnd l (outer_nodes rnd heating 0 C (khat x) chi a x r) ->
 LogBound (outer_beta et ek l) (eval_outer rnd heating 0 C (khat x) chi a x r)
 (exact_outer heating 0 (chi*saturate(C*kappa x)) (a*x^4) r).
Proof.
 intros Hl HC Hc Ha Hx Hr HK N; unfold outer_nodes in N.
 apply RoundNodes_app in N as [NB N]; apply RoundNodes_app in N as [NV NO].
 pose proof (B_graph_budget rnd l a x Hl Ha Hx NB) as HB.
 pose proof (V_graph_budget rnd l ek C (khat x) (kappa x) chi Hl HC Hc (HK x Hx) NV) as HV.
 rewrite exact_B_quartic in HB; unfold eval_outer.
 now apply outer_core_zero_graph_budget.
Qed.
Theorem planck_outer_zero_RN64_budget choice et ek kappa khat heating C chi a x r :
 0<C -> 0<chi -> 0<a -> 0<x -> 0<r -> PlanckValueContract kappa khat ek ->
 Forall (fun z => 0<z /\ normal64 z)
 (outer_nodes (RN64 choice) heating 0 C (khat x) chi a x r) ->
 LogBound (outer_beta et ek lambda64)
 (eval_outer (RN64 choice) heating 0 C (khat x) chi a x r)
 (exact_outer heating 0 (chi*saturate(C*kappa x)) (a*x^4) r).
Proof.
 intros HC Hc Ha Hx Hr HK N; apply planck_outer_zero_graph_budget; auto.
 - unfold lambda64; apply log_unit_nonneg; pose proof u64_range; lra.
 - now apply RN64_nodes.
Qed.
Theorem planck_outer_zero_finite64_budget choice et ek kappa khat heating C chi a x r :
 0<C -> 0<chi -> 0<a -> 0<x -> 0<r -> PlanckValueContract kappa khat ek ->
 FiniteNormalNodes choice (outer_nodes (RN64 choice) heating 0 C (khat x) chi a x r) ->
 LogBound (outer_beta et ek lambda64)
 (eval_outer (RN64 choice) heating 0 C (khat x) chi a x r)
 (exact_outer heating 0 (chi*saturate(C*kappa x)) (a*x^4) r).
Proof.
 intros HC Hc Ha Hx Hr HK N; apply planck_outer_zero_graph_budget; auto.
 - unfold lambda64; apply log_unit_nonneg; pose proof u64_range; lra.
 - apply RN64_rounded_nodes; now apply finite_nodes_rounded_normal.
Qed.
Print Assumptions planck_outer_zero_finite64_budget.
