(* A physical-variable-opacity scalar can have at least three roots although
   the common opacity's logarithmic slope is exactly -2, inside the old gray
   epsilon=1 band. Exact Planck integral bounds are explicit analytic inputs. *)
From Coq Require Import Reals Ranalysis5 Psatz Field.
From Coquelicot Require Import Coquelicot.
From BlackBox Require Import GasMap.
Open Scope R_scope.

Definition counter_gas x := gas_map 1 (1/100) 100 x.
Definition counter_F (B:R->R) x := counter_gas x-100+100*B x/(1+x*x).
Definition counter_kappa x := / (x*x).

Lemma inverse_forward_lower A D T x t :
 0<A -> 0<D -> 0<T -> 0<x -> 0<t -> forward_map A D T t<x ->
 t<gas_map A D T x.
Proof.
 intros HA HD HT HX Ht HF.
 destruct (gas_map_spec A D T x HA HD HT HX) as [HG Hmap].
 destruct (Rtotal_order t (gas_map A D T x)) as [HL|[HE|HU]]; [exact HL| |].
 - rewrite HE in HF; lra.
 - pose proof (forward_increasing A D T (gas_map A D T x) t HA HD HT HG HU); lra.
Qed.
Lemma inverse_forward_upper A D T x t :
 0<A -> 0<D -> 0<T -> 0<x -> 0<t -> x<forward_map A D T t ->
 gas_map A D T x<t.
Proof.
 intros HA HD HT HX Ht HF.
 destruct (gas_map_spec A D T x HA HD HT HX) as [HG Hmap].
 destruct (Rtotal_order (gas_map A D T x) t) as [HL|[HE|HU]]; [exact HL| |].
 - rewrite <-HE in HF; lra.
 - pose proof (forward_increasing A D T t (gas_map A D T x) HA HD HT Ht HU); lra.
Qed.

Lemma counter_gas_one_lower : 91<counter_gas 1.
Proof.
 unfold counter_gas; apply inverse_forward_lower; try lra.
 pose proof (sqrt_def 91 ltac:(lra)) as HS.
 pose proof (sqrt_lt_R0 91 ltac:(lra)) as HP.
 assert (HU:sqrt 91<10) by nra.
 unfold forward_map.
 apply (Rmult_lt_reg_r (sqrt 91)); [exact HP|].
 field_simplify; nra.
Qed.
Lemma counter_gas_five_upper : counter_gas 5<92.
Proof.
 unfold counter_gas; apply inverse_forward_upper; try lra.
 pose proof (sqrt_def 92 ltac:(lra)) as HS.
 pose proof (sqrt_lt_R0 92 ltac:(lra)) as HP.
 assert (HL:800<87*sqrt 92) by nra.
 unfold forward_map.
 apply (Rmult_lt_reg_r (sqrt 92)); [exact HP|].
 field_simplify; nra.
Qed.

Lemma counter_F_signs B :
 0<=B (1/100) -> B (1/100)<=1/100 -> 4/21<=B 1 ->
 0<=B 5 -> B 5<=5/3 -> 0<B 100 ->
 counter_F B (1/100)<0 /\ 0<counter_F B 1 /\
 counter_F B 5<0 /\ 0<counter_F B 100.
Proof.
 intros HB0 HB0u HB1 HB5 HB5u HB100.
 pose proof counter_gas_one_lower as HG1.
 pose proof counter_gas_five_upper as HG5.
 pose proof (gas_map_increasing 1 (1/100) 100 (1/100) 5
  ltac:(lra) ltac:(lra) ltac:(lra) ltac:(lra) ltac:(lra)) as HG0.
 change (counter_gas (1/100)<counter_gas 5) in HG0.
 assert (HG100:counter_gas 100=100).
 { unfold counter_gas; apply gas_map_equilibrium; lra. }
 unfold counter_F; rewrite HG100.
 repeat split; field_simplify; nra.
Qed.

Lemma counter_F_continuous B x : 0<x -> continuity_pt B x -> continuity_pt (counter_F B) x.
Proof.
 intros HX HB; unfold counter_F.
 apply continuity_pt_plus.
 - apply continuity_pt_minus.
   + unfold counter_gas; apply gas_map_continuous; lra.
   + apply continuity_pt_const; unfold constant; reflexivity.
 - apply continuity_pt_div.
   + apply continuity_pt_mult; [apply continuity_pt_const; unfold constant; reflexivity|exact HB].
   + apply continuity_pt_plus; [apply continuity_pt_const; unfold constant; reflexivity|].
     apply continuity_pt_mult; apply continuity_pt_id.
   + nra.
Qed.

Lemma strict_sign_change_root F lo hi :
 lo<hi -> F lo<0 -> 0<F hi ->
 (forall x, lo<=x<=hi -> continuity_pt F x) ->
 exists x, lo<x<hi /\ F x=0.
Proof.
 intros Hlh HL HH HC.
 destruct (f_interv_is_interv F lo hi 0 Hlh) as [x [[Hxl Hxh] HF]].
 - split; lra.
 - intros x Hx; apply HC; exact Hx.
 - exists x; split; [split; destruct (Req_dec x lo); destruct (Req_dec x hi); subst; try lra|exact HF].
Qed.

Theorem variable_opacity_three_positive_roots B :
 (forall x, 0<x -> continuity_pt B x) ->
 0<=B (1/100) -> B (1/100)<=1/100 -> 4/21<=B 1 ->
 0<=B 5 -> B 5<=5/3 -> 0<B 100 ->
 exists x1 x2 x3,
 1/100<x1<1 /\ 1<x2<5 /\ 5<x3<100 /\
 counter_F B x1=0 /\ counter_F B x2=0 /\ counter_F B x3=0.
Proof.
 intros HC HB0 HB0u HB1 HB5 HB5u HB100.
 destruct (counter_F_signs B HB0 HB0u HB1 HB5 HB5u HB100) as [H0 [H1 [H5 H100]]].
 destruct (strict_sign_change_root (counter_F B) (1/100) 1 ltac:(lra) H0 H1)
  as [x1 [HX1 HF1]].
 - intros x Hx; apply counter_F_continuous; [lra|apply HC; lra].
 - destruct (strict_sign_change_root (fun x=> -counter_F B x) 1 5
    ltac:(lra) ltac:(lra) ltac:(lra)) as [x2 [HX2 HF2]].
   + intros x Hx; apply continuity_pt_opp; apply counter_F_continuous; [lra|apply HC; lra].
   + destruct (strict_sign_change_root (counter_F B) 5 100 ltac:(lra) H5 H100)
      as [x3 [HX3 HF3]].
     * intros x Hx; apply counter_F_continuous; [lra|apply HC; lra].
     * exists x1,x2,x3; repeat split; try lra.
Qed.

Lemma counter_kappa_positive x : 0<x -> 0<counter_kappa x.
Proof. intros HX; unfold counter_kappa; apply Rinv_0_lt_compat; nra. Qed.
Lemma counter_kappa_is_derive x : 0<x -> is_derive counter_kappa x (-2/(x*x*x)).
Proof.
 intro HX; unfold counter_kappa; auto_derive; [nra|field; lra].
Qed.
Lemma counter_kappa_log_slope x : 0<x -> x*(-2/(x*x*x))/counter_kappa x = -2.
Proof. intro HX; unfold counter_kappa; field; lra. Qed.
Lemma counter_old_gray_band : -4+1 < -2 < 1-1.
Proof. lra. Qed.

Print Assumptions variable_opacity_three_positive_roots.

Definition counter_energy (B:R->R) x := B x/(1+x*x).
Definition counter_physical (B:R->R) x t E :=
 0<x /\ 0<t /\ 0<=E /\
 E=counter_kappa x*(B x-E) /\
 100*E=(1/100)*sqrt t*(t-x) /\
 t-100= -(1/100)*sqrt t*(t-x).

Theorem counter_root_reconstructs_physical B x :
 0<x -> 0<=B x -> counter_F B x=0 ->
 counter_physical B x (counter_gas x) (counter_energy B x).
Proof.
 intros HX HB HF.
 pose proof (proj1 (gas_map_spec 1 (1/100) 100 x
  ltac:(lra) ltac:(lra) ltac:(lra) HX)) as HGpos.
 pose proof (gas_map_collision 1 (1/100) 100 x
  ltac:(lra) ltac:(lra) ltac:(lra) HX) as HG.
 change (0<counter_gas x) in HGpos.
 change (collision 1 (1/100) 100 x (counter_gas x)=0) in HG.
 unfold collision in HG.
 unfold counter_physical; split; [exact HX|]; split; [exact HGpos|]; split.
 - unfold counter_energy; apply Rdiv_le_0_compat; nra.
 - split.
   + unfold counter_energy,counter_kappa; field; nra.
   + unfold counter_F in HF; unfold counter_energy.
     replace (100*(B x/(1+x*x))) with (100*B x/(1+x*x)) by (unfold Rdiv; ring).
     split; lra.
Qed.

Theorem variable_opacity_three_physical_solutions B :
 (forall x, 0<x -> continuity_pt B x) ->
 (forall x, 0<x -> 0<=B x) ->
 B (1/100)<=1/100 -> 4/21<=B 1 -> B 5<=5/3 -> 0<B 100 ->
 exists x1 x2 x3,
 1/100<x1<1 /\ 1<x2<5 /\ 5<x3<100 /\
 counter_physical B x1 (counter_gas x1) (counter_energy B x1) /\
 counter_physical B x2 (counter_gas x2) (counter_energy B x2) /\
 counter_physical B x3 (counter_gas x3) (counter_energy B x3).
Proof.
 intros HC HB H0 H1 H5 H100.
 destruct (variable_opacity_three_positive_roots B HC (HB (1/100) ltac:(lra)) H0 H1
  (HB 5 ltac:(lra)) H5 H100) as [x1 [x2 [x3 [HX1 [HX2 [HX3 [HF1 [HF2 HF3]]]]]]]].
 exists x1,x2,x3; split; [exact HX1|]; split; [exact HX2|]; split; [exact HX3|].
 split; [apply counter_root_reconstructs_physical; try assumption; try lra; apply HB; lra|].
 split; apply counter_root_reconstructs_physical; try assumption; try lra; apply HB; lra.
Qed.

Print Assumptions variable_opacity_three_physical_solutions.
