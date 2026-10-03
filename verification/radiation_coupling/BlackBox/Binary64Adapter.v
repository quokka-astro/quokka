(* IEEE754 finite-operation correspondence, including the upper exponent. *)
From Coq Require Import Reals Psatz Field ZArith Bool.
Require Import Flocq.Core.Core Flocq.Prop.Relative Flocq.IEEE754.BinarySingleNaN.
From BlackBox Require Import FloatingPoint.
Open Scope R_scope.
Local Instance ieee64_prec : Prec_gt_0 53. Proof. unfold Prec_gt_0; lia. Qed.
Local Instance ieee64_emax : Prec_lt_emax 53 1024. Proof. unfold Prec_lt_emax; lia. Qed.
Definition IEEE64 := binary_float 53 1024.
Definition nearest_even_choice (z:Z) := negb(Z.even z).
Definition RNE64 := RN64 nearest_even_choice.

Lemma IEEE64_plus_correspondence (x y:IEEE64) :
 is_finite x=true -> is_finite y=true ->
 finite64 (RNE64 (B2R x+B2R y)) ->
 B2R (Bplus mode_NE x y)=RNE64 (B2R x+B2R y) /\ is_finite (Bplus mode_NE x y)=true.
Proof.
 intros Fx Fy Hf.
 pose proof (Bplus_correct 53 1024 ieee64_prec ieee64_emax mode_NE x y Fx Fy) as H.
 rewrite Rlt_bool_true in H by exact Hf.
 split; [exact (proj1 H)|exact (proj1(proj2 H))].
Qed.
Lemma IEEE64_mult_correspondence (x y:IEEE64) :
 is_finite x=true -> is_finite y=true ->
 finite64 (RNE64 (B2R x*B2R y)) ->
 B2R (Bmult mode_NE x y)=RNE64 (B2R x*B2R y) /\ is_finite (Bmult mode_NE x y)=true.
Proof.
 intros Fx Fy Hf.
 pose proof (Bmult_correct 53 1024 ieee64_prec ieee64_emax mode_NE x y) as H.
 rewrite Rlt_bool_true in H by exact Hf.
 split; [exact (proj1 H)|].
 rewrite (proj1(proj2 H)),Fx,Fy; reflexivity.
Qed.
Lemma IEEE64_div_correspondence (x y:IEEE64) :
 is_finite x=true -> B2R y<>0 ->
 finite64 (RNE64 (B2R x/B2R y)) ->
 B2R (Bdiv mode_NE x y)=RNE64 (B2R x/B2R y) /\ is_finite (Bdiv mode_NE x y)=true.
Proof.
 intros Fx Ny Hf.
 pose proof (Bdiv_correct 53 1024 ieee64_prec ieee64_emax mode_NE x y Ny) as H.
 rewrite Rlt_bool_true in H by exact Hf.
 split; [exact (proj1 H)|].
 rewrite (proj1(proj2 H)); exact Fx.
Qed.
Lemma IEEE64_sqrt_correspondence (x:IEEE64) :
 B2R (Bsqrt mode_NE x)=RNE64 (sqrt(B2R x)).
Proof. exact (proj1 (Bsqrt_correct 53 1024 ieee64_prec ieee64_emax mode_NE x)). Qed.

Lemma IEEE64_plus_log (x y:IEEE64) :
 is_finite x=true -> is_finite y=true ->
 0<RNE64 (B2R x+B2R y) -> normal64 (RNE64 (B2R x+B2R y)) ->
 finite64 (RNE64 (B2R x+B2R y)) ->
 LogBound lambda64 (B2R(Bplus mode_NE x y)) (B2R x+B2R y).
Proof.
 intros Fx Fy Hp Hn Hf; rewrite (proj1 (IEEE64_plus_correspondence x y Fx Fy Hf)).
 apply RN64_log_rounded_normal; assumption.
Qed.
Lemma IEEE64_mult_log (x y:IEEE64) :
 is_finite x=true -> is_finite y=true ->
 0<RNE64 (B2R x*B2R y) -> normal64 (RNE64 (B2R x*B2R y)) ->
 finite64 (RNE64 (B2R x*B2R y)) ->
 LogBound lambda64 (B2R(Bmult mode_NE x y)) (B2R x*B2R y).
Proof.
 intros Fx Fy Hp Hn Hf; rewrite (proj1 (IEEE64_mult_correspondence x y Fx Fy Hf)).
 apply RN64_log_rounded_normal; assumption.
Qed.
Lemma IEEE64_div_log (x y:IEEE64) :
 is_finite x=true -> B2R y<>0 ->
 0<RNE64 (B2R x/B2R y) -> normal64 (RNE64 (B2R x/B2R y)) ->
 finite64 (RNE64 (B2R x/B2R y)) ->
 LogBound lambda64 (B2R(Bdiv mode_NE x y)) (B2R x/B2R y).
Proof.
 intros Fx Ny Hp Hn Hf; rewrite (proj1 (IEEE64_div_correspondence x y Fx Ny Hf)).
 apply RN64_log_rounded_normal; assumption.
Qed.
Lemma IEEE64_sqrt_log (x:IEEE64) :
 0<RNE64 (sqrt(B2R x)) -> normal64 (RNE64 (sqrt(B2R x))) ->
 LogBound lambda64 (B2R(Bsqrt mode_NE x)) (sqrt(B2R x)).
Proof. intros; rewrite IEEE64_sqrt_correspondence; apply RN64_log_rounded_normal; assumption. Qed.

Print Assumptions IEEE64_plus_correspondence.
Print Assumptions IEEE64_mult_correspondence.
Print Assumptions IEEE64_div_correspondence.
Print Assumptions IEEE64_sqrt_correspondence.

Lemma generic_format_FLX_shift64 x e :
 generic_format radix2 (FLX_exp 53) x ->
 generic_format radix2 (FLX_exp 53) (x*bpow radix2 e).
Proof.
 intro H; destruct (FLX_format_generic radix2 53 x H) as [[m ex] Hx Hm].
 apply generic_format_FLX; apply (FLX_spec _ _ _ (Float radix2 m (ex+e))).
 - rewrite Hx; unfold F2R; simpl; rewrite bpow_plus; ring.
 - exact Hm.
Qed.
Lemma power2_scaling_exact64 choice x e :
 generic_format radix2 (FLT_exp (-1074) 53) x ->
 normal64 (x*bpow radix2 e) -> finite64 (x*bpow radix2 e) ->
 RN64 choice (x*bpow radix2 e)=x*bpow radix2 e /\ finite64 (RN64 choice (x*bpow radix2 e)).
Proof.
 intros Hx Hn Hf.
 assert (Hfmt:generic_format radix2 (FLT_exp (-1074) 53) (x*bpow radix2 e)).
 { apply generic_format_FLT_FLX; [exact Hn|].
   apply generic_format_FLX_shift64.
   now apply (generic_format_FLX_FLT radix2 (-1074) 53). }
 assert (He:RN64 choice (x*bpow radix2 e)=x*bpow radix2 e) by (unfold RN64; apply round_generic; [typeclasses eauto|exact Hfmt]).
 split; auto; now rewrite He.
Qed.
Lemma power2_scaling_preserves_log_bound e h x k :
 LogBound e h x -> LogBound e (h*bpow radix2 k) (x*bpow radix2 k).
Proof.
 intro H; pose proof (LogBound_exact (bpow radix2 k) (bpow_gt_0 radix2 k)) as Hs.
 pose proof (LogBound_mult _ _ _ _ _ _ H Hs) as Hm.
 eapply LogBound_mono; [|exact Hm]; lra.
Qed.
Print Assumptions power2_scaling_exact64.
