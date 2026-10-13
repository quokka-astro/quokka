(* Exact representation of stored boundary and guard constants. *)
From Coq Require Import Reals Psatz Field ZArith List.
Require Import Flocq.Core.Core.
From BlackBox Require Import FloatingPoint Binary64Adapter Guards GuardFloatBridge.
Open Scope R_scope.
Import ListNotations.

Lemma stored_half_exact64 choice T :
 format64 T -> normal64 (T/2) -> finite64 (T/2) -> RN64 choice (T/2)=T/2.
Proof.
 intros HT Hn Hf.
 replace (T/2) with (T*bpow radix2 (-1)) in * by reflexivity.
 exact (proj1 (power2_scaling_exact64 choice T (-1) HT Hn Hf)).
Qed.
Lemma stored_half_format64 T :
 format64 T -> normal64 (T/2) -> format64 (T/2).
Proof.
 intros HT Hn; unfold format64 in *.
 apply generic_format_FLT_FLX; [exact Hn|].
 replace (T/2) with (T*bpow radix2 (-1)) by reflexivity.
 apply generic_format_FLX_shift64.
 now apply (generic_format_FLX_FLT radix2 (-1074) 53).
Qed.

(* All guard offsets are even multiples of u. At exponent -52 the exact
   significand is 2^52+k, for offset 2*k*u, including offsets below one. *)
Lemma stored_even_guard_format64 k : (-64<=k<=64)%Z ->
 format64 (1+2*IZR k*binary64_u).
Proof.
 intro Hk; unfold format64; apply generic_format_FLT.
 apply (FLT_spec _ _ _ _ (Float radix2 (4503599627370496+k) (-52))).
 - change (1+2*IZR k*binary64_u=IZR(4503599627370496+k)*bpow radix2 (-52)).
   rewrite plus_IZR.
   unfold binary64_u; change (1+2*IZR k*/9007199254740992=(4503599627370496+IZR k)*/4503599627370496).
   field.
 - change (Z.abs (4503599627370496+k)<9007199254740992)%Z; lia.
 - simpl; lia.
Qed.
Lemma stored_even_guard_normal_finite k : (-64<=k<=64)%Z ->
 0<1+2*IZR k*binary64_u /\ normal64 (1+2*IZR k*binary64_u) /\ finite64 (1+2*IZR k*binary64_u).
Proof.
 intro Hk; assert (HK: -64<=IZR k<=64).
 { split; apply IZR_le; lia. }
 pose proof binary64_u_bounds as Hu.
 assert (Hv:/2<1+2*IZR k*binary64_u<2) by nra.
 split; [lra|]; split.
 - unfold normal64; rewrite Rabs_pos_eq by lra.
   assert (bpow radix2 (-1022)<= /2).
   { change (bpow radix2 (-1022)<=bpow radix2 (-1)); apply bpow_le; lia. }
   lra.
 - unfold finite64; rewrite Rabs_pos_eq by lra.
   assert (2<bpow radix2 1024).
   { change (bpow radix2 1<bpow radix2 1024); apply bpow_lt; lia. }
   lra.
Qed.
Lemma stored_format64_exact choice c : format64 c -> RN64 choice c=c.
Proof. intro H; unfold RN64; apply round_generic; [typeclasses eauto|exact H]. Qed.
Lemma stored_even_guard_exact64 choice k : (-64<=k<=64)%Z ->
 RN64 choice (1+2*IZR k*binary64_u)=1+2*IZR k*binary64_u.
Proof. intro Hk; apply stored_format64_exact; now apply stored_even_guard_format64. Qed.

Lemma inner_lower_format64 : format64 (1-16*binary64_u).
Proof. replace (1-16*binary64_u) with (1+2*IZR (-8)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma inner_lower_exact64 choice : RN64 choice (1-16*binary64_u)=1-16*binary64_u.
Proof. apply stored_format64_exact, inner_lower_format64. Qed.

Lemma inner_upper_format64 : format64 (1+16*binary64_u).
Proof. replace (1+16*binary64_u) with (1+2*IZR (8)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma inner_upper_exact64 choice : RN64 choice (1+16*binary64_u)=1+16*binary64_u.
Proof. apply stored_format64_exact, inner_upper_format64. Qed.

Lemma initial_lower_format64 : format64 (1-64*binary64_u).
Proof. replace (1-64*binary64_u) with (1+2*IZR (-32)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma initial_lower_exact64 choice : RN64 choice (1-64*binary64_u)=1-64*binary64_u.
Proof. apply stored_format64_exact, initial_lower_format64. Qed.

Lemma initial_upper_format64 : format64 (1+64*binary64_u).
Proof. replace (1+64*binary64_u) with (1+2*IZR (32)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma initial_upper_exact64 choice : RN64 choice (1+64*binary64_u)=1+64*binary64_u.
Proof. apply stored_format64_exact, initial_upper_format64. Qed.

Lemma outer_lower_format64 : format64 (1-128*binary64_u).
Proof. replace (1-128*binary64_u) with (1+2*IZR (-64)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma outer_lower_exact64 choice : RN64 choice (1-128*binary64_u)=1-128*binary64_u.
Proof. apply stored_format64_exact, outer_lower_format64. Qed.

Lemma outer_upper_format64 : format64 (1+128*binary64_u).
Proof. replace (1+128*binary64_u) with (1+2*IZR (64)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma outer_upper_exact64 choice : RN64 choice (1+128*binary64_u)=1+128*binary64_u.
Proof. apply stored_format64_exact, outer_upper_format64. Qed.

Lemma outward_lower_format64 : format64 (1-8*binary64_u).
Proof. replace (1-8*binary64_u) with (1+2*IZR (-4)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma outward_lower_exact64 choice : RN64 choice (1-8*binary64_u)=1-8*binary64_u.
Proof. apply stored_format64_exact, outward_lower_format64. Qed.

Lemma outward_upper_format64 : format64 (1+8*binary64_u).
Proof. replace (1+8*binary64_u) with (1+2*IZR (4)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma outward_upper_exact64 choice : RN64 choice (1+8*binary64_u)=1+8*binary64_u.
Proof. apply stored_format64_exact, outward_upper_format64. Qed.

Lemma lower_endpoint_factor_format64 : format64 (1-32*binary64_u).
Proof. replace (1-32*binary64_u) with (1+2*IZR (-16)*binary64_u) by (simpl; ring); apply stored_even_guard_format64; lia. Qed.
Lemma lower_endpoint_factor_exact64 choice : RN64 choice (1-32*binary64_u)=1-32*binary64_u.
Proof. apply stored_format64_exact, lower_endpoint_factor_format64. Qed.

Definition stored_guard_constants64 := [1-16*binary64_u;1+16*binary64_u;1-64*binary64_u;1+64*binary64_u;1-128*binary64_u;1+128*binary64_u;1-8*binary64_u;1+8*binary64_u;1-32*binary64_u].
Lemma stored_guard_constants_format64 : Forall format64 stored_guard_constants64.
Proof. unfold stored_guard_constants64; repeat constructor; auto using inner_lower_format64, inner_upper_format64, initial_lower_format64, initial_upper_format64, outer_lower_format64, outer_upper_format64, outward_lower_format64, outward_upper_format64, lower_endpoint_factor_format64. Qed.
Lemma stored_guard_constants_exact64 choice : Forall (fun c => RN64 choice c=c) stored_guard_constants64.
Proof. apply Forall_forall; intros c Hc; apply stored_format64_exact; apply (proj1 (Forall_forall _ _) stored_guard_constants_format64 c Hc). Qed.

Print Assumptions stored_half_exact64.
Print Assumptions stored_guard_constants_exact64.
