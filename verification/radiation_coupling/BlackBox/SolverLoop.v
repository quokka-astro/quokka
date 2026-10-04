(* A finite, executable safeguarded scalar control loop over indexed values.
   The data functions are abstract; their arithmetic and grid contracts below
   are explicit. This is not a refinement theorem for production C++ code. *)
From Coq Require Import Reals Psatz Lia Arith Bool Ranalysis5.
From BlackBox Require Import Guards Brackets FloatingPoint GuardFloatBridge.
Open Scope R_scope.

Inductive scalar_result : Type :=
| ResidualAccepted (lower upper trial:nat)
| BracketAccepted (lower upper trial:nat)
| BudgetExhausted (lower upper:nat).

Definition residual_window (h:R) : bool :=
  if Rle_dec (1-128*binary64_u) h then
    if Rle_dec h (1+128*binary64_u) then true else false
  else false.
Definition width_window (w:R) : bool :=
  if Rle_dec w (16*binary64_u) then true else false.
Definition lower_side (rising:bool) (h:R) : bool :=
  if rising then if Rlt_dec h 1 then true else false
  else if Rlt_dec 1 h then true else false.
Definition oriented_balance (rising:bool) (balance:R->R) (x:R) : R :=
  if rising then balance x-1 else 1-balance x.

Fixpoint scalar_solve (fuel:nat) (rising:bool)
  (measured:nat->R) (width:nat->nat->R) (proposal:nat->nat->nat)
  (lower upper:nat) : scalar_result :=
  match fuel with
  | O => BudgetExhausted lower upper
  | S fuel' =>
    if width_window (width lower upper) then BracketAccepted lower upper lower
    else
      let trial:=safeguarded_index lower upper (proposal lower upper) in
      if residual_window (measured trial) then ResidualAccepted lower upper trial
      else if lower_side rising (measured trial) then
        scalar_solve fuel' rising measured width proposal trial upper
      else scalar_solve fuel' rising measured width proposal lower trial
  end.

Lemma residual_window_spec h :
  residual_window h=true <-> 1-128*binary64_u<=h<=1+128*binary64_u.
Proof.
 unfold residual_window; destruct Rle_dec; [destruct Rle_dec|]; split; intro H;
 try discriminate; try reflexivity; lra.
Qed.
Lemma width_window_spec w : width_window w=true <-> w<=16*binary64_u.
Proof. unfold width_window; destruct Rle_dec; split; intro H; auto; discriminate. Qed.
Lemma residual_window_reject h : residual_window h=false ->
 h<1-128*binary64_u \/ 1+128*binary64_u<h.
Proof.
 unfold residual_window; destruct Rle_dec; [destruct Rle_dec|]; intro H;
 try discriminate; [right|left]; lra.
Qed.

Lemma guarded_lower_side rising h s :
  0<h -> 0<s -> Rabs(ln h-ln s)<=71*binary64_lambda ->
  residual_window h=false -> lower_side rising h=true ->
  (if rising then s-1 else 1-s)<0.
Proof.
 intros Hh Hs Herr Hreject Hside.
 pose proof binary64_u_bounds as Hu.
 apply residual_window_reject in Hreject.
 destruct rising; unfold lower_side in Hside;
 destruct Rlt_dec; try discriminate; simpl;
 destruct Hreject as [Hlo|Hhi].
 - pose proof (outer_lower_sign h s Hh Hs Hlo Herr); lra.
 - exfalso; nra.
 - exfalso; nra.
 - pose proof (outer_upper_sign h s Hh Hs Hhi Herr); lra.
Qed.
Lemma guarded_upper_side rising h s :
  0<h -> 0<s -> Rabs(ln h-ln s)<=71*binary64_lambda ->
  residual_window h=false -> lower_side rising h=false ->
  0<(if rising then s-1 else 1-s).
Proof.
 intros Hh Hs Herr Hreject Hside.
 pose proof binary64_u_bounds as Hu.
 apply residual_window_reject in Hreject.
 destruct rising; unfold lower_side in Hside;
 destruct Rlt_dec; try discriminate; simpl;
 destruct Hreject as [Hlo|Hhi].
 - exfalso; nra.
 - pose proof (outer_upper_sign h s Hh Hs Hhi Herr); lra.
 - pose proof (outer_lower_sign h s Hh Hs Hlo Herr); lra.
 - exfalso; nra.
Qed.

Section VerifiedLoop.
Variables (rising:bool) (value:nat->R) (balance:R->R)
  (measured:nat->R) (width:nat->nat->R) (proposal:nat->nat->nat).
Variables (initial_lower initial_upper:nat) (root:R).

(* A finite normal binary64 grid. Completeness of adjacent indices is explicit;
   the theorem does not claim a particular C++ bit/rank encoding is verified. *)
Hypothesis grid_positive : forall i,
  (initial_lower<=i<=initial_upper)%nat -> 0<value i.
Hypothesis grid_increasing : forall i j,
  (initial_lower<=i)%nat -> (i<j)%nat -> (j<=initial_upper)%nat -> value i<value j.
Hypothesis grid_format : forall i,
  (initial_lower<=i<=initial_upper)%nat -> format64 (value i).
Hypothesis grid_normal : forall i,
  (initial_lower<=i<=initial_upper)%nat -> normal64 (value i).
Hypothesis grid_adjacent_complete : forall i,
  (initial_lower<=i)%nat -> (S i<=initial_upper)%nat ->
  forall x, format64 x -> ~(value i<x<value (S i)).

(* Each width call is exactly subtraction then division, with bounded relative
   errors. GuardFloatBridge.RN64_relative_factor discharges these hypotheses
   for normal exact operation results; exact subtraction permits zero error. *)
Hypothesis width_arithmetic : forall a b,
  (initial_lower<=a)%nat -> (a<b)%nat -> (b<=initial_upper)%nat ->
  exists es ed, Rabs es<=binary64_u /\ Rabs ed<=binary64_u /\
    width a b=((value b-value a)*(1+es)/value a)*(1+ed).
Hypothesis evaluator : forall i,
  (initial_lower<=i<=initial_upper)%nat ->
  0<measured i /\ 0<balance (value i) /\
  Rabs(ln(measured i)-ln(balance(value i)))<=71*binary64_lambda.
Hypothesis physical_monotonicity :
  nondecreasing_on (oriented_balance rising balance)
    (value initial_lower) (value initial_upper).
Hypothesis exact_root : balance root=1.

Definition loop_state (a b:nat) : Prop :=
  (initial_lower<=a /\ a<b /\ b<=initial_upper)%nat /\
  root_bracket (value a) (value b) root.

Lemma grid_le i j :
  (initial_lower<=i)%nat -> (i<=j)%nat -> (j<=initial_upper)%nat -> value i<=value j.
Proof. intros Hi Hij Hj; apply Nat.lt_eq_cases in Hij; destruct Hij as [H|H];
 [left; apply grid_increasing; assumption|subst; reflexivity]. Qed.

Lemma local_monotonicity a b : loop_state a b ->
  nondecreasing_on (oriented_balance rising balance) (value a) (value b).
Proof.
 intros [[Ha [Hab Hb]] Hr] x y Hx Hxy Hy.
 assert (Hla:value initial_lower<=value a) by (apply grid_le; lia).
 assert (Hbu:value b<=value initial_upper) by (apply grid_le; lia).
 apply physical_monotonicity; lra.
Qed.

Lemma state_width_certificate a b : loop_state a b -> width_window(width a b)=true ->
  Rabs(ln(value a/root))<32*binary64_lambda.
Proof.
 intros Hstate Hwidth; destruct Hstate as [[Ha [Hab Hb]] Hr].
 apply width_window_spec in Hwidth.
 destruct (width_arithmetic a b Ha Hab Hb) as [es [ed [Hes [Hed Hw]]]].
 apply (rounded_width_acceptance (value a) (value b) root (value a) es ed (width a b));
 try assumption.
 - apply grid_positive; lia.
 - split; [reflexivity|left; apply grid_increasing; assumption].
Qed.

Lemma rejected_width_has_interior a b : loop_state a b -> width_window(width a b)=false ->
  (a+1<b)%nat.
Proof.
 intros [[Ha [Hab Hb]] Hr] Hreject.
 destruct (Nat.lt_ge_cases (a+1) b) as [Hgap|Hgap]; [exact Hgap|].
 assert (Hbnext:b=S a) by lia; subst b.
 destruct (width_arithmetic a (S a) Ha ltac:(lia) Hb) as [es [ed [Hes [Hed Hw]]]].
 assert (Hwpass:width a (S a)<=16*binary64_u).
 { apply (adjacent_normal_bracket_passes (value a) (value (S a)) es ed (width a (S a)));
   try assumption.
   - apply grid_positive; lia.
   - apply grid_normal; lia.
   - apply grid_format; lia.
   - apply grid_format; lia.
   - apply grid_increasing; lia.
   - intros; eapply grid_adjacent_complete; eauto. }
 apply width_window_spec in Hwpass; congruence.
Qed.

Lemma trial_inside a b : loop_state a b -> width_window(width a b)=false ->
  (a<safeguarded_index a b (proposal a b)<b)%nat.
Proof. intros Hs Hw; apply safeguarded_index_inside; eapply rejected_width_has_interior; eauto. Qed.

Lemma lower_step_state a b i : loop_state a b -> (a<i<b)%nat ->
  residual_window(measured i)=false -> lower_side rising (measured i)=true ->
  loop_state i b.
Proof.
 intros Hstate Hi Hreject Hside.
 pose proof (local_monotonicity a b Hstate) as Hmono.
 destruct Hstate as [[Ha [Hab Hb]] Hr].
 destruct (evaluator i ltac:(lia)) as [Hh [Hs He]].
 pose proof (guarded_lower_side rising (measured i) (balance(value i)) Hh Hs He Hreject Hside) as Hsign.
 split; [lia|].
 apply (lower_update_preserves_root (oriented_balance rising balance) (value a) (value b) root (value i));
 try assumption.
 - split; left; apply grid_increasing; lia.
 - unfold oriented_balance; destruct rising; rewrite exact_root; ring.
Qed.

Lemma upper_step_state a b i : loop_state a b -> (a<i<b)%nat ->
  residual_window(measured i)=false -> lower_side rising (measured i)=false ->
  loop_state a i.
Proof.
 intros Hstate Hi Hreject Hside.
 pose proof (local_monotonicity a b Hstate) as Hmono.
 destruct Hstate as [[Ha [Hab Hb]] Hr].
 destruct (evaluator i ltac:(lia)) as [Hh [Hs He]].
 pose proof (guarded_upper_side rising (measured i) (balance(value i)) Hh Hs He Hreject Hside) as Hsign.
 split; [lia|].
 apply (upper_update_preserves_root (oriented_balance rising balance) (value a) (value b) root (value i));
 try assumption.
 - split; left; apply grid_increasing; lia.
 - unfold oriented_balance; destruct rising; rewrite exact_root; ring.
Qed.

Definition result_certificate (result:scalar_result) : Prop :=
  match result with
  | ResidualAccepted a b i =>
      loop_state a b /\ (a<i<b)%nat /\
      Rabs(ln(balance(value i)))<=200*binary64_lambda
  | BracketAccepted a b i =>
      loop_state a b /\ i=a /\ Rabs(ln(value i/root))<32*binary64_lambda
  | BudgetExhausted a b => loop_state a b
  end.

Theorem scalar_solve_certificate fuel a b : loop_state a b ->
  result_certificate (scalar_solve fuel rising measured width proposal a b).
Proof.
 revert a b; induction fuel as [|fuel IH]; intros a b Hstate; simpl; [exact Hstate|].
 destruct (width_window(width a b)) eqn:Hw.
 - simpl; split; [exact Hstate|split; [reflexivity|eapply state_width_certificate; eauto]].
 - set (i:=safeguarded_index a b (proposal a b)).
   assert (Hi:(a<i<b)%nat) by (unfold i; apply trial_inside; assumption).
   destruct (residual_window(measured i)) eqn:Hres.
   + simpl; split; [exact Hstate|split; [exact Hi|]].
     apply residual_window_spec in Hres.
     destruct Hstate as [[Ha [Hab Hb]] Hr].
     destruct (evaluator i ltac:(lia)) as [Hh [Hs He]].
     apply (outer_exact_window (measured i) (balance(value i))); assumption.
   + destruct (lower_side rising (measured i)) eqn:Hside; apply IH.
     * apply (lower_step_state a b i); assumption.
     * apply (upper_step_state a b i); assumption.
Qed.

Definition successful (result:scalar_result) : Prop :=
  match result with BudgetExhausted _ _=>False | _=>True end.

Theorem scalar_solve_sufficient_fuel fuel a b :
  loop_state a b -> (b-a<=fuel)%nat ->
  successful (scalar_solve fuel rising measured width proposal a b).
Proof.
 revert a b; induction fuel as [|fuel IH]; intros a b Hstate Hfuel.
 - destruct Hstate as [[Ha [Hab Hb]] Hr]; lia.
 - simpl; destruct (width_window(width a b)) eqn:Hw; [exact I|].
   set (i:=safeguarded_index a b (proposal a b)).
   assert (Hi:(a<i<b)%nat) by (unfold i; apply trial_inside; assumption).
   destruct (residual_window(measured i)) eqn:Hres; [exact I|].
   destruct (lower_side rising (measured i)) eqn:Hside.
   + apply IH; [apply (lower_step_state a b i); assumption|lia].
   + apply IH; [apply (upper_step_state a b i); assumption|lia].
Qed.

Theorem scalar_solve_accepted_or_exhausted fuel a b : loop_state a b ->
  (exists l h i, scalar_solve fuel rising measured width proposal a b=ResidualAccepted l h i /\
       loop_state l h /\ (l<i<h)%nat /\ Rabs(ln(balance(value i)))<=200*binary64_lambda) \/
  (exists l h i, scalar_solve fuel rising measured width proposal a b=BracketAccepted l h i /\
       loop_state l h /\ i=l /\ Rabs(ln(value i/root))<32*binary64_lambda) \/
  (exists l h, scalar_solve fuel rising measured width proposal a b=BudgetExhausted l h /\ loop_state l h).
Proof.
 intro Hstate; pose proof (scalar_solve_certificate fuel a b Hstate) as Hcert.
 destruct (scalar_solve fuel rising measured width proposal a b) as [l h i|l h i|l h] eqn:Hresult;
 simpl in Hcert.
 - left; exists l,h,i; auto.
 - right; left; exists l,h,i; auto.
 - right; right; exists l,h; auto.
Qed.
End VerifiedLoop.

(* Initialization is by checked reliable endpoint signs. An implementation must
   retry/enlarge endpoints until these checks succeed or report failure. There
   is deliberately no unsupported claim that "a few" outward steps suffice. *)
Theorem checked_endpoint_root_exists rising balance measured_lower measured_upper a b :
  a<b -> (forall x, a<=x<=b -> continuity_pt (oriented_balance rising balance) x) ->
  0<measured_lower -> 0<measured_upper -> 0<balance a -> 0<balance b ->
  Rabs(ln measured_lower-ln(balance a))<=71*binary64_lambda ->
  Rabs(ln measured_upper-ln(balance b))<=71*binary64_lambda ->
  residual_window measured_lower=false -> lower_side rising measured_lower=true ->
  residual_window measured_upper=false -> lower_side rising measured_upper=false ->
  exists root, root_bracket a b root /\ balance root=1.
Proof.
 intros Hab Hcont Hla Hlb Hsa Hsb Hea Heb Hwa Hloa Hwb Hlob.
 pose proof (guarded_lower_side rising measured_lower (balance a) Hla Hsa Hea Hwa Hloa) as Ha.
 pose proof (guarded_upper_side rising measured_upper (balance b) Hlb Hsb Heb Hwb Hlob) as Hb.
 destruct (IVT_interv (oriented_balance rising balance) a b Hcont Hab Ha Hb) as [root [Hr He]].
 exists root; split; [exact Hr|].
 unfold oriented_balance in He; destruct rising; lra.
Qed.

Print Assumptions scalar_solve_certificate.
Print Assumptions scalar_solve_sufficient_fuel.
Print Assumptions checked_endpoint_root_exists.
