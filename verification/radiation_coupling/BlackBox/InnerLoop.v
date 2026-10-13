(* Inner 16u/8lambda safeguarded loop, with rounded 4u chart-width guard.
   A finite, executable safeguarded scalar control loop over indexed values.
   The data functions are abstract; their arithmetic and grid contracts below
   are explicit. This is not a refinement theorem for production C++ code. *)
From Coq Require Import Reals Psatz Lia Arith Bool Ranalysis5.
From BlackBox Require Import Guards Brackets FloatingPoint GuardFloatBridge SolverLoop.
Open Scope R_scope.

Definition inner_residual_window (h:R) : bool :=
  if Rle_dec (1-16*binary64_u) h then
    if Rle_dec h (1+16*binary64_u) then true else false
  else false.
Definition inner_width_window (w:R) : bool :=
  if Rle_dec w (4*binary64_u) then true else false.
Definition inner_lower_side (rising:bool) (h:R) : bool :=
  if rising then if Rlt_dec h 1 then true else false
  else if Rlt_dec 1 h then true else false.
Definition inner_oriented_balance (rising:bool) (balance:R->R) (x:R) : R :=
  if rising then balance x-1 else 1-balance x.

Fixpoint inner_scalar_solve (fuel:nat) (rising:bool)
  (measured:nat->R) (width:nat->nat->R) (proposal:nat->nat->nat)
  (lower upper:nat) : scalar_result :=
  match fuel with
  | O => BudgetExhausted lower upper
  | S fuel' =>
    if inner_width_window (width lower upper) then BracketAccepted lower upper lower
    else
      let trial:=safeguarded_index lower upper (proposal lower upper) in
      if inner_residual_window (measured trial) then ResidualAccepted lower upper trial
      else if inner_lower_side rising (measured trial) then
        inner_scalar_solve fuel' rising measured width proposal trial upper
      else inner_scalar_solve fuel' rising measured width proposal lower trial
  end.

Lemma inner_residual_window_spec h :
  inner_residual_window h=true <-> 1-16*binary64_u<=h<=1+16*binary64_u.
Proof.
 unfold inner_residual_window; destruct Rle_dec; [destruct Rle_dec|]; split; intro H;
 try discriminate; try reflexivity; lra.
Qed.
Lemma inner_width_window_spec w : inner_width_window w=true <-> w<=4*binary64_u.
Proof. unfold inner_width_window; destruct Rle_dec; split; intro H; auto; discriminate. Qed.
Lemma inner_residual_window_reject h : inner_residual_window h=false ->
 h<1-16*binary64_u \/ 1+16*binary64_u<h.
Proof.
 unfold inner_residual_window; destruct Rle_dec; [destruct Rle_dec|]; intro H;
 try discriminate; [right|left]; lra.
Qed.

Lemma inner_guarded_lower_side rising h s :
  0<h -> 0<s -> Rabs(ln h-ln s)<=8*binary64_lambda ->
  inner_residual_window h=false -> inner_lower_side rising h=true ->
  (if rising then s-1 else 1-s)<0.
Proof.
 intros Hh Hs Herr Hreject Hside.
 pose proof binary64_u_bounds as Hu.
 apply inner_residual_window_reject in Hreject.
 destruct rising; unfold inner_lower_side in Hside;
 destruct Rlt_dec; try discriminate; simpl;
 destruct Hreject as [Hlo|Hhi].
 - pose proof (inner_lower_sign h s Hh Hs Hlo Herr); lra.
 - exfalso; nra.
 - exfalso; nra.
 - pose proof (inner_upper_sign h s Hh Hs Hhi Herr); lra.
Qed.
Lemma inner_guarded_upper_side rising h s :
  0<h -> 0<s -> Rabs(ln h-ln s)<=8*binary64_lambda ->
  inner_residual_window h=false -> inner_lower_side rising h=false ->
  0<(if rising then s-1 else 1-s).
Proof.
 intros Hh Hs Herr Hreject Hside.
 pose proof binary64_u_bounds as Hu.
 apply inner_residual_window_reject in Hreject.
 destruct rising; unfold inner_lower_side in Hside;
 destruct Rlt_dec; try discriminate; simpl;
 destruct Hreject as [Hlo|Hhi].
 - exfalso; nra.
 - pose proof (inner_upper_sign h s Hh Hs Hhi Herr); lra.
 - pose proof (inner_lower_sign h s Hh Hs Hlo Herr); lra.
 - exfalso; nra.
Qed.

Section VerifiedInnerLoop.
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
  Rabs(ln(measured i)-ln(balance(value i)))<=8*binary64_lambda.
Hypothesis physical_monotonicity :
  nondecreasing_on (inner_oriented_balance rising balance)
    (value initial_lower) (value initial_upper).
Hypothesis exact_root : balance root=1.

Definition inner_loop_state (a b:nat) : Prop :=
  (initial_lower<=a /\ a<b /\ b<=initial_upper)%nat /\
  root_bracket (value a) (value b) root.

Lemma inner_grid_le i j :
  (initial_lower<=i)%nat -> (i<=j)%nat -> (j<=initial_upper)%nat -> value i<=value j.
Proof. intros Hi Hij Hj; apply Nat.lt_eq_cases in Hij; destruct Hij as [H|H];
 [left; apply grid_increasing; assumption|subst; reflexivity]. Qed.

Lemma inner_local_monotonicity a b : inner_loop_state a b ->
  nondecreasing_on (inner_oriented_balance rising balance) (value a) (value b).
Proof.
 intros [[Ha [Hab Hb]] Hr] x y Hx Hxy Hy.
 assert (Hla:value initial_lower<=value a) by (apply inner_grid_le; lia).
 assert (Hbu:value b<=value initial_upper) by (apply inner_grid_le; lia).
 apply physical_monotonicity; lra.
Qed.

Lemma inner_state_width_certificate a b : inner_loop_state a b -> inner_width_window(width a b)=true ->
  Rabs(ln(value a/root))<8*binary64_lambda.
Proof.
 intros Hstate Hwidth; destruct Hstate as [[Ha [Hab Hb]] Hr].
 apply inner_width_window_spec in Hwidth.
 destruct (width_arithmetic a b Ha Hab Hb) as [es [ed [Hes [Hed Hw]]]].
 assert (Hpos:0<value a) by (apply grid_positive; lia).
 assert (Habval:value a<value b) by (apply grid_increasing; assumption).
 pose proof (bracket_log_error (value a) (value b) root (value a) Hpos Hr ltac:(lra)).
 pose proof (rounded_inner_bracket_width (value a) (value b) es ed (width a b) Hpos ltac:(lra) Hes Hed Hw Hwidth).
 lra.
Qed.

Lemma inner_rejected_width_has_interior a b : inner_loop_state a b -> inner_width_window(width a b)=false ->
  (a+1<b)%nat.
Proof.
 intros [[Ha [Hab Hb]] Hr] Hreject.
 destruct (Nat.lt_ge_cases (a+1) b) as [Hgap|Hgap]; [exact Hgap|].
 assert (Hbnext:b=S a) by lia; subst b.
 destruct (width_arithmetic a (S a) Ha ltac:(lia) Hb) as [es [ed [Hes [Hed Hw]]]].
 assert (Hwpass:width a (S a)<=4*binary64_u).
 { apply (adjacent_normal_inner_bracket_passes (value a) (value (S a)) es ed (width a (S a)));
   try assumption.
   - apply grid_positive; lia.
   - apply grid_normal; lia.
   - apply grid_format; lia.
   - apply grid_format; lia.
   - apply grid_increasing; lia.
   - intros; eapply grid_adjacent_complete; eauto. }
 apply inner_width_window_spec in Hwpass; congruence.
Qed.

Lemma inner_trial_inside a b : inner_loop_state a b -> inner_width_window(width a b)=false ->
  (a<safeguarded_index a b (proposal a b)<b)%nat.
Proof. intros Hs Hw; apply safeguarded_index_inside; eapply inner_rejected_width_has_interior; eauto. Qed.

Lemma inner_lower_step_state a b i : inner_loop_state a b -> (a<i<b)%nat ->
  inner_residual_window(measured i)=false -> inner_lower_side rising (measured i)=true ->
  inner_loop_state i b.
Proof.
 intros Hstate Hi Hreject Hside.
 pose proof (inner_local_monotonicity a b Hstate) as Hmono.
 destruct Hstate as [[Ha [Hab Hb]] Hr].
 destruct (evaluator i ltac:(lia)) as [Hh [Hs He]].
 pose proof (inner_guarded_lower_side rising (measured i) (balance(value i)) Hh Hs He Hreject Hside) as Hsign.
 split; [lia|].
 apply (lower_update_preserves_root (inner_oriented_balance rising balance) (value a) (value b) root (value i));
 try assumption.
 - split; left; apply grid_increasing; lia.
 - unfold inner_oriented_balance; destruct rising; rewrite exact_root; ring.
Qed.

Lemma inner_upper_step_state a b i : inner_loop_state a b -> (a<i<b)%nat ->
  inner_residual_window(measured i)=false -> inner_lower_side rising (measured i)=false ->
  inner_loop_state a i.
Proof.
 intros Hstate Hi Hreject Hside.
 pose proof (inner_local_monotonicity a b Hstate) as Hmono.
 destruct Hstate as [[Ha [Hab Hb]] Hr].
 destruct (evaluator i ltac:(lia)) as [Hh [Hs He]].
 pose proof (inner_guarded_upper_side rising (measured i) (balance(value i)) Hh Hs He Hreject Hside) as Hsign.
 split; [lia|].
 apply (upper_update_preserves_root (inner_oriented_balance rising balance) (value a) (value b) root (value i));
 try assumption.
 - split; left; apply grid_increasing; lia.
 - unfold inner_oriented_balance; destruct rising; rewrite exact_root; ring.
Qed.

Definition inner_result_certificate (result:scalar_result) : Prop :=
  match result with
  | ResidualAccepted a b i =>
      inner_loop_state a b /\ (a<i<b)%nat /\
      Rabs(ln(balance(value i)))<=25*binary64_lambda
  | BracketAccepted a b i =>
      inner_loop_state a b /\ i=a /\ Rabs(ln(value i/root))<8*binary64_lambda
  | BudgetExhausted a b => inner_loop_state a b
  end.

Theorem inner_scalar_solve_certificate fuel a b : inner_loop_state a b ->
  inner_result_certificate (inner_scalar_solve fuel rising measured width proposal a b).
Proof.
 revert a b; induction fuel as [|fuel IH]; intros a b Hstate; simpl; [exact Hstate|].
 destruct (inner_width_window(width a b)) eqn:Hw.
 - simpl; split; [exact Hstate|split; [reflexivity|eapply inner_state_width_certificate; eauto]].
 - set (i:=safeguarded_index a b (proposal a b)).
   assert (Hi:(a<i<b)%nat) by (unfold i; apply inner_trial_inside; assumption).
   destruct (inner_residual_window(measured i)) eqn:Hres.
   + simpl; split; [exact Hstate|split; [exact Hi|]].
     apply inner_residual_window_spec in Hres.
     destruct Hstate as [[Ha [Hab Hb]] Hr].
     destruct (evaluator i ltac:(lia)) as [Hh [Hs He]].
     apply (inner_exact_window (measured i) (balance(value i))); assumption.
   + destruct (inner_lower_side rising (measured i)) eqn:Hside; apply IH.
     * apply (inner_lower_step_state a b i); assumption.
     * apply (inner_upper_step_state a b i); assumption.
Qed.

Definition inner_successful (result:scalar_result) : Prop :=
  match result with BudgetExhausted _ _=>False | _=>True end.

Theorem inner_scalar_solve_sufficient_fuel fuel a b :
  inner_loop_state a b -> (b-a<=fuel)%nat ->
  inner_successful (inner_scalar_solve fuel rising measured width proposal a b).
Proof.
 revert a b; induction fuel as [|fuel IH]; intros a b Hstate Hfuel.
 - destruct Hstate as [[Ha [Hab Hb]] Hr]; lia.
 - simpl; destruct (inner_width_window(width a b)) eqn:Hw; [exact I|].
   set (i:=safeguarded_index a b (proposal a b)).
   assert (Hi:(a<i<b)%nat) by (unfold i; apply inner_trial_inside; assumption).
   destruct (inner_residual_window(measured i)) eqn:Hres; [exact I|].
   destruct (inner_lower_side rising (measured i)) eqn:Hside.
   + apply IH; [apply (inner_lower_step_state a b i); assumption|lia].
   + apply IH; [apply (inner_upper_step_state a b i); assumption|lia].
Qed.

Theorem inner_scalar_solve_accepted_or_exhausted fuel a b : inner_loop_state a b ->
  (exists l h i, inner_scalar_solve fuel rising measured width proposal a b=ResidualAccepted l h i /\
       inner_loop_state l h /\ (l<i<h)%nat /\ Rabs(ln(balance(value i)))<=25*binary64_lambda) \/
  (exists l h i, inner_scalar_solve fuel rising measured width proposal a b=BracketAccepted l h i /\
       inner_loop_state l h /\ i=l /\ Rabs(ln(value i/root))<8*binary64_lambda) \/
  (exists l h, inner_scalar_solve fuel rising measured width proposal a b=BudgetExhausted l h /\ inner_loop_state l h).
Proof.
 intro Hstate; pose proof (inner_scalar_solve_certificate fuel a b Hstate) as Hcert.
 destruct (inner_scalar_solve fuel rising measured width proposal a b) as [l h i|l h i|l h] eqn:Hresult;
 simpl in Hcert.
 - left; exists l,h,i; auto.
 - right; left; exists l,h,i; auto.
 - right; right; exists l,h; auto.
Qed.
End VerifiedInnerLoop.

(* Initialization is by checked reliable endpoint signs. An implementation must
   retry/enlarge endpoints until these checks succeed or report failure. There
   is deliberately no unsupported claim that "a few" outward steps suffice. *)
Theorem inner_checked_endpoint_root_exists rising balance measured_lower measured_upper a b :
  a<b -> (forall x, a<=x<=b -> continuity_pt (inner_oriented_balance rising balance) x) ->
  0<measured_lower -> 0<measured_upper -> 0<balance a -> 0<balance b ->
  Rabs(ln measured_lower-ln(balance a))<=8*binary64_lambda ->
  Rabs(ln measured_upper-ln(balance b))<=8*binary64_lambda ->
  inner_residual_window measured_lower=false -> inner_lower_side rising measured_lower=true ->
  inner_residual_window measured_upper=false -> inner_lower_side rising measured_upper=false ->
  exists root, root_bracket a b root /\ balance root=1.
Proof.
 intros Hab Hcont Hla Hlb Hsa Hsb Hea Heb Hwa Hloa Hwb Hlob.
 pose proof (inner_guarded_lower_side rising measured_lower (balance a) Hla Hsa Hea Hwa Hloa) as Ha.
 pose proof (inner_guarded_upper_side rising measured_upper (balance b) Hlb Hsb Heb Hwb Hlob) as Hb.
 destruct (IVT_interv (inner_oriented_balance rising balance) a b Hcont Hab Ha Hb) as [root [Hr He]].
 exists root; split; [exact Hr|].
 unfold inner_oriented_balance in He; destruct rising; lra.
Qed.

Print Assumptions inner_scalar_solve_certificate.
Print Assumptions inner_scalar_solve_sufficient_fuel.
Print Assumptions inner_checked_endpoint_root_exists.

From Coquelicot Require Import Coquelicot.
From BlackBox Require Import InnerCharts LogCalculus.

Lemma inner_heating_zero T A D d : Hh T A D d 0=0.
Proof. unfold Hh,Rdiv; ring. Qed.
Lemma inner_weak_zero T A D d : Hw T A D d 0=0.
Proof. unfold Hw,Rdiv; ring. Qed.
Lemma inner_heating_upper T A D d :
  0<T -> 0<A -> 0<D -> 0<d -> 1<Hh T A D d d.
Proof.
 intros HT HA HD Hd; unfold Hh.
 assert (Hs:0<sqrt(T+d)) by (apply sqrt_lt_R0; lra).
 assert (Hq:0<A*d/(D*sqrt(T+d))) by (apply Rdiv_lt_0_compat; nra).
 apply (Rmult_lt_reg_r d); [lra|].
 replace ((d+A*d/(D*sqrt(T+d)))/d*d) with (d+A*d/(D*sqrt(T+d))) by (field; lra).
 lra.
Qed.
Lemma inner_strong_lower T A D x :
  0<A -> 0<D -> 0<x<T -> 1<Hs T A D x x.
Proof.
 intros HA HD Hx; unfold Hs.
 assert (Hs:0<sqrt x) by (apply sqrt_lt_R0; lra).
 assert (Hq:0<A*(T-x)/(D*sqrt x)) by (apply Rdiv_lt_0_compat; nra).
 apply (Rmult_lt_reg_r x); [lra|].
 replace ((x+A*(T-x)/(D*sqrt x))/x*x) with (x+A*(T-x)/(D*sqrt x)) by (field; lra).
 lra.
Qed.

Lemma inner_derivative_monotone f df lo hi :
 (forall x, lo<=x<=hi -> is_derive f x (df x)) ->
 (forall x, lo<=x<=hi -> 0<=df x) -> nondecreasing_on f lo hi.
Proof.
 intros Hd Hp x y Hlo Hxy Hhi.
 assert (Hmin:Rmin x y=x) by (apply Rmin_left; lra).
 assert (Hmax:Rmax x y=y) by (apply Rmax_right; lra).
 destruct (mvt_interior_property f df x y (fun v=>0<=v) 0 ltac:(lra)) as [v [Hv Heq]].
 - intros z Hz; rewrite Hmin,Hmax in Hz; apply Hd; lra.
 - intros z Hz; rewrite Hmin,Hmax in Hz.
   eapply is_derive_continuity_pt; apply Hd; lra.
 - intros z Hz; rewrite Hmin,Hmax in Hz; apply Hp; lra.
 - nra.
Qed.

Lemma inner_heating_chart_monotone T A D d lo hi :
 0<T -> 0<A -> 0<D -> 0<d -> 0<lo -> lo<=hi ->
 nondecreasing_on (inner_oriented_balance true (Hh T A D d)) lo hi.
Proof.
 intros HT HA HD Hd Hlo Hhi.
 apply (inner_derivative_monotone _ (Hh' T A D d)); intros z Hz.
 - unfold inner_oriented_balance; simpl.
   replace (Hh' T A D d z) with (Hh' T A D d z-0) by ring.
   apply (is_derive_minus (Hh T A D d) (fun _ : R=>1) z (Hh' T A D d z) 0).
   + apply Hh_derivative; lra.
   + exact (is_derive_const (1:R) z).
 - pose proof (Hh_positive T A D d z HT HA HD Hd ltac:(lra)) as Hp.
   pose proof (Hh_elasticity T A D d z HT HA HD Hd ltac:(lra)) as He.
   assert (Hmul:z*Hh' T A D d z/Hh T A D d z*Hh T A D d z=z*Hh' T A D d z) by (field; lra).
   assert (0<z*Hh' T A D d z/Hh T A D d z) by lra.
   nra.
Qed.

Lemma inner_weak_chart_monotone T A D d lo hi :
 0<T -> 0<A -> 0<D -> 0<d -> 0<lo -> lo<=hi -> hi<=T/2 ->
 nondecreasing_on (inner_oriented_balance true (Hw T A D d)) lo hi.
Proof.
 intros HT HA HD Hd Hlo Hhi Hhalf.
 apply (inner_derivative_monotone _ (Hw' T A D d)); intros z Hz.
 - unfold inner_oriented_balance; simpl.
   replace (Hw' T A D d z) with (Hw' T A D d z-0) by ring.
   apply (is_derive_minus (Hw T A D d) (fun _ : R=>1) z (Hw' T A D d z) 0).
   + apply Hw_derivative; lra.
   + exact (is_derive_const (1:R) z).
 - pose proof (Hw_positive T A D d z HA HD Hd ltac:(lra)) as Hp.
   pose proof (Hw_elasticity T A D d z HT HA HD Hd ltac:(lra)) as He.
   assert (Hmul:z*Hw' T A D d z/Hw T A D d z*Hw T A D d z=z*Hw' T A D d z) by (field; lra).
   assert (0<z*Hw' T A D d z/Hw T A D d z) by lra.
   nra.
Qed.

Lemma inner_strong_chart_monotone T A D x lo hi :
 0<T -> 0<A -> 0<D -> 0<x -> 0<lo -> lo<=hi -> hi<=T/2 ->
 nondecreasing_on (inner_oriented_balance false (Hs T A D x)) lo hi.
Proof.
 intros HT HA HD Hx Hlo Hhi Hhalf.
 apply (inner_derivative_monotone _ (fun z=>-Hs' T A D x z)); intros z Hz.
 - unfold inner_oriented_balance; simpl.
   replace (-Hs' T A D x z) with (0-Hs' T A D x z) by ring.
   apply (is_derive_minus (fun _ : R=>1) (Hs T A D x) z 0 (Hs' T A D x z)).
   + exact (is_derive_const (1:R) z).
   + apply Hs_derivative; lra.
 - pose proof (Hs_positive T A D x z HA HD Hx ltac:(lra)) as Hp.
   pose proof (Hs_elasticity T A D x z HT HA HD Hx ltac:(lra)) as He.
   assert (Hmul:(-z*Hs' T A D x z)/Hs T A D x z*Hs T A D x z=-z*Hs' T A D x z) by (field; lra).
   assert (0<(-z*Hs' T A D x z)/Hs T A D x z) by lra.
   nra.
Qed.

Definition positive_chart_seed (Tref A D d:R) : R := d/(1+A/(D*sqrt Tref)).

Lemma inner_seed_basic Tref A D d :
  0<Tref -> 0<A -> 0<D -> 0<d ->
  0<positive_chart_seed Tref A D d<=d /\
  positive_chart_seed Tref A D d*(1+A/(D*sqrt Tref))=d.
Proof.
 intros HT HA HD Hd.
 assert (Hs:0<sqrt Tref) by (apply sqrt_lt_R0; lra).
 assert (Hk:0<A/(D*sqrt Tref)) by (apply Rdiv_lt_0_compat; nra).
 unfold positive_chart_seed.
 remember (A/(D*sqrt Tref)) as k in *.
 assert (He:d/(1+k)*(1+k)=d) by (field; lra).
 split; [split|exact He].
 - apply Rdiv_lt_0_compat; lra.
 - apply (Rmult_le_reg_r (1+k)); [lra|].
   rewrite He; nra.
Qed.

Lemma inner_denominator_antitone x a b :
  0<x -> 0<a -> a<=b -> x/b<=x/a.
Proof.
 intros Hx Ha Hab.
 apply (Rmult_le_reg_r (a*b)); [nra|].
 field_simplify; nra.
Qed.

Lemma inner_heating_positive_seed T A D d :
  0<T -> 0<A -> 0<D -> 0<d ->
  0<positive_chart_seed T A D d<=d /\
  Hh T A D d (positive_chart_seed T A D d)<=1.
Proof.
 intros HT HA HD Hd.
 pose proof (inner_seed_basic T A D d HT HA HD Hd) as [Hz He].
 split; [exact Hz|].
 set (z:=positive_chart_seed T A D d) in *.
 assert (Hs:0<sqrt T) by (apply sqrt_lt_R0; lra).
 assert (Hsz:sqrt T<=sqrt(T+z)) by (apply sqrt_le_1_alt; lra).
 assert (Hk:A/(D*sqrt(T+z))<=A/(D*sqrt T)).
 { apply inner_denominator_antitone; nra. }
 unfold Hh.
 apply (Rmult_le_reg_r d); [lra|].
 replace ((z+A*z/(D*sqrt(T+z)))/d*d) with (z+A*z/(D*sqrt(T+z))) by (field; lra).
 replace (A*z/(D*sqrt(T+z))) with (z*(A/(D*sqrt(T+z)))) by (unfold Rdiv; ring).
 nra.
Qed.

Lemma inner_weak_positive_seed T A D d :
  0<T -> 0<A -> 0<D -> 0<d -> 1<=Hw T A D d (T/2) ->
  0<positive_chart_seed (T/2) A D d<=Rmin d (T/2) /\
  Hw T A D d (positive_chart_seed (T/2) A D d)<=1.
Proof.
 intros HT HA HD Hd Hboundary.
 pose proof (inner_seed_basic (T/2) A D d ltac:(lra) HA HD Hd) as [Hz He].
 set (z:=positive_chart_seed (T/2) A D d) in *.
 assert (Hs:0<sqrt(T/2)) by (apply sqrt_lt_R0; lra).
 assert (Hk0:0<A/(D*sqrt(T/2))) by (apply Rdiv_lt_0_compat; nra).
 assert (Hhalf:z<=T/2).
 { unfold Hw in Hboundary.
   replace (T-T/2) with (T/2) in Hboundary by lra.
   assert (Hdprod:1*d<=((T/2+A*(T/2)/(D*sqrt(T/2)))/d)*d) by (apply Rmult_le_compat_r; lra).
   replace (((T/2+A*(T/2)/(D*sqrt(T/2)))/d)*d) with
     (T/2+A*(T/2)/(D*sqrt(T/2))) in Hdprod by (field; lra).
   replace (A*(T/2)/(D*sqrt(T/2))) with ((T/2)*(A/(D*sqrt(T/2)))) in Hdprod by (unfold Rdiv; ring).
   nra. }
 split.
 - split; [lra|apply Rmin_glb; lra].
 - assert (Hsz:sqrt(T/2)<=sqrt(T-z)) by (apply sqrt_le_1_alt; lra).
   assert (Hk:A/(D*sqrt(T-z))<=A/(D*sqrt(T/2))).
   { apply inner_denominator_antitone; nra. }
   unfold Hw; apply (Rmult_le_reg_r d); [lra|].
   replace ((z+A*z/(D*sqrt(T-z)))/d*d) with (z+A*z/(D*sqrt(T-z))) by (field; lra).
   replace (A*z/(D*sqrt(T-z))) with (z*(A/(D*sqrt(T-z)))) by (unfold Rdiv; ring).
   nra.
Qed.

Lemma inner_weak_upper T A D d :
  0<T -> 0<A -> 0<D -> 0<d -> 1<=Hw T A D d (T/2) ->
  1<=Hw T A D d (Rmin d (T/2)).
Proof.
 intros HT HA HD Hd Hb; destruct (Rle_dec d (T/2)) as [Hsmall|Hlarge].
 - rewrite Rmin_left by assumption; unfold Hw.
   assert (Hs:0<sqrt(T-d)) by (apply sqrt_lt_R0; lra).
   assert (Hq:0<A*d/(D*sqrt(T-d))) by (apply Rdiv_lt_0_compat; nra).
   apply (Rmult_le_reg_r d); [lra|].
   replace ((d+A*d/(D*sqrt(T-d)))/d*d) with (d+A*d/(D*sqrt(T-d))) by (field; lra).
   lra.
 - rewrite Rmin_right by lra; exact Hb.
Qed.

(* Exact analytic seeds are positive even for arbitrarily small transfers.
   No representability or normal-range claim follows from positivity. The
   executable initialization below separately checks domain/range and signs. *)
Inductive inner_initialization : Type :=
| InnerInitialAccepted (endpoint:nat)
| InnerInitialBracket (lower upper:nat)
| InnerInitialRangeFailure
| InnerInitialSignFailure.

Definition initialize_inner (range_ok:bool) (rising:bool)
  (measured:nat->R) (lower upper:nat) : inner_initialization :=
  if range_ok then
    if inner_residual_window(measured lower) then InnerInitialAccepted lower
    else if inner_residual_window(measured upper) then InnerInitialAccepted upper
    else if inner_lower_side rising (measured lower) then
      if inner_lower_side rising (measured upper) then InnerInitialSignFailure
      else InnerInitialBracket lower upper
    else InnerInitialSignFailure
  else InnerInitialRangeFailure.

Theorem initialize_inner_bracket_checked range_ok rising measured lower upper a b balance :
  initialize_inner range_ok rising measured lower upper=InnerInitialBracket lower upper ->
  a<b -> (forall x, a<=x<=b -> continuity_pt (inner_oriented_balance rising balance) x) ->
  0<measured lower -> 0<measured upper -> 0<balance a -> 0<balance b ->
  Rabs(ln(measured lower)-ln(balance a))<=8*binary64_lambda ->
  Rabs(ln(measured upper)-ln(balance b))<=8*binary64_lambda ->
  exists root, root_bracket a b root /\ balance root=1.
Proof.
 unfold initialize_inner; destruct range_ok; [|discriminate].
 destruct (inner_residual_window(measured lower)) eqn:Hl; [discriminate|].
 destruct (inner_residual_window(measured upper)) eqn:Hu; [discriminate|].
 destruct (inner_lower_side rising (measured lower)) eqn:Hsl; [|discriminate].
 destruct (inner_lower_side rising (measured upper)) eqn:Hsu; [discriminate|].
 intros _ Hab Hcont Hml Hmu Hba Hbb Hel Heu.
 exact (inner_checked_endpoint_root_exists rising balance (measured lower) (measured upper) a b Hab Hcont Hml Hmu Hba Hbb Hel Heu Hl Hsl Hu Hsu).
Qed.

Theorem initialize_inner_acceptance_checked range_ok rising measured lower upper endpoint (balance:R->R) (value:nat->R) :
  initialize_inner range_ok rising measured lower upper=InnerInitialAccepted endpoint ->
  (forall i, i=lower \/ i=upper ->
    Rabs(ln(measured i)-ln(balance(value i)))<=8*binary64_lambda) ->
  Rabs(ln(balance(value endpoint)))<=25*binary64_lambda.
Proof.
 unfold initialize_inner; destruct range_ok; [|discriminate].
 destruct (inner_residual_window(measured lower)) eqn:Hl.
 - intros He Herror; inversion He; subst endpoint.
   apply (inner_exact_window (measured lower)); [apply inner_residual_window_spec; assumption|apply Herror; auto].
 - destruct (inner_residual_window(measured upper)) eqn:Hu.
   + intros He Herror; inversion He; subst endpoint.
     apply (inner_exact_window (measured upper)); [apply inner_residual_window_spec; assumption|apply Herror; auto].
   + destruct (inner_lower_side rising (measured lower)); [destruct (inner_lower_side rising (measured upper))|]; discriminate.
Qed.

Lemma inner_heating_seed_brackets_root T A D d z :
  0<T -> 0<A -> 0<D -> 0<d -> 0<z -> Hh T A D d z=1 ->
  positive_chart_seed T A D d<=z<=d.
Proof.
 intros HT HA HD Hd Hz Hroot.
 pose proof (inner_seed_basic T A D d HT HA HD Hd) as [Hseed Hseed_eq].
 assert (Hs:0<sqrt T) by (apply sqrt_lt_R0; lra).
 assert (Hsz:sqrt T<=sqrt(T+z)) by (apply sqrt_le_1_alt; lra).
 assert (Hk0:0<A/(D*sqrt T)) by (apply Rdiv_lt_0_compat; nra).
 assert (Hk:0<A/(D*sqrt(T+z))) by (apply Rdiv_lt_0_compat; nra).
 assert (Hkle:A/(D*sqrt(T+z))<=A/(D*sqrt T)) by (apply inner_denominator_antitone; nra).
 assert (He:z*(1+A/(D*sqrt(T+z)))=d).
 { unfold Hh in Hroot.
   assert (Hmul:((z+A*z/(D*sqrt(T+z)))/d)*d=1*d) by now rewrite Hroot.
   replace (((z+A*z/(D*sqrt(T+z)))/d)*d) with (z+A*z/(D*sqrt(T+z))) in Hmul by (field; lra).
   replace (A*z/(D*sqrt(T+z))) with (z*(A/(D*sqrt(T+z)))) in Hmul by (unfold Rdiv; ring).
   nra. }
 split; nra.
Qed.

Lemma inner_weak_seed_brackets_root T A D d z :
  0<T -> 0<A -> 0<D -> 0<d -> 0<z<=T/2 -> Hw T A D d z=1 ->
  positive_chart_seed (T/2) A D d<=z<=Rmin d (T/2).
Proof.
 intros HT HA HD Hd Hz Hroot.
 pose proof (inner_seed_basic (T/2) A D d ltac:(lra) HA HD Hd) as [Hseed Hseed_eq].
 assert (Hs:0<sqrt(T/2)) by (apply sqrt_lt_R0; lra).
 assert (Hsz:sqrt(T/2)<=sqrt(T-z)) by (apply sqrt_le_1_alt; lra).
 assert (Hk0:0<A/(D*sqrt(T/2))) by (apply Rdiv_lt_0_compat; nra).
 assert (Hk:0<A/(D*sqrt(T-z))) by (apply Rdiv_lt_0_compat; nra).
 assert (Hkle:A/(D*sqrt(T-z))<=A/(D*sqrt(T/2))) by (apply inner_denominator_antitone; nra).
 assert (He:z*(1+A/(D*sqrt(T-z)))=d).
 { unfold Hw in Hroot.
   assert (Hmul:((z+A*z/(D*sqrt(T-z)))/d)*d=1*d) by now rewrite Hroot.
   replace (((z+A*z/(D*sqrt(T-z)))/d)*d) with (z+A*z/(D*sqrt(T-z))) in Hmul by (field; lra).
   replace (A*z/(D*sqrt(T-z))) with (z*(A/(D*sqrt(T-z)))) in Hmul by (unfold Rdiv; ring).
   nra. }
 split; [nra|apply Rmin_glb; nra].
Qed.

Lemma inner_strong_brackets_root T A D x z :
  0<T -> 0<A -> 0<D -> 0<x -> 0<z<=T/2 -> Hs T A D x z=1 ->
  x<z<=T/2.
Proof.
 intros HT HA HD Hx Hz Hroot.
 assert (Hsqrt:0<sqrt z) by (apply sqrt_lt_R0; lra).
 assert (Hq:0<A*(T-z)/(D*sqrt z)) by (apply Rdiv_lt_0_compat; nra).
 unfold Hs in Hroot.
 assert (Hmul:((x+A*(T-z)/(D*sqrt z))/z)*z=1*z) by now rewrite Hroot.
 replace (((x+A*(T-z)/(D*sqrt z))/z)*z) with (x+A*(T-z)/(D*sqrt z)) in Hmul by (field; lra).
 lra.
Qed.

Print Assumptions inner_heating_seed_brackets_root.
Print Assumptions inner_weak_seed_brackets_root.
Print Assumptions initialize_inner_bracket_checked.
