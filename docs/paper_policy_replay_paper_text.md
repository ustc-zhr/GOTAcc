# Section 4 paper-ready text

## Methods

We evaluated machine-aware diagnostic policies using a deterministic offline
replay of the production GOTAcc policy classes at commit
`010518fbece77a4630684065101033ff8588613d`. Eight fixed, three-sample objective
sequences represented nominal variation, a constant-zero ambiguity control, a
constant nonzero proxy for a frozen readback, a sustained high-level signal, an
isolated (10^8)-level spike, slow drift, and NaN/Inf boundary cases. All
sequences were synthetic. The spike amplitude reflects the reported order of
magnitude of anomalous IRFEL energy-meter readings, but no replayed sequence
was recorded from the machine.

For each sequence, the arithmetic mean, mean absolute amplitude, maximum
absolute amplitude, peak-to-peak variation, and post-policy value were
recorded. We compared (i) no policy, (ii) the production
`FelEnergyGuardPolicy`, which replaces the reduced objective with zero when the
mean absolute sample amplitude exceeds (10^6) or the peak-to-peak variation
is below (10^{-6}), and (iii) a registry-built
`SampleGuardObjectivePolicy`, configured for the named `fel_energy` target with
`max_abs > 10^6 OR peak_to_peak < 10^{-6}` and a declarative zero-replacement
action. These two guards were not assumed to be equivalent for arbitrary
inputs; they were compared only on the fixed replay. A separate constraint
case configured `SampleGuardConstraintPolicy` to map an all-zero `orbit_x`
sample sequence outside the accepted interval ([-1,1]).

The experiment called only the production `post_reduce()` methods on in-memory
NumPy arrays. It did not instantiate an EPICS backend, load pyepics, connect to
an IOC, or issue a PV read or write. Policy events were recorded only when the
production class emitted them through its event sink.

## Results

The fixed expectation check passed without adjusting the expected counts. No
policy changed any of the eight arithmetic means and six outputs were finite.
Both the dedicated FEL guard and the declarative Sample Guard changed four of
eight outputs, produced seven finite outputs, retained both nominal finite
sequences, and changed all three constructed anomaly sequences. In particular,
both guards mapped the synthetic ([2,10^8,2]) sequence from its mean of
(3.3333334667\times10^7) to zero. The replay therefore provides no evidence
of an isolated-spike detection defect in the existing FEL guard.

The declarative Sample Guard emitted five observable objective events: for the
constant-zero control, the constant nonzero frozen-value proxy, the sustained
high-level sequence, the (10^8)-level spike, and the sequence containing
infinity. The constant-zero event performed a numerical (0\rightarrow0)
replacement, so it increased the event count without increasing the
changed-output count. The production FEL guard emitted no events because its
implementation does not call the event-emission method; no internal trigger
state was inferred from its processed outputs.

NaN propagated with or without either guard. Infinity propagated without a
policy but was replaced by both guards because it satisfied their amplitude
predicates. This observed result is not evidence of comprehensive nonfinite
handling. In the constraint replay, the normal BPM mean remained (0.3),
whereas the all-zero sequence was mapped to (1.2), outside ([-1,1]), and
produced one directly observed policy event.

## Discussion

The replay demonstrates the architectural contribution rather than a ranking
between the two guards. Machine-specific diagnostic interpretation can be
applied at a common evaluation boundary, independently of the optimizer, while
the general Sample Guard expresses the same kinds of selected validity rules
through named configuration. The successful (10^8)-scale replay confirms
that the existing mean-absolute-threshold FEL guard handles the selected
three-sample spike; Sample Guard should therefore be presented as a
generalization and configuration mechanism, not as a repair for a demonstrated
FEL-guard defect.

The evidence is deliberately limited. The inputs are synthetic rather than
recorded machine data, and the experiment does not exercise acquisition,
write/readback verification, snapshot restoration, GUI transitions, optimizer
convergence, or live-machine operation. A constant valid signal may be
indistinguishable from a frozen diagnostic, and replacement by zero is
objective-dependent. These limitations prevent claims of classification
accuracy, improved convergence, or online accelerator safety.

## LaTeX table

```latex
\begin{table}[t]
\centering
\caption{Observed deterministic replay results for eight synthetic objective
sequences. Changed denotes a post-policy scalar different from the arithmetic
mean and is not a classification-accuracy measure.}
\label{tab:policy-replay}
\begin{tabular}{lrrrr}
\hline
Configuration & Changed & Finite & Nominal retained & Observable events \\
\hline
No policy & 0/8 & 6/8 & 2/2 & 0 \\
FEL energy guard & 4/8 & 7/8 & 2/2 & 0$^{\ast}$ \\
Declarative sample guard & 4/8 & 7/8 & 2/2 & 5 \\
\hline
\end{tabular}
\\[-1mm]
\footnotesize{$^{\ast}$The production FEL guard does not emit policy events;
zero events therefore indicates unavailable event observability, not a false
predicate.}
\end{table}
```
