# Trait mode: arbitrary discrete traits, transition histories and dates

This document describes the DTA pipeline as a general discrete-trait framework:
arbitrary categorical traits, directional acquisition and loss counts, complete
transition histories, posterior distributions of transition dates in calendar
time, prevalence over time and TMRCAs.

There is one analysis, not two modes.  A trait's states come either from an
annotation sheet (`--trait-source table`) or from the taxon labels themselves
(`--trait-source label`), which is how the geographic analyses work; everything
downstream is identical.

## 1. Trait inputs (WP1)

`workflow/scripts/trait_data.py` turns a trait source plus a tree into a
normalised assignment.  With `--source table` the states come from an annotation
sheet; with `--source label` they are extracted from the taxon labels (the
legacy geographic behaviour, `--label-index`, `--label-delimiter`,
`--taxon-regex`).

```
taxon,cfrD
EPI_ISL_2546697,1
EPI_ISL_2521906,0
EPI_ISL_2131520,NA
```

* **Taxon matching.** `--match-mode exact` compares the whole tree label,
  `field` splits the label on a delimiter and takes one field (this is the
  legacy geographic behaviour), `regex` extracts a capture group.  Unmatched
  metadata rows and unmatched tree taxa are reported, never silently dropped.
* **Missing observations.** Tokens such as `NA`, `?`, `-` or an empty cell are
  coded as BEAST's unknown state (`?`) by default.  A fixed empirical tree
  distribution cannot simply drop tips, so coding as unknown is the correct
  treatment; `--on-missing error` is available when a complete table is
  required.
* **Validation.** Duplicate taxa, unknown columns, observed states outside the
  declared state set, a constant trait and state codes that would corrupt the
  XML or the history log are all hard errors.  Declared but unobserved states
  produce a warning, since their rates are not identifiable.
* **State codes.** BEAST 1.10's `completeHistoryLogger` parses its own history
  strings with a generic value parser that turns anything number-like into an
  `Integer`/`Double` and then casts the result back to `String`, which aborts
  the run with a `ClassCastException`.  Numeric state codes are therefore
  prefixed (`0` → `s0`) before the XML is written, and mapped back in every
  summary.  The mapping is recorded in `<prefix>.traits.json` under
  `state_code_map`.  `--state-code-style as-is` disables the rewrite.

Outputs: `<prefix>.traits.tsv` (`taxon<TAB>state`, tree order, `?` for unknown)
and `<prefix>.traits.json` (states, counts, missing taxa, code map).

## 2. Models (WP2)

The BEAST model is a `generalSubstitutionModel` over a `generalDataType` whose
states are the trait's state codes, with a strict clock whose rate is the
overall trait transition rate, and a `markovJumpsTreeLikelihood`.  The rate
matrix is given row-major in the order of the state codes, which is also the
order used for every register.

| `--trait-model` | rates | priors | operators |
| --- | --- | --- | --- |
| `equal` | all relative rates fixed at 1 | — | — (the clock rate carries the scale) |
| `asymmetric` | one free rate per ordered pair | gamma on all rates | scale all independently |
| `bssvs` | free rates with indicators | gamma on rates, Poisson on the number of non-zero rates | scale + bit flip |

Root state prior (`--trait-root-state`):

* `uniform` — equal frequencies, estimated (delta exchange, uniform prior);
* `empirical` — observed tip frequencies as the starting values, still estimated;
* `fixed:<state>` — the root is pinned to one state (no prior, no operator).

The clock rate has the same hierarchical exponential/gamma prior as the legacy
geographic analysis (`clock_prior_shape`, `clock_prior_scale`), and the rate
prior is configurable (`rate_prior_shape`, `rate_prior_scale`).

For `bssvs`, the Poisson prior on the number of non-zero rates defaults to
`k(k-1)/2`, half of the ordered routes, which is the diffuse prior recommended
by [Gao et al. (2023)](https://doi.org/10.1093/bioinformatics/btac849) and what
PrioriTree specifies.  BEAST's own asymmetric default is `k-1`, a strongly
sparse prior, and a fixed value cannot be right for every number of states
(`trait_bssvs_prior_mean` overrides the default).

Markov jump registers are always written: `Into.<state>` (transitions into a
state — acquisitions), `OutOf.<state>` (transitions out of a state — losses),
`total`, one register per ordered pair (`<trait>.<from>.<to>`) and one reward
register per state (time spent in the state).

## 3. Complete transition histories (WP3)

Setting `log_complete_history` (default on in trait mode) adds

```xml
<markovJumpsTreeLikelihood ... useUniformization="true" numberOfSimulants="1" saveCompleteHistory="true"/>
```

and a `completeHistoryLogger` inside the MCMC block, which writes
`<prefix>.jumpHistory.log`.  Each line is one posterior sample:

```
state<TAB>totalChangeCount<TAB>completeHistory_1
300000	1.0	{{1,2.954822623502875,s0,s1}} 1
```

Every event is `{site, height, source, destination}`.  The events are sampled by
uniformization and conditioned on the tip states and the sampled ancestral
states, so the log contains *individual* events — origin, destination, position
along the branch — and not just their totals.  `workflow/scripts/jump_history.py`
parses the nested format and `totalChangeCount` is checked against the number of
events on every line; mismatches are reported.

## 4. Transition dates (WP4)

BEAST reports node heights measured from the most recent tip, in the time unit
of the phylogeny (years for every tree this pipeline accepts).  A transition at
height `h` therefore happened `h` years before the most recent sampling date:

```
date = mrsd - height
```

`mrsd` is not a free parameter.  A dated tree puts `date + height == mrsd` at
every tip, so the Snakefile reads the MRSD off the newest tip date in the labels
of `input_trees` (`dates.infer_mrsd`; the offsets must agree across at least 95%
of the tips, otherwise the trees are not a clock in years and there is no
calendar axis).  `--mrsd` still wins when given, and a disagreement with the
newest tip is printed as a warning, since it would shift every date by the same
amount.  Undated labels — the simulated example — fall back to `--mrsd`.

`workflow/scripts/summarize_jumps.py` writes:

* `<prefix>.events.tsv` — one row per transition of every retained sample
  (`state`, `event`, `height`, `date`, `date_iso`, `source`, `dest`).
* `<prefix>.event_counts.tsv` — per direction: total events, posterior mean,
  median, sd, 95% interval, `P(>0)` and the maximum.  **Samples with no event of
  that direction contribute a zero**, so `P(>0)` is a genuine posterior
  probability that the transition occurred at all.
* `<prefix>.event_dates.tsv` — per direction: the distribution of the earliest
  transition (`first_*`), of the latest (`last_*`) and of all transition times
  pooled over samples (`pooled_*`).  `first_*` is computed only over samples that
  have at least one event of that direction, which is the correct summary when a
  direction may be absent; `pooled_*` is the density of transition times and is
  the quantity to compare with an independently dated event.
* `<prefix>.root_states.tsv` — posterior root state probabilities, read from the
  `logTree` annotation of every retained tree.
* `<prefix>.acquisitions.tsv` — per state, the posterior of the *first*
  transition into it from any source (`first_*`, with the date the state was
  first reached in each sample) and `p_acquired`, the posterior probability that
  the state was entered at all.  This is the number to quote when dating an
  acquisition.  Summarising the per-direction "earliest event" columns instead
  would be late (the minimum of per-sample minima is at most the minimum of
  those medians) and would ignore samples in which a given source was never
  used.  Every ordered direction is reported, including those that never
  occurred, so an absent direction shows `P(>0) = 0` instead of disappearing
  from the table.
* `<prefix>.count_check.tsv` — the complete histories and BEAST's Markov jump
  registers are two estimates of the same quantity from the same samples; their
  posterior means must agree to within Monte Carlo error.  This is the main
  internal consistency check on the history parser.

Burn-in is controlled by `burnin_fraction` (default 0.1) and is applied to
**each chain separately**, both to the log and to the tree index: every chain
numbers its states from zero, so trimming a concatenation would keep the
burn-in of every chain but the first.  `logcombiner -burnin` therefore drops the
prefix of each chain when they are combined, and the consumers of the combined
files (treeannotator, the violin plots, the chord diagram) apply no further
burn-in; the consumers that read the per-chain files (`summarize_jumps`,
`trait_through_time`, `diagnostics`) trim each file themselves.  The input tree
posterior has its own burn-in, `input_burnin_fraction` (default 0.1): BEAST
gives every tree left in the file the same weight, so the unconverged start of
an upstream tree posterior must be dropped before the analysis, not after.

## 5. Workflow (WP5)

The workflow is packaged as a CLI with [snk](https://snk.wytamma.com/):
`snk install wytamma/dta` generates the `dta` command from
`config/config.yaml` (every key becomes `--<key-with-dashes>`; the annotated
subset, declared in `snk.yaml`, appears in `dta run --help`).  Files that live
inside the workflow are copied into the working directory for the run with
`--resource`, and the workflow's own `resources/` folder is copied
automatically.


`workflow/Snakefile` runs one set of rules for every trait:
`preprocess_traits`, `generate_beast_xml`, `run_beast`, `collect_chains`,
`max_clade_credibility_tree`, `max_clade_credibility_tree_render`, `plot_results`,
`plot_circlize`, `check_convergence`, `summarize_events`, `plot_event_counts`,
`plot_event_dates`, `plot_root_states`, `trait_through_time`,
`plot_trait_through_time` and `plot_trait_arrivals`.  Nothing is conditional on
where the states came from; only the calendar-time figures are skipped when
neither `mrsd` nor a dated tip label supplies a calendar origin.

`--beast-chains > 1` runs independent chains and combines the logs and trees with
BEAST's `logcombiner`; each chain writes its own jump history, and the summary
reads all of them.

## 6. Validation (WP6)

### Unit and integration tests

```bash
python -m pytest tests -q                                   # 96 fast tests
BEAST_BIN=/path/to/beast python -m pytest tests -q -m slow   # + real BEAST run
```

The slow test simulates a single acquisition, runs the real pipeline (trait
preprocessing → XML → BEAST 1.10.4 → summaries) and checks that

* the acquisition is supported (`P(>0) > 0.5`) and the posterior mean count is
  within 1 of the simulated count,
* the simulated date falls inside both the `first_*` and the `pooled_*` 95%
  intervals,
* the complete history agrees with the Markov jump registers,
* the root state probabilities sum to one and cover both states.

### Simulation-based calibration

`workflow/scripts/calibration.py` repeats the whole pipeline on simulated
datasets with a known history for five scenarios (single gain, recurrent gains,
single loss, near fixation, and 50% missing observations), 16 replicates each,
24 tips, 400 000 MCMC steps:

```bash
python workflow/scripts/calibration.py --output-prefix calib --replicates 16 \
    --chain-length 400000 --sample-every 4000 --beast-bin /path/to/beast
```

Reported coverage is the fraction of replicates in which the true value falls
inside the posterior 95% interval; for a well calibrated analysis it should be
close to 0.95 (with 16 replicates the binomial standard error is about 0.05, so
values from ~0.7 upward are consistent with 0.95).  The raw output of the run
reported below is in `docs/calibration/` (`coverage.json` per scenario,
`replicates.tsv` per replicate).

Results (24-tip coalescent tree, 400 000 states / 100 samples, 16 replicates per
scenario):

| scenario | n | earliest date | pooled dates | count | `P(>0)` | `P(true root)` | mean count (true) | max register diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single gain | 16 | 1.00 | 1.00 | 1.00 | 0.68 | 0.62 | 0.92 (1) | 0.10 |
| recurrent gains | 15 | 1.00 | 0.93 | 1.00 | 0.66 | 0.36 | 2.35 (2.47) | 0.26 |
| single loss | 16 | 1.00 | 1.00 | 1.00 | 0.69 | 0.62 | 0.88 (1) | 0.14 |
| near fixation | 16 | 1.00 | 1.00 | 1.00 | 0.58 | 0.53 | 0.79 (1) | 0.10 |
| sparse (50% missing) | 15 | 1.00 | 1.00 | 1.00 | 0.74 | 0.59 | 1.52 (1) | 0.21 |

Coverage is the fraction of replicates in which the true value lies inside the
posterior 95% interval; `P(>0)` is the mean posterior probability that the
simulated direction occurred at least once; `max register diff` is the largest
absolute difference between the complete-history counts and BEAST's Markov jump
registers, pooled over replicates.

Reading the table:

* **Transition dates are recovered conservatively.** The true acquisition (or
  loss) date is inside the 95% interval in every replicate, and the posterior
  count intervals always contain the simulated count.  Coverage of 1.00 rather
  than 0.95 means the intervals are wide, not that they are wrong: a single
  binary pattern on 24 tips carries little information about *when* a
  transition happened, and the pipeline reports that honestly.
* **The pooled transition-time distribution is the sharper summary.** For
  recurrent gains one replicate in fifteen fell outside its interval, i.e.
  roughly nominal coverage for the quantity that is actually identifiable.
* **Directional support is moderate.** `P(>0)` is 0.58–0.74 across scenarios:
  the analysis usually finds the acquisition, but with one pattern the evidence
  is not overwhelming.  Real datasets (many tips, a longer tree, several
  correlated traits) carry much more information.
* **The root state is the weakest quantity.** The true root state receives
  0.36–0.62 posterior probability.  A single binary trait does not identify the
  root state confidently, and Stage 2 conclusions should not rest on it.
* **Two replicates were rejected, correctly**: the simulated trait was constant
  at the tips (every tip in state 1), so there was nothing to reconstruct and
  `trait_data.py` refuses the analysis instead of fitting an unidentified model.
* **The register cross-check differences (0.10–0.26)** are the Monte Carlo error
  of the stochastic mapping: BEAST's registers are the *expected* jump counts
  given the sampled states, while the logged history is one random draw from
  that conditional distribution.  A parsing error would show up as a gross
  difference, not a tenth of a transition.

The calibration run is deliberately short (100 samples); the accompanying
`<prefix>.ess.tsv` reports the effective sample size of every parameter and
should be consulted before interpreting a real analysis.

### Geographic analyses are the label source

A geographic analysis is now simply a trait whose states come from the taxon
labels.  Re-running
`dta run --input-trees data/delphy.trees --resource data --trait-source label \
--trait location --trait-label-index 1` (1220 taxa, 202 locations) goes through
the same 17 steps as any other trait and keeps
the historical identifiers — `location.model`, `location.rates`,
`location.treeLikelihood`, `Into.*`, `OutOf.*` — plus a `location` trait table, a
jump-history log, event counts and dates, prevalence over time and TMRCAs.

Three intended differences from the historical behaviour:

* the transition model defaults to `asymmetric` for every trait; geography used
  to default to `bssvs` (pass `--trait-model bssvs` to restore it);
* the state order is deterministic (sorted) instead of depending on Python set
  iteration order, and the reward registers lost their doubled
  `rewards_rewards_` prefix;
* the ordered pair registers are named after the trait (`c_location.<from>.<to>`)
  instead of the fixed `c_Location.<from>.<to>`; `--trait-pair-prefix Location`
  restores the old names.

## 7. Reporting (WP7)

| figure | script | contents |
| --- | --- | --- |
| `<prefix>.event_counts.png` | `plot_events.py counts` | posterior transition counts per direction with 95% CI and `P(>0)` |
| `<prefix>.event_dates.png` | `plot_events.py dates` | transition time distributions with the earliest event marked |
| `<prefix>.root_states.png` | `plot_events.py root` | posterior root state probabilities |
| `<prefix>.mcc.mean<ext>` | `plotMCC.R` | MCC tree with ancestral states |
| `<prefix>.mcc.mean.<trait>.prob<ext>` | `plotMCC.R` | the same with posterior state probabilities at the nodes |
| `<prefix>.circlize<ext>` | `circlize.R` | chord diagram of mean transition counts |
| `<prefix>.log_<trait>.png` | `violinplot.py` | posterior violins for rates, root frequencies and counts |
| `<prefix>.trait_through_time.png` | `plot_events.py through_time` | prevalence over time plus one emergence-date density per state |
| `<prefix>.trait_arrivals.png` | `plot_events.py arrivals` | prevalence over time plus, per state, every sampled transition into it |
| `<prefix>.mcc.mean.<trait>.introductions.png` / `.pdf` / `.csv` | `plotMCC_introductions.R` | MCC tree coloured by state, with the entries into `introduction_states` marked and dated |

## 8. Prevalence over time and TMRCA (WP7)

`workflow/scripts/trait_through_time.py` reads the `logTree` file(s) and writes
`<prefix>.prevalence.tsv`, `<prefix>.tmrca.tsv` and `<prefix>.tmrca_samples.tsv`.
`plot_events.py through_time` renders them as
`<prefix>.trait_through_time.png`: a prevalence panel on top and one
emergence-date density per state below, sharing a calendar-date axis.

**Lineage prevalence at time *t*** is the fraction of branches crossing *t* whose
sampled state at *t* is *s* — the share of circulating lineages carrying the
trait, which rises as the trait spreads and falls if it is lost.  The state of a
branch comes from the per-branch transition history that BEAST writes when
complete-history logging is on (`logCompleteHistory="true"`, which the template
adds together with a `<trait name="history">` element in the `logTree` block).
A branch is in the destination of its youngest sampled transition, in the
destination of the younger of two transitions between them, and in the source of
its oldest transition above that; with no transition it keeps the state of its
child node.  Without the annotation the tool falls back to node states and warns.

Three caveats matter for real data:

* the number of lineages alive deep in the tree is small (2 at the root), so
  prevalence near the root is coarse and dominated by uncertainty about the root
  state;
* the curve runs from the root to the *present*, not to the earliest sampling
  date: the lineages crossing a time are those of the tips sampled after it, so
  the pool thins out towards the present and a share there can rest on a single
  lineage, which reads as 0% or 100%.  Where fewer than `--min-lineages`
  (default 5) cross, both the prevalence and the arrival figures stop drawing
  and shade the stretch instead — that is also what keeps the prevalence panel
  and the arrival histograms on one common window.  The rows stay in
  `prevalence.tsv`; the figures just decline to plot a proportion nobody should
  read;
* this is a *lineage* proportion, not a population prevalence.

**Emergence date.**  `<prefix>.tmrca.tsv` reports, per state, the age of the most
recent common ancestor of the tips carrying it — the crown of the carrier clade,
as `tmrca_*` (years before the MRSD) and `date_*` (calendar time) — together with
`appears_in`, the fraction of posterior trees in which the state occurs anywhere
in the reconstruction.  This is the density plotted in the outbreak.info-style
figure.  It is *younger* than the acquisition date whenever the trait arose
part-way along a branch: the acquisition is in `<prefix>.acquisitions.tsv`
(`first_*`: in each sample, the earliest transition into the state from *any*
source, summarised across samples; `p_acquired` says how often the state was
entered at all).  That is the number to quote when dating an acquisition, and it
is drawn on the figure as an orange dotted line, annotated with `P=...` whenever
the state is not acquired in every sample.  With a single fixed tree the
emergence date is identical in every sample and the density collapses to a
spike; a tree posterior gives a genuine distribution.

**When the carriers are not a clade.**  A reversible trait that is gained and
lost many times leaves the tips carrying each state paraphyletic, so their MRCA
is the tree root in every sample: `at_root` is 1, the emergence density is a
spike at the root, and `through_time` cannot say anything about the state.  The
complementary question — *when was the state entered?* — is answered by
`<prefix>.trait_arrivals.png` (`plot_events.py arrivals`): the same prevalence
panel on top, and below it, per state, a histogram of **every** sampled
transition into that state (all samples, every source), with the posterior
median of the first arrival as a dotted line, its 95% interval as a whisker at
the foot of the panel, and `P(acquired)` beside it.  Arrival times stay
informative however mixed the carriers are, because they come from the sampled
histories rather than from the tip labels.  Use `through_time` when the state is
a clade and its crown age is the quantity of interest, and `arrivals` when it is
not; both figures are produced by default.

**Entries into a state on the MCC tree.**  `<prefix>.mcc.mean.<trait>.introductions.png`
(`plotMCC_introductions.R`, derived from the analysis script used for the
Australia introductions) draws the maximum clade credibility tree with tips
coloured by their sampled state and internal nodes filled by the inferred state
— opacity from `location.prob` (nodes below `--min-prob`, 0.7, are faded) and
size from the node posterior.  A node counts as an **entry** into a state when
it is inferred in one of `introduction_states` while its parent is inferred in a
different state; those nodes are marked and the earliest
`introduction_max_labels` of them (15) are labelled with their tMRCA.  The
companion CSV lists every entry with its node, posterior, inferred and parental
state, tMRCA and 95% interval.

The interval is computed here rather than read from the tree: BEAST 1.10's
`treeannotator` annotates node heights but not their intervals, so the clade
below each entry is located in every posterior tree (by tip label, since tip
numbering differs) and the HPD of its MRCA height is taken.  Leave
`introduction_states` empty for a trait that is gained and lost continuously:
every node would then be an entry, which is true but not informative — the tree
is still drawn, without markers.

## 9. Scope

Trait mode reconstructs **observed discrete-trait changes on a dated
phylogeny**.  It does not model plasmid transfer networks, homologous
recombination or codon evolution, and an inferred acquisition date is the date
of the trait transition on the tree, not the molecular origin of the underlying
gene or allele.

## 10. Reproducing everything

```bash
# unit tests
python -m pytest tests -q

# install the workflow as a CLI, then run the shipped example
snk install wytamma/dta
dta run --cores 4

# calibration study
python workflow/scripts/calibration.py --output-prefix calib --replicates 16
```
