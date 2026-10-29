# DTA — discrete trait analysis on dated phylogenies

DTA reconstructs the history of a discrete trait on a posterior distribution of
dated trees.  It fits a continuous-time Markov chain (CTMC) in BEAST, uses Markov
jump registers for directional transition counts, samples the *complete*
transition history so that individual acquisition and loss events can be placed
in calendar time, and summarises prevalence over time and TMRCAs.

A trait is any categorical genome feature, and there is one analysis.  The only
choice is where the states come from:

| source | state comes from | use for |
| --- | --- | --- |
| `label` | a field of each taxon label (`A\|India\|...`) | geography, or any state already encoded in the labels |
| `table` | an annotation sheet (`taxon,state`) | gene acquisitions/losses, resistance mutations, phenotypes |

Both sources produce the same artefacts: ancestral states, directional
transition counts, complete transition histories, transition dates, prevalence
over time and per-state TMRCAs.

## Installation

DTA is a Snakemake workflow, packaged as a CLI with
[snk](https://snk.wytamma.com/):

```bash
pip install snk                     # or: conda create -n snk bioconda::snk
snk install wytamma/dta             # installs the `dta` command

# from a local checkout instead:
snk install . --name dta --editable
```

`dta --help` lists the subcommands, and `dta run --help` lists the workflow
options (generated from `config/config.yaml`).  The workflow runs BEAST 1.10.4,
Python and R from the conda environments in `workflow/envs`, which snk builds
when conda is available.

## Usage

Running with no options executes the simulated example that ships in
`resources/`:

```bash
dta run --cores 8
```

### States from an annotation sheet

```bash
dta run \
    --input-trees resources/simulated.trees \
    --trait-data resources/simulated.traits.csv \
    --trait cfrD \
    --trait-model asymmetric \
    --log-complete-history \
    --mrsd 2021-06-30 \
    --prefix results/cfrD
```

That runs the simulated example shipped in `resources/`; swap in your own trees
and sheet for a real analysis.  `--mrsd` is needed here only because the
simulated tips are labelled `t1..t100`; when the labels carry dates the newest
one is used and the flag can be left out.  `data/traits.csv` is a table with one
row per taxon:

```csv
taxon,cfrD
EPI_ISL_2546697,1
EPI_ISL_2521906,0
EPI_ISL_2131520,NA
```

Taxa with no observation (or a missing token such as `NA`) are passed to BEAST as
unknown (`?`), which is the correct treatment for a fixed tree.  Taxon labels are
matched to the sheet by exact string, by a delimited field or by a regular
expression (`--trait-match-mode`).

### States from the taxon labels

```bash
dta run \
    --input-trees data/delphy.trees \
    --trait-source label \
    --trait location \
    --trait-label-index 1 \
    --prefix results/dta \
    --resource data
```

`--trait-label-index 1` takes field 1 of each `|`-separated label
(`A|India|...`).  The trait is named `location`, which reproduces the historical
`location.*` parameter names; pass `--trait-pair-prefix Location` if you have
downstream scripts that read the old `c_Location.*` registers.  The calendar
axis needs no `--mrsd` here: the labels end in the sampling date, so the most
recent one is used (pass `--mrsd` to override, and a mismatch is reported).

### Inputs and resources

`--input-trees` and `--trait-data` are resolved in the working directory.  Files
that live inside the workflow are copied there for the run (and removed
afterwards) with `--resource <path>`, which copies by basename:

```bash
dta run --input-trees delphy.trees --resource data/delphy.trees   # -> ./delphy.trees
dta run --input-trees data/delphy.trees --resource data           # -> ./data/delphy.trees
```

The workflow's own `resources/` folder is copied automatically, which is why
`dta run` works out of the box.

### Everything else

Unrecognised arguments go to Snakemake, e.g. `--use-conda`, `--rerun-triggers`,
`--profile slurm` (profiles live in `workflow/profiles`).  `dta run -n` is a dry
run, `dta run --dag dag.pdf` writes the DAG, `dta config` prints the resolved
configuration and `dta run --config my.yaml` overrides it from a file.

## Options

Every key in `config/config.yaml` can be set as `--<key-with-dashes>`; the
annotated subset appears in `dta run --help`.  The most useful:

| key | meaning |
| --- | --- |
| `input_trees` | NEXUS file with dated trees (branch lengths in years) |
| `prefix` | output prefix: `results/dta` writes `results/dta.*` |
| `mrsd` | most recent sampling date, `YYYY-MM-DD`; read off dated tip labels when omitted |
| `trait_source` | `label` or `table` |
| `trait` | the trait's name; becomes the BEAST id prefix (`location` for label-sourced runs) |
| `trait_data` | (source `table`) annotation sheet, `taxon,state` |
| `trait_label_index` | (source `label`) which `\|`-separated label field holds the state |
| `trait_match_mode` | how a tree label is matched to a sheet row: `exact`, `field`, `regex` |
| `trait_model` | `asymmetric` (default), `equal` (symmetric/ER), `bssvs` |
| `trait_root_state` | `uniform`, `empirical`, `fixed:<state>` |
| `trait_states` | fix the state order, e.g. `0,1` |
| `trait_pair_prefix` | prefix of the ordered pair registers (default: the trait name) |
| `log_complete_history` | sample every transition and annotate every branch (default: on) |
| `burnin_fraction` | fraction of each chain discarded as burn-in, default `0.1` |
| `input_burnin_fraction` | fraction of the leading input trees dropped before BEAST samples them, default `0.1` |
| `introduction_states` | state(s) whose entries are marked on the MCC figure, e.g. `"Bheri"`; empty draws the tree without markers |
| `figure_ext` | `.svg` (needs cairo), `.png` or `.pdf` |
| `beast_chain_length`, `beast_samples` | MCMC length and log frequency |
| `beast_chains` | independent chains, combined with `logcombiner` |
| `beast_bin`, `beast_flags` | BEAST executable and flags (add `-beagle_GPU` to use a GPU) |

## Outputs

| file | contents |
| --- | --- |
| `<prefix>.log` | BEAST log: rates, root frequencies, Markov jump counts (`c_Into.*`, `c_OutOf.*`, `c_<trait>.<from>.<to>`) |
| `<prefix>.trees` | trees with reconstructed ancestral states at every node |
| `<prefix>.jumpHistory.log` | complete transition histories (one line per sample) |
| `<prefix>.events.tsv` | every transition of every retained sample, with height and date |
| `<prefix>.event_counts.tsv` | posterior number of transitions per direction (mean, 95% CI, `P(>0)`) |
| `<prefix>.event_dates.tsv` | posterior transition dates per direction (earliest, latest, pooled) |
| `<prefix>.acquisitions.tsv` | per state: first transition into it from any source, and `P(acquired)` |
| `<prefix>.root_states.tsv` | posterior root state probabilities |
| `<prefix>.prevalence.tsv` | share of reconstructed lineages carrying each state through time |
| `<prefix>.tmrca.tsv` | per state: emergence date (MRCA of the carriers) and how often the state appears |
| `<prefix>.tmrca_samples.tsv` | per-sample emergence dates (the data behind the density plot) |
| `<prefix>.count_check.tsv` | complete histories vs Markov jump registers (consistency check) |
| `<prefix>.ess.tsv` | effective sample size for every logged parameter |
| `<prefix>.mcc.mean.nex`, `.mcc.mean<ext>` | maximum clade credibility tree, rendered with ancestral states |
| `<prefix>.event_counts.png`, `.event_dates.png`, `.root_states.png` | trait figures |
| `<prefix>.trait_through_time.png` | prevalence over time + per-state emergence-date densities |
| `<prefix>.trait_arrivals.png` | prevalence + when each state was entered (all sampled arrivals; stays informative when the carriers are paraphyletic) |
| `<prefix>.mcc.mean.<trait>.introductions.png/.pdf/.csv` | MCC tree coloured by state, with the entries into `introduction_states` marked and dated |
| `<prefix>.circlize<ext>` | chord diagram of mean transition counts |

## Notes and limitations

* **Numeric state codes.** BEAST 1.10's `completeHistoryLogger` crashes when a
  state code parses as a number, because it casts its own parsed values back to
  strings.  DTA therefore rewrites such codes (`0` → `s0`) before generating the
  XML and maps them back in every summary; the mapping is recorded in
  `<prefix>.traits.json` (`state_code_map`).  Use `--trait-state-code-style
  as-is` to opt out (and then do not enable complete history logging).
* The reconstructed acquisition date is the date of the *trait transition on the
  phylogeny*.  It is not the molecular origin of the underlying gene or allele.
* Trees are treated as fixed: DTA integrates over the supplied tree posterior
  rather than re-estimating the phylogeny.
* Two different dates are reported for a state.  The **emergence date**
  (`<prefix>.tmrca.tsv`) is the age of the most recent common ancestor of the
  tips carrying it, i.e. the crown of the carrier clade; the **acquisition
  date** (`<prefix>.event_dates.tsv`) is when the state first appeared on the
  tree, which is older.  They coincide only when the trait arose exactly at a
  node.
* The tree log is written to `<prefix>.trees`, so the input trees must not be at
  that path (the workflow refuses to overwrite them); keep the inputs in
  `resources/` or `data/` and the outputs in `results/`.
* The transition model defaults to `asymmetric` for every trait, including
  label-sourced geography, which used to default to `bssvs`.  Pass
  `--trait-model bssvs` for the old behaviour.

## Validation

`workflow/scripts/simulate_trait.py` generates datasets with a known history and
`workflow/scripts/calibration.py` runs the whole pipeline on them and reports
credible-interval coverage.  `docs/traits.md` documents the model, the event
dating arithmetic and the simulation results.

```bash
python -m pytest tests -q                                    # unit tests (fast)
BEAST_BIN=/path/to/beast python -m pytest tests -q -m slow    # + a real BEAST run
python workflow/scripts/calibration.py --output-prefix calib --replicates 16
```

## Sampling bias

https://academic.oup.com/ve/article/9/1/vead010/7028398#s6
