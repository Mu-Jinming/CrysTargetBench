# Tasks and metrics

A task (`ctb.task.v2`) names measurements, units, input or relaxed reference geometry, optional fidelity constraints, and predicates. It contains data, never imports, executables, Python code or provider installation URLs. Site configuration selects trusted software; the protocol defines how to measure; a predicate determines whether the measured value satisfies a target.

| Property ID | Definition / unit | Route / reference |
| --- | --- | --- |
| `space_group_number` | spglib space-group category, 1…230 | native; exact requested geometry |
| `band_gap_eV` | generalized Kohn–Sham occupied/unoccupied gap on the locked uniform k mesh, eV | DFT eigenvalues + occupations; not optical gap or path-only gap |
| `phonon_path_min_frequency_THz` | minimum signed harmonic frequency along locked path, THz | stationary reference, MLFF or DFT force constants |
| `phonon_mesh_min_frequency_THz` | minimum signed frequency on specified mesh, THz | separate scope from path; no full-temperature stability claim |
| `bulk_modulus_eos_GPa` | BM3 EOS derivative after fixed-shape internal relaxation, GPa | existing complete ASE EOS recipe; not Voigt |
| `bulk_modulus_voigt_GPa`, `bulk_modulus_reuss_GPa`, `bulk_modulus_hill_GPa` | distinct elastic definitions, GPa | capability remains blocked unless that recipe exists |
| `dielectric_electronic_trace_over3` | trace of static electronic relative permittivity / 3, dimensionless | real response, geometry/definition checked |
| `dielectric_ionic_bec_trace_over3` | ionic contribution / 3, dimensionless | compatible ionic response; not a nominal-charge proxy |
| `dielectric_total_static_trace_over3` | electronic + ionic tensor, trace / 3, dimensionless | static fixed-strain response, both components or proven total |

Bundled task thresholds are examples, not validated materials criteria. `measure` reports distributions/coverage only. `screen` reports the AND of explicit constraints. `constrained_rank` reports feasibility and ranks accepted ranking observations without inventing a κ cutoff or κ success rate.

For each submitted item: a qualified value satisfying a predicate is pass; a qualified violation is fail; missing, unsupported, unconverged, malformed or wrong-fidelity data are unknown. A reliable necessary-condition fail makes an AND fail even when downstream response is skipped. Calculation failure itself is never a material failure. Invalid inputs and duplicate entries stay in N; shared calculations may be deduplicated without dropping submissions.

`verified_yield = pass / N`; `unknown_fraction = unknown / N`; `determined_fraction = (pass+fail)/N`. `pass_fraction_determined` has its explicitly smaller denominator. Per-measurement coverage is accepted observations / N. Space groups use a histogram, never an arithmetic mean of their IDs. Continuous quantities retain accepted scalar distributions. Zero denominators yield null, not zero. A rank task reports `constraint_yield`, `rankable_yield`, ranking values and coverage; no artificial threshold is added.

External imports retain `synthetic`, `reference`, `reported` or `live` provenance per observation. Hash verification proves integrity and binding, not independent authentication or scientific accuracy. A synthetic plugin can exercise the scorer but cannot satisfy a DFT fidelity requirement or acquire `benchmark_eligible=true`. Native CPU analysis is real geometry analysis, not physics validation. All candidate profiles remain scientifically unvalidated.

External costs are null unless supplied with evidence; CTB file collection itself invokes zero engines. Generation/training cost and novelty/SUN remain unavailable without the corresponding trusted metadata/reference libraries. Report every target separately before any explicitly defined cross-target aggregation.
