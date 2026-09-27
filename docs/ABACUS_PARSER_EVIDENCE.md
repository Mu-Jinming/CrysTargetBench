# ABACUS parser evidence

Evidence scope: **authored synthetic parser fragments only**. No actual ABACUS engine output or licensed reference dataset has been acquired. No engine was installed or executed; no pseudopotential/orbital was downloaded, copied, or generated. The test fixtures are not reference DFT or materials-science validation.

The candidate profile pins 3.7.4. The official [3.7.4 running-log documentation](https://abacus.deepmodeling.com/en/v3.7.4/advanced/output_files/running_scf.log.html) specifies labelled final energy, forces, stress, completion/convergence and k-point tables. That page's own sample header says 3.7.0, so documentation-path version is not claimed as proof of actual 3.7.4 execution. The adapter conservatively accepts one documented-format profile; a genuine version/build validation remains outstanding.

The [3.7.4 STRU specification](https://abacus.deepmodeling.com/en/v3.7.4/advanced/input_files/stru.html) defines lattice scale and Direct/Cartesian coordinates. The implementation independently writes a Bohr lattice scale corresponding to one angstrom, groups species stably, records atom mapping and checks round-trip conversion. The force tensor is normalized to original atom order; raw compressive-positive kbar stress is sign-flipped and converted to ASE units. Synthetic analytic unit tests check both ordering and numerical factors.

The [3.9.0 output introduction](https://abacus.deepmodeling.com/en/v3.9.0/quick_start/output.html) documents the istate table shape. This is cross-version format guidance, explicitly **not** a 3.7.4 reference fixture. Our grammar candidate reads indexed bands/energies/occupations and cross-checks the actual direct k-point table; nspin=1 weighted occupations are normalized to per-state values. The supported unreduced grid has equal weights summing to two before normalization. An independently obtained 3.7.4 output is needed before widening compatibility or declaring reference_tested.

Sources were read on 2026-09-25; no third-party source or real material output is copied into the package. Authored strings exercise schema, complete/partial records, energy/force/stress units, relaxation posterior checks, mapping and bands. Full synthetic examples live under `docs/s3/examples/synthetic/abacus`; `reference/README.md` records the absence of real reference evidence.

| Evidence layer | Status |
| --- | --- |
| Software interface and preparation | implemented |
| Generic contract and negative tests | tested |
| Authored parser fragments | tested, synthetic |
| Genuine fixed-build reference output | not available / not tested |
| Local ABACUS execution | not_run |
| Electronic/Born/ionic response parser | not implemented |
| Complete real High-K | blocked_capability |
| Material accuracy / scientific eligibility | false |

A receipt binds externally reported effective settings and actual input/output hashes; it does not independently authenticate who ran an engine. CTB rejects mismatches rather than silently adopting different settings. Ionic qualification requires reported ionic convergence and actual posterior F/stress limits. A normal process finish alone never qualifies a result.

The pinned [Phonopy interface documentation](https://phonopy.github.io/phonopy/abacus.html) describes the external-force pattern; CTB continues using its previously pinned 2.38.2 numerical implementation. Its standardized angstrom/eV-per-angstrom route has eV/angstrom² IFCs; it is not interchangeable with ABACUS-native Bohr-length IFCs. The S3 numerical check uses analytic pair-spring forces and is labelled analytic_test/synthetic throughout.
