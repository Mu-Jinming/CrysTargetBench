# Third-party software and redistribution

CTB-authored source and authored synthetic fixtures use the repository MIT license. The ABACUS adapter is CTB-authored format-conversion code, not redistributed ABACUS source. No engine source/binary, pseudopotential, orbital or model weight is included. A subprocess boundary does not grant redistribution rights.

Runtime dependencies are installed by users under their own licenses: NumPy (BSD), ASE (LGPL), spglib (BSD), jsonschema (MIT); optional Phonopy (BSD), SeeK-path (MIT), SciPy (BSD), MatterSim and its dependency/model licenses apply separately. Check installed distributions' LICENSE/METADATA and upstream terms for the exact version before redistributing any dependency or model. Wheels here contain CTB and adapter code only, not dependency wheels.

ABACUS engine and asset installation are outside CTB. See the official [ABACUS documentation](https://abacus.deepmodeling.com/en/v3.7.4/) for user installation and format descriptions. The adapter's parser evidence cites official fixed documentation; its authored fragments contain no copied material-specific reference data. There is no acquired genuine ABACUS output fixture in this release, and no claim of live solver validation. See `docs/ABACUS_PARSER_EVIDENCE.md`.
