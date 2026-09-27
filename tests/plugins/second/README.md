# Second independently installed conformance plugin

`python -m pip install ./tests/plugins/second` installs `ctb-second-example`, provider `second_example`. Core has no import or dependency on it. It reads authored-observations.json using the same request/method/geometry contract, and always labels its observations synthetic. It never returns metrics or pass/fail, never runs a calculator, and cannot satisfy real DFT fidelity.

This is a test fixture and extension example, not a scientific backend. See tests/adapter_conformance for actual installed entry-point discovery, staged prepare/collect, full denominator, rank/screen and negative cases. For your own engine, implement the same three methods and static manifest in a separately installed distribution; do not edit the core registry.
