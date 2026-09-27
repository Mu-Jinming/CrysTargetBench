# Security and trust boundaries

Tasks, structures and returned JSON are untrusted data. The SDK bounds records and verifies paths, hashes, identities and units. Keep untrusted archives outside the working tree until you have checked their contents; CTB does not unpack arbitrary return archives. No pickle or task-controlled Python import is accepted.

Installed plugins and site executable configuration are trusted local code. Metadata discovery is not plugin sandboxing. Hash binding establishes integrity, not authenticity or scientific accuracy of external claims. Run only an executable you trust, in a least-privilege account with an explicit budget. The thin runner is not a hardened operating-system sandbox.

Ordinary bugs and usage questions belong in [Issues](https://github.com/Mu-Jinming/CrysTargetBench/issues). A private security-reporting channel has not yet been designated, and GitHub Private vulnerability reporting has not been verified as enabled. You may ask for a private contact in an ordinary issue, without disclosing vulnerability details. Do not post exploit details, enabled permits, credentials, private sessions or sensitive raw results publicly.

GitHub Private vulnerability reporting is the proposed channel, subject to owner approval and verification after it is enabled. No commit email or GitHub username is a designated security contact.
