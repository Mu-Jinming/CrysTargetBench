"""Wheel-only resource copy. Needs only installed CTB; never calls a backend."""
from importlib.resources import files
from pathlib import Path

root = Path('ctb-examples')
root.mkdir()  # Refuse to overwrite user files; use a fresh directory.
source = files('crystargetbench').joinpath('resources/examples')
for folder, names in {
    'structures': ['ideal_fcc.cif', 'ideal_bcc.cif'],
    'tasks': ['spacegroup_225.json', 'bandgap_measure.json', 'highk_screen.json'],
}.items():
    (root / folder).mkdir()
    for name in names:
        (root / folder / name).write_bytes(source.joinpath(folder, name).read_bytes())
print(root)
