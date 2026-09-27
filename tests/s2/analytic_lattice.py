"""Manufactured translation-invariant crystal, strictly test-only.

The two sites of a cubic unit cell are coupled by eight isotropic vector springs.
Optional same-sublattice axial springs produce a dispersive unstable optical
branch with stable regions elsewhere in the Brillouin zone. This is an analytic
interface fixture, not a material model and never a production provider.
"""
from itertools import product

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


def lattice_atoms(a=3.0):
    return Atoms("NaCl", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                 cell=np.eye(3) * a, pbc=True, masses=[20.0, 40.0])


class AnalyticLattice(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    family = "synthetic"
    evidence_kind = "analytic_test"

    def __init__(self, reference=None, *, spring=1.0, same_sublattice_spring=0.0, drift=(0.0, 0.0, 0.0)):
        super().__init__()
        self.reference = (reference if reference is not None else lattice_atoms()).copy()
        self.spring = spring
        self.same_spring = same_sublattice_spring
        self.drift = np.asarray(drift)
        self.forward_calls = 0

    def bonds(self, atoms):
        base_cell = self.reference.cell.array
        inverse = np.linalg.inv(base_cell)
        basis = self.reference.get_scaled_positions(wrap=False)
        ideal_positions, sublattices, translations = [], [], []
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
            fractional = position @ inverse
            candidates = []
            for index, base_position in enumerate(basis):
                if self.reference[index].symbol != symbol:
                    continue
                translation = np.rint(fractional - base_position).astype(int)
                ideal = (translation + base_position) @ base_cell
                candidates.append((np.linalg.norm(position - ideal), index, ideal, translation))
            _, index, ideal, translation = min(candidates, key=lambda candidate: candidate[0])
            ideal_positions.append(ideal)
            sublattices.append(index)
            translations.append(translation)
        ideal_positions = np.array(ideal_positions)
        ideal_supercell = np.rint(atoms.cell.array @ inverse).astype(int) @ base_cell
        inverse_supercell = np.linalg.inv(ideal_supercell)

        def destination(target, sublattice):
            for index, (position, candidate) in enumerate(zip(ideal_positions, sublattices)):
                if sublattice != candidate:
                    continue
                image = np.rint((target - position) @ inverse_supercell).astype(int)
                if np.allclose(target, position + image @ ideal_supercell, atol=1e-9, rtol=0):
                    return index, image
            raise ValueError("Manufactured lattice mapping failed")

        bonds = []
        for index, (sublattice, translation) in enumerate(zip(sublattices, translations)):
            if sublattice == 0:
                for offset in product((-1, 0), repeat=3):
                    target = (translation + np.array(offset) + basis[1]) @ base_cell
                    neighbour, image = destination(target, 1)
                    bonds.append((index, neighbour, image, target - ideal_positions[index], self.spring))
            if self.same_spring:
                for offset in np.eye(3, dtype=int):
                    target = (translation + offset + basis[sublattice]) @ base_cell
                    neighbour, image = destination(target, sublattice)
                    bonds.append((index, neighbour, image, target - ideal_positions[index], self.same_spring))
        return bonds

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.forward_calls += 1
        energy = 0.0
        forces = np.zeros((len(atoms), 3))
        stress = np.zeros((3, 3))
        for first, second, image, ideal, spring in self.bonds(atoms):
            bond = atoms.positions[second] + image @ atoms.cell.array - atoms.positions[first]
            displacement = bond - ideal
            energy += 0.5 * spring * np.dot(displacement, displacement)
            forces[first] += spring * displacement
            forces[second] -= spring * displacement
            stress += spring * np.outer(displacement, bond)
        stress = (stress + stress.T) / (2 * atoms.get_volume())
        self.results = {"energy": float(energy), "forces": forces + self.drift,
                        "stress": stress.flat[[0, 4, 8, 5, 2, 1]]}

    def exact_ifc(self, atoms):
        """Independent graph Hessian; never calls Phonopy or numerical fitting."""
        constants = np.zeros((len(atoms), len(atoms), 3, 3))
        for first, second, _image, _ideal, spring in self.bonds(atoms):
            constants[first, first] += spring * np.eye(3)
            constants[second, second] += spring * np.eye(3)
            constants[first, second] -= spring * np.eye(3)
            constants[second, first] -= spring * np.eye(3)
        return constants

    def exact_frequencies(self, qpoints, factor):
        """Independent analytic 2 x 2 Bloch matrix, tripled Cartesian branches."""
        mass_a, mass_b = self.reference.get_masses()
        output = []
        for qpoint in np.asarray(qpoints):
            diagonal = 8 * self.spring + 2 * self.same_spring * np.sum(1 - np.cos(2 * np.pi * qpoint))
            offdiagonal = -8 * self.spring * np.prod(np.cos(np.pi * qpoint))
            dynamical = [[diagonal / mass_a, offdiagonal / np.sqrt(mass_a * mass_b)],
                         [offdiagonal / np.sqrt(mass_a * mass_b), diagonal / mass_b]]
            eigenvalues = np.linalg.eigvalsh(dynamical)
            signed = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * factor
            output.append(np.sort(np.repeat(signed, 3)))
        return np.asarray(output)
