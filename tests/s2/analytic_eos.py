"""Test-only translationally invariant BM3 crystal with internal-coordinate DOF.

This manufactured energy has no trained weights, no physical-material claim,
and is not registered as a production MLFF. Stress uses ASE tensile-positive
units eV/angstrom^3 and forces are analytical Cartesian derivatives.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class AnalyticEOSCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    physics_fidelity = "synthetic"
    evidence_origin = "analytic_test"

    def __init__(self, *, v0=64.0, b0=0.5, bprime=4.2, e0=-8.0,
                 spring=3.0, internal_volume_coupling=0.2, shear=0.3,
                 before_evaluate=None, stress_bias=None, **kwargs):
        super().__init__(**kwargs)
        self.v0, self.b0, self.bprime, self.e0 = v0, b0, bprime, e0
        self.spring, self.internal_volume_coupling, self.shear = spring, internal_volume_coupling, shear
        self.before_evaluate = before_evaluate
        self.stress_bias = np.zeros((3, 3)) if stress_bias is None else np.asarray(stress_bias, dtype=float)

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        if self.before_evaluate:
            self.before_evaluate(atoms)
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        if len(atoms) != 2:
            raise ValueError("Manufactured EOS calculator requires its two-site test cell")
        cell = np.asarray(atoms.cell)
        volume = atoms.get_volume()
        x = (self.v0 / volume) ** (2 / 3)
        z = x - 1
        prefactor = 9 * self.v0 * self.b0 / 16
        energy_bm = self.e0 + prefactor * (2 * z**2 + (self.bprime - 4) * z**3)
        derivative_bm = prefactor * (4 * z + 3 * (self.bprime - 4) * z**2) * (-2 * x / (3 * volume))

        target = np.array([0.25 + self.internal_volume_coupling * (volume / self.v0 - 1), 0.25, 0.25])
        scaled = atoms.get_scaled_positions(wrap=False)
        fractional_error = scaled[1] - scaled[0] - target
        fractional_error -= np.rint(fractional_error)
        error = fractional_error @ cell
        internal_energy = 0.5 * self.spring * np.dot(error, error)
        forces = np.vstack([self.spring * error, -self.spring * error])
        target_prime_cartesian = np.array([self.internal_volume_coupling / self.v0, 0, 0]) @ cell
        stress_internal = self.spring * np.outer(error, error) / volume - self.spring * np.dot(error, target_prime_cartesian) * np.eye(3)

        shape = cell.T @ cell / volume ** (2 / 3)
        shape_error = shape - np.eye(3)
        shape_energy = self.shear * self.v0 / 4 * np.sum(shape_error**2)
        product = shape_error @ shape
        stress_shape = self.shear * self.v0 / volume * (product - np.eye(3) * np.trace(product) / 3)
        stress = np.eye(3) * derivative_bm + stress_internal + stress_shape + self.stress_bias
        self.results = {"energy": float(energy_bm + internal_energy + shape_energy),
                        "forces": forces, "stress": stress}


def manufactured_bm3(volumes, *, v0=64.0, b0=0.5, bprime=4.2, e0=-8.0):
    """Independent closed-form data used to check the actual ASE nonlinear fit."""
    volumes = np.asarray(volumes, dtype=float)
    strain = 0.5 * ((v0 / volumes) ** (2 / 3) - 1)
    return e0 + 4.5 * b0 * v0 * strain**2 * (1 + (bprime - 4) * strain)


from .analytic_lattice import AnalyticLattice


class AnalyticLatticeBM3(AnalyticLattice):
    """One manufactured model for shared-reference phonons and exact BM3 EOS.

    The volume-only term replaces the spring lattice's uniform-strain E(V)
    with BM3, with its matching isotropic stress derivative. It contributes no
    Cartesian force at fixed cell, so the independently known lattice IFC and
    nonzero-q acoustic/optical dispersion remain unchanged. Cell energy is
    extensive in the number of two-site primitive cells.
    """
    def __init__(self, *, b0=0.5, bprime=4.2, e0=-4.0):
        super().__init__()
        self.b0, self.bprime, self.e0 = b0, bprime, e0
        self.v0 = self.reference.get_volume()
        self.uniform_spring_coefficient = sum(
            0.5 * spring * np.dot(ideal, ideal)
            for _, _, _, ideal, spring in self.bonds(self.reference))

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cells = len(atoms) / len(self.reference)
        volume = atoms.get_volume() / cells
        scale = (volume / self.v0) ** (1 / 3)
        uniform_energy = self.uniform_spring_coefficient * (scale - 1)**2
        uniform_derivative = 2 * self.uniform_spring_coefficient * (scale - 1) * scale / (3 * volume)
        x = (self.v0 / volume) ** (2 / 3)
        z = x - 1
        prefactor = 9 * self.v0 * self.b0 / 16
        bm_energy = self.e0 + prefactor * (2 * z**2 + (self.bprime - 4) * z**3)
        bm_derivative = prefactor * (4 * z + 3 * (self.bprime - 4) * z**2) * (-2 * x / (3 * volume))
        self.results["energy"] += float(cells * (bm_energy - uniform_energy))
        self.results["stress"] = np.asarray(self.results["stress"]).copy()
        self.results["stress"][:3] += bm_derivative - uniform_derivative
