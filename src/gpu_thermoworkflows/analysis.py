"""Numerical Hessians and fixed-configuration harmonic thermochemistry."""
import csv
from pathlib import Path
import numpy as np
from .config import BOHR_A, prepare
from .engines import verified
from .storage import read, save

R = 8.31446261815324
H = 6.62607015e-34
KB = 1.380649e-23
C = 299792458.0
AMU = 1.66053906660e-27
EH_J = 4.3597447222071e-18
EH_KJ = 2625.4996394799
MASSES = dict(H=1.00794, He=4.002602, Li=6.941, B=10.811, C=12.0107, N=14.0067,
              O=15.9994, F=18.998403163, Na=22.98976928, Mg=24.3050, Al=26.9815385,
              Si=28.0855, P=30.973761998, S=32.065, Cl=35.453, K=39.0983, Ca=40.078,
              Mn=54.938044, Fe=55.845, Co=58.933194, Ni=58.6934, Cu=63.546, Zn=65.38,
              Br=79.904, Cd=112.411, I=126.90447)


def projection(masses, positions, periodic):
    n = len(masses)
    vectors = []
    for axis in range(3):
        v = np.zeros((n, 3))
        v[:, axis] = np.sqrt(masses)
        vectors.append(v.ravel())
    if not periodic:
        centered = positions - np.average(positions, axis=0, weights=masses)
        for axis in np.eye(3):
            vectors.append((np.cross(axis, centered) * np.sqrt(masses)[:, None]).ravel())
    matrix = np.column_stack(vectors)
    u, s, _ = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(s > max(s[0] * 1e-10, 1e-12)))
    return u[:, rank:], rank


def vib_thermo(frequencies, temperature, cutoff=None):
    f = np.asarray(frequencies)
    x = H * C * 100 * f / (KB * temperature)
    occupation = 1 / np.expm1(x)
    energy = R * temperature * np.sum(x * (0.5 + occupation)) / 1000
    entropy_modes = R * (x * occupation - np.log(-np.expm1(-x)))
    if cutoff:
        nu = f * C * 100
        inertia = H / (8 * np.pi ** 2 * nu)
        effective = inertia * 1e-44 / (inertia + 1e-44)
        rotor = R * (0.5 + np.log(np.sqrt(8 * np.pi ** 3 * effective * KB * temperature / H ** 2)))
        weight = 1 / (1 + (cutoff / f) ** 4)
        entropy_modes = weight * entropy_modes + (1 - weight) * rotor
    entropy = float(np.sum(entropy_modes))
    return float(energy), entropy


def analyze(c, temperature=None):
    work = Path(c["paths"]["work_dir"])
    manifest = prepare(c)
    if not c["calculation"]["hessian"]:
        raise ValueError("Set hessian=true and complete displacements before frequency analysis")
    for task in manifest["tasks"]:
        if not verified(c, task, manifest["science_sha256"]):
            raise ValueError(f"Missing force point {task['id']}; resume the queue")
    n = len(c["atoms"])
    d = c["calculation"]["displacement"]
    step_bohr = d["value"] if d["unit"] == "bohr" else d["value"] / BOHR_A
    matrix = np.empty((3*n, 3*n))
    center = read(work / "tasks/center/result.json")
    curvature = []
    for q in range(3*n):
        minus = read(work / "tasks" / f"q{q:04d}_minus/result.json")
        plus = read(work / "tasks" / f"q{q:04d}_plus/result.json")
        matrix[:, q] = -(np.array(plus["forces_Ha_Bohr"]) - np.array(minus["forces_Ha_Bohr"])).ravel() / (2*step_bohr)
        energy_curvature = (plus["energy_Ha"] - 2*center["energy_Ha"] + minus["energy_Ha"]) / step_bohr**2
        curvature.append({"coordinate": q, "energy_curvature_Ha_Bohr2": energy_curvature,
                          "force_Hessian_diagonal_Ha_Bohr2": float(matrix[q, q])})
    norm = np.linalg.norm(matrix)
    asymmetry = float(np.linalg.norm(matrix - matrix.T) / norm) if norm else 0.0
    symmetric = (matrix + matrix.T) / 2
    settings = c.get("analysis", {})
    masses = {**MASSES, **settings.get("atomic_masses_amu", {})}
    try:
        mass = np.array([masses[a[0]] for a in c["atoms"]])
    except KeyError as error:
        raise ValueError(f"Provide analysis.atomic_masses_amu for {error.args[0]}") from error
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("Atomic masses must be finite and positive")
    positions = np.array([a[1] for a in c["atoms"]])
    weights = np.repeat(mass, 3)
    dynamic = symmetric / np.sqrt(weights[:, None] * weights[None, :])
    periodic = c["calculation"]["periodic"]
    basis, removed = projection(mass, positions, periodic)
    values = np.linalg.eigvalsh(basis.T @ dynamic @ basis)
    factor = np.sqrt(EH_J / (AMU * (BOHR_A * 1e-10)**2)) / (2*np.pi*C*100)
    frequencies = np.sign(values) * np.sqrt(np.abs(values)) * factor
    tolerance = settings.get("imaginary_tolerance_cm_1", 5.0)
    imaginary = frequencies[frequencies < -tolerance]
    usable = frequencies[frequencies > settings.get("zero_tolerance_cm_1", 1e-5)]
    excluded = frequencies[frequencies <= settings.get("zero_tolerance_cm_1", 1e-5)]
    t = temperature or settings.get("temperature_K", 298.15)
    if t <= 0 or not np.isfinite(t):
        raise ValueError("temperature must be finite and positive")
    output = work / "analysis"
    output.mkdir(exist_ok=True)
    np.save(output / "Hessian_Ha_Bohr2.npy", matrix)
    with (output / "frequencies_cm_1.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "signed_frequency_cm_1"])
        writer.writerows(enumerate(frequencies.tolist(), 1))
    result = {"science_sha256": manifest["science_sha256"], "engine": c["engine"],
              "synthetic": c["engine"] == "mock", "temperature_K": t,
              "geometry_optimized": False, "source_gradient_RMS_Ha_Bohr":
              float(np.sqrt(np.mean(np.array(center["forces_Ha_Bohr"])**2))),
              "removed_rigid_modes": removed, "signed_frequencies_cm_1": frequencies.tolist(),
              "imaginary_modes_beyond_tolerance": imaginary.tolist(), "excluded_nonpositive_modes": excluded.tolist(),
              "Hessian_relative_antisymmetry": asymmetry, "energy_force_curvature": curvature,
              "scope": "Gamma cell only; no Brillouin-zone integration" if periodic else "fixed molecular configuration",
              "treatments": {}}
    if len(imaginary) or asymmetry > settings.get("max_Hessian_antisymmetry", 0.005):
        result["state"] = "FREQUENCIES_REPORTED_THERMAL_CORRECTION_UNAVAILABLE"
    else:
        result["state"] = "FIXED_CONFIGURATION_THERMAL_ESTIMATE"
        gas_h = 0.0
        gas_s = 0.0
        standard = settings.get("standard_state", "1bar")
        if periodic:
            if standard != "1bar":
                raise ValueError("Periodic harmonic Helmholtz contribution has no molecular concentration conversion")
            divisor = settings.get("formula_units_per_cell", 1)
            if divisor <= 0:
                raise ValueError("formula_units_per_cell must be positive")
        else:
            divisor = 1
            pressure = settings.get("pressure_Pa", 100000.0)
            symmetry = settings.get("symmetry_number", 1)
            if pressure <= 0 or symmetry <= 0:
                raise ValueError("pressure and symmetry_number must be positive")
            molecular_mass = np.sum(mass) * AMU
            q_translation = (2*np.pi*molecular_mass*KB*t/H**2)**1.5 * KB*t/pressure
            gas_s = R * (np.log(q_translation) + 2.5)
            gas_h = 2.5 * R * t / 1000
            centered = (positions - np.average(positions, axis=0, weights=mass)) * 1e-10
            inertia = sum(AMU*m*(np.dot(v, v)*np.eye(3) - np.outer(v, v)) for m, v in zip(mass, centered))
            moments = np.linalg.eigvalsh(inertia)
            if n > 1:
                linear = removed == 5
                if linear:
                    q_rotation = 8*np.pi**2*np.mean(moments[1:])*KB*t/(symmetry*H**2)
                    gas_s += R*(np.log(q_rotation) + 1)
                    gas_h += R*t/1000
                else:
                    q_rotation = np.sqrt(np.pi)/symmetry*(8*np.pi**2*KB*t/H**2)**1.5*np.sqrt(np.prod(moments))
                    gas_s += R*(np.log(q_rotation) + 1.5)
                    gas_h += 1.5*R*t/1000
            if settings.get("include_spin_entropy", False):
                gas_s += R*np.log(c["calculation"]["multiplicity"])
        conversion = R*t*np.log(R*t*1000/settings.get("pressure_Pa", 100000))/1000 if standard == "1M" else 0
        if standard not in ("1bar", "1M"):
            raise ValueError("standard_state must be 1bar or 1M")
        result["standard_state"] = standard
        result["formula_units_divisor"] = divisor
        for cutoff in [None, *settings.get("qrrho_cutoffs_cm_1", [50, 100])]:
            u, s = vib_thermo(usable, t, cutoff)
            label = "RRHO" if cutoff is None else f"qRRHO{cutoff:g}"
            correction = (u + gas_h - t*(s + gas_s)/1000)/divisor + conversion
            result["treatments"][label] = {
                "G_correction_kJ_mol": correction,
                "G_total_Ha": center["energy_Ha"]/divisor + correction/EH_KJ,
                "H_correction_kJ_mol": (u+gas_h)/divisor,
                "S_J_mol_K": (s+gas_s)/divisor,
                "concentration_conversion_kJ_mol": conversion}
        if len(excluded):
            result["state"] = "THERMAL_ESTIMATE_EXCLUDING_REPORTED_NONPOSITIVE_MODES"
    save(output / "thermochemistry.json", result)
    return result


def compare(reference_path, candidate_path, energy_tolerance=1e-7, force_tolerance=1e-5):
    a, b = read(reference_path), read(candidate_path)
    if a["science_sha256"] != b["science_sha256"] or a["coordinates_A"] != b["coordinates_A"]:
        raise ValueError("Comparison requires matching scientific fingerprints and coordinates")
    energy = abs(a["energy_Ha"] - b["energy_Ha"])
    force = float(np.max(np.abs(np.array(a["forces_Ha_Bohr"]) - np.array(b["forces_Ha_Bohr"]))))
    return {"energy_difference_Ha": energy, "max_force_difference_Ha_Bohr": force,
            "accepted": energy <= energy_tolerance and force <= force_tolerance,
            "energy_tolerance_Ha": energy_tolerance, "force_tolerance_Ha_Bohr": force_tolerance}
