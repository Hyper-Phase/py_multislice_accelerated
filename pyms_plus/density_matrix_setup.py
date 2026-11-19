"""ASE/abTEM specimen preparation for transition-potential multislice."""

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from abtem import Potential

from .Probe import make_contrast_transfer_function, plane_wave_illumination
from .utils.numpy_utils import bandwidth_limit_array, fourier_shift


def tqdm_handler(show_progress):
    if isinstance(show_progress, str) and show_progress.lower() == "notebook":
        from tqdm.notebook import tqdm
        return False, tqdm
    from tqdm import tqdm
    return not show_progress if isinstance(show_progress, bool) else False, tqdm


def size_of_bandwidth_limited_array(shape, limit=1):
    return [int(round(s * limit)) for s in shape]


def tile_out_ionization_image(image, tiling):
    tiled = np.zeros_like(image)
    for y in range(tiling[0]):
        for x in range(tiling[1]):
            tiled += fourier_shift(image, [y / tiling[0], x / tiling[1]], pixel_units=False)
    return tiled


class StructureASE:
    """An ASE crystal with fixed-slice abTEM transmission functions."""

    def __init__(self, atoms):
        self.atoms = atoms.copy()
        self.unitcell = np.diag(np.asarray(self.atoms.cell)).copy()
        self._positions = self.atoms.get_positions()
        self._numbers = self.atoms.get_atomic_numbers()
        if "debye_waller_factor" not in self.atoms.arrays:
            dwf = next((np.asarray(self.atoms.get_array(key), dtype=float)
                        for key in ("dwf", "u_iso", "u", "U_iso")
                        if key in self.atoms.arrays), None)
            if dwf is None:
                dwf = np.full(len(atoms), 0.01)
            self.atoms.new_array("debye_waller_factor", np.clip(dwf, 0, None))

    def get_target_positions(self, atomic_number):
        positions = self._positions[self._numbers == atomic_number].copy()
        positions[:, :2] /= self.unitcell[:2]
        return positions

    def make_transmission_functions_abtem(
        self, gridshape, eV, total_thickness, slice_thickness=1.0,
        tiling=(1, 1), displacements=False, seed=None, projection="finite",
        bandwidth_limit=2 / 3, device=None, dtype=torch.float32,
        parametrization="kirkland", fftout=False,
    ):
        nslices = int(np.ceil(total_thickness / slice_thickness))
        nz = int(np.ceil(total_thickness / self.unitcell[2]))
        atoms = self.atoms.repeat((tiling[0], tiling[1], nz)) if tuple(tiling) != (1, 1) or nz > 1 else self.atoms.copy()
        if displacements:
            rng = np.random.default_rng(seed)
            dwf = np.asarray(atoms.get_array("debye_waller_factor"), dtype=float)
            positions = atoms.get_positions().copy()
            positions[:, :2] += rng.normal(scale=np.sqrt(np.clip(dwf, 0, None))[:, None], size=(len(atoms), 2))
            atoms.set_positions(positions)
        real_dim = np.diag(np.asarray(atoms.cell))[:2]
        sampling = [real_dim[i] / gridshape[i] for i in range(2)]
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        potential_slices = np.zeros((nslices, *gridshape), dtype=np.float32)
        for i in range(nslices):
            zmin = i * slice_thickness
            zmax = min((i + 1) * slice_thickness, total_thickness)
            inside = (positions[:, 2] >= zmin) & (positions[:, 2] < zmax)
            if i == nslices - 1:
                inside = (positions[:, 2] >= zmin) & (positions[:, 2] <= total_thickness)
            if not inside.any():
                continue
            local_positions = positions[inside].copy()
            local_positions[:, 2] -= zmin
            dz = zmax - zmin
            slice_atoms = Atoms(
                symbols=[symbol for symbol, selected in zip(symbols, inside) if selected],
                positions=local_positions, cell=[*real_dim, dz + 0.1],
                pbc=[True, True, False],
            )
            potential = Potential(
                slice_atoms, sampling=sampling, slice_thickness=dz + 0.1,
                projection=projection, parametrization=parametrization,
            )
            built = potential.build().array
            if hasattr(built, "compute"):
                built = built.compute(scheduler="single-threaded")
            plane = np.sum(np.asarray(built), axis=0)
            if plane.shape != tuple(gridshape):
                from scipy.ndimage import zoom
                plane = zoom(plane, [gridshape[j] / plane.shape[j] for j in range(2)], order=1)
            potential_slices[i] = plane
        m0c2 = 5.109989461e5
        hc = 1.23984193e4
        gamma = 1 + eV / m0c2
        wavelength = hc / np.sqrt(eV * (2 * m0c2 + eV))
        sigma = 2 * np.pi * gamma * wavelength / (hc**2 / m0c2)
        transmission_fft = np.fft.fft2(np.exp(1j * sigma * potential_slices), axes=(-2, -1))
        transmission = np.stack([
            bandwidth_limit_array(plane, limit=bandwidth_limit, qspace_in=True, qspace_out=fftout)
            for plane in transmission_fft
        ])
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        return torch.as_tensor(transmission, dtype=complex_dtype, device=device)


def load_structure(filename, repeat=(1, 1, 1)):
    atoms = read(filename)
    if tuple(repeat) != (1, 1, 1):
        atoms = atoms.repeat(repeat)
    return StructureASE(atoms)


def multislice_precursor(
    structure, gridshape, eV, total_thickness, slice_thickness=1.0,
    tiling=(1, 1), specimen_tilt=(0, 0), tilt_units="mrad", nT=5,
    device=None, dtype=torch.float32, showProgress=True,
    band_width_limiting=(2 / 3, 2 / 3),
):
    nslices = int(np.ceil(total_thickness / slice_thickness))
    rsize = np.asarray(structure.unitcell[:2]) * np.asarray(tiling)
    gridmax = np.asarray(gridshape) / rsize / 2
    app = np.hypot(*gridmax)
    propagator = bandwidth_limit_array(
        make_contrast_transfer_function(
            gridshape, rsize, eV, app, df=slice_thickness,
            app_units="invA", optic_axis=specimen_tilt, tilt_units=tilt_units,
        ), limit=band_width_limiting[0],
    )
    disabled, tqdm = tqdm_handler(showProgress)
    n_phonons = max(1, nT)
    transmissions = torch.empty(
        (n_phonons, nslices, *gridshape),
        dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
        device=device,
    )
    for i in tqdm(range(n_phonons), desc="Making projected potentials", disable=disabled):
        transmissions[i] = structure.make_transmission_functions_abtem(
            gridshape, eV, total_thickness, slice_thickness, tiling,
            displacements=nT > 1, seed=i if nT > 1 else None,
            bandwidth_limit=band_width_limiting[1], device=device, dtype=dtype,
        )
    return propagator, transmissions, nslices
