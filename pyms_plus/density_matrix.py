"""Projected transition-potential density matrices using pyms_plus propagation.

The specimen and optical setup is prepared locally; both elastic and ionized
waves are propagated by :func:`pyms_plus.multislice`.
"""

import numpy as np
import torch

from .py_multislice import multislice


def _shift_potentials(ion_fft, positions, shape):
    ny, nx = shape
    fy = torch.fft.fftfreq(ny, device=positions.device)
    fx = torch.fft.fftfreq(nx, device=positions.device)
    phase_y = torch.exp(-2j * torch.pi * ny * positions[:, 0:1] * fy[None])
    phase_x = torch.exp(-2j * torch.pi * nx * positions[:, 1:2] * fx[None])
    return torch.fft.ifftn(
        phase_y[:, None, :, None] * phase_x[:, None, None, :]
        * ion_fft[None], dim=(-2, -1)
    )


def calculate_dm(
    structure, gridshape, eV, app, thicknesses, Ztarget, n, ell, epsilon,
    Hn0=None, coeffs=None, slice_thickness=1.0, device_type=None,
    tiling=(1, 1), dtype=torch.float32, beam_tilt=(0, 0),
    specimen_tilt=(0, 0), tilt_units="mrad", aberrations=(),
    showProgress=True, ionization_cutoff=None, nT=5, P=None, T=None,
    axis=0, limit=1, fast=True, return_components=False,
):
    """Calculate the projected density matrix with pyms_plus multislice.

    The signature preserves the original transition-potential simulation API.
    ``Hn0`` must contain the transition potentials.  ``n``, ``ell``,
    ``epsilon`` and ``ionization_cutoff`` are retained for call compatibility.
    When ``return_components`` is true, return ``(total, components)`` where
    ``components[i]`` is the density matrix from ``Hn0[i]`` alone.
    """
    from .density_matrix_setup import (
        plane_wave_illumination, make_contrast_transfer_function,
        multislice_precursor, size_of_bandwidth_limited_array,
        tile_out_ionization_image, tqdm_handler,
    )

    if Hn0 is None:
        raise ValueError("Hn0 transition potentials are required")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if not (0 < limit <= 1):
        raise ValueError("limit must be in (0, 1]")
    device = torch.device(device_type or ("cuda" if torch.cuda.is_available() else "cpu"))
    complex_dtype = torch.complex64 if dtype in (torch.float32, torch.complex64) else torch.complex128
    gridshape = tuple(int(v) for v in gridshape)
    rsize = np.asarray(structure.unitcell[:2]) * np.asarray(tiling)
    nslices = int(np.ceil(float(thicknesses) / slice_thickness))
    probe = torch.as_tensor(
        plane_wave_illumination(gridshape, rsize, eV, beam_tilt, tilt_units),
        dtype=complex_dtype, device=device,
    )
    outshape = size_of_bandwidth_limited_array(gridshape, limit=limit)
    ctf = torch.as_tensor(
        make_contrast_transfer_function(
            outshape, rsize, eV, app, df=0, aberrations=aberrations,
        ), dtype=complex_dtype, device=device,
    )
    coords = np.array(structure.get_target_positions(Ztarget), copy=True)
    coords[:, :2] /= np.asarray(tiling)
    h = torch.as_tensor(np.stack(Hn0) if isinstance(Hn0, list) else Hn0,
                        dtype=complex_dtype, device=device)
    if h.ndim == 2:
        h = h[None]
    if coeffs is not None:
        h = h * torch.as_tensor(coeffs, dtype=complex_dtype, device=device)[:, None, None]
    h_fft = torch.fft.fftn(h, dim=(-2, -1))
    positions = torch.as_tensor(coords[:, :2].astype(np.float32), device=device)
    atom_slices = np.clip((coords[:, 2] / slice_thickness).astype(int), 0, nslices - 1)
    length = outshape[0] if axis == 0 else outshape[1]
    total = np.zeros((length, length), dtype=np.complex64 if complex_dtype == torch.complex64 else np.complex128)
    component_total = np.zeros((h.shape[0], length, length), dtype=total.dtype) if return_components else None
    disabled, tqdm = tqdm_handler(showProgress)

    with torch.no_grad():
        for fph in tqdm(range(max(1, nT)), desc="Frozen phonon iterations:", disable=disabled):
            if P is None or T is None:
                P, T, nslices = multislice_precursor(
                    structure, gridshape, eV, float(thicknesses), slice_thickness,
                    tiling, specimen_tilt, tilt_units, nT, device, dtype,
                    showProgress, [limit, limit],
                )
            prop = torch.as_tensor(P, dtype=complex_dtype, device=device)
            transmissions = torch.as_tensor(T, dtype=complex_dtype, device=device)
            # The legacy forward pass selects one frozen-phonon grating per
            # slice with a single seeded RandomState.  Freeze that selection
            # before invoking pyms_plus on each wave batch.
            rng = np.random.RandomState(fph)
            chosen = np.array([rng.randint(transmissions.shape[0]) for _ in range(nslices)])
            selected = transmissions[chosen, np.arange(nslices)][None]
            seeds = np.zeros(nslices, dtype=np.int64)
            wave = probe
            dm = torch.zeros((length, length), dtype=complex_dtype, device=device) if not return_components else None
            dm_components = torch.zeros((h.shape[0], length, length), dtype=complex_dtype, device=device) if return_components else None
            for iz in tqdm(range(nslices), desc="Density-matrix slices", disable=disabled):
                atom_ids = np.nonzero(atom_slices == iz)[0]
                if len(atom_ids):
                    shifted = _shift_potentials(h_fft, positions[atom_ids], gridshape)
                    ionized = (shifted * wave[None, None]).reshape(-1, *gridshape)
                    exit_wave = multislice(
                        ionized, np.arange(iz, nslices), prop, selected,
                        device_type=device, seed=seeds, return_numpy=False,
                        output_to_bandwidth_limit=False,
                    )
                    filtered = torch.fft.ifftn(
                        torch.fft.fftn(exit_wave, dim=(-2, -1)) * ctf,
                        dim=(-2, -1),
                    )
                    if limit < 1:
                        from .utils.torch_utils import crop_to_bandwidth_limit_torch
                        filtered = crop_to_bandwidth_limit_torch(
                            filtered, limit=limit, qspace_in=False, qspace_out=False,
                        )
                    if return_components:
                        per_atom = filtered.reshape(len(atom_ids), h.shape[0], *filtered.shape[-2:])
                        if axis == 0:
                            dm_components += torch.einsum("acij,ackj->cik", per_atom, per_atom.conj())
                        else:
                            dm_components += torch.einsum("acji,acjk->cik", per_atom, per_atom.conj())
                    elif axis == 0:
                        dm += torch.einsum("bij,bkj->ik", filtered, filtered.conj())
                    else:
                        dm += torch.einsum("bji,bjk->ik", filtered, filtered.conj())
                wave = multislice(
                    wave, np.array([iz]), prop, selected, device_type=device,
                    seed=seeds, return_numpy=False, output_to_bandwidth_limit=False,
                )
            norm = outshape[1] if axis == 0 else outshape[0]
            dm_tiling = [tiling[0]] * 2 if axis == 1 else [tiling[1]] * 2
            if return_components:
                for component in range(h.shape[0]):
                    component_total[component] += tile_out_ionization_image(
                        (dm_components[component] / norm).cpu().numpy()[None], dm_tiling
                    )[0]
                total = component_total.sum(axis=0)
            else:
                total += tile_out_ionization_image((dm / norm).cpu().numpy()[None], dm_tiling)[0]
    return (total, component_total) if return_components else total
