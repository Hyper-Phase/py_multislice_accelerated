"""GPU transition-potential contraction for EFSTEM PRISM.

Fourier shifts are batched across atoms and components. Matrix products are
performed one atom and component at a time to bound GPU memory. Only
diffraction intensity is accumulated; no density matrix is constructed.
"""

import numpy as np
import torch

def window_indices(center, windowsize, gridshape):
    """Generate flattened indices for a cropped periodic window."""
    window = []
    for i, wind in enumerate(windowsize):
        indices = np.arange(-wind // 2, wind // 2, dtype=int) + wind % 2
        indices += int(np.floor(center[i] * gridshape[i]))
        indices = np.mod(indices, gridshape[i])
        window.append(indices)
    return (window[0][:, None] * gridshape[0] + window[1][None, :]).ravel()


@torch.no_grad()
def accumulate_transition_intensity(
    output, S1, S2, atoms, Hn0s, crop, scan, scan_array, prism_factor,
    atom_batch_size=4,
    scan_batch_size=1024,
    progress_callback=None,
):
    """Add one slice's ionization intensity to ``output`` in place."""
    if len(atoms) == 0:
        return
    device = S1.S.device
    grid = S1.stored_gridshape
    hy, hx = Hn0s.shape[-2:]
    # Keep NumPy's complex128 shift precision until casting to the wave dtype.
    potentials = torch.as_tensor(np.asarray(Hn0s), device=device)
    fy = torch.fft.fftfreq(hy, device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(hx, device=device, dtype=torch.float64)
    s1 = S1.S.reshape(S1.S.shape[0], -1)
    s2 = S2.S.reshape(S2.S.shape[0], -1)
    scan_flat = scan.reshape(-1, 2)
    factor = np.asarray(prism_factor)
    full_scan = np.all(factor == 1)
    full_window = tuple(crop) == tuple(grid) and grid[0] == grid[1]
    selected_all = scan_array.to(S1.S.dtype).T if full_scan else None
    completed = 0

    for start in range(0, len(atoms), atom_batch_size):
        positions = np.asarray(atoms[start:start + atom_batch_size])
        indices = np.stack([window_indices(p, crop, grid) for p in positions])
        indices = torch.as_tensor(indices, device=device, dtype=torch.long)
        remainder = np.remainder(positions * np.asarray(grid), 1.0)
        shifts = torch.as_tensor(remainder, device=device, dtype=torch.float64)
        phase_y = torch.exp(-2j * torch.pi * shifts[:, 0, None] * fy)
        phase_x = torch.exp(-2j * torch.pi * shifts[:, 1, None] * fx)
        shifted = torch.fft.ifft2(
            potentials[None] * phase_y[:, None, :, None] * phase_x[:, None, None, :]
        )
        shifted = torch.fft.fftshift(shifted, dim=(-2, -1)).reshape(
            len(positions), len(Hn0s), -1
        ).to(S1.S.dtype)

        for i, atom in enumerate(positions):
            if full_scan:
                selected = selected_all
                scan_indices = None
            else:
                wrapped = np.abs((scan_flat - atom + 0.5) % 1.0 - 0.5)
                mask = np.all(wrapped <= 1 / factor / 2, axis=1)
                if not np.any(mask):
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, len(atoms))
                    continue
                selected = scan_array[mask].to(S1.S.dtype).T
                scan_indices = torch.as_tensor(np.flatnonzero(mask), device=device)
            # Keep only one detector window in memory. Broadcasting it across
            # all atoms and transitions requires several GiB on large grids.
            if full_window:
                # The crop indices are a permutation of the entire grid.
                # Reorder the small potential instead of copying both large
                # scattering matrices for every target atom.
                s1_window, s2_window = s1, s2
            else:
                s1_window = s1[:, indices[i]]
                s2_window = s2[:, indices[i]]
            for potential in shifted[i]:
                if full_window:
                    global_potential = torch.empty_like(potential)
                    global_potential[indices[i]] = potential
                    weighted_s1 = s1_window * global_potential
                else:
                    weighted_s1 = s1_window * potential
                transition = s2_window @ weighted_s1.T
                for first in range(0, selected.shape[1], scan_batch_size):
                    last = min(first + scan_batch_size, selected.shape[1])
                    waves = transition @ selected[:, first:last]
                    intensity = waves.real.square() + waves.imag.square()
                    if full_scan:
                        output[first:last] += intensity.T
                    else:
                        output[scan_indices[first:last]] += intensity.T
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(atoms))
