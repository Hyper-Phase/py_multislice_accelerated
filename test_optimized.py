"""Small numerical regression for the PRISM transition contraction."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch


def window_indices(center, windowsize, gridshape):
    window = []
    for i, wind in enumerate(windowsize):
        indices = np.arange(-wind // 2, wind // 2, dtype=int) + wind % 2
        indices += int(np.floor(center[i] * gridshape[i]))
        window.append(np.mod(indices, gridshape[i]))
    return (window[0][:, None] * gridshape[0] + window[1][None, :]).ravel()


package = types.ModuleType('pyms_plus')
package.window_indices = window_indices
sys.modules['pyms_plus'] = package
spec = importlib.util.spec_from_file_location(
    'pyms_plus.optimized', Path(__file__).parent / 'pyms_plus' / 'optimized.py'
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

rng = np.random.default_rng(7)
grid = (8, 8)
crop = (4, 4)
atoms = rng.random((5, 2))
scan = rng.random((3, 4, 2))
complex_numpy = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
s1 = torch.tensor(complex_numpy((3, *grid)), dtype=torch.complex64)
s2 = torch.tensor(complex_numpy((4, *grid)), dtype=torch.complex64)
H = complex_numpy((2, *crop))
scan_array = torch.tensor(complex_numpy((12, 3)), dtype=torch.complex128)
S1 = types.SimpleNamespace(S=s1, stored_gridshape=grid)
S2 = types.SimpleNamespace(S=s2, stored_gridshape=grid)
actual = torch.zeros(12, 4)
module.accumulate_transition_intensity(
    actual, S1, S2, atoms, H, crop, scan, scan_array, (2, 2), atom_batch_size=3
)

expected = torch.zeros_like(actual)
for atom in atoms:
    index = torch.from_numpy(window_indices(atom, crop, grid))
    shift = np.remainder(atom * grid, 1)
    fy = np.fft.fftfreq(crop[0])[:, None]
    fx = np.fft.fftfreq(crop[1])[None, :]
    for potential in H:
        shifted = np.fft.fftshift(np.fft.ifft2(
            potential * np.exp(-2j * np.pi * (fy * shift[0] + fx * shift[1]))
        )).ravel()
        shifted = torch.tensor(shifted, dtype=torch.complex64)
        transition = torch.zeros(4, 3, dtype=torch.complex64)
        for beam in range(3):
            transition[:, beam] = s2.reshape(4, -1)[:, index] @ (
                shifted * s1[beam].flatten()[index]
            )
        mask = np.all(np.abs((scan.reshape(-1, 2) - atom + .5) % 1 - .5) <= .25, axis=1)
        wave = transition @ scan_array[mask].to(torch.complex64).T
        expected[mask] += (wave.real.square() + wave.imag.square()).T

torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
print('Batched transition intensity matches scalar reference')
