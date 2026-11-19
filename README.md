# py_multislice_accelerated
GPU accelerated inelastic scattering computation.

## Run the example from scratch

The commands below use Python 3.12 on Linux and install the CPU build of PyTorch. The example runs without a GPU.

```bash
git clone https://github.com/Hyper-Phase/py_multislice_accelerated.git
cd py_multislice_accelerated
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
python -m pip install -r requirements.txt
python -m pip check
```

If you already have the checkout, start in the repository root. The explicit PyTorch install selects the CPU wheel before pip resolves the remaining dependencies.

To run interactively, start Jupyter from the activated environment with `jupyter notebook example.ipynb`, select **Python 3**, and run all cells. Keep the notebook in the repository root so it can import `pyms_plus` and read the bundled `structures/STO3bc.cif`.

To execute all cells and save the outputs into the notebook from the terminal, run this from the same directory with the environment still active.

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.record_timing=False example.ipynb
```

The example prints the device and a 48 × 48 density matrix summary, then displays the real part of the matrix and its diagonal intensity.
