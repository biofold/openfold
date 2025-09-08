#!/usr/bin/env bash
# setup_esmfold_full.sh
# Fully featured ESMFold environment setup for single-GPU inference

# ---- 1. Create conda environment ----
conda create -n esmtest python=3.9 -y
conda activate esmtest

# ---- 2. Upgrade pip ----
pip install --upgrade pip setuptools wheel

# ---- 3. Install PyTorch (CUDA 12.x wheels) ----
# RTX A5000 works fine with cu124 build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ---- 4. Install ESMFold ----
pip install "fair-esm[esmfold]"

# ---- 5. Install OpenFold from GitHub ----
#pip install git+https://github.com/biofold/openfold.git
pip install git+https://github.com/sokrypton/openfold.git

# ---- 5a. upgrade fair-esm ----
pip install --upgrade "fair-esm[esmfold]"

# ---- 6. Install extra dependencies ----
pip install modelcif biopython "numpy<=1.23.5"

# ---- 7. Install PyTorch Geometric ecosystem ----
# Ensure versions compatible with PyTorch 2.x + CUDA 12.x
#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install torch-geometric torch-scatter

# ---- 8. Optional: remove DeepSpeed (not needed for inference) ----
pip uninstall -y deepspeed || true

# ---- 9. Verify installation ----
echo
echo "✅ ESMFold environment installation complete!"
echo "Activate environment with:   conda activate esmfold"
echo "Test installation with:     python -c 'from esm import pretrained; m = pretrained.esmfold_v1(); print(\"ESMFold OK\")'"

