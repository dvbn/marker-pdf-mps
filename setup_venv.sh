#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
SURYA_FORK_DIR="${SCRIPT_DIR}/surya-fork"
SURYA_REPO="https://github.com/VikParuchuri/surya.git"
SURYA_TAG="v0.17.1"
SURYA_BRANCH="mps-table-rec-fix"

echo "=== marker-pdf MPS Acceleration: venv setup ==="

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "venv already exists at ${VENV_DIR}, skipping creation."
else
    echo "Creating venv at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install marker-pdf (brings in surya-ocr from PyPI)
echo "Installing marker-pdf==1.10.2..."
pip install 'marker-pdf==1.10.2'

# Clone and patch surya-ocr fork
if [ -d "$SURYA_FORK_DIR" ]; then
    echo "surya-fork already exists, skipping clone."
else
    echo "Cloning surya-ocr ${SURYA_TAG} into ${SURYA_FORK_DIR}..."
    git clone "$SURYA_REPO" "$SURYA_FORK_DIR"
    cd "$SURYA_FORK_DIR"
    git checkout "$SURYA_TAG"
    git checkout -b "$SURYA_BRANCH"

    echo "Applying MPS patches..."

    # Patch 1: loader.py — keep device=mps, force float32
    python3 -c "
import pathlib
p = pathlib.Path('surya/table_rec/loader.py')
src = p.read_text()
old = '''    if device == \"mps\":
            logger.warning(
                \"\`TableRecEncoderDecoderModel\` is not compatible with mps backend. Defaulting to cpu instead\"
            )
            device = \"cpu\"
            dtype = \"float32\"'''
new = '''    if device == \"mps\":
            logger.info(
                \"TableRecEncoderDecoderModel: using float32 on MPS for kernel compatibility\"
            )
            dtype = torch.float32'''
p.write_text(src.replace(old, new))
print('  Patched loader.py')
"

    # Patch 2: __init__.py — replace cumsum with arange
    python3 -c "
import pathlib
p = pathlib.Path('surya/table_rec/__init__.py')
src = p.read_text()
old = 'decoder_position_ids = torch.ones_like(batch_input_ids[0, :, 0], dtype=torch.int64, device=self.model.device).cumsum(\n            0) - 1'
new = 'decoder_position_ids = torch.arange(batch_input_ids.shape[1], dtype=torch.int64, device=self.model.device)'
p.write_text(src.replace(old, new))
print('  Patched __init__.py')
"

    # Patch 3: decoder.py — extend mask fixup to MPS
    python3 -c "
import pathlib
p = pathlib.Path('surya/common/adetr/decoder.py')
src = p.read_text()
old = 'if attention_mask is not None and attention_mask.device.type == \"cuda\":'
new = 'if attention_mask is not None and attention_mask.device.type in (\"cuda\", \"mps\"):'
p.write_text(src.replace(old, new))
print('  Patched decoder.py')
"

    git add -A
    git commit -m "Enable MPS (Apple Silicon GPU) for TableRecEncoderDecoderModel"
    cd "$SCRIPT_DIR"
fi

# Install the fork in editable mode (overrides PyPI surya-ocr)
echo "Installing surya-ocr fork (editable)..."
pip install -e "$SURYA_FORK_DIR"

# Verify installation
echo "Verifying installation..."
python -c "
import surya.settings
import marker.settings
print(f'surya  TORCH_DEVICE_MODEL: {surya.settings.settings.TORCH_DEVICE_MODEL}')
print(f'marker TORCH_DEVICE_MODEL: {marker.settings.settings.TORCH_DEVICE_MODEL}')
print('Installation OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source ${VENV_DIR}/bin/activate"
