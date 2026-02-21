# marker-pdf-mps

Run [marker-pdf](https://github.com/VikParuchuri/marker) on Apple Silicon GPUs (MPS) — including table recognition.

## The problem

`marker-pdf` uses `surya-ocr` for table recognition. In surya v0.17.1, `TableRecEncoderDecoderModel` is explicitly blocked from running on MPS and forced to CPU. This makes PDF extraction significantly slower on Macs with Apple Silicon.

## The fix

Three surgical changes (6 lines total) to surya-ocr enable MPS with float32, which avoids the float16 kernel issues that caused the original block:

1. **`surya/table_rec/loader.py`** — Keep `device="mps"`, only force `dtype=float32`
2. **`surya/table_rec/__init__.py`** — Replace `cumsum` (unreliable with int64 on MPS) with `torch.arange`
3. **`surya/common/adetr/decoder.py`** — Extend the SDPA `_unmask_unattended` fix to MPS

float32 on MPS is ~2-5x faster than float32 on CPU for this model on Apple Silicon.

## Quick start

```bash
git clone https://github.com/dvbn/marker-pdf-mps.git
cd marker-pdf-mps
bash setup_venv.sh
source venv/bin/activate
python convert_pdf.py your_file.pdf
```

## Use in your own project

```bash
pip install marker-pdf==1.10.2
pip install git+https://github.com/dvbn/surya.git@mps-table-rec-fix
```

Then in your code:

```python
from fix_marker_mps import apply_fix
apply_fix()  # call before importing marker/surya

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

model_dict = create_model_dict()
converter = PdfConverter(artifact_dict=model_dict)
rendered = converter("your_file.pdf")
print(rendered.markdown)
```

Or skip `fix_marker_mps.py` entirely — the patched surya fork is the actual fix. `apply_fix()` just sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and `KMP_DUPLICATE_LIB_OK=TRUE` as safety nets.

## Verify it works

```bash
source venv/bin/activate
python test_fix.py                # settings-level tests
python test_fix.py --with-models  # loads models, runs table_rec inference on MPS
python convert_pdf.py --verify-only
```

## Pinned versions

| Package | Version | Why |
|---------|---------|-----|
| marker-pdf | 1.10.2 | Latest at time of fix |
| surya-ocr | 0.17.1 (patched fork) | The 3-line MPS fix targets this version |
| torch | >=2.7.0 | MPS support required |

See `requirements.txt` for details. Future versions of surya may fix this upstream, making this repo unnecessary.

## Credits

This fix was built by David Benatia (https://davidbenatia.com).
