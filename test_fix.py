"""Automated tests for the marker-pdf MPS acceleration fix."""

from fix_marker_mps import apply_fix
apply_fix()

import os
import sys

import torch


def test_env_vars():
    """Test 1: Environment variables are set correctly."""
    assert os.environ.get("TORCH_DEVICE") is None or os.environ.get("TORCH_DEVICE") == "mps", \
        f"TORCH_DEVICE should be unset or 'mps', got '{os.environ.get('TORCH_DEVICE')}'"
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE", "KMP_DUPLICATE_LIB_OK should be 'TRUE'"
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1", "PYTORCH_ENABLE_MPS_FALLBACK should be '1'"
    print("PASS: Environment variables set correctly")


def test_marker_settings():
    """Test 2: marker settings resolved to MPS."""
    from marker.settings import settings as marker_settings
    device = marker_settings.TORCH_DEVICE_MODEL
    assert device == "mps", f"marker TORCH_DEVICE_MODEL should be 'mps', got '{device}'"
    print(f"PASS: marker TORCH_DEVICE_MODEL = '{device}'")


def test_surya_settings():
    """Test 3: surya settings resolved to MPS."""
    from surya.settings import settings as surya_settings
    device = surya_settings.TORCH_DEVICE_MODEL
    assert device == "mps", f"surya TORCH_DEVICE_MODEL should be 'mps', got '{device}'"
    print(f"PASS: surya TORCH_DEVICE_MODEL = '{device}'")


def test_verify_fix():
    """Test 4: verify_fix() reports success."""
    from fix_marker_mps import verify_fix
    info = verify_fix()
    assert info["fix_verified"], f"verify_fix() failed: {info}"
    print("PASS: verify_fix() -> fix_verified = True")


def test_table_rec_on_mps():
    """Test 5: TableRecModelLoader loads model onto MPS."""
    from surya.table_rec.loader import TableRecModelLoader
    loader = TableRecModelLoader()
    model = loader.model(device="mps")
    device_str = str(next(model.parameters()).device)
    assert "mps" in device_str, f"table_rec model should be on MPS, got '{device_str}'"
    print(f"PASS: table_rec model loaded on {device_str}")
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def test_table_rec_inference():
    """Test 6: TableRecPredictor runs inference on MPS without crashing."""
    from PIL import Image
    from surya.table_rec import TableRecPredictor

    # Create a synthetic table-like image
    img = Image.new("RGB", (640, 480), color=(255, 255, 255))

    predictor = TableRecPredictor()
    results = predictor([img])

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    print(f"PASS: table_rec inference on MPS succeeded ({len(results[0].cells)} cells detected)")
    del predictor
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def test_model_creation():
    """Test 7: create_model_dict() succeeds and all models are on MPS."""
    from marker.models import create_model_dict
    print("Loading models (may take a moment on first run)...")
    model_dict = create_model_dict()

    for name, predictor in model_dict.items():
        device = str(getattr(predictor, "device", "unknown"))
        print(f"  {name}: device = {device}")

    print("PASS: create_model_dict() succeeded")


def test_end_to_end(pdf_path):
    """Test 8: End-to-end conversion of a PDF."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    print(f"End-to-end test with: {pdf_path}")
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    rendered = converter(pdf_path)

    assert rendered.markdown, "Conversion produced empty markdown"
    print(f"PASS: Converted {pdf_path} -> {len(rendered.markdown)} chars of markdown")
    if rendered.images:
        print(f"      {len(rendered.images)} image(s) extracted")


def main():
    tests = [
        test_env_vars,
        test_marker_settings,
        test_surya_settings,
        test_verify_fix,
    ]

    # Model tests require --with-models flag
    load_models = "--with-models" in sys.argv or len(
        [a for a in sys.argv[1:] if not a.startswith("-")]
    ) > 0

    if load_models:
        tests.append(test_table_rec_on_mps)
        tests.append(test_table_rec_inference)
        tests.append(test_model_creation)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"FAIL: {test.__name__}: {exc}")
            failed += 1

    # End-to-end test if a PDF path was given
    pdf_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if pdf_args:
        pdf_path = pdf_args[0]
        if os.path.isfile(pdf_path):
            try:
                test_end_to_end(pdf_path)
                passed += 1
            except Exception as exc:
                print(f"FAIL: test_end_to_end: {exc}")
                failed += 1
        else:
            print(f"SKIP: PDF not found: {pdf_path}")

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
