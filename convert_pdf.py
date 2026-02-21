"""CLI wrapper for marker-pdf with MPS acceleration."""

from fix_marker_mps import apply_fix
apply_fix()

import argparse
import os
import sys

from fix_marker_mps import get_status_report, verify_fix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using marker-pdf (MPS-accelerated)."
    )
    parser.add_argument("input_pdf", nargs="?", help="Path to input PDF file.")
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory (default: ./output).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Print status report and exit without converting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Always print the status report
    print(get_status_report())
    print()

    info = verify_fix()
    if not info["fix_verified"]:
        print("ERROR: Fix verification failed. Aborting.")
        sys.exit(1)

    if args.verify_only:
        print("--verify-only: exiting.")
        return

    if not args.input_pdf:
        print("ERROR: input_pdf is required (unless --verify-only).")
        sys.exit(2)

    if not os.path.isfile(args.input_pdf):
        print(f"ERROR: File not found: {args.input_pdf}")
        sys.exit(2)

    # Imports after fix is applied
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    print("Loading models (this may take a moment on first run)...")
    model_dict = create_model_dict()
    print("Models loaded.")

    converter = PdfConverter(artifact_dict=model_dict)
    print(f"Converting: {args.input_pdf}")
    rendered = converter(args.input_pdf)

    # Save output
    os.makedirs(args.output, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.input_pdf))[0]
    out_path = os.path.join(args.output, f"{basename}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rendered.markdown)

    print(f"Output saved to: {out_path}")

    # Save images if any
    if rendered.images:
        img_dir = os.path.join(args.output, f"{basename}_images")
        os.makedirs(img_dir, exist_ok=True)
        for img_name, img in rendered.images.items():
            img_path = os.path.join(img_dir, img_name)
            img.save(img_path)
        print(f"Images saved to: {img_dir}")


if __name__ == "__main__":
    main()
