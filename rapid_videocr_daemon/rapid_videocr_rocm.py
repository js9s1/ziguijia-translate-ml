#!/home/ziguijia/子归家/code_ml/rapid_videocr_daemon/.venv-rocm/bin/python
"""
rapid_videocr entry point using the ROCm (torch) backend, used by the
rapid_videocr_daemon worker and by batch's OCR_MODE=subprocess fallback.

  - loads the bundled copy of rapid_videocr from this directory (sys.path),
  - uses the ROCm PyTorch backend (rapidocr torch engine on the AMD GPU).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # repo root — rocm_env.py

from rocm_env import setup as _rocm_setup  # noqa: E402

_rocm_setup()  # before torch/rapidocr load ROCm libs

from rapid_videocr.export import OutputFormat
from rapid_videocr.main import RapidVideOCR, RapidVideOCRInput

ROM_DETECT_MODELS = Path(
    "/home/ziguijia/.pyenv/versions/3.11.14/lib/python3.11/site-packages/rapidocr/models"
).resolve()

OCR_PARAMS = os.environ.get(
    "RAPID_VIDEOCR_ROCM_OCR_PARAMS", "1"
).strip().lower() not in {"0", "false", "no"}


def build_ocr_params():
    if not OCR_PARAMS:
        return None
    from rapidocr.utils.typings import EngineType

    return {
        "Det.engine_type": EngineType.TORCH,
        "Cls.engine_type": EngineType.TORCH,
        "Rec.engine_type": EngineType.TORCH,
        "EngineConfig.torch.use_cuda": True,
        "Global.model_root_dir": ROM_DETECT_MODELS,
    }


def main():
    if sys.stdout is not None:
        sys.stdout.reconfigure(line_buffering=True)
    if sys.stderr is not None:
        sys.stderr.reconfigure(line_buffering=True)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--img_dir",
        type=str,
        required=True,
        help="The full path of RGBImages or TXTImages.",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="outputs",
        help='The path of saving the recognition result. Default is "outputs" under the current directory.',
    )
    parser.add_argument(
        "-f",
        "--file_name",
        type=str,
        default="result",
        help='The name of the resulting file name. Default is "result".',
    )
    parser.add_argument(
        "-o",
        "--out_format",
        type=str,
        default=OutputFormat.ALL.value,
        choices=[v.value for v in OutputFormat],
        help='Output file format. Default is "all".',
    )
    parser.add_argument(
        "--is_batch_rec",
        action="store_true",
        default=False,
        help="Which mode to run (concat recognition or single recognition). Default is False.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10,
        help="The batch of concating image nums in concat recognition mode. Default is 10.",
    )
    args = parser.parse_args()

    ocr_params = build_ocr_params()

    ocr_input_params = RapidVideOCRInput(
        is_batch_rec=args.is_batch_rec,
        batch_size=args.batch_size,
        out_format=args.out_format,
        ocr_params=ocr_params,
    )
    extractor = RapidVideOCR(ocr_input_params)
    extractor(args.img_dir, args.save_dir, args.file_name)


if __name__ == "__main__":
    main()