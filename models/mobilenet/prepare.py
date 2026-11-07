"""固定照片及其水平翻转的独立参考；只在准备资产时需要 ONNX/NumPy/Pillow。"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from PIL import Image
import PIL

ROOT = Path(__file__).resolve().parent
MODEL_SHA = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
IMAGE_SHA = "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a"


def main():
    data = ROOT / "data"
    for name, expected in (("mobilenetv2-12.onnx", MODEL_SHA), ("dog.jpg", IMAGE_SHA)):
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected
    model = onnx.load(data / "mobilenetv2-12.onnx")
    onnx.checker.check_model(model)
    image = Image.open(data / "dog.jpg").convert("RGB")
    width, height = image.size
    ratio = 256 / min(width, height)
    size = (int(round(width * ratio)), int(round(height * ratio)))
    image = image.resize(size, Image.BILINEAR)
    left, top = size[0] // 2 - 112, size[1] // 2 - 112
    rgb = np.asarray(image.crop((left, top, left + 224, top + 224)))
    evaluator = ReferenceEvaluator(model)
    hashes = {}
    for sample, pixels in enumerate((rgb, rgb[:, ::-1, :])):
        tensor = pixels.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
        for channel, (mean, std) in enumerate(
            zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ):
            tensor[channel] = (tensor[channel] - mean) / std
        tensor = np.ascontiguousarray(tensor[None], dtype="<f4")
        start = time.monotonic()
        result = evaluator.run(["output"], {"input": tensor})[0]
        assert result.shape == (1, 1000) and result.dtype == np.float32
        assert np.isfinite(result).all()
        for label, array in (("input", tensor), ("expected", result)):
            name = f"{label}-{sample}.f32"
            payload = array.astype("<f4").tobytes()
            (data / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        print(
            f"sample={sample} reference_seconds={time.monotonic()-start:.3f} "
            f"top1={result.argmax()} input_bytes={tensor.nbytes}",
            flush=True,
        )
    assert hashes["input-0.f32"] != hashes["input-1.f32"]
    value = {
        "model_sha256": MODEL_SHA,
        "image_sha256": IMAGE_SHA,
        "reference": "onnx.reference.ReferenceEvaluator (NumPy; no ONNX Runtime)",
        "onnx": onnx.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000],
        "sample_count": 2,
        "files": hashes,
    }
    (ROOT / "fixtures.json").write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
