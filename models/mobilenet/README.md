# Fixed MobileNetV2 CPU fixture

Model: [ONNX Model Zoo MobileNetV2-12 FP32](https://huggingface.co/onnxmodelzoo/mobilenetv2-12/tree/23a2a25eaaa6fcd61865414c8a1acf13dda66b7a),
revision `23a2a25eaaa6fcd61865414c8a1acf13dda66b7a`, Apache-2.0 as stated by upstream.
SHA256: `c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5`.
The graph identifies PyTorch 1.8 as producer, IR 7 / opset 12. Names are `input`
and `output`, both FP32. Exported batch is symbolic; this application always binds
batch one: `[1,3,224,224]` → `[1,1000]` raw logits, without softmax.

Photo: [PyTorch Hub dog.jpg](https://github.com/pytorch/hub/blob/c3beaae7d32fca2a23fec30aa7938ef5c9b6e5d5/images/dog.jpg),
revision `c3beaae7d32fca2a23fec30aa7938ef5c9b6e5d5`;
SHA256 `f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a`.
Its individual reuse terms were not established; keep the downloaded image local,
outside Git, without assuming the model's license covers it.

`prepare.py` follows the Model Zoo preprocessing: RGB; bilinear resize of the
shorter side to 256 with rounded other dimension; center crop 224; NCHW;
FP32 `/255`; mean `[.485,.456,.406]`, std `[.229,.224,.225]`. Two fixed samples are
the crop and its horizontal flip, not two independent photographs. It computes
every expected logit with ONNX 1.17.0's NumPy ReferenceEvaluator, independently of
ONNX Runtime. Full-output checks use `atol=1e-4, rtol=1e-4` for accumulated FP32 CNN
rounding differences, not just top-1 agreement. This is a correctness fixture,
not an accuracy evaluation.

All model/image/tensor binaries stay in ignored `data/`; `fixtures.json` records
their provenance and hashes. To regenerate after downloading the model and photo
with `scripts/setup-inference.sh`, use ONNX 1.17.0, NumPy 1.26.4 and Pillow 9.0.1
in a temporary environment, then `python3 -B models/mobilenet/prepare.py`.
Neither Python ONNX nor Pillow is required by the C++ runtime or ROS2 runner.
From the repository root, use a Python 3.10 installation with pip for preparation
(the existing Humble runtime image does not include pip):

```sh
fixture_packages=$(mktemp -d)
python3.10 -m pip install --target "$fixture_packages" onnx==1.17.0 numpy==1.26.4 Pillow==9.0.1
PYTHONPATH="$fixture_packages" OPENBLAS_NUM_THREADS=1 python3.10 -B models/mobilenet/prepare.py
rm -rf -- "$fixture_packages"
```

Generation writes the four tensor files and updates `fixtures.json`; inspect any
manifest difference before treating the assets as the same historical fixture.
The runtime checks tensor bytes against that manifest. Preparation takes place
before measurements and is not part of the Source-to-Consumer age. Downloads
require network access to the pinned upstream resources; latency is not promised
to be deterministic across machines or runs.
