"""Export the best Cadence checkpoint to ONNX for fast CPU inference in the server.

Output: model/onnx/cadence.onnx  (FP32)
        model/onnx/cadence_q8.onnx  (INT8 dynamic quantization, ~4x smaller)

Run: make export
"""

from pathlib import Path

import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np

from model.cadence_model import CadenceModel

CKPT_DIR = Path("model/checkpoints")
ONNX_DIR = Path("model/onnx")
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 2 * SAMPLE_RATE  # 32 000


def export() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = CKPT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("No checkpoint — run make train first")

    model = CadenceModel()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy_input = torch.randn(1, WINDOW_SAMPLES)
    fp32_path = ONNX_DIR / "cadence.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"Exported FP32 model → {fp32_path}")

    # Verify
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_values": dummy_input.numpy()})
    assert out[0].shape == (1, 2), f"Unexpected output shape: {out[0].shape}"
    print(f"  ONNX verification passed: output shape {out[0].shape}")

    # INT8 quantisation
    q8_path = ONNX_DIR / "cadence_q8.onnx"
    quantize_dynamic(fp32_path, q8_path, weight_type=QuantType.QUInt8)
    print(f"Exported INT8 model → {q8_path}")

    fp32_size = fp32_path.stat().st_size / 1e6
    q8_size = q8_path.stat().st_size / 1e6
    print(f"  Size: FP32 {fp32_size:.1f}MB → INT8 {q8_size:.1f}MB ({q8_size/fp32_size:.0%})")

    # Benchmark latency on dummy data
    times = []
    for _ in range(100):
        import time
        t0 = time.perf_counter()
        sess.run(None, {"input_values": dummy_input.numpy()})
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  Median inference latency (CPU): {np.median(times):.1f}ms")


if __name__ == "__main__":
    export()
