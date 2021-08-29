import sys

import onnx
import onnxoptimizer

src_onnx_path = sys.argv[1]
dst_onnx_path = sys.argv[2]

onnx_model = onnx.load(src_onnx_path)
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = onnxoptimizer.optimize(onnx_model, passes)

onnx.save(optimized_model, dst_onnx_path)