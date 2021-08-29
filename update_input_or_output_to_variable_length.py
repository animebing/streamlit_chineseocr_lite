import onnx
from onnx.tools import update_model_dims

'''
based on https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#updating-models-inputs-outputs-dimension-sizes-with-variable-length
and https://github.com/onnx/onnx/blob/master/onnx/tools/update_model_dims.py
'''

# dbnet
src_dbnet_path = './models/dbnet_output_length_fixed.onnx'
dst_dbnet_path = './models/dbnet_output_length_variable.onnx'
src_dbnet = onnx.load(src_dbnet_path)
dst_dbnet = update_model_dims.update_inputs_outputs_dims(
    src_dbnet, {'input0': []}, 
    {'out1': [1, 1, 'shape_h', 'shape_w']}
)
onnx.save(dst_dbnet, dst_dbnet_path)

# crnn
src_crnn_path = './models/crnn_lite_lstm_remove_unused_nodes.onnx'
dst_crnn_path = './models/crnn_lite_lstm_output_length_variable.onnx'
src_crnn = onnx.load(src_crnn_path)
dst_crnn = update_model_dims.update_inputs_outputs_dims(
    src_crnn, {'input': []}, 
    {'out': ['seq_len', 'batch_size']}
)
onnx.save(dst_crnn, dst_crnn_path)