# streamlit_chineseocr_lite
[chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite) GUI based on [streamlit](https://streamlit.io/)

# install 
`pip install -r requirements.txt`

# run
`streamlit run app.py`

# modify some onnx file to prevent onnxruntime warning
1. remove unused nodes based on https://github.com/microsoft/onnxruntime/issues/1899#issuecomment-840596510
2. update output length to be variable based on https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#updating-models-inputs-outputs-dimension-sizes-with-variable-length
