from PIL import Image
import streamlit as st

from ocr_utils import get_ocr_result

st.set_page_config(page_title='OCR App based on chineseocr lite and streamlit', layout='wide')

col_0, col_1, col_2 = st.columns([4, 1, 4])
with col_0:
    uploaded_img = st.file_uploader(label='upload an image', type=('jpg', 'jpeg', 'png'))
    if uploaded_img is not None:
        st.image(uploaded_img)
    
with col_1:
    short_size = st.text_input('short edge size', '960')
    st.text('')
    ocr_button = st.button(label='recognize')

if ocr_button and uploaded_img is not None:
    if not short_size:
        st.error('the short edge size must be set')
    else:
        try:
            short_size = int(short_size)
        except Exception as e:
            st.error(str(e))
        else:
            short_size = max(short_size, 64)
            short_size = 32 * (short_size // 32)

            img = Image.open(uploaded_img)
            img = img.convert('RGB')
            img_detected, texts = get_ocr_result(img, int(short_size))

            with col_2:
                st.subheader('detection result')
                st.image(img_detected)

                st.subheader('texts')
                st.text(texts)