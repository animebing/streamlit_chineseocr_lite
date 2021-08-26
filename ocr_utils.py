import json

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from model import OcrHandle

ocr_handle = OcrHandle()


def draw_bbox(img, res):
    img_detected = img.copy()
    img_draw = ImageDraw.Draw(img_detected)
    colors = ['red', 'green', 'blue', "purple"]

    for i, r in enumerate(res):
        rect, txt, confidence = r

        x1,y1,x2,y2,x3,y3,x4,y4 = rect.reshape(-1)
        size = max(min(x2-x1,y3-y2) // 2 , 20 )

        myfont = ImageFont.truetype("仿宋_GB2312.ttf", size=size)
        fillcolor = colors[i % len(colors)]
        img_draw.text((x1, y1 - size ), str(i+1), font=myfont, fill=fillcolor)
        for xy in [(x1, y1, x2, y2), (x2, y2, x3, y3 ), (x3 , y3 , x4, y4), (x4, y4, x1, y1)]:
            img_draw.line(xy=xy, fill=colors[i % len(colors)], width=2)
    
    return img_detected


@st.cache
def get_ocr_result(img, short_size):
    res = ocr_handle.text_predict(img, short_size)
    img_detected = draw_bbox(img, res)

    texts = ''
    for bbox, text, score in res:
        texts += f'{text}\n'

    return img_detected, texts