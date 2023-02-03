
import inference_
from translation import translate
import tensorflow as tf
from PIL import Image
import os
import sys
import urllib.request
from gtts import gTTS
from playsound import playsound
import streamlit as st
from io import BytesIO, StringIO
import pandas as pd
#print("모든 모듈이 로드되었습니다.")
#except Exception as e:
    #print("몇 모듈이 실패 : {} ".format(e))



def caption_play(image):
    image_url = image
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file('image'+image_extension, origin=image_url)
    result, attention_plot = inference_.evaluate(image_path)
    for i in result:
        if i =='<end>':
            result.remove(i)
        
    result_caption = ' '.join(result)
    print('caption: ', result_caption)
    #trans_caption = transplations(result_caption)
    print(result_caption)
    Image.open(image_path)
    #if os.path.exists(image_path):
    #    os.remove(image_path)
    
    return result_caption,  image_path

#번역
# def transplations(caption):
#     translation_caption = translate(caption)
#     return translation_caption

#음성
def sound_play(translation_caption):
    if os.path.exists('./tts_.mp3'):
        os.remove('./tts_.mp3')
    file_name = 'tts_.mp3'
    tts_kr = gTTS(text = translation_caption, lang = 'en-us')
    tts_kr.save(file_name)

    #playsound(file_name)  #음성 재생
    return file_name

STYLE = """
<style>
img {
    max-width: 80%;
}
</style>
"""


def main():
    """ streamlit app display"""
    st.set_page_config(layout = "centered")
    st.header("이미지 URL을 입력하세요. (JPEG, PNG, GIF, BMP)")
    image_url = st.text_input('이미지 URL 주소 입력', '')

    if image_url != '':
        st.text(image_url)
        caption_text, image_path_ = caption_play(image_url)
        image_ = Image.open(image_path_)
        st.image(image_)
        os.remove(image_path_)
        st.text(caption_text)
        #st.text(trans_caption_)
        #st.markdown("![Alt Text](image_url)") 
        
        if caption_text:
            if st.button('음성'):
                tts_mp3 = sound_play(caption_text)
                audio_file = open(tts_mp3, 'rb')
                st.audio( audio_file.read() , format='audio/mp3')

    # if image_url is None:
    #     st.image_url("이미지 URL 입력 해주세요.")
    # else:
    #     st.text(image_url)
       
    #     caption_text, image_path_ = caption_play(image_url)
    #     image_ = Image.open(image_path_)
    #     st.image(image_)
    #     os.remove(image_path_)
    #     st.text(caption_text)
    #     #st.text(trans_caption_)
    #     #st.markdown("![Alt Text](image_url)") 
        

    #     if caption_text:
    #         if st.button('음성'):
    #             tts_mp3 = sound_play(caption_text)
    #             audio_file = open(tts_mp3, 'rb')
    #             st.audio( audio_file.read() , format='audio/mp3')

    
main()
