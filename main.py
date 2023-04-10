import streamlit as st
import cv2
from keras.models import load_model
from keras.utils import img_to_array,load_img
import numpy as np
st.set_page_config(page_title="Gender Detection",page_icon="https://i.pinimg.com/originals/ad/bc/27/adbc2786e5b2aaea10fe1d1d83ecb9be.png")
facemodel=cv2.CascadeClassifier("face.xml")
gendermodel=load_model("gender.h5")
st.title("GENDER DETECTION SYSTEM")
st.sidebar.image("https://viterbischool.usc.edu/wp-content/uploads/2022/12/USC-ISI-1200x600-29.png")
choice=st.sidebar.selectbox("MENU",("HOME","IMAGE","URL/IP CAM","WEB CAM"))
if(choice=="HOME"):
    st.image("https://repository-images.githubusercontent.com/203266777/4d5c8980-d3f1-11e9-803f-8b0963a65ea1")
    st.write("Gender Detection System is a Computer Vision Machine Learning Application which can be accessed through IP Cameras,it helps to detect the face of man and wooman.")
elif(choice=="IMAGE"):
    st.markdown('<center><h2>GENDER DETECTION</h2></center>',unsafe_allow_html=True)
    file=st.file_uploader("Upload an Image")
    if file:
        b=file.getvalue()
        a=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(a,cv2.IMREAD_COLOR)
        gender=facemodel.detectMultiScale(img)
        for (x,y,l,w) in gender:
            cv2.imwrite("temp.jpg",img[y:y+w,x:x+l])
            gender_img=load_img("temp.jpg",target_size=(150,150,3))
            gender_img=img_to_array(gender_img)
            gender_img=np.expand_dims(gender_img,axis=0)
            pred=gendermodel.predict(gender_img)[0][0]
            if(pred==1):
                cv2.rectangle(img,(x,y),(x+l,y+w),(0,0,255),8)
            else:
                cv2.rectangle(img,(x,y),(x+l,y+w),(0,255,0),8)
        st.image(img,channels='BGR')
elif(choice=='WEB CAM'):
    k=st.text_input("Enter 0 for Primary Camera or 1 for Secondary Camera") 
    btn=st.button('Start Camera')
    if btn:
        window=st.empty()               
        k=int(k)
        vid=cv2.VideoCapture(k)
        btn2=st.button("Stop Camera")
        if(btn2):
            vid.release()
            st.experimental_rerun()
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                gender=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in gender:
                    cv2.imwrite("temp.jpg",frame[y:y+w,x:x+l])
                    gender_img=load_img("temp.jpg",target_size=(150,150,3))
                    gender_img=img_to_array(gender_img)
                    gender_img=np.expand_dims(gender_img,axis=0)
                    pred=gendermodel.predict(gender_img)[0][0]
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),8)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),8)            
                window.image(frame,channels='BGR')
elif(choice=='URL/IP CAM'):
    k=st.text_input("Enter URL for the video") 
    btn=st.button('Start Camera')
    if btn:
        window=st.empty()      
        vid=cv2.VideoCapture(k)
        btn2=st.button("Stop Camera")
        if(btn2):
            vid.release()
            st.experimental_rerun()
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                gender=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in gender:
                    cv2.imwrite("temp.jpg",frame[y:y+w,x:x+l])
                    gender_img=load_img("temp.jpg",target_size=(150,150,3))
                    gender_img=img_to_array(gender_img)
                    gender_img=np.expand_dims(gender_img,axis=0)
                    pred=gendermodel.predict(gender_img)[0][0]
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),8)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),8)            
                window.image(frame,channels='BGR')







