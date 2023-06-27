import streamlit as st
import pandas as pd
import numpy as np
import pickle


file1 = open('pipe.pkl','rb')
rf = pickle.load(file1)
file1.close()


#Apple,mac,ultrabook, intecore,8 
data = pd.read_csv("traineddata.csv")

data["IPS"].unique()

st.title('Laptop Price Predictor')

company = st.selectbox('Brand',data['Company'].unique())

# type of laptop

type = st.selectbox('Type',data['Typename'].unique())

# ram present in laptop

ram = st.selectbox('Ram(in GB)',data[2,4,6,8,12,16,24,32,64].unique())

# os of laptop

os = st.selectbox('OS',data['OpSys'].unique())

#weight of laptop

weight = st.number_input('Weight of laptop')

# touch screen avialable or not

touchscreen = st.selectbox('TouchScreen',['NO','YES'])

#IPS

ips = st.selectbox('IPS',['NO','YES'])

# screen size

screen_size = st.number_input("Screen Size")

#resolution of laptop

resolution = st.selectbox('Screen resolution',['1920x1080','1366x768','1600x900',
                                               '3840x2160','2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu

cpu = st.selectbox('CPU',data['CPU_name'].unique())

# hdd

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

#ssd

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

#gpu

gpu = st.selectbox('GPU(in GB)',data['Gpu barnd'].unique())

if st.button('Predict Price'):

    ppi = None

    if touchscreen == 'YES':
        touchscreen=1
    else:
        touchscreen=0
    if ips == 'YES':
        ips = 1
    else:
        ips = 0

    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

    query = np.array([company, type, ram, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    prediction = int(np.exp(rf.predict(query)[0]))

    st.title("Predicted price for this laptop could be between " +
             str(prediction-1000)+"₹" + " to " + str(prediction+1000)+"₹")
