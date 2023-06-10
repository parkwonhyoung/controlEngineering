import streamlit as st

st.title("202021012 박원형 제어공학 ")

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sympy
# 전달함수 G(s)의 분자와 분모 계수
s = sympy.symbols('s')

numG= [100]
denG= [1,5,6]

numH = [100]
denH= [1, 5, 106]

# 폐루프 전달함수 H(s) 계산
G = ( 100/( (s+2)*(s+3) ) )
H = G / ( 1 + G )

st.write("폐 루프 전달함수 : ",H)

st.write(" \n 단위계단입력에 대한 응답곡선")
G0 = signal.lti([100],[1])  # 비례요소
G1 = signal.lti([1],[1,2] )  #적분요소
G2 = signal.lti([1],[1,3])   #1차지연요소
G3 = signal.lti([100],[1,5,6])  #Overall 


t = np.linspace(0, 10, 1000)  # 시간 범위 설정
u = np.ones_like(t)  # 단위계단입력 생성

# 응답곡선 계산
t, y = signal.step((numG, den), T=t)

# 응답곡선 그래프 그리기
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response')
plt.grid(True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


st.write(" \n 주파수 응답에 대한 보드선도")
#주파수 범위 정의
Frequency = np.logspace(-2,2,500)

#주파수 응답 계산
systems = [G0,G1,G2,G3]
labels =['Proportional Element','Integral Element','First_Order Lag Element','Overall System']
colors = ['b','r','c','m']

plt.figure(figsize=(12,8))

#Bode Magnitude plot
plt.subplot(2,1,1)
for sys, label, color in zip(systems, labels ,colors):
    w, mag, _ = sys.bode(Frequency)
    plt.semilogx(w, mag, color=color, label=label)

plt.title('Bode Plot')
plt.ylabel('Magnitude [dB]')
plt.legend()

#Bode Phase plot
plt.subplot(2,1,2)
for sys, _, color in zip(systems, labels ,colors):
    w, _, phase = sys.bode(Frequency)
    plt.semilogx(w, phase, color=color)
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency [Hz]')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
