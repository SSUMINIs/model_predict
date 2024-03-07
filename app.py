import streamlit as st
import pandas as pd
from joblib import load
import os

model_directory = 'model'
model_path = os.path.join(model_directory, 'tip_prediction_pipeline.joblib')

def predict_tip(model_path, total_bill, size, sex, smoker, day, time):
    
    # 모델 불러오기
    pipeline = load(model_path)
    
    # 예측 데이터 생성
    df = pd.DataFrame([{'total_bill': total_bill, 'size': size, 'sex': sex, 'smoker': smoker, 'day': day, 'time': time}])
    
    # 예측값 생성
    prediction = pipeline.predict(df)
    return prediction[0]

def main():

    st.title('팁 예측 모델')
    st.write('total_bill과 다른 요인을 고려하여 tip 예측 모델 생성')

    total_bill = st.number_input('Total Bill ($)', min_value=0.0, format='%f')
    size = st.number_input('Size of the Party', min_value=1, step=1)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    smoker = st.selectbox('Smoker', ['Yes', 'No'])
    day = st.selectbox('Day', ['Thur', 'Fri', 'Sat', 'Sun'])
    time = st.selectbox('Time', ['Lunch', 'Dinner'])

    if st.button('예상 Tip 예측'):
        result = predict_tip(model_path, total_bill, size, sex, smoker, day, time)
        st.success(f'예측 Tip: ${result:.2f}')

if __name__ == "__main__":
    main()