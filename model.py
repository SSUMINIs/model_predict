import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
import os

# 데이터셋 불러오기
tips = sns.load_dataset('tips')

# 데이터셋 컬럼 추출
categorical_features = ['sex', 'smoker', 'day', 'time']
numerical_features = ['total_bill', 'size']

# pipeline 모델 만들기
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LinearRegression())])

# 데이터셋 분류 / 종속 변수 tip을 예측하는 모델
X = tips.drop('tip', axis=1)
y = tips['tip']

# 데이터셋 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
pipeline.fit(X_train, y_train)

# 모델 저장
model_directory = 'model'

if not os.path.exists(model_directory):
    os.makedirs(model_directory)
# model 폴더 내에서, tip_pre~joblib 파일명으로 모델을 저장하겠다.
model_path = os.path.join(model_directory, 'tip_prediction_pipeline.joblib')
dump(pipeline, model_path)