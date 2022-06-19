#!/usr/bin/env python
# coding: utf-8


#Import all required libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# %matplotlib inline


st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Diabetes Prediction App
This app predicts the **Chances of getting diagnosed with Diabetes**!
""")
st.write('---')
st.write('**Description of Dataset**')
st.write('**Pregnancies** - Number of times pregnant')
st.write('**Glucose** - Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
st.write('**BloodPressure** - Diastolic blood pressure (mm Hg)')
st.write('**SkinThickness** - Triceps skin fold thickness (mm)')
st.write('**Insulin** - 2-Hour serum insulin (mu U/ml)')
st.write('**BMI** -  Body mass index (weight in kg/(height in m)^2)')
st.write('**DiabetesPedigreeFunction** - Diabetes pedigree function')
st.write('**Age** -  Age (years)')

diabetes = pd.read_csv('data/diabetes.csv')
st.write(diabetes)

sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes.drop(['Outcome'], axis=1)),
                 columns=['PREGNANCIES', 'GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI',
                          'DIABETESPEDIGREEFUNCTION', 'AGE'])

# X = pd.DataFrame(X.iloc[:, 0:8].values, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
#                           'DiabetesPedigreeFunction', 'Age'])
y = pd.DataFrame(diabetes.iloc[:, -1].values, columns=["DIABETIC OUTCOME"])


st.sidebar.header('Specify Input Parameters')


def user_input_features():
    PREGNANCIES = st.sidebar.slider('PREGNANCIES', float(diabetes.Pregnancies.min()), float(diabetes.Pregnancies.max()), float(diabetes.Pregnancies.mean()))
    GLUCOSE = st.sidebar.slider('GLUCOSE', float(diabetes.Glucose.min()), float(diabetes.Glucose.max()), float(diabetes.Glucose.mean()))
    BLOODPRESSURE = st.sidebar.slider('BLOODPRESSURE', float(diabetes.BloodPressure.min()), float(diabetes.BloodPressure.max()), float(diabetes.BloodPressure.mean()))
    SKINTHICKNESS = st.sidebar.slider('SKINTHICKNESS', float(diabetes.SkinThickness.min()), float(diabetes.SkinThickness.max()), float(diabetes.SkinThickness.mean()))
    INSULIN = st.sidebar.slider('INSULIN', float(diabetes.Insulin.min()), float(diabetes.Insulin.max()), float(diabetes.Insulin.mean()))
    BMI = st.sidebar.slider('BMI', float(diabetes.BMI.min()), float(diabetes.BMI.max()), float(diabetes.BMI.mean()))
    DIABETESPEDIGREEFUNCTION = st.sidebar.slider('DIABETESPEDIGREEFUNCTION', float(diabetes.DiabetesPedigreeFunction.min()), float(diabetes.DiabetesPedigreeFunction.max()), float(diabetes.DiabetesPedigreeFunction.mean()))
    AGE = st.sidebar.slider('AGE', float(diabetes.Age.min()), float(diabetes.Age.max()), float(diabetes.Age.mean()))

    data = {'PREGNANCIES': PREGNANCIES,
            'GLUCOSE': GLUCOSE,
            'BLOODPRESSURE': BLOODPRESSURE,
            'SKINTHICKNESS': SKINTHICKNESS,
            'INSULIN': INSULIN,
            'BMI': BMI,
            'DIABETESPEDIGREEFUNCTION': DIABETESPEDIGREEFUNCTION,
            'AGE': AGE}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build ML Model
model = LogisticRegression(random_state=0)
model.fit(X, y)

# Apply model to make inference
prediction = model.predict(df)
print(prediction)
st.header('Prediction of Diabetes outcome')
st.write(prediction)
st.write('---')