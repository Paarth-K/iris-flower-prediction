# S10.1: Copy this code cell in 'iris_app.py' using the Sublime text editor. You have already created this ML model in the previous class(es).

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LogisticRegression as lr

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


@st.cache()
def prediction(model, SepalLength, SepalWidth, PetalLength, PetalWidth):
  model.fit(X_train, y_train)
  score = model.score(X_train, y_train)
  species = model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris-setosa", score
  elif species == 1:
    return "Iris-virginica", score
  else:
    return "Iris-versicolor", score



st.title("Iris Flower Prediction App")
sepal_len = st.slider("Sepal Lengh (CM)", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
sepal_width = st.slider("Sepal Width (CM)", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
petal_len = st.slider("Petal Lengh (CM)", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
petal_width = st.slider("Petal Width (CM)", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))
model = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVC)", "Logistic Regression", "Random Forest Classifier"))

# Creating the SVC model and storing the accuracy score in a variable 'score'.



if st.sidebar.button("Predict"):
  if model == "Support Vector Machine (SVC)":
    pred_model = SVC(kernel = 'linear')
  

  elif model == "Logistic Regression":
    pred_model = rfc(n_jobs=-1, n_estimators=100)


  elif model == "Random Forest Classifier":
    pred_model = lr(n_jobs=-1)

  predict = prediction(pred_model, sepal_len, sepal_width, petal_len, petal_width)
  st.write(f"Species Prediction: {predict[0]}")
  st.write(f"Score: {round(predict[1]*100, 2)}%")

