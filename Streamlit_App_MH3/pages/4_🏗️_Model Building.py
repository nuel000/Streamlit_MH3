import streamlit as st
from PIL import Image

st.subheader("Model Building")

image = Image.open("resources/model2.png")
st.image(image, width=400)



st.markdown("""In developing this web App we trained and applied 5 models, these include:

- **Support Vector Classifier**
- **Random Forest**
- **Naive Bayes**
- **K Nearest Neigbours**
- **Logistic Regression**



After cleaning and Exploratory Data Analysis we gained some insights as regards The most common words, most popular users and what they believe in and we also gained
some information on the popular trends regarding climate change.

Preparation of text data involved converting into some numeric format which the machine learning models can understand. We achieved this by using:



-  **CountVectorizer** 

- **TFIDF( Term Frequency–Inverse Document Frequency)**

### Procedure for Model Building

In building our models, we took the following steps

1. Built the Pipeline for Models
2. Splitted our dataset into train and test sets.
3. Fitted each models using X and y trains
4. Making classifications using the fitted models
5. Evaluated performance for each model using sklearn metrics

### Performance of each model

A model's performance is determined using a term called **F1_Score**.


The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.

To know more about F1_Score, Please read the following Article [Learn More on F1_Score](https://www.educative.io/answers/what-is-the-f1-score)

Ultimately after training our models, we calculated the F1_scores for all 5 models and plotted a bar chart as seen below.


""")


image = Image.open("resources/download.png")
st.image(image, width=600)

st.markdown(""" As observed from the chart above, Linear SVC was our best performing model, hence was our model of choice moving forward,



Hope it will be yours too!! 😄



""")




