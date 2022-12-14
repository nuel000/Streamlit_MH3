"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os


import pandas as pd
import numpy as np
import re
from nlppreprocess import NLP
from nltk import pos_tag
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nlppreprocess import NLP
from nltk import pos_tag
import pickle
import json

# Vectorizer
news_vectorizer = open("resources/vectorizer3.pk","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file



# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app

st.markdown('### Enter Tweet Below')

data_source = ['Single text', 'Dataset'] ## differentiating between a single text and a dataset inpit

source_selection = st.selectbox('Choose Type of Input?', data_source)




def load_models(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_models

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key
if source_selection == 'Single text':
    st.subheader('Single tweet classification')

    input_text = st.text_area('Enter Text :') ##user entering a single text to classify and predict
    all_ml_models = ["Support Vector Classifier","Logistic Regression","Naive Bayes","Random Forest","K Nearest Neighbor"]
    model_choice = st.selectbox("Choose ML Model",all_ml_models)

    prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}

    if st.button('Classify'):
        st.text("Original test ::\n{}".format(input_text))
        pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+' ## Remove urls
        subs_url = r''
        input_text= re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", input_text) ## replace all urls with clean space
        input_text = input_text.lower()

        def remove_pattern(input_text, pattern):
                r = re.findall(pattern, input_text)
                for i in r:
                    input_text = re.sub(i, '', input_text)
                    return input_text
                input_text= np.vectorize(remove_pattern)(input_text, "@[\w]*")
        import string
        def remove_punctuation(input_text):
            return ''.join([l for l in input_text if l not in string.punctuation])

            input_text = remove_punctuation(input_text)
            input_text= input_text.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

        def decontracted(input_text):
                input_text = input_text.rstrip() #returns a copy of the string in which all characters have been stripped from the end of the string
                input_text = ' '.join(input_text.split()) #Join all items in a tuple into a string
                input_text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', input_text) #remove special cases
                input_text = re.sub('@[\w]+','',input_text) # remove @ mentioned users
                input_text = re.sub(r'[^\x00-\x7f]',r'', input_text) #remove all non-ASCII characters
                return input_text
        input_text = decontracted(input_text)
        input_text =[input_text]

            
    if model_choice == 'Support Vector Classifier':
        model_load_path = "resources/real_svc_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(input_text)
               

    elif model_choice == 'Logistic Regression':
        model_load_path = "resources/real_lr_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(input_text)
                

    elif model_choice == 'Naive Bayes':
        model_load_path = "resources/real_nb_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(input_text)

    elif model_choice == 'Random Forest':
        
        model_load_path = "resources/real_rf_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(input_text)

    elif model_choice == 'K Nearest Neighbor':
                predictor = load_models("resources/real_knn_model.pkl")
                prediction = predictor.predict(input_text)

    final_result = get_keys(prediction,prediction_labels)
    st.success("Tweet Categorized as: {}".format(final_result))


if source_selection == 'Dataset':

    st.subheader('Dataset tweet classification')

    all_ml_models = ["Support Vector Classifier","Logistic Regression","Naive Bayes","Random Forest","K Nearest Neighbor"]
    model_choice = st.selectbox("Choose ML Model",all_ml_models)

    prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
    text_input = st.file_uploader("Upload a CSV file", type="csv")
    if text_input is not None:
        text_input = pd.read_csv(text_input)

    uploaded_dataset = st.checkbox('view uploaded dataset')
    if uploaded_dataset:
        st.dataframe(text_input.head(25))
    col = st.text_area('choose column to classify')

    if st.button('Classify'):

        st.text("Original test ::\n{}".format(text_input))

        pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+' ## Remove urls
        subs_url = r''
        text_input['clean'] = text_input[col].replace(to_replace = pattern_url, value = subs_url, regex = True)
        text_input['clean'] = text_input['clean'].str.lower()

        def remove_pattern(text_input, pattern):
            r = re.findall(pattern, text_input)
            for i in r:
                text_input = re.sub(i, '', text_input)
            return text_input
        text_input['clean']= np.vectorize(remove_pattern)(text_input['clean'], "@[\w]*")

        import string
        def remove_punctuation(text_input):

            return ''.join([l for l in text_input if l not in string.punctuation])
        text_input['clean'] = text_input['clean'].apply(remove_punctuation)
        text_input['clean']= text_input['clean'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

        def decontracted(text_input):
                text_input = text_input.rstrip() #returns a copy of the string in which all characters have been stripped from the end of the string
                text_input = ' '.join(text_input.split()) #Join all items in a tuple into a string
                text_input = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text_input) #remove special cases
                text_input = re.sub('@[\w]+','',text_input) # remove @ mentioned users
                text_input = re.sub(r'[^\x00-\x7f]',r'', text_input) #remove all non-ASCII characters
                return text_input
        text_input['clean'] = text_input['clean'].apply(decontracted)
        clean = text_input['clean']




    if model_choice == 'Support Vector Classifier':
        model_load_path = "resources/real_svc_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(clean)
            text_input['sentiment'] = prediction
            result = text_input[['sentiment',col]]
            result = result.apply(lambda x: x.replace({1:'Positive', -1:'Negative',0:'Neutral',2:'News'}, regex=True))
            st.info("RESULT")
            st.write(result)


    if model_choice == 'Logistic Regression':
        model_load_path = "resources/real_lr_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(clean)
            text_input['sentiment'] = prediction
            result = text_input[['sentiment',col]]
            result = result.apply(lambda x: x.replace({1:'Positive', -1:'Negative',0:'Neutral',2:'News'}, regex=True))
            st.info("RESULT")
            st.write(result)

    if model_choice == 'Naive Bayes':
        model_load_path = "resources/real_nb_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(clean)
            text_input['sentiment'] = prediction
            result = text_input[['sentiment',col]]
            result = result.apply(lambda x: x.replace({1:'Positive', -1:'Negative',0:'Neutral',2:'News'}, regex=True))
            st.info("RESULT")
            st.write(result)

    if model_choice == 'Random Forest':
        model_load_path = "resources/real_rf_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(clean)
            text_input['sentiment'] = prediction
            result = text_input[['sentiment',col]]
            result = result.apply(lambda x: x.replace({1:'Positive', -1:'Negative',0:'Neutral',2:'News'}, regex=True))
            st.info("RESULT")
            st.write(result)

    if model_choice == 'K Nearest Neighbor':
        model_load_path = "resources/real_knn_model.pkl"
        with open(model_load_path,'rb') as file:
            unpickled_model = pickle.load(file)
            prediction = unpickled_model.predict(clean)
            text_input['sentiment'] = prediction
            result = text_input[['sentiment',col]]
            result = result.apply(lambda x: x.replace({1:'Positive', -1:'Negative',0:'Neutral',2:'News'}, regex=True))
            st.info("RESULT")
            st.write(result)
            
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(result)
    st.download_button(
                "Download Classification Result",csv,
                "file.csv",
                "text/csv",
                key='download-csv'
                )
    



            
    #final_result = get_keys(prediction,prediction_labels)
    #st.success("Tweets Categorized as:: {}".format(final_result))

    #csv = text_input.to_csv(index=False)
    #b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

    #st.markdown(href, unsafe_allow_html=True)
