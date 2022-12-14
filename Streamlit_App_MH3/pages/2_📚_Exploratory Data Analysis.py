import streamlit as st
from PIL import Image

st.title('Exploratory Data Analysis')


st.markdown(""" Exploratory Data Analysis is a very important aspect of machine learning. It refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.


It is a good practice to understand the data first and try to gather as many insights from it. EDA is all about making sense of data in hand,before getting into proper data investigation.""")

st.markdown("### Insights from Exploratory Data Analysis")


data_source = ['BarCharts', 'WordClouds']

source_selection = st.selectbox('Select a section of EDA', data_source)
if source_selection == 'BarCharts':
    st.markdown("""##

                1. number of users with positive sentiments was higher than other classes """)
    image = Image.open('resources/sentiment_c.png')
    st.image(image, width=500)


    st.markdown("""##

                2. most popular users and their sentiements """)
    image = Image.open('resources/users.png')
    st.image(image, width=600)


    st.markdown("""##

                3. most popular users and their sentiements """)
    image = Image.open('resources/top10.png')
    st.image(image, width=500)


    st.markdown("""##

                4. Top 10 users with pro sentiments """)
    image = Image.open('resources/anti.png')
    st.image(image, width=500)

if source_selection == 'WordClouds':

    st.markdown("""##### Word Cloud of the most common words: A word cloud is a collection, or cluster, of words depicted in different sizes. The bigger and bolder the word appears, the more often it’s mentioned within a given text and the more important it is.""")


    st.markdown("""

                1. WordCloud for all Messages """)
    image = Image.open('resources/word cloud.png')
    st.image(image, width=500)


    st.markdown("""2. Word Cloud of trending HastTags """)
    image = Image.open('resources/hashtags.png')
    st.image(image, width=500)


    














