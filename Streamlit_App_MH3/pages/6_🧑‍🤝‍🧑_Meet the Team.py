import streamlit as st
from PIL import Image



image = Image.open("resources/1.png")
st.image(image, width=300)

st.subheader("Meet Our Wonderful Team of Talented Individuals")



col1, col2, col3, col4, col5 = st.columns(5,gap="large")

with col1:
    image = Image.open("resources/tebo.jpeg")
    st.image(image, width=100)




with col2:

    image = Image.open("resources/idongg.jpeg")
    st.image(image, width=98)


with col3:
   image = Image.open("resources/emmanuel_m.jpeg")
   st.image(image, width=98)

with col4:
   image = Image.open("resources/aloy.jpeg")
   st.image(image, width=104)

with col5:
   image = Image.open("resources/maryam.jpg")
   st.image(image, width=100)
   st.markdown('Maryam')




st.markdown("""At **TREXTO INC** we strive for excellence through hardwork and determination!""")
#image1 = Image.open('sunrise.jpg')









