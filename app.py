from utils import get_emission_and_vocab, my_preprocess, predict_pos
import streamlit as st
from pandas import read_csv


#app sidebar
st.sidebar.subheader('About the App')
st.sidebar.write("POS tagging app using a simple tagger.")
st.sidebar.write("This is just a small model that predict the part of speech for each word in the sentence with respect to context.")
st.sidebar.write("The model is not perfect, so it may miss out some words...")
st.sidebar.write("Don't mind the crude display of the tags :)")
st.sidebar.write("Below is a table showing the tags and their meaning")
total_tags = read_csv('tags.csv')
st.sidebar.write(total_tags)


#start the user interface
st.title("POS (Part of Speech) Tagging App.")
st.write("Check the left sidebar for more information.")
st.write("Type in your sentence below and don't forget to press the enter button before clicking/pressing the button below.")

my_text = st.text_input("Enter your sentence...", "A sample sentence.", max_chars=100, key='to_classify')



if st.button('Get POS tags', 'run_model'):
    vocab, emission_count = get_emission_and_vocab()
    orig, prep = my_preprocess(vocab, my_text)
    final = predict_pos(prep, emission_count, vocab)

    to_display = {}

    for i in range(len(orig)):
        to_display[orig[i]] = final[i]

    st.write("The POS tags for your sentence are:", to_display)
