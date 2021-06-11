import os
import sys

# import pandas as pd
import streamlit as st

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState
from utils import feedback_doc
from utils import retrieve_doc
from utils import upload_doc


# Define state
state_question = SessionState.get(
    random_question="Stock Market News", run_query="false"
)

# Initialize variables
random_question = "Stock Market News"

# UI search bar and sidebar
st.write("# Mapintel App")
st.sidebar.header("Options")
top_k_reader = st.sidebar.slider("Number of displayed documents", min_value=1, max_value=20, value=10, step=1)
top_k_retriever = st.sidebar.slider("Number of candidate documents", min_value=1, max_value=200, value=100, step=1)
debug = st.sidebar.checkbox("Show debug info")

st.sidebar.write("## File Upload:")
data_file = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"])
# Upload file
if data_file:
    raw_json = upload_doc(data_file)
    st.sidebar.write(raw_json['status'])

# Search bar
question = st.text_input("Please provide your query:", value=random_question)
if state_question and state_question.run_query:
    run_query = state_question.run_query
    st.button("Run")
else:
    run_query = st.button("Run")
    state_question.run_query = run_query

raw_json_feedback = ""

# Get results for query
if run_query:
    with st.spinner(
        "Performing neural search on documents... 🧠 \n "
        "Do you want to optimize speed or accuracy? \n"
        "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "
    ):
        results, raw_json = retrieve_doc(question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever)

    st.write("## Retrieved answers:")

    # Make every button key unique
    count = 0

    for result in results:
        col1, mid, col2 = st.beta_columns([6, 1, 3])
        with col1:
            st.write(result["answer"])
        with col2:
            if result['image_url'] is not None:
                image_url = result['image_url']
            else:
                image_url = 'document_placeholder.png'

            if result['url'] is not None:
                st.markdown(
                    f"""
                    <a href={result['url']}>
                        <img src={image_url} alt={result['url']} style="width:600px;"/>
                    </a>
                    """
                    , unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <img src={image_url} alt="Placeholder Image" style="width:600px;">
                    """
                    , unsafe_allow_html=True
                )

        "**Relevance:** ", result["relevance"], "**Source:** ", result["source"], "**Published At:** ", result["publishedat"]

        # Define columns for feedback buttons
        button_col1, button_col2, button_col3 = st.beta_columns([1, 1, 8])
        if button_col1.button("👍", key=(result["answer"] + str(count)), help="Relevant document"):
            raw_json_feedback = feedback_doc(
                question, result["answer"], result["document_id"], 1, "true", "true"
            )
            st.success("Thanks for your feedback")
        if button_col2.button("👎", key=(result["answer"] + str(count)), help="Irrelevant document"):
            raw_json_feedback = feedback_doc(
                question, result["answer"], result["document_id"], 1, "false", "false"
            )
            st.success("Thanks for your feedback!")
        count += 1
        st.write("___")
    if debug:
        st.subheader("REST API JSON response")
        st.write(raw_json)
