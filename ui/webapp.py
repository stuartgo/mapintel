from datetime import date, timedelta
import pandas as pd
import streamlit as st
from bokeh.io import curdoc

from ui_components import umap_page
from utils import (
    feedback_doc, 
    retrieve_doc, 
    get_all_docs,
    upload_doc, 
    umap_query,
    umap_inference,
    doc_count
)

# TODO: A problem with the application is that when setting the slider value the query
# will execute even if we didn't finish selecting the value we want. Can we make a rule 
# like: If the slider value hasn't changed in 2 seconds, then execute the query?
# TODO: When selecting a point in the UMAP, show the KNN points highlighted on the plot
# and the listed below (the action of selecting the point will call the query endpoint)

# Init variables
default_question = "Stock Market News"
debug_mode = False
batch_size = 10000
umap_sample_size = 10000
only_umap = True
filters = []

# Set page configuration
st.set_page_config(layout="centered")
curdoc().theme = 'dark_minimal'  # bokeh dark theme

# Title
st.write("# Mapintel App")

# UI sidebar
with st.sidebar:
    st.header("Options:")
    st.write("Global Options")
    end_of_week = date.today() + timedelta(6 - date.today().weekday())
    _, mid, _ = st.beta_columns([1, 10, 1])
    with mid:  # Use columns to avoid slider labels being off-window
        filter_date = st.slider(
            "Date range", 
            value=(date(2020,1,1), end_of_week),
            step=timedelta(7), 
            format="DD-MM-YY"
        )
    with st.beta_expander("Query Options"):
        # TODO: remove hard encoding of unique categories. When creating the topic modeling endpoint, 
        # put the unique categories in the results and pass it here
        unique_categories = ['Business', 'Entertainment', 'General', 'Health', 'Science', 'Sports', 'Technology']
        filter_category = st.multiselect(
            "Category",
            options = unique_categories
        )        
    with st.beta_expander("Results Options"):
        top_k_reader = st.slider(
            "Number of returned documents",
            min_value=1, 
            max_value=20, 
            value=10, 
            step=1
        )
        top_k_retriever = st.slider(
            "Number of candidate documents", 
            min_value=1, 
            max_value=200, 
            value=100, 
            step=1
        )
        if debug_mode:
            debug = st.checkbox("Show debug info")
        else:
            debug = None
    with st.beta_expander("Visualization Options"):
        umap_perc = st.slider(
            "Percentage of documents displayed", 
            min_value=1, 
            max_value=100, 
            value=1, 
            step=1, 
            help="Display a randomly sampled percentage of the documents to improve performance"
        )
    st.header("File Upload:")
    data_file = st.file_uploader("", type=["pdf", "txt", "docx"])
    # Upload file
    if data_file:
        raw_json1 = upload_doc(data_file)  # Upload documents to the doc store
        raw_json2 = umap_inference()  # Get respective umap embedding
        if raw_json1['status'] == "Success" and raw_json2['status'] == "Success":
            st.write("Success")
        else:
            st.write("Fail")

# Prepare filters
if filter_category:
    filters.append(
        {
            "terms": {
                "category": list(map(lambda x: x.lower(), filter_category))
            }
        }
    )
filters.append(
    {
        "range": {
            "publishedat": {
                "gte": filter_date[0].strftime("%Y-%m-%d"),
                "lte": filter_date[1].strftime("%Y-%m-%d")
            }
        }
    }
)

# Search bar
question = st.text_input(label="Please provide your query:", value=default_question)

# TODO: create a umatrix endpoint much like the umap one and use it to display the umatrix?
# Sampling the docs and passing them to the UMAP plot
doc_num = doc_count(filters)
sample_size = int(umap_perc/100 * doc_num)
st.subheader("UMAP")
with st.spinner(
    "Getting documents from database... \n "
    "Documents will be plotted when ready."
):
    # Read data for umap plot (create generator)
    umap_docs = get_all_docs(
        filters=filters, 
        batch_size=batch_size, 
        sample_size=sample_size
    )

# Plot the completed UMAP plot
umap_ids = umap_page(
    documents=pd.DataFrame(umap_docs), 
    query=umap_query(question)
)
st.write("___")

# If there is any UMAP selection, then retrieve_doc will only consider those documents
if umap_ids:
    pass
    # TODO: What do we want to do with the selected docs? Maybe some summary 
    # characteristics of them? I don't think that using the set as the basis of
    # the query is that useful (because most likely it won't contain the KNN and
    # we aren't limited by searching on the whole doc store because of Aprox-KNN)

# Get results for query
with st.spinner(
    "Performing neural search on documents... 🧠 \n "
    "Do you want to optimize speed or accuracy? \n"
    "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "
):
    results, raw_json = retrieve_doc(
        query=question, 
        filters=filters,
        top_k_reader=top_k_reader, 
        top_k_retriever=top_k_retriever
    )

st.write("## Retrieved answers:")

# Make every button key unique
count = 0
raw_json_feedback = ""

for result in results:
    # Define columns for answer text and image
    col1, _, col2 = st.beta_columns([6, 1, 3])
    with col1:
        st.write(result["answer"])
    with col2:
        # TODO: resolve issue of document_placeholder.png not showing
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
    button_col1, button_col2, _ = st.beta_columns([1, 1, 8])
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
