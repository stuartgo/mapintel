from datetime import date, timedelta
import streamlit as st
from utils import feedback_doc, retrieve_doc, upload_doc

# Init variables
default_question = "Stock Market News"
umap_sample_size = 10000
only_umap = True
filters = []

# Set page configuration
st.set_page_config(layout="centered")

# Title
st.write("# Mapintel App")

# UI sidebar
with st.sidebar:
    st.header("Options")
    with st.beta_expander("Query Options"):
        end_of_week = date.today() + timedelta(6 - date.today().weekday())
        filter_test = st.slider(
            "Date range", 
            value=(date(2020,1,1), end_of_week),
            step=timedelta(7), 
            format="DD-MM-YY"
        )
        # filter_test = st.date_input(  # A different alternative
        #     "Date range",
        #     value=(date(2020,1,1), end_of_week),
        # )

    with st.beta_expander("Results Options"):
        top_k_reader = st.slider("Number of displayed documents", min_value=1, max_value=20, value=10, step=1)
        top_k_retriever = st.slider("Number of candidate documents", min_value=1, max_value=200, value=100, step=1)
        debug = st.checkbox("Show debug info")

    st.write("## File Upload:")
    data_file = st.file_uploader("", type=["pdf", "txt", "docx"])
    # Upload file
    if data_file:
        raw_json = upload_doc(data_file)
        st.write(raw_json['status'])

# Search bar
question = st.text_input(label="Please provide your query:", value=default_question)

# Plot view
from os.path import join, abspath, dirname, pardir
import pickle
from ui_components import umap_page, umatrix_page
from bokeh.io import curdoc
curdoc().theme = 'dark_minimal'  # set dark theme

OUTPUTS_PATH = join(dirname(abspath(__file__)), pardir, 'outputs', 'ui_outputs')

# Read data
encoder = pickle.load(open(join(OUTPUTS_PATH, 'encoder2.pkl'), 'rb'))
ml_engine = pickle.load(open(join(OUTPUTS_PATH, 'ml_engine2.pkl'), 'rb'))
df = ml_engine['df']
doc_embeddings = ml_engine['doc_embeddings']
projections = ml_engine['projections']
bmus = ml_engine['bmus']

# Setup variables to be used as colors
color_vars = {
    'Clusters': 'cluster_label',
    'U-matrix': 'umat_val',
    'Business': 'business',
    'Entertainment': 'entertainment',
    'General': 'general',
    'Health': 'health',
    'Science': 'science',
    'Sports': 'sports',
    'Technology': 'technology',
}

if only_umap:
    st.subheader("UMAP")
    umap_page(df[:umap_sample_size], projections[:umap_sample_size])  # Get umap
else:
    # Define columns for plots
    col1, col2 = st.beta_columns(2)
    with col1:
        st.subheader("U-matrix")
        umatrix_page(bmus, df, color_vars)  # Get umatrix
    with col2:
        st.subheader("UMAP")
        for i in range(6): st.text("")  # add spacing after subheadin to align the plots
        umap_page(df[:umap_sample_size], projections[:umap_sample_size])  # Get umap


st.write("___")

# Get results for query
with st.spinner(
    "Performing neural search on documents... 🧠 \n "
    "Do you want to optimize speed or accuracy? \n"
    "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "
):
    if len(filters) == 0:
        filters = None
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
