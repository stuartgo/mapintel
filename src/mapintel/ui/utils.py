import logging
import os

import requests
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:30000")
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_REQUEST_GENERATOR = "all_docs_generator"
UMAP_QUERY = "umap_query"
TOPIC_NAMES = "topic_names"
NUM_DOCS = "doc_count"

logger = logging.getLogger(__name__)


@st.cache(show_spinner=False)  # If the input parameters didn't change then streamlit doesn't execute the function again
def retrieve_doc(query, filters=None, top_k_reader=10, top_k_retriever=100):
    # Query Haystack API
    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    request_params = {
        "query": query,
        "filters": filters,
        "top_k_retriever": top_k_retriever,
        "top_k_reader": top_k_reader,
    }
    response_raw = requests.post(url, json=request_params).json()

    # Format response

    result = []
    answers = response_raw["answers"]
    for i in range(len(answers)):
        score = answers[i]["_score"]
        answer = answers[i]["_source"]
        if answer:
            document_id = answer["document_id"]
            relevance = round(score, 2)
            url = answer["url"]
            title = answer["title"]
            timestamp = answer["timestamp"]
            topic_label = answer["topic_label"]
            topic_number = answer["topic_number"]
            umap_embeddings = answer["umap_embeddings"]
            image_url = answer.get("image_url")
            snippet = answer["snippet"]
            result.append(
                {
                    "answer": answer,
                    "relevance": relevance,
                    "timestamp": timestamp,
                    "topic_label": topic_label,
                    "topic_number": topic_number,
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "image_url": image_url,
                    "umap_embeddings": umap_embeddings,
                    "document_id": document_id,
                },
            )
    return result, response_raw


@st.cache(show_spinner=False)
def get_all_docs(batch_size, filters=None, sample_size=None):
    # Query Haystack API
    url = f"{API_ENDPOINT}/{DOC_REQUEST_GENERATOR}"
    request_params = {"filters": filters, "batch_size": batch_size}
    response_generator = requests.post(
        url,
        json=request_params,  # , stream=True
    ).json()  # .iter_lines(delimiter=b"#SEP#")
    if sample_size is None:
        sample_size = float("inf")
    final_docs = []
    i = 0
    for line in response_generator["generator"]:
        if i % batch_size == 0:
            if i == 0:
                logger.info("Begin generator.")
            else:
                logger.info(f"Iteration done: {i}.")
        # Filter out keep-alive new lines
        if line:
            # doc = json.loads(line.decode("utf-8"))["hits"]["hits"][0]["_source"]
            doc = line["_source"]
            document_id = doc["document_id"]
            url = doc["url"]
            title = doc["title"]
            timestamp = doc["timestamp"]
            topic_label = doc["topic_label"]
            topic_number = doc["topic_number"]
            umap_embeddings = doc["umap_embeddings"]
            image_url = doc["image_url"]
            snippet = doc["snippet"]
            umap_embeddings_x = None if umap_embeddings is None else umap_embeddings[0]
            umap_embeddings_y = None if umap_embeddings is None else umap_embeddings[1]
            final_docs.append(
                {
                    # "answer": answer,
                    "timestamp": timestamp,
                    "topic_label": topic_label,
                    "topic_number": topic_number,
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "image_url": image_url,
                    "umap_embeddings": umap_embeddings,
                    "umap_embeddings_x": umap_embeddings_x,
                    "umap_embeddings_y": umap_embeddings_y,
                    "document_id": document_id,
                },
            )
            i += 1
        # Exit the loop when we reach sample_size
        if i == sample_size:
            logger.info(f"Final iteration number: {i}")
            break
    return final_docs


@st.cache(show_spinner=False)
def get_query_umap(query):
    url = f"{API_ENDPOINT}/{UMAP_QUERY}"
    request_params = {"query": query}
    response_raw = requests.post(url, json=request_params).json()
    return response_raw


def get_topic_names():
    url = f"{API_ENDPOINT}/{TOPIC_NAMES}"
    response_raw = requests.get(url, json={}).json()
    return response_raw["topic_names"]


@st.cache(show_spinner=False)
def count_docs(filters=None):
    url = f"{API_ENDPOINT}/{NUM_DOCS}"
    request_params = {"filters": filters}
    response_raw = requests.post(url, json=request_params).json()
    return response_raw["num_documents"]


def feedback_answer(question, answer, document_id, model_id, is_correct_answer, is_correct_document):
    # Feedback Haystack API
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    request_params = {
        "question": question,
        "answer": answer,
        "document_id": document_id,
        "model_id": model_id,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document,
    }
    response_raw = requests.post(url, json=request_params).json()
    return response_raw
