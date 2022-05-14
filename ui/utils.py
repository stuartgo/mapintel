import json
import logging
import os

import requests
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_REQUEST_GENERATOR = "all-docs-generator"
UMAP_QUERY = "umap-query"
TOPIC_NAMES = "topic-names"
NUM_DOCS = "doc-count"

logger = logging.getLogger(__name__)


@st.cache(
    show_spinner=False
)  # If the input parameters didn't change then streamlit doesn't execute the function again
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
        answer = answers[i]["answer"]
        if answer:
            document_id = answers[i]["document_id"]
            relevance = round(answers[i]["score"], 1)
            url = answers[i]["meta"]["url"]
            title = answers[i]["meta"]["title"]
            timestamp = answers[i]["meta"]["timestamp"]
            topic_label = answers[i]["meta"]["topic_label"]
            topic_number = answers[i]["meta"]["topic_number"]
            umap_embeddings = answers[i]["meta"]["umap_embeddings"]
            image_url = answers[i]["meta"].get("image_url")
            snippet = answers[i]["meta"].get("snippet", answer[:500])
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
                }
            )
    return result, response_raw


@st.cache(show_spinner=False)
def get_all_docs(batch_size, filters=None, sample_size=None):
    # Query Haystack API
    url = f"{API_ENDPOINT}/{DOC_REQUEST_GENERATOR}"
    request_params = {"filters": filters, "batch_size": batch_size}
    response_generator = requests.post(
        url, json=request_params, stream=True
    ).iter_lines(delimiter=b"#SEP#")

    if sample_size is None:
        sample_size = float("inf")

    final_docs = []
    i = 0
    for line in response_generator:
        if i % batch_size == 0:
            if i == 0:
                logger.info(f"Begin generator.")
            else:
                logger.info(f"Iteration done: {i}.")
        # Filter out keep-alive new lines
        if line:
            doc = json.loads(line.decode("utf-8"))
            answer = doc["answer"]
            if answer:
                document_id = doc["document_id"]
                url = doc["meta"]["url"]
                title = doc["meta"]["title"]
                timestamp = doc["meta"]["timestamp"]
                topic_label = doc["meta"]["topic_label"]
                topic_number = doc["meta"]["topic_number"]
                umap_embeddings = doc["meta"]["umap_embeddings"]
                image_url = doc["meta"].get("image_url")
                snippet = doc["meta"].get("snippet", answer[:500])
                umap_embeddings_x = (
                    None if umap_embeddings is None else umap_embeddings[0]
                )
                umap_embeddings_y = (
                    None if umap_embeddings is None else umap_embeddings[1]
                )
                final_docs.append(
                    {
                        "answer": answer,
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
                    }
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


def feedback_answer(
    question, answer, document_id, model_id, is_correct_answer, is_correct_document
):
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
