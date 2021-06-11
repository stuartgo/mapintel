import os

import requests
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"


@st.cache(show_spinner=False)
def retrieve_doc(query, filters=None, top_k_reader=10, top_k_retriever=100):
    # Query Haystack API
    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    req = {"query": query, "filters": filters, "top_k_retriever": top_k_retriever, "top_k_reader": top_k_reader}
    response_raw = requests.post(url, json=req).json()

    # Format response
    result = []
    answers = response_raw["answers"]
    for i in range(len(answers)):
        answer = answers[i]["answer"]
        if answer:
            relevance = round(answers[i]["score"] * 100, 2)
            meta_source = answers[i]["meta"].get("source", None)
            meta_publishedat = answers[i]["meta"].get("publishedat", None)
            meta_topic = answers[i]["meta"].get("category", None) 
            meta_url = answers[i]["meta"].get("url", None)
            meta_imageurl = answers[i]["meta"].get("urltoimage", None)
            document_id = answers[i]["document_id"]
            result.append(
                {
                    "answer": answer,
                    "relevance": relevance,
                    "source": meta_source,
                    "publishedat": meta_publishedat,
                    "topic": meta_topic,
                    "url": meta_url,
                    "image_url": meta_imageurl,
                    "document_id": document_id
                }
            )
    return result, response_raw


def feedback_doc(question, answer, document_id, model_id, is_correct_answer, is_correct_document):
    # Feedback Haystack API
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "question": question,
        "answer": answer,
        "document_id": document_id,
        "model_id": model_id,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document
    }
    response_raw = requests.post(url, json=req).json()
    return response_raw


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("file", file)]
    response_raw = requests.post(url, files=files).json()
    return response_raw
