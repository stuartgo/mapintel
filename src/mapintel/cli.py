import mlflow
import cloudpickle
from pathlib import Path
import numpy as np
import platform
import requests
from elasticsearch import Elasticsearch
from datetime import datetime,timedelta
from elasticsearch.helpers import scan,bulk
import pandas as pd
from datetime import datetime

from sklearn.decomposition import LatentDirichletAllocation as LDA
import requests
from sklearn.pipeline import Pipeline
import shutil
import mlflow
import umap
from sklearn.feature_extraction.text import CountVectorizer
import os
from sentence_transformers import SentenceTransformer


class HFWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.minilm_model = cloudpickle.load(Path(context.artifacts["minilm_model"]).open(mode="rb"))
    def predict(self, context, model_input):
        predictions=self.minilm_model.encode(model_input)
        return predictions

def vectorisation_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    artifacts = {
            "minilm_model": "./model1.pkl"
        }
    cloudpickle.dump(model,Path(artifacts["minilm_model"]).open(mode="wb"))
    #creates and saves mlflow model
    mlflow_pyfunc_model_path = "./model1/"
    conda_env = { #not sure if this actually does anything
        'channels': ['defaults'],
        'dependencies': [
        'python={}'.format(platform.python_version()),
        'pip',
        {
            'pip': [
            'mlflow',
            'cloudpickle=={}'.format(cloudpickle.__version__),
            'sentence-transformers'
            ],
        },
        ],
        'name': 'minilm_env'
    }   
    mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path, python_model=HFWrapper(), artifacts=artifacts,
        conda_env=conda_env)
    #loads model from saved filesmlflow_model = mlflow.pyfunc.load_model("./model2/")
    #zips file
    shutil.make_archive("./model1", 'zip', "./model1/")
    with open("./model1.zip", "rb") as f:
        response_post = requests.post("http://localhost:3000/vectorisation/model",files={"file": ("filename", f, "application/zip")})
    response_post.json()

class UMAPWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.umap_model = cloudpickle.load(Path(context.artifacts["umap_model"]).open(mode="rb"))
    def predict(self, context, model_input):
        predictions=self.umap_model.transform(model_input)
        return predictions

def dimensionality_reduction(embeddings):
    model=umap.UMAP()
    model.fit(embeddings)
    artifacts = {
            "umap_model": "./model2.pkl"
        }
    cloudpickle.dump(model,Path(artifacts["umap_model"]).open(mode="wb"))
    #creates and saves mlflow model
    mlflow_pyfunc_model_path = "./model2/"
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
        'python={}'.format(platform.python_version()),
        'pip',
        {
            'pip': [
            'mlflow',
            'cloudpickle=={}'.format(cloudpickle.__version__),
            'umap-learn=={}'.format(umap.__version__),
            ],
        },
        ],
        'name': 'umap_env'
    }   
    mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path, python_model=UMAPWrapper(), artifacts=artifacts,
        conda_env=conda_env)
    #loads model from saved filesmlflow_model = mlflow.pyfunc.load_model("./model2/")
    #zips file
    shutil.make_archive("./model2", 'zip', "./model2/")
    with open("./model2.zip", "rb") as f:
        response_post = requests.post("http://localhost:3000/dim_reduc/model",files={"file": ("filename", f, "application/zip")})



class LDAWrapper(mlflow.pyfunc.PythonModel):
    def top_words(self,model, feature_names, n_top_words):
        topics=[]
        for topic_idx, topic in enumerate(model.components_):
            topics.append(str(topic_idx)+"_"+"_".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])) 
        return topics

    def load_context(self, context):
        self.pipeline = cloudpickle.load(Path(context.artifacts["lda_model"]).open(mode="rb"))
    def predict(self, context, model_input):
        predictions=self.pipeline.transform(model_input)
        return list(map(lambda x: np.argmax(x),predictions))
    def topics(self):
        feature_names=self.pipeline.steps[0][1].get_feature_names()
        return self.top_words(self.pipeline.steps[1][1],feature_names,5)


# class PipelineWrapper(Pipeline):
#     def topics(self):
#         print(self.steps)
#         feature_names=self.lda_model.get_feature_names()
#         return self.top_words(self.lda_model,feature_names,5)

def topic_modelling(docs):
    model=Pipeline([("tf",CountVectorizer(max_df=0.95, min_df=2, max_features=50, stop_words='english')),("LDA",LDA(random_state=8))])
    model.fit(docs)
    artifacts = {
            "lda_model": "./model3.pkl"
        }
    cloudpickle.dump(model,Path(artifacts["lda_model"]).open(mode="wb"))
    #creates and saves mlflow model
    mlflow_pyfunc_model_path = "./model3/"
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
        'python={}'.format(platform.python_version()),
        'pip',
        {
            'pip': [
            'mlflow',
            'cloudpickle=={}'.format(cloudpickle.__version__),
            'scikit-learn=={}'.format("1.1.2"),
            ],
        },
        ],
        'name': 'lda_env'
    }   
    mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path, python_model=LDAWrapper(), artifacts=artifacts,
        conda_env=conda_env)
    #loads model from saved files
    mlflow_model = mlflow.pyfunc.load_model("./model3/")
    #zips file
    shutil.make_archive("./model3", 'zip', "./model3/")
    with open("./model3.zip", "rb") as f:
        response_post = requests.post("http://localhost:3000/topic/model",files={"file": ("filename", f, "application/zip")})



def __init__(train,db_data):
    """_summary_

    Args:
        train (list): Contains list of documents/strings to train models
        db_data (DataFrame): Contains the features: text,date,link,title,media
    """
    vectorisation_model()
    response_post=requests.post("http://localhost:3000/vectorisation/vectors",json={"docs":train})
    embeddings_train=response_post.json()["embeddings"]
    dimensionality_reduction(embeddings_train)
    topic_modelling(train)
    response_get=requests.get("http://localhost:3000/topic/topic_names",json={})
    topic_names=response_get.json()["topic_names"]
    response_post=requests.post("http://localhost:3000/vectorisation/vectors",json={"docs":db_data.text.tolist()})
    embeddings_db=response_post.json()["embeddings"]
    response_post=requests.post("http://localhost:3000/dim_reduc/vectors",json={"docs":embeddings_db})
    embeddings_2d=response_post.json()["embeddings"]
    response_post=requests.post("http://localhost:3000/topic",json={"docs":db_data.text.tolist()})
    topics_docs=response_post.json()["topics"]  
    db_data_filled=db_data.copy()
    db_data_filled["embeddings"]=embeddings_db
    db_data_filled["embeddings_2d"]=embeddings_2d
    db_data_filled["topic"]=topics_docs
    #connects to db
    es = Elasticsearch(cloud_id="Testin123Mapintel:ZXVyb3BlLXdlc3QzLmdjcC5jbG91ZC5lcy5pbzo0NDMkNWZlZjhmODQ5YjViNDMwOGFjZGJiMTJiYzVhMmFmMjAkYzdkYTM0OTJiZjIyNDBlM2I2ODBjOTVmNDQyMTVkNWI=",
    basic_auth=('elastic', 'G3lzQNlF392UoiKQTeFxM3te'),)
    #empties indices
    es.options(ignore_status=[400,404]).indices.delete(index='topics')
    es.options(ignore_status=[400,404]).indices.delete(index='docs')
    #puts docs and topics into database
    docs=[]
    db_data_filled.apply(lambda article:
    docs.append(
        {
            "_index":"docs",
            "document_id":str(len(docs)+1),
            "timestamp":article.date,#datetime.strptime(article.date, '%Y-%m-%dT%H:%M:%SZ'),
            "url":article.link,
            "title": article.title,
            "topic_label": topic_names[article.topic],
            "umap_embeddings":article.embeddings_2d,
            "image_url":article.media,
            "snippet": article.text,
            "topic_number":article.topic,
        }
    ),axis=1)
    es.indices.create(index="docs",body={"mappings":{"properties":{"meta.umap_embeddings":{"type":"dense_vector","dims":2,"index":True,"similarity":"l2_norm"}}}})
    bulk(es,docs)
    topics=[]
    for topic_name in topic_names:
        topics.append({
            "_index":"topics",
            "topic":topic_name
        })
    bulk(es,topics)

dir_path = os.path.abspath('')
file_path = os.path.join(dir_path, "other things",'catcherapi_data2.json')

data=pd.read_json(file_path,lines=True)
data=data[~data.excerpt.isna()]
data_to_use=data[:500]
data_to_use=data_to_use[["title","excerpt","link","media","published_date"]]
data_to_use.columns=["title","text","link","media","date"]
__init__(data_to_use.text.tolist(),data_to_use)



    
