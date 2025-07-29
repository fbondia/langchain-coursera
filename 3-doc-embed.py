import os
import openai
from dotenv import load_dotenv, find_dotenv

import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import NotionDirectoryLoader

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings


_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.environ['OPENAI_API_KEY']

persist_directory = 'docs/chroma/'

loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()




def check_similarity():
    sentence1 = "Gatos correm pelo gramado"
    sentence2 = "Árvores florescem no campo"
    sentence3 = "João é um homem de negócios"

    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)

    print(np.dot(embedding1, embedding2))
    print(np.dot(embedding1, embedding3))
    print(np.dot(embedding2, embedding3))

def store_docs():

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    vectordb.persist()


def search():

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    #question = "is there an email i can ask for help"
    #docs = vectordb.similarity_search(question,k=3)
    #print(len(docs))
    #print(docs[0].page_content)
    #print("===========")

    #question = "what did they say about matlab?"
    #docs = vectordb.similarity_search(question,k=5)
    #print(len(docs))
    #print("===========")
    #print(docs[0])
    #print("===========")
    #print(docs[1])

    question = "what did they say about regression in the third lecture?"
    docs = vectordb.similarity_search(question,k=5)
    for doc in docs:
        print(doc.metadata)
    #print(docs[4].page_content)


# check_similarity()
# store_docs()
search()
