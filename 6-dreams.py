import os
import openai
import sys
import json

from dotenv import load_dotenv, find_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader,  FileSystemBlobLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_community.document_loaders import JSONLoader

from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())
openai.api_key  = os.environ['OPENAI_API_KEY']

embedding = OpenAIEmbeddings()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

persist_directory = 'vectordb/dreams/'

def load_json_metadata(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    metadata["date"] = record.get("date")

    return metadata

def load_json():

    loader = JSONLoader(
        file_path='docs/json/dreams.json',
        jq_schema='.[]',
        content_key='text',
        metadata_func=load_json_metadata
    )

    docs = loader.load()

    #print(len(docs))
    #print(docs[0].page_content[0:200])
    #print(docs[0].metadata)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        separators=["\n\n", "\\n\\n", "\n", "\\n", ".", " ", ""]
    )

    all_chunks = []

    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(
                page_content=chunk,
                metadata=doc.metadata
            ))
        
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    print("Base vetorial salva com sucesso.")

def pretty_print_dream(dream):
    print (f"""{dream.metadata["date"]} - {dream.metadata["title"]}""")
    print (dream.page_content)
    #print (dream.metadata)
    print ("================================================")


def search():

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    question = "Quais sonhos relacionados a morte?"
    results = vectordb.similarity_search(question, k=10)

    print ("================================================")
    print ("=== RESULTADOS SIMILARES =======================")
    print ("================================================")

    for result in results:
        pretty_print_dream(result)

    print ("================================================")
    print ("=== RESULTADOS DIVERGENTES =====================")
    print ("================================================")

    results = vectordb.max_marginal_relevance_search(question,k=3, fetch_k=10)

    for result in results:
        pretty_print_dream(result)

    print ("================================================")
        


#load_json()

search()