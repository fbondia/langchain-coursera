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
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import SVMRetriever
from langchain_community.retrievers import TFIDFRetriever

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.environ['OPENAI_API_KEY']

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())




template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template,)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def query_stuff():

    #qa_chain = RetrievalQA.from_chain_type(
    #    llm,
    #    retriever=vectordb.as_retriever()
    #)
    
    #result = qa_chain({"query": question})
    #
    #print(result["result"])


    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    question = "Is probability a class topic?"

    result = qa_chain({"query": question})

    print(result["result"])
    #pretty_print_docs(result["source_documents"])

def query_map_reduce():

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        chain_type="map_reduce"
    )

    question = "Is probability a class topic?"

    result = qa_chain({"query": question})

    print(result["result"])
    #pretty_print_docs(result["source_documents"])

def query_refine():

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        chain_type="refine"
    )

    question = "Is probability a class topic?"

    result = qa_chain({"query": question})

    print(result["result"])
    #pretty_print_docs(result["source_documents"])

def query_map_rerank():

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        chain_type="map_rerank"
    )

    question = "Is probability a class topic?"

    result = qa_chain({"query": question})

    print(result["result"])
    #pretty_print_docs(result["source_documents"])

def chat():

    print(llm.predict("Hello world!"))

    
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )

    question = "Is probability a class topic?"

    result = qa({"query": question})

    print(result["result"])

    question = "why are those prerequesites needed?"

    result = qa({"question": question})


# query_stuff()
# query_map_reduce()
# query_refine()
# query_map_rerank()
chat()

