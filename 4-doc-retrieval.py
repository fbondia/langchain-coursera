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

from langchain.chains.query_constructor.base import AttributeInfo

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import SVMRetriever
from langchain_community.retrievers import TFIDFRetriever


_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.environ['OPENAI_API_KEY']

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def mmr_search():

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    print(vectordb._collection.count())

        
    texts = [
        """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
        """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
        """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    ]

    smalldb = Chroma.from_texts(texts, embedding=embedding)
    question = "Tell me about all-white mushrooms with large fruiting bodies"
    print(smalldb.similarity_search(question, k=2))
    print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))

def filter_search():

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    question = "what did they say about regression in the third lecture?"

    # Manual
    #docs = vectordb.similarity_search(
    #    question,
    #    k=3,
    #    filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
    #)
    #
    #for d in docs:
    #    print(d.metadata)

    # Usando um LLM

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the lecture",
            type="integer",
        ),
    ]

    document_content_description = "Lecture notes"
    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True
    )

    question = "what did they say about regression in the third lecture?"

    docs = retriever.invoke(question)

    for d in docs:
        print(d.metadata)

def compression_search():

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
    compressor = LLMChainExtractor.from_llm(llm)

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever()
    )

    question = "what did they say about matlab?"
    compressed_docs = compression_retriever.invoke(question)
    pretty_print_docs(compressed_docs)


def combined_search():

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
    compressor = LLMChainExtractor.from_llm(llm)

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever(search_type = "mmr")
    )

    question = "what did they say about matlab?"
    compressed_docs = compression_retriever.invoke(question)
    pretty_print_docs(compressed_docs)

def other_search():
    loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    pages = loader.load()
    all_page_text=[p.page_content for p in pages]
    joined_page_text=" ".join(all_page_text)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
    splits = text_splitter.split_text(joined_page_text)

    # Retrieve
    svm_retriever = SVMRetriever.from_texts(splits,embedding)
    tfidf_retriever = TFIDFRetriever.from_texts(splits)

    question = "What are major topics for this class?"
    docs_svm=svm_retriever.invoke(question)
    docs_svm[0]

    question = "what did they say about matlab?"
    docs_tfidf=tfidf_retriever.invoke(question)
    docs_tfidf[0]


#mmr_search()
#filter_search()
#compression_search()
#combined_search()
other_search()

