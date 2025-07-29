import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader,  FileSystemBlobLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import NotionDirectoryLoader

_ = load_dotenv(find_dotenv())
openai.api_key  = os.environ['OPENAI_API_KEY']

def load_pdf():
    loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    pages = loader.load()

    print("===== PDF ======")
    print("==================")

    print("Number of pages:")
    print(len(pages))

    page = pages[0]

    print("==================")

    print("First page contents:")
    print(page.page_content[0:500])

    print("==================")

    print("First page metadata:")
    print(page.metadata)


def load_youtube():

    print("===== YOUTUBE ======")

    print("==================")

    print("Download:")

    url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir="docs/youtube/"
    loader = GenericLoader(
        #YoutubeAudioLoader([url],save_dir),  # fetch from youtube
        FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally
        OpenAIWhisperParser()
    )
    docs = loader.load()

    print("==================")

    print("Transcript:")

    print(docs[0].page_content[0:500])




def load_url():
    print("===== URL ======")

    print("==================")

    loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
    docs = loader.load()
    print(docs[0].page_content[:500])


def load_notion():
    print("===== NotionDB ======")

    print("==================")

    loader = NotionDirectoryLoader("docs/Notion_DB")
    docs = loader.load()

    print(docs[0].page_content[0:200])

    print(docs[0].metadata)



#load_pdf()
#load_youtube()
#load_url()
#load_notion()