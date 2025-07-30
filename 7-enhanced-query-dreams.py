import os
import openai
import sys
import json
import numpy as np

from dotenv import load_dotenv, find_dotenv

from colorama import Fore, Style

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib
import pickle

from gliner import GLiNER

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

gliner = GLiNER.from_pretrained("urchade/gliner_base")

def extract_entities(text):
    labels = ["Pessoa", "Local", "Animal", "Conceito", "Data", "Cor"]
    return gliner.predict_entities(text, labels, threshold=0.5)

from colorama import Fore, Style, init
init(autoreset=True)

def pretty_print_dream(dream, score=None):
    from colorama import Fore, Style

    title = dream.metadata.get("title", "Sem título")
    date = dream.metadata.get("date", "Sem data")
    
    print(Fore.YELLOW + "🕰️  Data: " + Fore.CYAN + f"{date}")
    print(Fore.YELLOW + "🌙  Título: " + Fore.CYAN + f"{title}")
    print(Fore.YELLOW + "📖  Conteúdo:\n" + Fore.WHITE + f"{dream.page_content.strip()}")

    entities = extract_entities(dream.page_content.strip())

    print(Fore.YELLOW + "📖  Entidades:")
    for entity in entities:
        print(entity["text"], "=>", entity["label"])

    if score is not None:
        print(f"{Fore.YELLOW}Score: {score:.4f}{Style.RESET_ALL}")

    print(Fore.MAGENTA + "🔮" + "═" * 60)

persist_directory = 'vectordb/dreams/'
tfidf_vectorizer_file = persist_directory+'tfidf_vectorizer.pkl'
tfidf_matrix_file = persist_directory+'tfidf_matrix.npz'
chunks_file = persist_directory + 'documents.pkl'

def load_json_metadata(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    metadata["date"] = record.get("date")

    return metadata

def load_json():

    if os.path.exists(chunks_file):
        with open(chunks_file, 'rb') as f:
            all_chunks = pickle.load(f)
            return all_chunks

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

    with open(chunks_file, 'wb') as f:
        pickle.dump(all_chunks, f)

    return all_chunks

def build_semantic_index(all_chunks):

    if not all_chunks:
        all_chunks = load_json()
        
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    vectordb.persist()

    print("Base vetorial salva com sucesso.")

def build_tfidf_index(all_chunks):
    
    if not all_chunks:
        all_chunks = load_json()

    texts = [doc.page_content for doc in all_chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 1. Salva o modelo
    joblib.dump(vectorizer, tfidf_vectorizer_file)

    # 2. Salva a matriz esparsa
    sparse.save_npz(tfidf_matrix_file, tfidf_matrix)

def semantic_search(query, top_k=5):

    if not os.path.exists(persist_directory):
        build_semantic_index()
    else:
        print("Base já existe. Pulando criação.")
        
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    question = query
    results = vectordb.similarity_search_with_score(question, k=top_k)

    return results

    # print ("================================================")
    # print ("=== RESULTADOS DIVERGENTES =====================")
    # print ("================================================")
    # 
    # results = vectordb.max_marginal_relevance_search(question,k=3, fetch_k=10)
    # 
    # for result in results:
    #     pretty_print_dream(result)


def tdidf_search(query, top_k=5):

    all_chunks = load_json()

    if not os.path.exists(tfidf_vectorizer_file) or not os.path.exists(tfidf_matrix_file):
        build_tfidf_index(all_chunks)
    else:
        print("Base tdidf já existe. Pulando criação.")

    tfidf_vectorizer = joblib.load(tfidf_vectorizer_file)
    tfidf_matrix = sparse.load_npz(tfidf_matrix_file)

    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

    top_indices = np.argsort(tfidf_scores)[::-1][:top_k]

    return [(all_chunks[i], tfidf_scores[i]) for i in top_indices]

def hybrid_search(query, alpha=0.5, top_k=5):
    """
    alpha = peso do resultado semântico (0.0 a 1.0)
    """

    all_chunks = load_json()

    if not os.path.exists(persist_directory):
        build_semantic_index(all_chunks)
    else:
        print("Base semântica já existe. Pulando criação.")



    if not os.path.exists(tfidf_vectorizer_file) or not os.path.exists(tfidf_matrix_file):
        build_tfidf_index(all_chunks)
    else:
        print("Base tdidf já existe. Pulando criação.")

    tfidf_vectorizer = joblib.load(tfidf_vectorizer_file)
    tfidf_matrix = sparse.load_npz(tfidf_matrix_file)

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # --- Parte 1: TF-IDF ---
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

    # --- Parte 2: Embeddings semânticos ---
    semantic_results = vectordb.similarity_search_with_score(query, k=len(all_chunks))
    semantic_scores = np.zeros(len(all_chunks))

    # Indexar por conteúdo (assumindo que o conteúdo é igual ao de TF-IDF)
    content_to_index = {doc.page_content: i for i, doc in enumerate(all_chunks)}
    for doc, score in semantic_results:
        idx = content_to_index.get(doc.page_content)
        if idx is not None:
            semantic_scores[idx] = 1 - score  # 1 - distância para virar "similaridade"

    # --- Combinar scores ---
    final_scores = alpha * semantic_scores + (1 - alpha) * tfidf_scores
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [all_chunks[i] for i in top_indices]


def run_menu():
    while True:
        print(f"\n{Fore.CYAN}=== MENU DE BUSCA ==={Style.RESET_ALL}")
        print("1. Buscar com TF-IDF 🔍")
        print("2. Buscar com Semantic 🧠")
        print("3. Buscar com Hybrid 🧪")
        print("4. Sair 🚪")

        choice = input("Escolha uma opção (1-4): ").strip()

        if choice == "1" or choice == "2" or choice == "3":
            query = input("Digite sua busca: ")
            print(Fore.BLUE + "═" * 60)
            print(f"{Fore.GREEN}=== {query.upper()} {'═' * max(0, 60 - len(query) - 5)}{Style.RESET_ALL}")
            print(Fore.BLUE + "═" * 60)

        if choice == "1":
            results = tdidf_search(query, top_k=5)
            for result, score in results:
                pretty_print_dream(result, score)

        elif choice == "2":
            results = semantic_search(query)
            for result, score in results:
                pretty_print_dream(result, score)

        elif choice == "3":
            try:
                alpha = float(input("Peso do embedding (0.0 a 1.0): "))
            except ValueError:
                alpha = 0.5
            results = hybrid_search(query, alpha=alpha, top_k=5)
            for result in results:
                pretty_print_dream(result, score)

        elif choice == "4":
            print(f"{Fore.YELLOW}Saindo...{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Opção inválida. Tente novamente.{Style.RESET_ALL}")



if __name__ == "__main__":
    run_menu()
    