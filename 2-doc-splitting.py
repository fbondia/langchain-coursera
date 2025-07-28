import os
import openai
from dotenv import load_dotenv, find_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import NotionDirectoryLoader

# 🔐 Carrega chave da OpenAI do .env
_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.environ['OPENAI_API_KEY']

# CharacterTextSplitter
# Faz cortes brutos com base em um único separador e tamanho fixo (ex: espaço ou \n)
# Não considera contexto semântico → pode cortar no meio de uma frase ou palavra

# RecursiveCharacterTextSplitter
# Tenta cortar de forma "inteligente", seguindo uma hierarquia de separadores (parágrafos > linhas > frases > palavras > caractere bruto)
# Preserva melhor o contexto entre chunks, ideal para textos longos e estruturados

# TokenTextSplitter
# Separa os chunks por número de tokens. Pode ser útil ao considerar que o contexto colocado no prompt tem limites baseados em número de tokens.

# MarkdownHeaderTextSplitter
# Separa os chunks por headers no markdown e armazena seus metadados

def split_raw_string():
    chunk_size = 26
    chunk_overlap = 4

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    c_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 🔤 Texto contínuo sem espaços (apenas letras do alfabeto)
    text1 = 'abcdefghijklmnopqrstuvwxyz'
    print("\n🔍 Recursive split (text1):")
    print(r_splitter.split_text(text1))

    text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
    print("\n🔍 Recursive split (text2):")
    print(r_splitter.split_text(text2))

    # 🧩 Texto com espaços entre letras
    text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"

    print("\n🔍 Recursive split (text3 - com espaços):")
    print(r_splitter.split_text(text3))

    print("\n🔍 Character split (text3 - com espaços):")
    print(c_splitter.split_text(text3))

    # 🔧 Força o separador do CharacterTextSplitter como ' '
    c_splitter_custom = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=' '
    )
    print("\n🔍 Character split (com separador explícito ' '):")
    print(c_splitter_custom.split_text(text3))

def split_document_text():
    # 📄 Texto estruturado com parágrafos, frases e espaços
    some_text = """Ao escrever documentos, os autores utilizam a estrutura do documento para agrupar o conteúdo.
    Isso pode transmitir ao leitor quais ideias estão relacionadas. Por exemplo, ideias intimamente relacionadas estão em sentenças. Ideias semelhantes estão em parágrafos. Os parágrafos formam um documento.

    Parágrafos geralmente são delimitados por uma quebra de linha ou duas quebras de linha.
    As quebras de linha são o "barra n" (\\n) que você vê incorporado nessa string.
    As sentenças terminam com um ponto final, mas também têm um espaço.
    E as palavras são separadas por espaços."""

    print(f"\n📏 Tamanho do texto original: {len(some_text)} caracteres\n")

    # 🔹 CharacterTextSplitter com separador por espaço
    c_splitter = CharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separator=' '
    )

    # 🔷 RecursiveCharacterTextSplitter com separadores customizados
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # 🔷 RecursiveCharacterTextSplitter com separadores padrão
    r_default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0
    )

    print("🔹 CharacterTextSplitter (por espaço):")
    print(*c_splitter.split_text(some_text), sep="\n")

    print("\n🔷 RecursiveCharacterTextSplitter (com separadores explícitos):")
    print(*r_splitter.split_text(some_text), sep="\n")

    print("\n🔷 RecursiveCharacterTextSplitter (com separadores padrão):")
    print(*r_default_splitter.split_text(some_text), sep="\n")


    # Reduzindo um chunk e incluindo o ponto como separador

    print("\n🔷 RecursiveCharacterTextSplitter (\\n\\n, \\n, ., ' ', ''):")
    r_splitter1 = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print(*r_splitter1.split_text(some_text), sep="\n")

    print("\n🔷 RecursiveCharacterTextSplitter (\\n\\n, \\n, (?<=\\. ), ' ', ''):")
    r_splitter2 = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    print(*r_splitter2.split_text(some_text), sep="\n")

def split_pdf():
    loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    docs = text_splitter.split_documents(pages)
    print(len(docs))
    print(len(pages))

def split_notion():
    loader = NotionDirectoryLoader("docs/Notion_DB")
    notion_db = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    docs = text_splitter.split_documents(notion_db)
    print(len(notion_db))
    print(len(docs))

def split_token():

    loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    pages = loader.load()

    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    text1 = "foo bar bazzyfoo"
    print(*text_splitter.split_text(text1), sep="\n")

    text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    print(docs[0])
    print(pages[0].metadata)

def split_markdown():

    markdown_document = """# Title\n\n \
    ## Chapter 1\n\n \
    Hi this is Jim\n\n Hi this is Joe\n\n \
    ### Section \n\n \
    Hi this is Lance \n\n 
    ## Chapter 2\n\n \
    Hi this is Molly"""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)

    print(md_header_splits[0])
    print(md_header_splits[1])


    loader = NotionDirectoryLoader("docs/Notion_DB")
    docs = loader.load()
    txt = ' '.join([d.page_content for d in docs])

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    md_header_splits = markdown_splitter.split_text(txt)

    print (*md_header_splits[0], sep="\n")



# split_raw_string()
# split_document_text()
# split_pdf()
# split_notion()
# split_token()
split_markdown()