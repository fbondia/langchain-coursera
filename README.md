# 🧪 Laboratório LangChain com seus próprios dados

Este repositório reúne anotações e códigos experimentais baseados no curso ["LangChain: Chat with Your Data"](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/snupv/introduction), com foco em compreender e testar as etapas principais de processamento de documentos: **LOAD → SPLIT → EMBED → RETRIEVE**.


## Baseado no curso em:

https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/snupv/introduction


## 📄 Documento usado nos testes

O PDF referenciado em alguns exemplos de código vem do curso de Stanford. Para baixá-lo, execute:

```bash
curl -O https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf
```

Salve em: **./docs/cs229_lectures/**

## Recomendável criar o ambiente com virtualenv

```
virtualenv env
source env/bin/activate
```

## Dependências

```
pip install langchain pypdf yt_dlp pydub dotenv openai langchain-community beautifulsoup4 tiktoken
```

## O processo com o LangChain ocorre nas etapas LOAD, SPLIT, EMBED, RETRIEVE

### 1 - LOAD

Uma parte importante do LangChain são seus DocumentLoaders, que podem carregar:

**Dados públicos**
- dados não estruturados: youtube, wikipedia, twiter etc.
- dados estruturados: datasets, apis, csv etc.

**Dados privados**
- dados não estruturados: power-point, notion, whatsapp etc.
- dados estruturados: pandas, excel, stripe, elastic etc.


#### 2 - SPLIT
Após o carregamento, os documentos precisam ser divididos em pedaços menores (chunks) para que possam ser processados e indexados eficientemente — especialmente em sistemas de busca vetorial.

##### Desafios do split
Dividir um texto apenas por número de caracteres pode cortar frases ou ideias importantes no meio, dificultando a recuperação de informações relevantes.

##### Estratégias para preservar contexto
Uma prática comum é usar overlapping entre chunks:

Por exemplo, se o chunk A vai de 0 a 500 tokens, o chunk B pode começar no token 400 e ir até o 900, criando uma sobreposição de 100 tokens. Isso ajuda a manter o contexto fluido entre trechos consecutivos.


##### Principais Tipos de Text Splitters no LangChain

Os Text Splitters são responsáveis por dividir documentos em pedaços menores (**chunks**) para facilitar o processamento, indexação e busca. A escolha do splitter certo pode impactar diretamente a qualidade das respostas em sistemas baseados em RAG (retrieval-augmented generation).

###### CharacterTextSplitter

Divide o texto com base em um número fixo de caracteres, sem considerar a estrutura semântica. É simples e rápido, mas pode cortar ideias no meio se não for bem parametrizado.

```python
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
```

###### MarkdownHeaderTextSplitter
Divide documentos com base na hierarquia de cabeçalhos Markdown (#, ##, ### etc.), preservando a estrutura do conteúdo. Ideal para documentação técnica.

```
from langchain.text_splitter import MarkdownHeaderTextSplitter
```

###### TokenTextSplitter
Usa contagem de tokens em vez de caracteres. É útil para evitar que o chunk ultrapasse o limite de tokens de modelos como GPT.

```
from langchain.text_splitter import TokenTextSplitter
```

###### SentenceTransformersTokenTextSplitter
Usa embeddings para identificar pontos de corte semântico entre sentenças, mantendo a coesão do texto. Requer a instalação do sentence-transformers.

```
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
```

###### RecursiveCharacterTextSplitter
O splitter mais recomendado na maioria dos casos. Tenta dividir primeiro por seções maiores (como parágrafos, frases) e vai recursivamente até atingir o tamanho ideal. Preserva bem o contexto.

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

###### NLTKTextSplitter
Baseado na biblioteca NLTK, divide textos em sentenças respeitando a pontuação e a gramática. Requer o nltk instalado e seus recursos baixados.

```
from langchain.text_splitter import NLTKTextSplitter
```

###### SpacyTextSplitter
Semelhante ao NLTKTextSplitter, mas usa o spaCy para análise sintática mais precisa. Pode reconhecer entidades, frases nominais, etc.

```
from langchain.text_splitter import SpacyTextSplitter
```

###### Language (Code Splitter)

Pensado para código-fonte. Usa a linguagem de programação como base para dividir trechos de código por classes, funções ou blocos. Muito útil em agentes que lidam com bases de código.

```
from langchain.text_splitter import Language
```


##### Informações interessantes
- Os chunks armazenam metadados relacionados a sua fonte - um chunk separado a partir de um pdf pode ter o nome do arquivo e a página em que foi produzido; um markdown pode ter a estrutura de cabeçalhos na qual se originou; etc


### 3 - EMBED
