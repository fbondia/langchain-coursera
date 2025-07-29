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
pip install langchain pypdf yt_dlp pydub dotenv openai langchain-community beautifulsoup4 tiktoken langchain-openai langchain_chroma chromadb lark scikit-learn jq
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

Após carregar o documento e dividi-lo em pedaços menores (chunks), é necessário transformar cada chunk em um vetor numérico que represente seu conteúdo de forma semântica. Esse processo é chamado de embedding, e o vetor resultante é conhecido como um embedding vector (ou simplesmente embedding).

Esses vetores geralmente são armazenados em uma base vetorial (vector store), permitindo buscas por similaridade com base no conteúdo, em vez de apenas palavras-chave.

Neste exemplo usamos o base vetorial Chroma, mas poderia ser o FAISS, Weaviate etc.

Importante notar que neste caso estamos usando o OpenAIEmbeddings, consumindo a API da OpenAI. A vetorização dos chunks **gera custo financeiro**. Em relação a isso, algumas dicas:

### 📊 Preços (julho/2025)

- `text-embedding-3-small`: **0.02 USD por 1 milhão de tokens**
- `text-embedding-3-large`: **0.13 USD por 1 milhão de tokens**

Fonte oficial: [https://openai.com/pricing](https://openai.com/pricing)

### 🔢 Exemplo prático

Se você tiver 100 documentos com 500 tokens cada:

- Total de tokens: 100 × 500 = **50.000 tokens**
- Custo estimado com `text-embedding-3-small`:  
  → 50.000 × 0.00002 USD = **0.01 USD**

### 💡 Dicas para economizar

- Prefira o modelo `text-embedding-3-small`, que é mais barato e eficiente.
- Ajuste o tamanho dos chunks para otimizar o número de tokens por embedding.
- Use modelos **locais e gratuitos** caso a máxima precisão da OpenAI não seja necessária.

### 🆓 Alternativas gratuitas (locais)

Você pode usar modelos open-source com `HuggingFaceEmbeddings` ou `LangChain`:

- `all-MiniLM-L6-v2`
- `intfloat/e5-small-v2`
- `thenlper/gte-small`

Esses modelos funcionam bem em pipelines RAG, sem custo por token.


### 4 - RETRIEVAL

Nesta etapa, consultamos de fato a base semântica (vector store). A busca retorna, por padrão, os **chunks mais similares** à consulta — ou seja, aqueles cujo *embedding* tem menor distância vetorial em relação ao *embedding* da pergunta.

No entanto, **os resultados mais similares nem sempre são os mais úteis**. Em muitos casos, pode ser mais interessante obter respostas que tragam **diversidade informativa**, e não apenas repetições do mesmo contexto. É aqui que entra o algoritmo **MMR (Maximal Marginal Relevance)**.

O MMR busca um **equilíbrio entre relevância e novidade**: resultados que sejam suficientemente **relacionados à consulta**, mas ao mesmo tempo **diferentes entre si**, evitando redundância.

#### 📌 Exemplo ilustrativo

Imagine que um cozinheiro faz a pergunta:  
**"Quais cogumelos são totalmente brancos?"**

- Os resultados mais similares podem descrever **uma única espécie em detalhes** (como o champignon).
- Porém, um trecho que mencione que **"uma espécie branca é venenosa"**, mesmo sendo menos similar, pode ser **crucial** — e o MMR ajuda a trazê-lo para os resultados.

➡️ Em resumo: **MMR busca relevância com a consulta, mas diversidade em relação aos demais resultados**.

---

### 🔎 Estratégias adicionais de recuperação

Além da busca por similaridade ou MMR, existem outras estratégias que podem ser aplicadas em sistemas baseados em embeddings:

#### 📘 LLM-Aided Retrieval

Algumas consultas possuem tanto um **elemento semântico** quanto um **filtro explícito**.  
Exemplo:  
**"Quais filmes de terror foram lançados em 1980?"**

- Parte semântica: *filmes de terror*  
- Parte estrutural: *ano de lançamento = 1980*

Esse tipo de consulta pode ser tratado com uma estratégia chamada **LLM-Aided Retrieval**, onde o LLM ajuda a **entender, expandir ou reformular a consulta**, e a engine de busca aplica filtros estruturados.

#### 🧠 Compressão com LLM

Após recuperar diversos chunks, é possível usar um LLM para **resumir, combinar ou comprimir** os resultados antes de adicioná-los ao prompt final.

Essa estratégia é útil para:
- **Reduzir o custo e o tamanho do prompt**
- **Aumentar a densidade de informação por token**

> ⚠️ Naturalmente, há um **trade-off**: você pode ganhar espaço e velocidade, mas corre o risco de perder nuances importantes se a compressão for excessiva ou mal conduzida.

## 5 - QUESTION - ANSWER

O fluxo básico é:

1 - questão é submetida para o vector store
2 - vector store provê n documentos relevantes
3 - questão original e chunks são enviados para LLM

### 🧠 Modos de resposta no LangChain: `stuff`, `map_reduce`, `refine` e `map_rerank`

Ao usar LangChain com múltiplos documentos (ou chunks), podemos escolher diferentes estratégias para combinar as informações e gerar uma resposta final com o LLM.

---

#### 1. `stuff` — Tudo de uma vez

> Envia todos os documentos concatenados diretamente no prompt do modelo.

- ✅ Simples e rápido
- ❌ Limitado pelo tamanho máximo de tokens do modelo
- Ideal para poucos documentos pequenos

```text
[documento1] + [documento2] + ... → LLM responde com base em todos
```

#### 2. map_reduce — Processamento paralelo + resumo

> O modelo processa cada chunk individualmente (map) e depois resume todas as respostas (reduce).

- 🗺 Map: o LLM responde cada documento separadamente
- 🧾 Reduce: combina as respostas em um único resumo final
- ✅ Escalável para muitos documentos
- ❌ Pode perder o contexto global

```
[doc1 → resp1], [doc2 → resp2], ... → resumo final das respostas
```

#### 3. refine — Construção progressiva

> Cria uma resposta inicial com o primeiro chunk e refina iterativamente com os demais.

- Cada nova iteração melhora ou expande a resposta anterior
- ✅ Mantém coerência entre passos
- ❌ Pode propagar erros do início se a resposta inicial for fraca

```
resp1 = resposta(doc1)  
resp2 = refinar(resp1 + doc2)  
resp3 = refinar(resp2 + doc3)  
... → resposta final
```

#### 4. map_rerank — Seleção da melhor resposta

> O modelo avalia cada chunk individualmente e atribui uma pontuação de relevância a cada resposta.

- ✅ Retorna a melhor resposta individual
- ❌ Pode ignorar outras informações relevantes

```
[doc1 → (resposta1, score1)], [doc2 → (resposta2, score2)], ...  
→ retorna a resposta com maior score
```

#### 🧭 Qual usar?

| Modo           | Quando usar                                          |
-----------------|------------------------------------------------------|
| stuff          | Poucos documentos curtos                             |
| map_reduce     | Muitos documentos, resposta resumida                 |
| refine         | Deseja construir a resposta progressivamente         |
| map_rerank     | Quer selecionar a melhor resposta entre as possíveis |

