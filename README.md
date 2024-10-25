# RAG Language Model using Pinecone, LangChain, and OpenAI API

Project was inspired by [this Youtube video](https://www.youtube.com/watch?v=BrsocJb-fAo&list=PLm6JKTYp_lsYob9yAS_yZmDXYKga3VciZ)
This project is an implementation of a Retrieval-Augmented Generation (RAG) model that combines retrieval from a custom knowledge base (using Pinecone) with OpenAI's language model API. The goal is to provide precise and contextually relevant responses by retrieving relevant data and passing it as input to the language model.

The project includes essential setup steps, API integration, transcription of YouTube videos, text processing, and retrieval-based question answering. Full code is provided in the `notebooks/testing.ipynb` file.

## Setup and Requirements

To run this project, you’ll need API keys for both Pinecone and OpenAI. Place your keys in a `.env` file as follows:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Make sure to have all necessary dependencies installed. You can install them using:
```bash
poetry install
poetry shell
```
## Project Structure

### 1. Load Environment Variables
The project begins by loading environment variables from the .env file for accessing API keys.
```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=cdiD-9MMpb0"
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
```

### 2. Model and Output Parsing
The OpenAI language model is initialized with a specified model type, and an output parser is set up for handling responses.
```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser = StrOutputParser()

chain = model | parser
chain.invoke("What is the meaning of life?")
```


### 3. Prompt Template
We define a prompt template to structure the question and context for the model's responses.
```python
from langchain.prompts import ChatPromptTemplate

template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
prompt.format(context="Mary's favorite food is pizza.", question="What is Mary's favorite food?")
```


### 4. YouTube Transcription
The project utilizes the Whisper model and pytube to transcribe audio from a YouTube video if a transcription is not already available.
```python
import tempfile
import whisper
from pytube import YouTube  

if not os.path.exists("transcription.txt"):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()
    whisper_model = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open("transcription.txt", "w") as f:
            f.write(transcription)

with open("transcription.txt", "r") as f:
    transcription = f.read()
```


### 5. Document Loading and Embedding Creation
Documents are loaded, split, and embedded using OpenAI embeddings. These embeddings are then indexed using Pinecone for efficient retrieval.
```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_pinecone import PineconeVectorStore

loader = TextLoader("transcription.txt")
text_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

embeddings = OpenAIEmbeddings()
index_name = "youtube-idx"

pinecone = PineconeVectorStore.from_documents(
    documents=documents, embedding=embeddings, index_name=index_name
)
```


### 6. Query Execution with Retrieval-Augmented Generation
A RAG chain is established where the retriever supplies context for the question-answering task. Pinecone serves as the retriever to gather relevant information based on similarity scores.
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()} 
    | prompt 
    | model 
    | parser
)

chain.invoke("Do aliens exist?")
```


## Usage

### Query the Model
To use the model, simply invoke chain.invoke(<YOUR_QUERY>) with any question. The model will retrieve context from the knowledge base and generate a response accordingly.

### Customization
You can modify the template variable to adjust the prompt structure or fine-tune the RecursiveCharacterTextSplitter parameters for different chunk sizes and overlaps, depending on the granularity of the documents.







