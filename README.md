# AI-powered document analysis and Q&A using Small Language Model(SLM)
# Objective:
The goal of this project is to implement a Retrieval-Augmented Generation (RAG) pipeline to answer user questions by retrieving relevant information from a PDF document. The pipeline leverages FAISS IVFFlat for efficient similarity search, Phi3:medium as the local language model for generating responses, and nomic-embed-text for creating high-quality embeddings. This approach combines retrieval and generation to provide accurate and context-aware answers.
# Why use SLMs over LLMs at all?
1. Runs Locally (No Internet Required): SLMs can function entirely offline, making them ideal for secure and privacy-sensitive applications.
2. Cost-Effective: Since they don’t require cloud API calls, they eliminate subscription fees associated with LLM services like OpenAI or Google Gemini.
3. Faster Response Time: Without network latency, SLMs provide real-time responses, making them ideal for low-latency applications.
4. Easier Mobile & Embedded Integration: SLMs can be deployed on mobile devices, Raspberry Pi, and IoT devices, enabling AI-powered applications without heavy infrastructure.
5. Better Data Privacy: Since no data is sent to external servers, SLMs are preferred for healthcare, legal, and enterprise applications where data privacy is critical.
6. Energy Efficient: Consumes less power, making them suitable for battery-operated devices and sustainable AI solutions.
# Codebase Overview

The codebase consists of two Jupyter notebooks:
- small-text.ipynb – Designed for processing smaller documents. It utilizes RecursiveCharacterTextSplitter for chunking and FAISS indexing for efficient retrieval.

- large-text.ipynb – Optimized for handling larger documents (megabyte-sized files). It implements batch processing along with RecursiveCharacterTextSplitter and uses the FAISS IndexIVFFlat structure to efficiently index and retrieve large text chunks. This optimization significantly reduces computational time for larger files.

Both notebooks are structured to be easily integrated into a Streamlit web application, making the system accessible for a wider range of users. The modular design allows seamless deployment, enabling users to upload PDFs, process them, and perform question-answering efficiently.

# Scope
1.	Legal Document Analysis: Extract and analyze key information from legal contracts, case files, or compliance documents.Provide quick answers to specific legal queries.
2.	Research Paper Summarization: Summarize lengthy research papers into concise, actionable insights.Retrieve specific sections or data points from academic documents.
3.	Technical Documentation Querying: Enable users to quickly find solutions or explanations from technical manuals or guides.Assist developers in navigating complex documentation.
4.	Educational Content Retrieval: Help students and educators access specific information from textbooks or study materials.Provide detailed explanations or summaries of educational content.
5.	Local Model Usage: The project uses Phi3:medium , a locally available language model, eliminating the need for internet access.Ensures data privacy and reduces dependency on external APIs or cloud services.


# Retrieval-Augmented Generation (RAG) Pipeline
Pipeline Overview

This RAG pipeline processes user-uploaded PDFs, extracts text, converts it into embeddings using nomic-embed-text, and retrieves relevant information using FAISS IndexIVFFlat. The retrieved text chunks are then passed to the phi3:medium language model to generate accurate responses based on user queries.
Pipeline Workflow
## 1. Upload and Process PDF

A PDF file is uploaded (hardcoded in the notebook for this implementation).Text is extracted and split into manageable chunks using RecursiveCharacterTextSplitter.

## 2. Chunking and Embedding Generation

Chunks are grouped into batches for efficient processing.Each batch is converted into embeddings using OllamaEmbeddings (nomic-embed-text).All generated embeddings are stored in the Embeddings variable.

## 3. FAISS Indexing and Storage

The embeddings are used to train a FAISS IndexIVFFlat for fast similarity search.
After training, all embeddings are indexed and stored.
    The FAISS index (including cluster centroids, inverted file structure, and embeddings) is saved to disk.
    The associated text chunks are also stored for retrieval.

## 4. Setting Up the Language Model

The phi3:medium model is loaded locally to generate responses based on retrieved context.

## 5. Query Processing and Retrieval

A Prompt Template is created with input variables (i.e., the user's question).
The prompt instructs the LLM to generate two alternative versions of the user’s question for improved retrieval.
    The MultiQueryRetriever is used to combine the vector database retriever with the LLM.

## 6. Retrieval Process

The user’s question is passed to the retriever.
The LLM generates two alternative questions to enhance retrieval.
    The vector database retrieves relevant documents using both the original and alternative questions.

## 7. Output Generation

The retrieved documents are returned to the user.
    The phi3:medium model uses these documents as context to generate the final answer.


# Why phi3:medium and phi3:mini both small language models are used?
## phi3:medium (Used in llm = ChatOllama(model=local_model))

### Role: 
Used in the MultiQueryRetriever to generate alternative queries for better document retrieval.
### Impact: 
Since this step only involves generating a few alternative queries, using a larger model (medium) ensures that the reformulated questions are high-quality and diverse.
        More computationally expensive but enhances retrieval accuracy.

## phi3:mini (Used in chain)
### Role: 
Used for answer generation after retrieving relevant documents.
### Impact:
Since phi3:mini is a smaller model, it processes the retrieved context faster and generates responses with lower latency.
        Response quality might be slightly lower compared to phi3:medium, but since the retrieval step ensures relevant context, the smaller model can still generate meaningful answers.

#### Here we have used a hybrid approach that  balances efficiency and accuracy— Phi3:medium provides retrieval benefits from a stronger model, while Phi3:mini responses are optimized for speed. If we want better answer quality, we can use Phi3:medium in both cases.If we want faster answer accuracy then we can use mini model


