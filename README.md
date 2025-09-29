##  RAG Pipeline for Kidney Disease Research Papers

I implemented a **Retrieval-Augmented Generation (RAG) pipeline** to collect and manage research papers related to **kidney diseases**.  
The pipeline consists of several key steps: **data ingestion, text chunking, embedding generation, vector database storage, and query-based retrieval.**

---

### Workflow Overview
1. **Data Ingestion** – Load research papers in PDF/text format.  
2. **Text Splitting** – Break documents into smaller chunks using LangChain’s `RecursiveCharacterTextSplitter` for better retrieval.  
3. **Embedding Generation** – Convert text chunks into dense vector embeddings using **Sentence Transformers (`all-MiniLM-L6-v2`)**.  
4. **Vector Store** – Store embeddings in **ChromaDB** for efficient similarity search.  
5. **RAG Retriever** – Retrieve the most relevant documents for user queries.

---

### Code Snippet: Text Splitting into Chunks
```python
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs
````

---

### Code Snippet: Embedding Generation

```python
embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)
```

---

### Code Snippet: Storing in Vector Database

```python
vectorstore = VectorStore(collection_name="pdf_documents")
vectorstore.add_documents(chunks, embeddings)
```

---

###  Code Snippet: Retrieval with RAG

```python
rag_retriever = RAGRetriever(vectorstore, embedding_manager)
results = rag_retriever.retrieve("What is Chronic Kidney Disease?")
```

---

###  Features

*  **PDF/Text ingestion** of research papers
*  **Chunking** with overlap for contextual understanding
*  **Semantic embeddings** with Sentence Transformers
*  **Vector database (ChromaDB)** for persistent storage
*  **Retriever module** for query-based search and ranking

This setup enables **efficient retrieval of kidney disease research papers**, supporting downstream tasks like **summarization, Q&A, and literature review automation**.

---
