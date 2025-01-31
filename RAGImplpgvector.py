from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Step 1: Load data from the text file
def load_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    # Create LangChain Document objects
    docs = [Document(page_content=line.strip(), metadata={"id": i}) for i, line in enumerate(lines)]
    return docs

# Step 2: Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 3: Configure the PostgreSQL connection
connection_string = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "rag_documents"

# Step 4: Initialize the vector store
def initialize_vector_store(connection_string, collection_name, embedding_length):
    # Create or connect to the vector store
    vector_store = PGVector(
        embeddings=embedding_model,
        connection=connection_string,
        collection_name=collection_name,
        embedding_length=embedding_length,  # Specify the embedding dimension
        use_jsonb=True,  # Use JSONB for metadata storage
    )
    return vector_store

# Step 5: Add documents to the vector store
def add_documents_to_vector_store(vector_store, embedding_model, docs):
    # Generate embeddings and add documents to the vector store
    vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs], embedding=embedding_model)

# Step 6: Initialize the Ollama LLM
def initialize_llm():
    llm = Ollama(
        model="llama3.2",  # Name of the model pulled via Ollama
        temperature=0.7,  # Sampling temperature
        num_ctx=2048,  # Context length
    )
    return llm

# Step 7: Build the RAG pipeline
def build_rag_pipeline(vector_store, llm):
    # Define a custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa_chain

# Main function
if __name__ == "__main__":
    # Load data from the text file
    file_path = "documents.txt"
    docs = load_data(file_path)

    # Initialize the vector store
    embedding_length = 768  # Dimensionality of the embeddings
    vector_store = initialize_vector_store(connection_string, collection_name, embedding_length)

    # Add documents to the vector store
    add_documents_to_vector_store(vector_store, embedding_model, docs)

    # Initialize the Ollama LLM
    llm = initialize_llm()

    # Build the RAG pipeline
    rag_chain = build_rag_pipeline(vector_store, llm)

    # Query the RAG system
    query = "What is artificial intelligence?"
    result = rag_chain({"query": query})

    # Print the results
    print("Query:", query)
    print("\nAnswer:", result["result"])
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(f"- {doc.page_content} [{doc.metadata}]")