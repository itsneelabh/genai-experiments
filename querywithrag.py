from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize the embedding model (same as before)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the persisted Chroma vector store
vector_store = Chroma(
    persist_directory="./chroma_db",  # Directory where the vector store is persisted
    embedding_function=embedding_model
)

# Initialize the locally running Ollama model (Llama 3.2)
llm = Ollama(model="llama3.2")  # Replace with your Ollama model name

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Query the RAG system
query = "What is artificial intelligence?"
response = qa_chain({"query": query})

# Print the response
print("Answer:", response["result"])
#print("Source Documents:", response["source_documents"])