from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter

# Load documents (replace with your own text file)
loader = TextLoader("./documents.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model (NeuML/pubmedbert-base-embeddings)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create a Chroma vector store from the documents
vector_store = Chroma.from_documents(texts, embedding_model, persist_directory="./chroma_db")

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
query = "What is the main topic of the document?"
response = qa_chain({"query": query})

# Print the response
print("Answer:", response["result"])
print("Source Documents:", response["source_documents"])