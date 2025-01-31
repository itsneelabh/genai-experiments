import chromadb

from RAGImplementation import embedding_model

client = chromadb.PersistentClient(path="./chroma_db", embedding_model="sentence-transformers/all-mpnet-base-v2")  # Specify the directory to store persistent data

collection = client.get_collection("langchain")

query_text = "What is common cold?"

results = collection.query(query_texts=[query_text], n_results=5)  # Get top

print(results)# 5 similar documents


# collection_names = client.list_collections()
# print("Available collections:", collection_names)