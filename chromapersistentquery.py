import chromadb

client = chromadb.PersistentClient(path="./chroma_db", emb)  # Specify the directory to store persistent data

collection = client.get_collection("langchain")

query_text = "What is common cold?"

results = collection.query(query_texts=[query_text], n_results=5)  # Get top

print(results)# 5 similar documents


# collection_names = client.list_collections()
# print("Available collections:", collection_names)