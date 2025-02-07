import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")

collection.add(
    documents=[
        "This is a document about New York",
        "This is a document about Zurich",
        "This is a document about New Delhi",
        "This is a document about Tokyo"
    ],
    ids=["id1", "id2", "id3", "id4"]
)

#documents = collection.get(ids=["id1"])
#print(documents)

#query to fetch data from Chroma collection
results = collection.query(
    query_texts=["alps"], # Chroma will embed this for you
    n_results=4 # how many results to return
)
print(results)
