from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings)

# print the embeddings into a file embeddings.txt
with open("embeddings.txt", "w") as f:
    for emb in embeddings:
        f.write(" ".join(str(x) for x in emb) + "\n")