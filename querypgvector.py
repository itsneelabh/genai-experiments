from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "rag_documents"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

results = vector_store.similarity_search(
    "tiger", k=2
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")