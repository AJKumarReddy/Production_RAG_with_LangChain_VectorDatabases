import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection("mycollection")

documents = [
    {"id":"doc1","text":"Hello"},
    {"id":"doc2","text":"How are you?"},
    {"id":"doc3","text":"Goodbye, see you later!"}
]

query = "Hello, World!"
# upsert : add or replace
# get  : retrieve
# delete : delete
# update : update
# add : add only if not exists

for doc in documents:
    collection.upsert(
        documents=[doc["text"]],
        ids=[doc["id"]],
        metadatas=[{"source":"local","id":doc["id"]}]
    )

results = collection.query(
    query_texts=query,
    n_results=3
)

print(f"\n Results:",results, "\n\n")
