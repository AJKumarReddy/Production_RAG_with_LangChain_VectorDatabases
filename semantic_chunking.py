from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

doument = '''
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
'''

# Recursice Chunking

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=400,
    chunk_overlap=50,
)

recursive_chunks = recursive_splitter.split_text(doument)
print(f"Recursive Chunks: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks):
    print(f"Chunk {i+1}: {len(chunk)} characters")
    print(chunk)



# Semantic Chunking

sematic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,
)

sematic_chunks = sematic_chunker.split_text(doument)
print(f"Semantic Chunks: {len(sematic_chunks)}")
for i, chunk in enumerate(sematic_chunks):
    print(f"Chunk {i+1}: {len(chunk)} characters")
    print(chunk)    



# Create Vector Store
recursive_vector_store = Chroma.from_texts(recursive_chunks, embeddings, collection_name="recursive_chunks")
print(f"Recursive Vector Store: {recursive_vector_store}")

semantic_vector_store = Chroma.from_texts(sematic_chunks, embeddings, collection_name="semantic_chunks")
print(f"Semantic Vector Store: {semantic_vector_store}")

test_queries = [
    "How to authenticate with OAuth2?",
    "What happens when I hit the rate limit?",
    "How are webhooks secured?",
    "What format are errors returned in?"
]

def test_retrieval(query, vector_store, name):
    print(f"\n--- Testing {name} ---")
    retrieved_docs = vector_store.similarity_search(query, k=1)
    for i, doc in enumerate(retrieved_docs):
        print("---"* 10)
        print(doc.page_content)
        print("---"* 10)

for query in test_queries:
    print("\n" + "#" * 20)  
    print(f"Query: {query}")
    print("#" * 20 + "\n")

    test_retrieval(query, recursive_vector_store, "Recursive Chunking")
    test_retrieval(query, semantic_vector_store, "Semantic Chunking")
    