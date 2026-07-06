from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Documents with both semantic content and specific identifiers
documents = [
    Document(
        page_content=(
            "Product SKU-7742X is our flagship router. "
            "It supports gigabit speeds and advanced QoS features."
        ),
        metadata={"type": "product"},
    ),

    Document(
        page_content=(
            "For network connectivity issues, first check the "
            "ethernet cable and router status lights."
        ),
        metadata={"type": "troubleshooting"},
    ),

    Document(
        page_content=(
            "Error code E_CONN_REFUSED indicates the server "
            "rejected the connection. Check firewall settings."
        ),
        metadata={"type": "error"},
    ),

    Document(
        page_content=(
            "The authentication process requires valid credentials. "
            "Use OAuth2 for secure API access."
        ),
        metadata={"type": "auth"},
    ),

    Document(
        page_content=(
            "Router configuration guide: Access the admin panel "
            "at 192.168.1.1 to modify settings."
        ),
        metadata={"type": "config"},
    ),

    Document(
        page_content=(
            "WCAG 2.1 compliance requires all images to have "
            "alt text and sufficient color contrast."
        ),
        metadata={"type": "compliance"},
    ),
]

print(f"loaded {len(documents)} documents")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="hybrid_db"
)

#Vector Retriver
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#BM25 Retriever
bm25 = BM25Retriever.from_documents(documents, k=3)

# Ensemble Retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25],
    weights=[0.5, 0.5] # closer to vector search or bm25
)


print ("Hybrid retriever Ready")

def test_query(query, name, retriever):
    '''Test and print the results of a query '''
    results = retriever.invoke(query)
    print(f'\nRetriver: {name}  - Query: {query}')
    for i, doc in enumerate(results, start=1):
        print(f' {i}. {doc.page_content}')

    return results

test_queries = [
    'SKU-7742X specifications',
    'E_CONN_REFUSED error',
    'How do I authenticate',
    'WCAG Compliance',
    'router configuration'
    ]

for query in test_queries:
    print('\n'+'='*60)
    vector_results = test_query(query, 'Vector', vector_retriever)
    bm25_results = test_query(query, 'BM25', bm25)
    ensemble_results = test_query(query, 'Ensemble', ensemble_retriever)