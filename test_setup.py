# save as test_setup.py in your project folder and run: python test_setup.py

import os
from dotenv import load_dotenv
load_dotenv()

print("Testing all connections...\n")

# Test Groq
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5
    )
    print("✓ Groq working:", response.choices[0].message.content)
except Exception as e:
    print("✗ Groq failed:", e)

# Test Voyage AI
try:
    import voyageai
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    result = vo.embed(["test financial document"], model="voyage-finance-2")
    print("✓ Voyage AI working: embedding dimension", len(result.embeddings[0]))
except Exception as e:
    print("✗ Voyage AI failed:", e)

# Test Cohere
try:
    import cohere
    co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
    result = co.rerank(
        model="rerank-english-v3.0",
        query="test query",
        documents=["doc one", "doc two"],
        top_n=1
    )
    print("✓ Cohere working: reranker responded")
except Exception as e:
    print("✗ Cohere failed:", e)

# Test ChromaDB
try:
    import chromadb
    client = chromadb.Client()
    col = client.create_collection("test")
    print("✓ ChromaDB working: local instance created")
except Exception as e:
    print("✗ ChromaDB failed:", e)

# Test LangSmith
try:
    from langsmith import Client
    ls = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
    print("✓ LangSmith working: client initialized")
except Exception as e:
    print("✗ LangSmith failed:", e)

print("\nSetup check complete.")