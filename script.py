import chromadb
from chromadb.utils import embedding_functions
import json

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create an embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get the collection with the specified embedding function
collection = client.get_or_create_collection(
    name="tweets",
    embedding_function=embedding_function
)

# Function to add tweets to the database
def add_tweets(tweets):
    ids = [str(tweet["id"]) for tweet in tweets]
    contents = [tweet["text"] for tweet in tweets]
    
    collection.add(
        ids=ids,
        documents=contents,
        metadatas=[{"text": content} for content in contents]
    )

# Function to query tweets
def query_tweets(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    return results

# Example usage
if __name__ == "__main__":
    # Example tweets
    tweets = [
        {"id": 1, "text": "Just had a great cup of coffee!"},
        {"id": 2, "text": "Excited about the new AI developments"},
        {"id": 3, "text": "Beautiful sunset at the beach today"}
    ]
    
    # Add tweets to the database
    add_tweets(tweets)
    
    # Query the database
    query = "What's new in AI?"
    results = query_tweets(query)
    
    print(f"Query: {query}")
    for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
        print(f"{i+1}. Tweet: {metadata['text']} (Distance: {distance})")
