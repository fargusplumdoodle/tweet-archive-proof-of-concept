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

def add_tweets(tweets):
    ids = [str(tweet["id"]) for tweet in tweets]
    contents = [tweet["text"] for tweet in tweets]

    collection.add(
        ids=ids,
        documents=contents,
        metadatas=[{"text": content} for content in contents]
    )

if __name__ == "__main__":
    # Load tweets from tweets.json
    with open('tweets.json', 'r') as f:
        tweets = json.load(f)

    # Add tweets to the database
    add_tweets(tweets)
    print(f"Added {len(tweets)} tweets to the database.")
