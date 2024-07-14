import chromadb
from chromadb.utils import embedding_functions
import json
import sys

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create an embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Get the collection
collection = client.get_collection(
    name="tweets",
    embedding_function=embedding_function
)

def query_tweets(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    return results

def get_tweet_by_id(tweet_id, tweets):
    return next((tweet for tweet in tweets if str(tweet["id"]) == tweet_id), None)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a query as an argument.")
        sys.exit(1)

    query = sys.argv[1]
    results = query_tweets(query)

    # Load all tweets from tweets.json
    with open('tweets.json', 'r') as f:
        all_tweets = json.load(f)

    print(f"Query: {query}")
    for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
        tweet_id = results['ids'][0][i]
        full_tweet = get_tweet_by_id(tweet_id, all_tweets)
        if full_tweet:
            print(f"{i+1}. Tweet ID: {tweet_id}")
            print(f"   Content: {full_tweet['text']}")
            print(f"   Distance: {distance}")
            print()
        else:
            print(f"{i+1}. Tweet ID: {tweet_id} (not found in tweets.json)")
            print(f"   Content: {metadata['text']}")
            print(f"   Distance: {distance}")
            print()
