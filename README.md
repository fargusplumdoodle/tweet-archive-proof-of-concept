# Tweet Archive Search Proof of Concept

```bash
pip install -r ./requirements.txt

# ... wait a long long time


# Load the tweets into the vector db
# (only needs to be done once per dataset)
python ./load_tweets.py


# Query the database for related tweets 
python ./query_tweets.py "Matcha Tea"
```

