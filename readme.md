## Advance RAG 

**Goal:** 

The overall goal of this code is...

Tecniques that are used:...

## Steps to follow:

**1. Creating the environment:**

* **Mac**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
* **Windows**
    ```bash
    python -m venv .venv
    .\venv\scripts\activate
    ```

**2. Getting the *REPLICATE_API_TOKRN***

visit here: https://replicate.com/account/api-tokens


**3. Installing the pakages** \
    ```bash
    pip install --upgrade --quiet -r requirements.txt
    ```



## Notes

**Notes for Pincone Index:** \
Note 1: The dimension of Google's "embedding-001" embeddings is 768. 

Note 2: For semantic similarity tasks, the cosine similarity between the embeddings is typically used to measure their relatedness. Embeddings with higher cosine similarity are considered more semantically similar. For retrieval tasks like "retrieval_query" and "retrieval_document", the relative distances between the query and document embeddings matter. \
[See more here](https://docs.llamaindex.ai/en/stable/examples/embeddings/gemini/)

Note 3: The Pinecode index can be creaed directly from the code too. See [this link](https://docs.pinecone.io/guides/indexes/create-an-index) for more information. 

**Notes for Google Gemini Models:**
Note 1: See this [link](https://ai.google.dev/gemini-api/docs/models/gemini#aqa) to get more infoamtion about Gemini's available models. 