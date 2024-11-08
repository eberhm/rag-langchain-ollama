# Llama Langchain RAG Project

This repository is dedicated to training on Retrieval-Augmented Generation (RAG) applications using Langchain (Python) and Ollama. The main reference for this project is the [DataCamp tutorial on Llama 3.1 RAG](https://www.datacamp.com/tutorial/llama-3-1-rag).

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

This project includes two main scripts:

1. **Loading Documents and Storing Embeddings**: This script loads the documents and stores them on disk as embeddings.

    ```bash
    python load_documents.py
    ```

2. **Querying the Model**: This script queries the model using the stored embeddings.

    ```bash
    python query.py
    ```

## References

The following resources have been instrumental in the development of this project:

- [Langchain Ollama Embeddings API Reference](https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html): Used for changing embeddings generation from OpenAI to Ollama (using Llama3 as the model).
- [RAG Using Langchain Part 2: Text Splitters and Embeddings](https://jayant017.medium.com/rag-using-langchain-part-2-text-splitters-and-embeddings-3225af44e7ea): Helped in understanding text splitters and embeddings.
- [Lilian Weng's Blog](https://lilianweng.github.io/): Provided general concepts and served as a source for tests.
- [Question Answering Over Documents](https://medium.com/emburse/question-answering-over-documents-e92658e7a405): A secondary source on RAG.
- [Official Langchain Documentation](https://python.langchain.com/docs/introduction/): The official documentation site for Langchain.

## Disclaimer

This README was generated using generative AI.
