import logging
from langchain_ollama import OllamaEmbeddings

from src.docs_store import DocsStore
from src.docs_embedder import DocsEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = OllamaEmbeddings(model="llama3.1")
vector_db_path = "./document_vector_db.parquet"
doc_urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embedder = DocsEmbedder(embedding_model, logger)
store = DocsStore(embedding_model, vector_db_path, logger)

store.store_docs(
    embedder.embed_docs(doc_urls)
)

logger.info("Documents loaded and persisted")

