from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocsEmbedder:
    def __init__(self, embedding_model, logger):
        self.embedding_model = embedding_model
        self.logger = logger

    def embed_docs(self, urls):
        doc_splits = self._load_docs(urls)
        self._load_embeddings(doc_splits)

        return doc_splits

    def _load_docs(self, urls):
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        self.logger.info("Documents loaded and prepared")

        self.logger.info("Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )

        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits

    def _load_embeddings(self, doc_splits):
        self.logger.info("Documents split into chunks")

        self.logger.info("Creating embeddings and storing in vector store")

        total_chunks = len(doc_splits)
        self.logger.info(f"Total chunks to process: {total_chunks}")

        embeddings = []

        for i, chunk in enumerate(doc_splits, start=1):
            self.logger.info(f"Creating embedding for chunk {i} out of {len(doc_splits)}")
            embedding = self.embedding_model.embed_query(chunk.page_content)
            embeddings.append(embedding)

        return embeddings

