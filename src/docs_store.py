from langchain_community.vectorstores import SKLearnVectorStore


class DocsStore:
    def __init__(self, embedding_model, persist_path, logger):
        self.persist_path = persist_path
        self.embedding_model = embedding_model
        self.logger = logger

    def store_docs(self, doc_splits):
        self.logger.info("Building Vector Store ...")

        vectorstore = self._create_store()

        vectorstore.add_documents(doc_splits)

        self.logger.info("Embeddings created and stored in vector store")

        vectorstore.persist()

        return vectorstore

    def get_retriever(self, k=5):
        self.logger.info("Building Vector Store ...")

        vectorstore = self._create_store()

        retriever = vectorstore.as_retriever(
            search_kwargs={'k': k}
        )

        self.logger.info("Embeddings loaded from vector store")

        return retriever
    

    def _create_store(self):
        vectorstore = SKLearnVectorStore(
            embedding=self.embedding_model,
            persist_path=self.persist_path,
            serializer="parquet"
        )
        
        return vectorstore