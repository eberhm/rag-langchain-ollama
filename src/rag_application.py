class RAGApplication:
    def __init__(self, retriever, rag_chain, logger):
        logger.info("Initializing the RAG application")
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.logger = logger

    def run(self, question):
        self.logger.info("Running RAG application")
        
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        self.logger.info("Documents retrieved")

        # Extract content from retrieved documents
        doc_texts = "\n".join([doc.page_content for doc in documents])
        self.logger.info("Content extracted from retrieved documents")

        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        self.logger.info("Answer obtained from the language model")

        return answer
