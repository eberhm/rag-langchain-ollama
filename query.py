import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.docs_store import DocsStore
from src.rag_application import RAGApplication



def build_rag_chain():
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    logger.info("LLM initialized")

    # Create a chain combining the prompt template and LLM
    logger.info("Creating a chain combining the prompt template and LLM")

    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = OllamaEmbeddings(model="llama3.1")
vector_db_path = "./document_vector_db.parquet"

store = DocsStore(embedding_model, vector_db_path, logger)
retriever = store.get_retriever(5)


rag_application = RAGApplication(retriever, build_rag_chain(), logger)

question = "What is prompt engineering"
logger.info(f"Running example usage with question: {question}")

answer = rag_application.run(question)

logger.info("Example usage completed")

print("Question:", question)
print("Answer:", answer)
