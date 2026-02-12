"""
Medical RAG Chatbot
LangChain 1.2+ + Ollama + FAISS
Fully Modern LCEL Implementation
"""

import logging
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# =====================================================
# CONFIG
# =====================================================

DB_FAISS_PATH = "vectorstore/db_faiss"
OLLAMA_MODEL = "mistral"
TOP_K = 4


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# LOAD LLM
# =====================================================

def load_llm():
    logger.info("Loading Ollama model...")
    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=0.3,
    )


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

def load_embeddings():
    logger.info("Loading embeddings...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =====================================================
# LOAD VECTORSTORE
# =====================================================

def load_vectorstore(embeddings):
    if not Path(DB_FAISS_PATH).exists():
        logger.error("FAISS database not found!")
        sys.exit(1)

    logger.info("Loading FAISS vectorstore...")
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =====================================================
# FORMAT DOCUMENTS
# =====================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================
# BUILD LCEL RAG PIPELINE
# =====================================================

def build_rag_chain():

    llm = load_llm()
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    prompt = ChatPromptTemplate.from_template("""
You are a professional medical AI assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say:
"I do not have enough medical information in the database."

Context:
{context}

Question:
{question}

Give a structured, medical-style answer:
""")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =====================================================
# MAIN LOOP
# =====================================================

def main():
    logger.info("Starting Medical RAG Chatbot...")
    rag_chain = build_rag_chain()
    logger.info("System Ready!")

    while True:
        query = input("\nEnter your medical question (or type 'exit'): ")

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            answer = rag_chain.invoke(query)

            print("\n==============================")
            print("Answer:\n")
            print(answer)
            print("==============================")

        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
