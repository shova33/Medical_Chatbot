import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Custom Prompt
# ---------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are a professional medical assistant.

Use the provided context to answer the user's medical question.
If the answer is not in the context, say:
"I don't have enough medical information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# ---------------------------
# Load Embeddings
# ---------------------------
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------------------
# Load FAISS Vector Store
# ---------------------------
def load_vectorstore():
    embeddings = load_embeddings()
    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

# ---------------------------
# Load Ollama LLM
# ---------------------------
def load_llm():
    return OllamaLLM(
        model="mistral",
        temperature=0.5
    )

# ---------------------------
# Build QA Chain
# ---------------------------
def build_qa_chain():
    llm = load_llm()
    db = load_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": set_custom_prompt()}
    )

    return qa_chain

# ---------------------------
# Main Chat Loop
# ---------------------------
if __name__ == "__main__":
    logging.info("Loading Medical QA Chatbot with Ollama...")

    try:
        qa = build_qa_chain()
        logging.info("System Ready!")

        while True:
            query = input("\nEnter your medical question (or type 'exit'): ")

            if query.lower() == "exit":
                print("Goodbye!")
                break

            try:
                response = qa.run(query)
                print("\nAnswer:\n", response)

            except Exception as e:
                logging.error(f"Error during query: {e}")

    except Exception as e:
        logging.error(f"Startup error: {e}")
