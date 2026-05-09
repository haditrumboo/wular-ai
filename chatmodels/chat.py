

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
import time
load_dotenv()
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory= CHROMA_PATH,
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k" : 6,
        # "fetch_k":10,
        # "lambda_mult" :0.5
    }
)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

         """You are a Kashmir travel assistant.

STRICT RULES:
- Answer ONLY from the context provided below
- Do NOT use your own knowledge
- Do NOT generate extra information
- If context does not have the answer, say "I don't have that information"
- Keep answers short and direct
- Only recommend agencyhi and places that are mentioned in the context
"""

        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)
print("Rag system created ")

print("press 0 to exit ")
while True:
    query = input("You : ")

    if query == "0":
        break 

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    print("\nAI is thinking", end="")

    for _ in range(3):
        time.sleep(0.4)
        print(".", end="", flush=True)

    print("\n")

    print("AI: ", end="")

    llm.invoke(final_prompt)

    print("\nSources:")
    for doc in docs:
        print(" -", doc.metadata.get("source"))

    print()
