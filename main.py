import os
import csv
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles  

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index_path = "faiss_index"

if not os.path.exists(index_path):
    print("Creating FAISS index (Semantic Chunking)...")
    docs = []
    with open("medquad.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = f"Question: {row['question']}\nAnswer: {row['answer']}"
            docs.append(Document(page_content=content))

    print(f"Loaded {len(docs)} QA pairs")

    text_splitter = SemanticChunker(embedding)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(index_path)
    print("FAISS index created!")

else:
    print("Loading existing FAISS index...")
    vectordb = FAISS.load_local(
        index_path,
        embedding,
        allow_dangerous_deserialization=True
    )

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2}
)


model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7
)

prompt = PromptTemplate(
    template="""
You are a medical assistant. Answer the question based ONLY on the context below. 
Suggest a healthy diet and relevant medicines where appropriate.

Context: {context}
Question: {question}
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()
app.mount("/static", StaticFiles(directory="static_page"), name="static")

@app.get("/")
def home():
    return FileResponse("static_page/index.html")  


@app.get("/ask")
def ask_question(query: str):
    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        chain = prompt | model | parser

        result = chain.invoke({
            "context": context,
            "question": query
        })

        return {
            "question": query,
            "answer": result   
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)