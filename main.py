import os
import csv
import re
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# LangChain
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from langchain_community.tools import DuckDuckGoSearchRun

#  APP
app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  STATIC 
app.mount("/static", StaticFiles(directory="static_page"), name="static")

@app.get("/")
def home():
    return FileResponse("static_page/index.html")

# MEMORY 
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

#  EMBEDDING 
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index_path = "faiss_index"

#  VECTOR DB 
if not os.path.exists(index_path):
    print("Creating FAISS index...")
    docs = []

    with open("medquad.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = f"Question: {row['question']}\nAnswer: {row['answer']}"
            docs.append(Document(page_content=content))

    splitter = SemanticChunker(embedding)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(index_path)

else:
    print("Loading FAISS index...")
    vectordb = FAISS.load_local(
        index_path,
        embedding,
        allow_dangerous_deserialization=True
    )

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

#  LLM 
model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.3
)

#  PROMPT 
prompt = PromptTemplate(
    template="""
You are a safe medical assistant.

- Use ONLY given context
- Do NOT give final diagnosis
- Do NOT prescribe medicines
- Give general advice only
- Always suggest consulting a doctor

Context: {context}
Question: {question}
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

#  TOOL 1: RAG MEDICAL QA 
def medical_qa(query: str):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    chain = prompt | model | parser
    return chain.invoke({"context": context, "question": query})

qa_tool = Tool(
    name="Medical_QA",
    func=medical_qa,
    description="Answer medical questions using dataset"
)

#  TOOL 2: LLM SYMPTOM ANALYZER 
def symptom_checker(text: str):
    try:
        chat_history = memory.load_memory_variables({})["chat_history"]

        prompt_text = f"""
You are a conversational medical assistant. Be clear and structured.

Conversation history:
{chat_history}

User reports: {text}

Respond in this exact structure:
**Problem Explanation:** (What this condition likely is, explain clearly)
**Risk Level:** LOW / MEDIUM / HIGH
**Possible Causes:** (Bullet points)
**Precautions to Take Right Now:** (Numbered steps the user should follow immediately)
**When to See a Doctor:** (Be specific about warning signs)
**Follow-up Questions:** (Ask 1-2 relevant questions to better understand the situation)

Rules:
- Do NOT prescribe medicines
- Do NOT give a final diagnosis
- Be empathetic, clear, and helpful
"""

        response = model.invoke(prompt_text)

        memory.save_context(
            {"input": text},
            {"output": response.content}
        )

        return response.content

    except Exception as e:
        return f"Error analyzing symptoms: {str(e)}"

symptom_tool = Tool(
    name="Symptom_Checker",
    func=symptom_checker,
    description="Analyze symptoms using LLM"
)

#  TOOL 3: EMERGENCY DETECTOR 
# India Emergency Numbers
INDIA_EMERGENCY_NUMBERS = """
 INDIA EMERGENCY CONTACTS:
• Ambulance (National):        102
• Emergency Helpline:          112
• Police:                      100
• Fire:                        101
• Disaster Management:         108 (also works as ambulance in many states)
• AIIMS (Delhi) Emergency:     011-26588500
• Poison Control (Delhi):      1800-116-117 (Toll-Free)

➡ Call 112 immediately — it connects to ambulance, police, and fire services.
"""

def emergency_detector(text: str):
    t = text.lower()

    emergency_keywords = [
        "chest pain", "heart attack", "not breathing", "can't breathe",
        "unconscious", "huge cut", "bleeding heavily", "stroke", "seizure",
        "overdose", "poisoning", "suicide", "drowning", "choking",
        "severe burn", "head injury", "collapsed", "unresponsive",
        "severe allergic", "anaphylaxis", "paralysis", "fainting",
        "severe bleeding", "no pulse", "stopped breathing"
    ]

    if any(k in t for k in emergency_keywords):
        return (
            " **MEDICAL EMERGENCY DETECTED!**\n\n"
            "**Do NOT wait — take immediate action:**\n"
            "1. Call 112 (National Emergency) or 102 (Ambulance) RIGHT NOW\n"
            "2. Keep the person calm and still\n"
            "3. Do NOT give food or water\n"
            "4. Stay on the line with the emergency operator\n\n"
            + INDIA_EMERGENCY_NUMBERS
        )

    return ""


#  TOOL 4: WEB SEARCH
def web_search_safe(query: str):
    try:
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)

        if not result:
            return "No news results found."

        return result  

    except Exception as e:
        return f"Search error: {str(e)}"

web_tool = Tool(
    name="Web_Search",
    func=web_search_safe,
    description="Search latest medical news and info"
)

# TOOL 5: HOSPITAL FINDER --------------------


#  AGENT 
tools = [qa_tool, symptom_tool, web_tool]

agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

#  MULTI-INTENT API 
@app.get("/agent")
def run_agent(query: str):
    try:
        response_parts = []
        q = query.lower()
        emergency = emergency_detector(query)
        if emergency:
            return {"response": emergency}

        symptom_keywords = [
            "pain", "fever", "deep cut", "blood", "injury", "headache", "nausea",
            "vomiting", "cough", "cold", "rash", "dizzy", "fatigue", "swelling",
            "itching", "burning", "sore", "ache", "breathe", "breathing",
            "infection", "diarrhea", "constipation", "allergy", "allergic",
            "stomach", "throat", "eyes", "ear", "weakness", "cramp", "spasm",
            "bleed", "wound", "fracture", "broken", "sprain", "anxiety",
            "depression", "insomnia", "sleep", "tired", "chills", "shiver",
            "urine", "discharge", "pus", "bump", "lump", "numb", "tingling"
        ]

#        hospital_keywords = [
#           "hospital", "nearby", "clinic", "doctor near",
#            "emergency room", "near me", "find hospital", "find clinic"
#        ]

        news_keywords = [
            "news", "latest", "recent", "update", "outbreak",
            "research", "study", "treatment", "vaccine", "drug"
        ]

        is_symptom  = any(word in q for word in symptom_keywords)
        #is_hospital = any(word in q for word in hospital_keywords)
        is_news     = any(word in q for word in news_keywords)

        if is_symptom:
            response_parts.append(symptom_checker(query))
            rag_answer = medical_qa(query)
            if rag_answer:
                response_parts.append(f"** Medical Reference:**\n{rag_answer}")


        if is_news or is_symptom:
            news_query = f"{query} medical news latest"
            news_result = web_search_safe(news_query)
            if news_result:
                response_parts.append(f"** Latest Related News:**\n{news_result}")

        # ---- STEP 6: Fallback to agent if nothing matched ----
        # ---- STEP 6: Fallback to agent if nothing matched ----
        if not response_parts:
            agent_response = agent.run(query)
            formatted_response = f"""
            **Problem Explanation:**
            {agent_response}
            **Risk Level:**
            LOW
            **Possible Causes:**
            • General informational query
            **Precautions to Take Right Now:**
                1. Stay informed from reliable sources
                2. Do not self-diagnose

            **When to See a Doctor:**
            If you experience symptoms or concerns

            **Follow-up Questions:**
                1. Do you want detailed symptoms?
                2. Should I find nearby hospitals?"""
            response_parts.append(formatted_response)

        return {"response": "\n\n---\n\n".join(response_parts)}

    except Exception as e:
        return {"error": str(e)}

# -------------------- RUN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)