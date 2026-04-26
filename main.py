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

# DuckDuckGo
from duckduckgo_search import DDGS

# -------------------- APP --------------------
app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- STATIC --------------------
app.mount("/static", StaticFiles(directory="static_page"), name="static")

@app.get("/")
def home():
    return FileResponse("static_page/index.html")

# -------------------- MEMORY --------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -------------------- EMBEDDING --------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index_path = "faiss_index"

# -------------------- VECTOR DB --------------------
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

# -------------------- LLM --------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.3
)

# -------------------- PROMPT --------------------
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

# -------------------- TOOL 1: RAG MEDICAL QA --------------------
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

# -------------------- TOOL 2: LLM SYMPTOM ANALYZER --------------------
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

# -------------------- TOOL 3: EMERGENCY DETECTOR --------------------
# India Emergency Numbers
INDIA_EMERGENCY_NUMBERS = """
🚨 INDIA EMERGENCY CONTACTS:
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
            "🚨 **MEDICAL EMERGENCY DETECTED!**\n\n"
            "**Do NOT wait — take immediate action:**\n"
            "1. Call 112 (National Emergency) or 102 (Ambulance) RIGHT NOW\n"
            "2. Keep the person calm and still\n"
            "3. Do NOT give food or water\n"
            "4. Stay on the line with the emergency operator\n\n"
            + INDIA_EMERGENCY_NUMBERS
        )

    return ""


# -------------------- TOOL 4: WEB SEARCH --------------------
def web_search_safe(query: str):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No news results found."

        formatted = []
        for r in results:
            title = r.get("title", "No Title")
            body = r.get("body", "")[:200]
            href = r.get("href", "")
            formatted.append(f"• **{title}**\n  {body}\n  🔗 {href}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {str(e)}"

web_tool = Tool(
    name="Web_Search",
    func=web_search_safe,
    description="Search latest medical news and info"
)

# -------------------- TOOL 5: HOSPITAL FINDER --------------------
def hospital_finder(query: str):
    try:
        location = query.lower()
        location = re.sub(
            r"(hospital|hospitals|nearby|in|find|search|clinic|clinics|doctor|doctors|near me)",
            "", location
        ).strip()

        lat, lon = None, None
        resolved_location = location

        # If no location extracted, use IP geolocation as fallback
        if not location:
            try:
                geo = requests.get("https://ipapi.co/json/", timeout=5).json()
                lat = geo.get("latitude")
                lon = geo.get("longitude")
                resolved_location = geo.get("city", "your location")
            except Exception:
                return (
                    "Could not detect your location automatically.\n"
                    "Please specify a location, e.g. 'hospitals in Delhi'."
                )
        else:
            # Use Nominatim (OpenStreetMap) for reliable geocoding — free, no API key needed
            try:
                geo_url = (
                    f"https://nominatim.openstreetmap.org/search"
                    f"?q={requests.utils.quote(location)}&format=json&limit=1"
                )
                geo_res = requests.get(
                    geo_url,
                    headers={"User-Agent": "medical-assistant-app"},
                    timeout=5
                ).json()

                if geo_res:
                    lat = geo_res[0]["lat"]
                    lon = geo_res[0]["lon"]
                    resolved_location = geo_res[0].get("display_name", location).split(",")[0]

            except Exception:
                pass

        if not lat or not lon:
            return f"Could not find the location '{location}'. Please try a more specific city name."

        # Query OpenStreetMap Overpass API for nearby hospitals
        query_osm = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:5000,{lat},{lon});
          node["amenity"="clinic"](around:5000,{lat},{lon});
        );
        out;
        """

        res = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query_osm},
            timeout=10
        )
        data = res.json()

        hospitals = []
        for el in data.get("elements", [])[:7]:
            tags = el.get("tags", {})
            name = tags.get("name", "Unknown Hospital/Clinic")
            phone = tags.get("phone") or tags.get("contact:phone") or "N/A"
            amenity_type = tags.get("amenity", "facility").capitalize()
            hospitals.append(f"• 🏥 **{name}** ({amenity_type}) | 📞 {phone}")

        if not hospitals:
            return (
                f"No hospitals or clinics found near **{resolved_location}** within 5 km.\n"
                "Try searching with a broader city name.\n\n"
                + INDIA_EMERGENCY_NUMBERS
            )

        result = f"**Nearby Hospitals & Clinics near {resolved_location}:**\n\n"
        result += "\n".join(hospitals)
        result += f"\n\n{INDIA_EMERGENCY_NUMBERS}"
        return result

    except Exception as e:
        return f"Error finding hospitals: {str(e)}"

hospital_tool = Tool(
    name="Hospital_Finder",
    func=hospital_finder,
    description="Find nearby hospitals and clinics"
)

# -------------------- AGENT --------------------
tools = [qa_tool, symptom_tool, web_tool, hospital_tool]

agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

# -------------------- MULTI-INTENT API --------------------
@app.get("/agent")
def run_agent(query: str):
    try:
        response_parts = []
        q = query.lower()

        # ---- STEP 1: Emergency check (always runs first, stops everything else) ----
        emergency = emergency_detector(query)
        if emergency:
            return {"response": emergency}

        # ---- STEP 2: Broader keyword lists for better intent detection ----
        symptom_keywords = [
            "pain", "fever", "cut", "blood", "injury", "headache", "nausea",
            "vomiting", "cough", "cold", "rash", "dizzy", "fatigue", "swelling",
            "itching", "burning", "sore", "ache", "breathe", "breathing",
            "infection", "diarrhea", "constipation", "allergy", "allergic",
            "stomach", "throat", "eyes", "ear", "weakness", "cramp", "spasm",
            "bleed", "wound", "fracture", "broken", "sprain", "anxiety",
            "depression", "insomnia", "sleep", "tired", "chills", "shiver",
            "urine", "discharge", "pus", "bump", "lump", "numb", "tingling"
        ]

        hospital_keywords = [
            "hospital", "nearby", "clinic", "doctor near",
            "emergency room", "near me", "find hospital", "find clinic"
        ]

        news_keywords = [
            "news", "latest", "recent", "update", "outbreak",
            "research", "study", "treatment", "vaccine", "drug"
        ]

        is_symptom  = any(word in q for word in symptom_keywords)
        is_hospital = any(word in q for word in hospital_keywords)
        is_news     = any(word in q for word in news_keywords)

        # ---- STEP 3: Symptom analysis + RAG medical context ----
        if is_symptom:
            response_parts.append(symptom_checker(query))
            rag_answer = medical_qa(query)
            if rag_answer:
                response_parts.append(f"**📚 Medical Reference:**\n{rag_answer}")

        # ---- STEP 4: Hospital finder ----
        if is_hospital:
            response_parts.append(hospital_finder(query))

        # ---- STEP 5: Always fetch latest news for symptom or news queries ----
        if is_news or is_symptom:
            news_query = f"{query} medical news latest"
            news_result = web_search_safe(news_query)
            if news_result:
                response_parts.append(f"**📰 Latest Related News:**\n{news_result}")

        # ---- STEP 6: Fallback to agent if nothing matched ----
        if not response_parts:
            response_parts.append(agent.run(query))

        return {"response": "\n\n---\n\n".join(response_parts)}

    except Exception as e:
        return {"error": str(e)}

# -------------------- RUN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)