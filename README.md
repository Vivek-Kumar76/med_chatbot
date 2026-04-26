#  MedChat Pro — AI Medical Assistant

A conversational AI-powered medical assistant built with **FastAPI**, **LangChain**, **FAISS**, and **Google Gemini**. Users can ask medical questions through a clean chat interface and receive context-aware answers grounded in a curated medical QA dataset.

---

##  Preview

> A sidebar chat UI where users type medical questions and receive AI-generated answers with diet and medication suggestions.

---

##  Features

-  **Semantic Search** — Uses FAISS vector store with semantic chunking for relevant document retrieval
-  **Gemini LLM** — Powered by Google's "gemini-3-flash-preview" model for fast, accurate responses
-  **Chat Interface** — Clean, responsive HTML/CSS frontend with chat history sidebar
-  **Persistent Index** — FAISS index is built once and reused across restarts
-  **REST API** — FastAPI backend with CORS support, easily extensible
-  **Holistic Responses** — Answers include diet suggestions and medicine recommendations

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python |
| LLM | Google Gemini ("gemini-3-flash-preview") via "langchain-google-genai" |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Text Splitting | Semantic Chunker (LangChain Experimental) |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Dataset | MedQuAD (Medical Question Answering Dataset) |

---

##  Project Structure

```
medchat-pro/
├── main.py                  # FastAPI app — routes, RAG pipeline, LLM chain
├── medquad.csv              # Medical QA dataset
├── faiss_index/             # Auto-generated FAISS vector index (created on first run)
├── static_page/
│   └── index.html           # Chat UI frontend
├── .env                     # API keys (not committed)
├── requirements.txt         # Python dependencies
└── README.md
```

---

##  Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/medchat-pro.git
cd medchat-pro
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
 On Windows: task\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

> Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Add the Dataset

Place `medquad.csv` in the root directory. The CSV must have `question` and `answer` columns.

> The MedQuAD dataset is publicly available from the [U.S. National Library of Medicine](https://www.nlm.nih.gov/).

### 6. Run the Server

```bash
python main.py
```

The server starts at `http://localhost:8000`.

On **first run**, the FAISS index is built from the dataset (this may take a few minutes). Subsequent runs load the saved index instantly.

---

##  Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Type a medical question in the input box
3. Press **Send** or hit **Enter**
4. Receive an AI-generated answer with diet and medicine suggestions

### Example Questions

- *"What are the symptoms of type 2 diabetes?"*
- *"How is hypertension treated?"*
- *"What causes chronic kidney disease?"*

---

##  API Reference

### `GET /ask`

Ask a medical question via the REST API.

**Query Parameters**

| Parameter | Type | Description |
|---|---|---|
| `query` | `string` | The medical question to ask |

**Example Request**

```bash
curl "http://localhost:8000/ask?query=What+are+symptoms+of+anemia"
```

**Example Response**

```json
{
  "question": "What are symptoms of anemia?",
  "answer": "Anemia symptoms include fatigue, weakness, pale skin, shortness of breath... [diet and medicine suggestions follow]"
}
```

--

##  Requirements

Create `requirements.txt` with:

```
fastapi
uvicorn
python-dotenv
langchain
langchain-community
langchain-experimental
langchain-huggingface
langchain-google-genai
faiss-cpu
sentence-transformers
```

---

##  How It Works

```
User Question
     │
     ▼
FastAPI /ask endpoint
     │
     ▼
FAISS Retriever (MMR Search, top-2 chunks)
     │
     ▼
Context assembled from retrieved MedQuAD chunks
     │
     ▼
PromptTemplate filled with context + question
     │
     ▼
Google Gemini LLM (gemini-2.0-flash)
     │
     ▼
Answer with diet & medicine suggestions → User
```

**First-run index creation pipeline:**

```
medquad.csv → LangChain Documents → Semantic Chunker → HuggingFace Embeddings → FAISS Index (saved to disk)
```
##  Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -m 'Add your feature)
4. Push to the branch (git push origin feature/your-feature)
5. Open a Pull Request

##  Acknowledgements

- [MedQuAD Dataset](https://www.nlm.nih.gov/) — U.S. National Library of Medicine
- [LangChain](https://www.langchain.com/) — LLM application framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) — LLM backbone
- [FAISS](https://faiss.ai/) — Efficient similarity search by Meta AI
- [HuggingFace Sentence Transformers](https://www.sbert.net/) — Embeddings model
