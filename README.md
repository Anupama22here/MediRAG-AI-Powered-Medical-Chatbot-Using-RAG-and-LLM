# MediRAG – AI‑Powered Medical Chatbot Using RAG & LLM

 
MediRAG is a medical question‑answering chatbot that combines Retrieval‑Augmented Generation (RAG) with Large Language Models (LLMs) to deliver accurate, contextually grounded medical responses. It is designed for clinicians, medical students, or health‑aware users who need reliable and explainable health guidance.

---

## 🔧 Features

- **PDF-based Knowledge Base**: Ingests medical documents (e.g. peer-reviewed papers, guidelines) as vector embeddings in Pinecone.
- **Retrieval Layer (RAG)**: Retrieves most relevant passages before answering to reduce hallucinations.
- **LLM Generation**: Uses an LLM (e.g. OpenAI GPT‑3.5 / open-source variant) to generate informed responses.
- **Iterative Follow‑Up Queries** (optional): Ask follow-up questions for deeper reasoning (inspired by i‑MedRAG architecture).  
- **Web UI (Flask)**: Interactive chat interface with real-time responses.
- **Returns Chat History & Feedback**: Optional logging of conversation metadata and user feedback.

---

## 🚀 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Anupama22here/MediRAG-AI-Powered-Medical-Chatbot-Using-RAG-and-LLM.git
   cd MediRAG-AI-Powered-Medical-Chatbot-Using-RAG-and-LLM
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # (macOS/Linux)
   venv\Scripts\activate     # (Windows PowerShell)
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add document PDFs**
   Place medical reference PDFs (e.g. PubMed guidelines) into `data/`.

5. **Ingest documents and build index**

   ```bash
   python src/store_index.py
   ```

6. **Run the web app (Flask)**

   ```bash
   python app.py
   ```

   Access at `http://127.0.0.1:8000` 

---

## 🧠 How It Works

1. **Vectorization**: `store_index.py` parses your PDFs into chunks, computes embeddings, and stores them in a vector index.
2. **Retrieval**: When a user submits a medical query, the system retrieves the most relevant segments from your knowledge base.
3. **LLM Prompting**: Retrieved content is injected into a prompt template to the LLM in `prompt.py`.
4. **Response Generation**: The LLM generates an answer grounded in retrieved medical evidence.
5. **Optional Follow‑Up Logic**: For ambiguous cases, the chatbot may ask clarifying questions to refine the response chain.

---

## 🧪 Example Use

```text
User: "What’s the recommended dosage of ibuprofen for a 12‑year‑old child with fever?"

Bot: "Based on pediatric dosage guidelines from trusted sources: 
– Ibuprofen 10 mg/kg every 6–8 hours, max 40 mg/kg per day. 
– For a 35 kg child, typically 350 mg every 6–8 hours. 
Please consult a pediatrician before administering."

Assistant may follow up: “Does the child have any known allergies to NSAIDs?”
```

---

## ⚙️ Customization & Configuration

* Modify LLM endpoint or model in `prompt.py`.
* Customize embedding model or vector DB (e.g., Chroma, FAISS, Qdrant) in `store_index.py`.
* Update prompt templates or add domain-specific instructions for fine-tuned control.
* Enable follow-up reasoning via an iterative prompt loop, inspired by i‑MedRAG 
---


## ⚠️ Disclaimer

**Not for clinical use.** Always validate answers with medical professionals. The chatbot is intended for educational and preliminary information only.

---
