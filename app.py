from flask import Flask, render_template, jsonify, request
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from src.prompt import system_prompt
from src.helper import download_hugging_face
from langchain_pinecone import PineconeVectorStore
import os
import re

app = Flask(__name__)
load_dotenv()

# --- CONFIGURATION ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# --- STATE MANAGEMENT ---
chat_history = []

# --- MODEL AND RETRIEVER SETUP ---
embeddings = download_hugging_face()
index_name = "m-chat"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 5})
llm = OpenAI(temperature=0.4, max_tokens=500)

# --- CONVERSATIONAL CHAIN ---
contextualize_q_system_prompt = """Given a chat history and the latest user question \nwhich might reference context in the chat history, formulate a standalone question \nwhich can be understood without the chat history. Do NOT answer the question, \njust reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = system_prompt + """

Here is the relevant context:
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- HELPER FUNCTION ---
def clean_response(text):
    """Removes unwanted prefixes from the LLM response."""
    text = text.strip()
    # More robustly handle prefixes, case-insensitively
    prefix_pattern = r'^(System|AI|it AI|Okay|Sure)\s*:\s*'
    cleaned_text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE).lstrip()
    return cleaned_text

# --- ROUTES ---
@app.route('/')
def chat_page():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    global chat_history
    msg = request.form['msg']
    print("User input:", msg)

    response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
    raw_answer = response.get('answer', "Sorry, I couldn't find an answer.")
    cleaned_answer = clean_response(raw_answer)
    
    print("Cleaned Response:", cleaned_answer)
    chat_history.extend([HumanMessage(content=msg), AIMessage(content=cleaned_answer)])
    
    return cleaned_answer

@app.route('/reset', methods=['POST'])
def reset_chat():
    global chat_history
    chat_history = []
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
