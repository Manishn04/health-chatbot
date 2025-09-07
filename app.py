import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from src.helper import download_embeddings

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Get API key for Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found! Please check your .env file.")

# Initialize embeddings + retriever (from helper.py)
retriever = download_embeddings(index_name="health-chatbot-index")

# Setup memory for conversation
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

# Use supported Groq model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0
)

# Create Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('message', '')

        if not query:
            return jsonify({'error': 'No message provided'}), 400

        # Get response from the chain
        result = qa_chain.invoke({"question": query})

        return jsonify({
            'response': result['answer']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == "__main__":
    # For local development
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
