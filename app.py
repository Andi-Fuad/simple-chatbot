import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# --- LLM Setup using Groq and Streamlit Secrets ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Groq API Key not found. Please add it to your Streamlit Cloud secrets!")
    st.stop()

llm = ChatGroq(
    model_name="llama3-8b-8192", 
    groq_api_key=groq_api_key
)

# --- UI Configuration ---
st.set_page_config(page_title="Fuad's AI", page_icon="🤖")

def set_custom_style():
    """Applies custom CSS for a personalized look."""
    st.markdown("""
        <style>
        .st-emotion-cache-16txtl3 {
            padding: 2rem 1rem 1rem;
        }
        .st-emotion-cache-1y4p8pa {
            width: 100%;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .st-chat-message-user {
            background-color: #4A90E2; /* A friendly blue for user messages */
            color: white;
        }
        .st-chat-message-assistant {
            background-color: #F5F5F5; /* A light grey for assistant messages */
        }
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# --- Title and Header ---
st.title("AI Version of Fuad")
st.header("Ask me anything!")

# --- Knowledge Base Loading and Processing ---
@st.cache_resource
def load_and_process_knowledge_base():
    """Loads documents, splits them, creates embeddings, and the vector store."""
    loader = DirectoryLoader('./knowledge_base/', glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

vectorstore = load_and_process_knowledge_base()

# --- LLM and Conversational Chain Setup ---
# llm = Ollama(model="llama3.2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever()

# In your app.py, before creating the qa_chain

from langchain.prompts import PromptTemplate

# --- Custom Prompt Template for Personalization ---
prompt_template = """
You are a specialized AI assistant. Your name is 'AI version of Fuad'.
You are the AI persona of Fuad Ahsan. Your sole purpose is to answer questions about Fuad Ahsan based on a knowledge base he has provided.
The user asking the questions is a third party who wants to learn about Fuad Ahsan. The user does not know him personally.

When you answer, you MUST follow these rules:
1.  Speak in the first person, as if you ARE Fuad Ahsan. For example, use "I", "my", "me" instead of "Fuad's".
2.  Use the provided "Context" below to find the answer. The context is a collection of facts about Fuad.
3.  Do not make up answers. If the information is not in the context, simply say "I don't have information on that topic, sorry." or "I'm not sure about that, I'd have to check."
4.  Be friendly, conversational, and helpful.
5.  Do not refer to yourself as an AI or a language model. You are the AI version of Fuad

Context:
{context}

Chat History:
{chat_history}

User's Question: {question}
Your Answer (as Fuad Ahsan):
"""

QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- Update the Conversational Chain with the Custom Prompt ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm,
#     retriever=retriever,
#     memory=memory
# )

# --- Chat History Management with Opening Message ---
if "messages" not in st.session_state:
    # Initialize the chat history with the opening message
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Nice to meet you, I'm Fuad Ahsan (AI Version). I'm a AI Engineer and data specialist. Feel free to ask me anything!"
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Chatbot Response ---
if prompt := st.chat_input("What would you like to know about me?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"question": prompt})
            st.markdown(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})