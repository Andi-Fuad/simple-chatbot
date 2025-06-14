import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceBgeEmbeddings # <-- NEW IMPORT
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- UI Configuration ---
st.set_page_config(page_title="AI of [Your Name]", page_icon="🤖")

# --- Title and Header ---
st.title("AI of [Your Name]")
st.header("Ask me anything about [Your Name]!")

# --- Knowledge Base & Embeddings ---
@st.cache_resource
def load_and_process_knowledge_base():
    # 1. Load documents
    loader = DirectoryLoader('./knowledge_base/', glob="**/*.txt", show_progress=True)
    documents = loader.load()
    
    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # 3. Define the cloud-friendly embeddings model
    # THIS IS THE KEY CHANGE TO FIX THE EMBEDDINGS ERROR
    model_name = "BAAI/bge-small-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}, # Run on CPU
        encode_kwargs=encode_kwargs
    )
    
    # 4. Create the vector store
    st.info("Creating the knowledge base embeddings. This might take a moment...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    st.success("Knowledge base is ready!")
    return vectorstore

vectorstore = load_and_process_knowledge_base()

# --- LLM and Conversational Chain Setup ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Groq API Key not found. Please add it to your Streamlit Cloud secrets!")
    st.stop()

# Use the cloud-based Groq model for chat
llm = ChatGroq(
    model_name="llama3-8b-8192", 
    groq_api_key=groq_api_key
)

# --- The Advanced Prompt Template ---
# (This section remains the same)
prompt_template = """
You are a specialized AI assistant. Your name is 'Digital [Your First Name]'.
You are the AI persona of [Your Full Name]. Your sole purpose is to answer questions about [Your Full Name] based on a knowledge base he has provided.
The user asking the questions is a third party who wants to learn about [Your Full Name].

When you answer, you MUST follow these rules:
1. Speak in the first person, as if you ARE [Your Full Name]. For example, use "I", "my", "me".
2. Use the provided "Context" below to find the answer.
3. Do not make up answers. If the information is not in the context, simply say "I don't have information on that topic, sorry."
4. Be friendly, conversational, and helpful.
5. Do not refer to yourself as an AI or a language model.

Context:
{context}

Chat History:
{chat_history}

User's Question: {question}
Your Answer (as [Your Full Name]):
"""

QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
retriever = vectorstore.as_retriever()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

# --- Chat History Management with Opening Message ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm the AI version of [Your Name]. Feel free to ask me anything about him."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Chatbot Response ---
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": prompt})
            response = result["answer"]
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})