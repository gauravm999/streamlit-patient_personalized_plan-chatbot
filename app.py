# app.py

# --- Import Libraries ---
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="🏥 Patient Healthcare Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Top Title and App Description ---
st.title("🏥 Patient Personalized Healthcare Recommendations Chatbot")
st.caption("📝 Ask personalized healthcare-related questions based on structured and unstructured patient records. Powered by RAG + OpenAI GPT-3.5.")

# --- Load Environment Variables ---
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("🚨 OpenAI API key not found! Please check your .env file or Streamlit secrets.")
    st.stop()

# --- Sidebar Section ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2972/2972357.png", width=120)
st.sidebar.markdown("# 🏥 Healthcare Chatbot")
st.sidebar.markdown("Helps doctors and healthcare workers personalize treatment plans based on patient data.")

st.sidebar.header("🛠️ Utilities")
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.success("✅ Chat history cleared!")

st.sidebar.header("📁 Upload Patient Data")
uploaded_csv = st.sidebar.file_uploader("Upload **patient_records.csv**", type="csv")
uploaded_txts = st.sidebar.file_uploader("Upload **patient_notes (.txt)**", type="txt", accept_multiple_files=True)

st.sidebar.header("💡 Sample Questions")
st.sidebar.info(
    "- Show me patient P0023's profile.\n"
    "- Suggest lifestyle changes for cardiac patients.\n"
    "- What is the blood pressure of patient P0045?\n"
    "- Compare treatment for diabetes vs hypertension."
)

# --- Main App Logic ---
if uploaded_csv and uploaded_txts:
    # --- Load Structured Data ---
    df = pd.read_csv(uploaded_csv)

    # --- Create Structured Texts ---
    structured_texts = df.apply(
        lambda row: f"Patient ID: {row['Patient_ID']}. Name: {row.get('Name', '')}. Age: {row['Age']} years. Gender: {row['Gender']}. BMI: {row['BMI']}. Blood Pressure: {row['Blood_Pressure']}. Cholesterol: {row['Cholesterol']} mg/dL. Diagnosis: {row['Diagnosis']}. Recommended Treatment: {row['Recommended_Treatment']}.",
        axis=1
    ).tolist()

    # --- Load Unstructured Data ---
    unstructured_texts = [file.read().decode('utf-8') for file in uploaded_txts]

    # --- Combine All Texts ---
    all_texts = structured_texts + unstructured_texts

    st.success(f"✅ Loaded {len(all_texts)} documents successfully!")

    # --- Setup FAISS Vector Store ---
    with st.spinner("🔄 Setting up vector store..."):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vector_store = FAISS.from_texts(all_texts, embeddings)

    # --- Setup Conversational Retrieval Chain ---
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        memory=memory
    )

    # --- Chat Section ---
    st.subheader("💬 Ask Personalized Healthcare Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Type your question:", key="input")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Ask"):
            if user_input:
                with st.spinner("🤔 Thinking..."):
                    response = rag_chain({"question": user_input})
                    answer = response['answer']

                    if "I don't have" in answer or "no information" in answer:
                        st.warning("🔎 No specific matching patient found. Please check Patient ID or rephrase your query.")
                    else:
                        st.success("✅ Response generated successfully!")
                        st.session_state.messages.append(("You", user_input))
                        st.session_state.messages.append(("Bot", answer))

    with col2:
        if st.button("🆕 Start New Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

    # --- Display Chat Messages with Bubbles ---
    for sender, message in st.session_state.messages:
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.markdown(message)

else:
    st.warning("📄 Please upload **patient_records.csv** and at least one **patient_notes.txt** file to start.")

