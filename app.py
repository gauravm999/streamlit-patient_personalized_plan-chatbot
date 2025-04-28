# app.py

# --- Import Libraries ---
import streamlit as st
import os
import glob
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Streamlit Page Config ---
st.set_page_config(page_title="Patient Healthcare Chatbot", page_icon="🩺")
st.title("🏥 Patient Personalized Healthcare Recommendations Chatbot")

# --- Load Environment Variables ---
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("🚨 OpenAI API key not found! Please check your .env file or Streamlit secrets.")
    st.stop()

# --- Sidebar: Upload Files ---
st.sidebar.header("📁 Upload Data Files")
uploaded_csv = st.sidebar.file_uploader("Upload patient_records.csv", type="csv")
uploaded_txts = st.sidebar.file_uploader("Upload patient_notes (.txt)", type="txt", accept_multiple_files=True)

# --- Sidebar: Utility Buttons ---
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.success("Chat history cleared! Start fresh.")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Sample Questions")
st.sidebar.info(
    "- Show me the profile of patient P0012.\n"
    "- Suggest lifestyle changes for cardiac patients.\n"
    "- What is the blood pressure of patient P0045?\n"
    "- Compare treatment plans for diabetes vs hypertension."
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

    if st.button("Ask"):
        if user_input:
            with st.spinner("🤔 Thinking..."):
                response = rag_chain({"question": user_input})
                answer = response['answer']

                if "I don't have" in answer or "no information" in answer:
                    st.warning("🔎 No specific matching patient found. Please check Patient ID or rephrase your query.")
                else:
                    st.success("✅ Response generated successfully!")
                    st.markdown(answer)

                st.session_state.messages.append(("You", user_input))
                st.session_state.messages.append(("Bot", answer))

    # --- Display Chat Messages ---
    for sender, message in st.session_state.messages:
        if sender == "You":
            st.markdown(f"**🧑‍💼 {sender}:** {message}")
        else:
            st.markdown(f"**🤖 {sender}:** {message}")

else:
    st.warning("📄 Please upload **patient_records.csv** and at least one **patient_notes.txt** file to start.")

