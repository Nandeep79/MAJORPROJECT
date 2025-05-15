import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("AIzaSyC_0cpKR2V2bmaxqLg8hIViXppvB9WRznc"))



# Streamlit setup
st.set_page_config(page_title="Cross-Border Logistics Validator", layout="wide")
st.title("Multi-Agent Document Validation for Cross-Border Logistics")

# Load Vector DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local("customs_faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 4})

retriever = load_vector_db() if os.path.exists("customs_faiss_index") else None

# Set up Google Gemini model
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
)

def run_rag_agent(query, agent_prompt):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            f"{agent_prompt}\n\n"
            "Use the following context to assist your answer:\n"
            "{context}\n\n"
            "Question: {question}"
        )
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    result = qa_chain({"query": query})
    return result["result"]

# Agent definitions
def validation_agent(query):
    prompt = (
        "You are a Validation Agent. Verify the shipment documents for completeness "
        "and correctness against international customs standards."
    )
    return run_rag_agent(query, prompt)

def readiness_score_agent(query):
    prompt = (
        "You are a Customs Readiness Scoring Agent. Based on the document data, "
        "calculate a readiness score between 0% and 100% indicating how prepared "
        "the shipment is for customs clearance."
    )
    return run_rag_agent(query, prompt)

def fraud_detection_agent(query):
    prompt = (
        "You are a Fraud Detection Agent. Analyze the documents and identify any "
        "fraud risks, inconsistencies or suspicious patterns. Provide a risk score "
        "between 0 and 1 and remediation suggestions."
    )
    return run_rag_agent(query, prompt)

# Streamlit input
query = st.text_area("Enter shipment or customs document query here:")

if st.button("Run Validation"):
    if not query:
        st.warning("Please enter a query to proceed.")
    else:
        with st.spinner("Running multi-agent validation..."):
            validation_report = validation_agent(query)
            readiness_score = readiness_score_agent(query)
            fraud_report = fraud_detection_agent(query)

        # Display results
        st.subheader("📋 Validation Report")
        st.write(validation_report)

        st.subheader("🛃 Customs Readiness Score")
        st.write(readiness_score)

        st.subheader("🚨 Fraud Alert Summary")
        st.write(fraud_report)