import io
import json
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from tqdm import tqdm
from datetime import datetime
import re

from azure.storage.blob import BlobServiceClient, ContentSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# ---- Check and Load Secrets from Streamlit ----
required_keys = [
    "AZURE_CONNECTION_STRING",
    "AZURE_CONTAINER_NAME",
    "GOOGLE_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT",
    "PINECONE_INDEX1"
]

missing_keys = [key for key in required_keys if key not in st.secrets]
if missing_keys:
    st.error(f"❌ Missing keys in `.streamlit/secrets.toml`: {', '.join(missing_keys)}")
    st.stop()

AZURE_CONNECTION_STRING = st.secrets["AZURE_CONNECTION_STRING"]
AZURE_CONTAINER_NAME = st.secrets["AZURE_CONTAINER_NAME"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX1"]

# ---- Configure APIs ----
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# ---- Streamlit UI ----
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("🤖 RAG Based LLM with GEMINI Vision Model")

# ---- Internal PDF Processing from Azure Blob ----
@st.cache_resource
def load_and_process_all_pdfs():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    all_chunks = []
    for blob in container_client.list_blobs():
        if not blob.name.endswith(".pdf"):
            continue
        blob_client = container_client.get_blob_client(blob.name)
        pdf_data = blob_client.download_blob().readall()
        pdf = fitz.open(stream=io.BytesIO(pdf_data), filetype="pdf")
        first_page_text = pdf[0].get_text()
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        candidate_name = lines[0] if lines else "Unknown"

        full_text = ""
        for page_num, page in enumerate(pdf, start=1):
            page_text = page.get_text().strip()
            full_text += f"\n\n--- Page {page_num} ---\n{page_text}"

        chunks = splitter.create_documents(
            texts=[full_text],
            metadatas=[{"filename": blob.name, "candidate_name": candidate_name}]
        )
        all_chunks.extend(chunks)
    return all_chunks

@st.cache_resource
def embed_chunks(_all_chunks):
    embedded = []
    for i, chunk in enumerate(_all_chunks):
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=chunk.page_content,
                task_type="retrieval_document"
            )
            embedding_vector = response["embedding"]
            embedded.append({
                "id": f"{chunk.metadata['filename']}_{i}",
                "text": chunk.page_content,
                "embedding": embedding_vector,
                "metadata": chunk.metadata
            })
        except Exception as e:
            st.warning(f"Embedding error: {e}")
    return embedded

@st.cache_resource
def initialize_index():
    if PINECONE_INDEX in [idx.name for idx in pc.list_indexes()]:
        pc.delete_index(PINECONE_INDEX)
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    return pc.Index(PINECONE_INDEX)

@st.cache_resource
def upsert_to_pinecone(embedded_chunks):
    index = initialize_index()
    for i in tqdm(range(0, len(embedded_chunks), 100)):
        batch = embedded_chunks[i:i + 100]
        vectors = [
            {
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": {"text": chunk["text"], **chunk["metadata"]}
            }
            for chunk in batch
        ]
        index.upsert(vectors=vectors)
    return index

# ---- Initial Preprocessing on Start ----
with st.spinner("📅 Loading and embedding all resumes from Azure Blob..."):
    all_chunks = load_and_process_all_pdfs()
    embedded_chunks = embed_chunks(all_chunks)
    index = upsert_to_pinecone(embedded_chunks)
    st.success(f"✅ {len(embedded_chunks)} chunks embedded and indexed from Azure Blob PDFs.")

# ---- Search & RAG Inference ----
st.sidebar.header("🔍 Ask a Question")
query = st.sidebar.text_input("Type your question", " ")
search_btn = st.sidebar.button("Search")

# ---- Helper Function to Evaluate Metrics ----
def calculate_metrics(retrieved_texts, correct_answer, model_answer):
    answer_accuracy = 1 if model_answer.lower() == correct_answer.lower() else 0
    relevant = [1 if correct_answer.lower() in text.lower() else 0 for text in retrieved_texts]
    precision_at_k = sum(relevant) / len(retrieved_texts) if retrieved_texts else 0
    recall_at_k = 1.0 if any(relevant) else 0
    mrr = 0
    for idx, relevant_flag in enumerate(relevant):
        if relevant_flag:
            mrr = 1 / (idx + 1)
            break
    return {
        "answer_accuracy": answer_accuracy,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr
    }

# ---- Azure Blob Upload Helper ----
def upload_json_to_blob(data: dict, blob_filename: str, container_name: str = AZURE_CONTAINER_NAME):
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_filename)

    blob_client.upload_blob(
        json.dumps(data, indent=4),
        overwrite=True,
        content_settings=ContentSettings(content_type='application/json')
    )

# ---- RAG Query Execution ----
if search_btn and query:
    st.info("🔍 Running hybrid search...")

    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = response["embedding"]
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    context = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])

    prompt = f"""
You are a highly knowledgeable assistant, and your task is to provide the most accurate and relevant answer to the user's question based on the given context. The context consists of information extracted from resumes.
You are a strict JSON generator. Your task is to return a JSON object with the following structure only, and nothing else:

You must do the following:
1. Read the context carefully and determine which candidate's resume best answers the user's question.
2. Provide the most accurate and relevant answer by picking the best matching candidate's information.
3. The answer must be in the following JSON format:
{{
  "top_candidate": "string",
  "experience_years": number,
  "filename": "string",
  "matched_chunks": ["string", ...]
}}
Rules:
- DO NOT add any preamble, explanation, or commentary.
- DO NOT use Markdown or code blocks.
- ONLY return the raw JSON object, starting with '{{' and ending with '}}'.

Context:
{context}

Question:
{query}
"""

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    rag_response = model.generate_content(prompt)

    def extract_json(text):
        try:
            json_text = re.search(r"\{.*\}", text, re.DOTALL).group()
            return json.loads(json_text)
        except Exception:
            return None

    answer_json = extract_json(rag_response.text)

    if answer_json:
        st.subheader("🤖 Gemini Answer")
        st.json(answer_json)

        correct_answer = answer_json.get("top_candidate", "").strip()
        if not correct_answer:
            st.warning("⚠️ No `top_candidate` found in Gemini response. Evaluation may be invalid.")

        eval_metrics = calculate_metrics(
            [match["metadata"]["text"] for match in results["matches"]],
            correct_answer,
            correct_answer
        )

        evaluation_result = {
            "query": query,
            "generated_answer": answer_json,
            "evaluation_metrics": eval_metrics,
            "timestamp": datetime.now().isoformat()
        }

        blob_filename = f"evaluations/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        upload_json_to_blob(evaluation_result, blob_filename)

        st.success(f"✅ Evaluation uploaded to Azure Blob: `{blob_filename}`")

        st.subheader("📊 Evaluation Metrics")
        st.write(f"**Answer Accuracy**: {eval_metrics['answer_accuracy']}")
        st.write(f"**Precision@5**: {eval_metrics['precision_at_k']}")
        st.write(f"**Recall@5**: {eval_metrics['recall_at_k']}")
        st.write(f"**MRR**: {eval_metrics['mrr']}")
    else:
        st.error("⚠️ Gemini response could not be parsed as JSON.")
        st.subheader("🔎 Raw Gemini Response")
        st.code(rag_response.text)
