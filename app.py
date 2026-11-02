import os, re, fitz, numpy as np, faiss, streamlit as st
from docx import Document
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── 1. Chargement ──────────────────────────────────────────
def load_cv(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join(page.get_text() for page in doc)
    else:
        doc = Document(file)
        return " ".join(p.text for p in doc.paragraphs)

# ── 2. Nettoyage ───────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── 3. Chunking ────────────────────────────────────────────
def chunk_document(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# ── 4. Embeddings ──────────────────────────────────────────
def build_embeddings(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings.astype("float32")

# ── 5. Indexation FAISS ────────────────────────────────────
def create_vector_store(chunks, embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return {"index": index, "chunks": chunks}

# ── 6. Recherche ───────────────────────────────────────────
def retrieve_relevant_chunks(question, store, k=3):
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    _, indices = store["index"].search(q_vec, k)
    return [store["chunks"][i] for i in indices[0]]

# ── 7. Prompt ──────────────────────────────────────────────
def build_prompt(question, context):
    return f"""Tu es un assistant qui répond UNIQUEMENT aux questions basées sur le CV ci-dessous.
Si la question n'est pas liée au CV, réponds exactement : "Je ne peux répondre qu'aux questions sur ce CV."

Contenu du CV :
{context}

Question : {question}
Réponse :"""

# ── 8. Génération ──────────────────────────────────────────
def generate_answer(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ── 9. Sécurité métier ─────────────────────────────────────
def answer_cv_only(question, store):
    chunks = retrieve_relevant_chunks(question, store)
    context = "\n".join(chunks)
    prompt = build_prompt(question, context)
    return generate_answer(prompt)

# ── 10. Reload ─────────────────────────────────────────────
def reload_vector_store(file):
    text = clean_text(load_cv(file))
    chunks = chunk_document(text)
    embeddings = build_embeddings(chunks)
    return create_vector_store(chunks, embeddings)

# ── 11. Interface Streamlit ────────────────────────────────
def run_streamlit_app():
    st.set_page_config(page_title="Chatbot RAG — CV", page_icon="🤖")
    st.title("🤖 Chatbot RAG — CV Assistant")

    # Initialiser session state
    if "store" not in st.session_state:
        st.session_state.store = None
    if "cv_name" not in st.session_state:
        st.session_state.cv_name = None
    if "history" not in st.session_state:
        st.session_state.history = []

    # Upload CV
    uploaded = st.file_uploader("📎 Charger un CV (PDF ou DOCX)", type=["pdf", "docx"])

    if uploaded:
        # Recharger si nouveau CV
        if st.session_state.cv_name != uploaded.name:
            with st.spinner("⏳ Chargement et indexation du CV..."):
                st.session_state.store = reload_vector_store(uploaded)
                st.session_state.cv_name = uploaded.name
                st.session_state.history = []
            st.success(f"✅ CV chargé : **{st.session_state.cv_name}**")
        else:
            st.info(f"📄 CV actuel : **{st.session_state.cv_name}**")

        # Afficher historique
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Zone de saisie
        question = st.chat_input("Posez votre question sur le CV...")

        if question:
            # Afficher question
            st.session_state.history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            # Générer réponse
            with st.chat_message("assistant"):
                with st.spinner("💭 Réflexion..."):
                    answer = answer_cv_only(question, st.session_state.store)
                st.write(answer)

            st.session_state.history.append({"role": "assistant", "content": answer})

    else:
        st.warning("⬆️ Veuillez charger un CV pour commencer.")

run_streamlit_app()