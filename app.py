import os
import re
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EmployeeOnboard AI",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    :root {
        --bg: #0a0e17;
        --surface: #141b2a;
        --surface-raised: #182236;
        --border: #243040;
        --text-primary: #eef1f7;
        --text-muted: #8d97ac;
        --accent-teal: #2dd4bf;
        --accent-indigo: #6366f1;
        --accent-gradient: linear-gradient(120deg, #2dd4bf 0%, #6366f1 100%);
    }

    .stApp {
        background: var(--bg);
        color: var(--text-primary);
    }

    section[data-testid="stSidebar"] {
        background: #0d1220;
        border-right: 1px solid var(--border);
    }

    .brand-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.3px;
        margin-bottom: 0px;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .brand-subtitle {
        color: var(--text-muted);
        font-size: 13px;
        margin-top: 4px;
        margin-bottom: 26px;
        line-height: 1.5;
    }

    .hero {
        position: relative;
        padding: 40px 36px;
        border: 1px solid var(--border);
        border-radius: 20px;
        background:
            radial-gradient(circle at 15% 25%, rgba(45, 212, 191, 0.14), transparent 55%),
            radial-gradient(circle at 85% 75%, rgba(99, 102, 241, 0.16), transparent 55%),
            var(--surface);
        margin-bottom: 28px;
        overflow: hidden;
    }

    .hero h1 {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.6px;
        margin-bottom: 10px;
        color: var(--text-primary);
    }

    .hero h1 .accent {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero p {
        color: var(--text-muted);
        font-size: 15.5px;
        line-height: 1.6;
        margin-bottom: 0px;
        max-width: 640px;
    }

    .status-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid transparent;
        border-image: var(--accent-gradient) 1;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .status-title {
        font-size: 11px;
        color: var(--text-muted);
        letter-spacing: 0.3px;
    }

    .status-value {
        font-size: 16px;
        font-weight: 700;
        margin-top: 4px;
        color: var(--text-primary);
    }

    .answer-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 26px 28px;
        margin-top: 22px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
    }

    .answer-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 16px;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .answer-title::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-gradient);
        display: inline-block;
    }

    .source-card {
        background: var(--surface-raised);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 13px 16px;
        margin-top: 8px;
        transition: border-color 0.15s ease;
    }

    .source-card:hover {
        border-color: var(--accent-indigo);
    }

    .source-name {
        font-weight: 600;
        font-size: 14px;
        color: var(--text-primary);
    }

    .source-meta {
        color: var(--text-muted);
        font-size: 12px;
        margin-top: 4px;
    }

    .domain-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background: var(--surface-raised);
        border: 1px solid var(--border);
        color: var(--text-primary);
        font-size: 12px;
        font-weight: 500;
        margin: 3px 4px 3px 0;
    }

    .footer {
        text-align: center;
        color: #57617a;
        font-size: 12px;
        padding: 34px 0 12px 0;
        border-top: 1px solid var(--border);
        margin-top: 20px;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
    }

    .stButton > button {
        background: var(--accent-gradient);
        color: #06121a;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 0;
        transition: filter 0.15s ease;
    }

    .stButton > button:hover {
        filter: brightness(1.08);
        color: #06121a;
    }

    .stTextArea textarea {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text-primary);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

KNOWLEDGE_DIR = Path("knowledge_base")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# You can change this if your Groq account provides another model.
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_PER_DOMAIN = 3
FINAL_CONTEXT_CHUNKS = 8


# ============================================================
# DOMAIN MAPPING
# ============================================================

DOMAIN_MAP = {
    "hr_handbook.pdf": "HR",
    "it_onboarding.pdf": "IT",
    "security_policy.pdf": "Security",
    "benefits_policy.pdf": "Benefits",
    "department_guidelines.pdf": "Department",
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand-title">EmployeeOnboard AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        "Multi-RAG Employee Knowledge Assistant"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Knowledge Domains")

    domains = [
        "HR",
        "IT",
        "Security",
        "Benefits",
        "Department",
    ]

    for domain in domains:
        st.markdown(
            f'<span class="domain-badge">{domain}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### System")

    st.markdown(
        """
        <div class="status-card">
            <div class="status-title">RAG Engine</div>
            <div class="status-value">● Ready</div>
        </div>

        <div class="status-card">
            <div class="status-title">Retrieval</div>
            <div class="status-value">Multi-Domain</div>
        </div>

        <div class="status-card">
            <div class="status-title">Generation</div>
            <div class="status-value">Groq AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.caption(
        "EmployeeOnboard AI retrieves information from "
        "predefined organizational knowledge documents."
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    """Clean extracted PDF text."""

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def create_text_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Create overlapping chunks.

    The function attempts to keep paragraph boundaries intact
    before falling back to word-based chunking.
    """

    text = clean_text(text)

    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) <= chunk_size:
            current += paragraph + "\n\n"

        else:
            if current.strip():
                chunks.append(current.strip())

            # If a single paragraph is larger than chunk size,
            # split it by words.
            if len(paragraph) > chunk_size:

                words = paragraph.split()
                temp = ""

                for word in words:

                    if len(temp) + len(word) + 1 <= chunk_size:
                        temp += word + " "

                    else:
                        if temp.strip():
                            chunks.append(temp.strip())

                        # Character-level overlap
                        previous = temp[-overlap:] if temp else ""
                        temp = previous + word + " "

                if temp.strip():
                    current = temp.strip() + "\n\n"
                else:
                    current = ""

            else:
                # Keep overlap from previous chunk
                previous_overlap = current[-overlap:] if current else ""
                current = previous_overlap + paragraph + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ============================================================
# PDF INGESTION
# ============================================================

@st.cache_data(show_spinner=False)
def load_pdf_documents():

    documents = []

    if not KNOWLEDGE_DIR.exists():
        return documents

    pdf_files = sorted(KNOWLEDGE_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:

        try:
            pdf = fitz.open(pdf_path)

            domain = DOMAIN_MAP.get(
                pdf_path.name.lower(),
                pdf_path.stem.replace("_", " ").title(),
            )

            for page_number, page in enumerate(pdf, start=1):

                page_text = clean_text(page.get_text())

                if not page_text:
                    continue

                chunks = create_text_chunks(page_text)

                for chunk_number, chunk in enumerate(chunks, start=1):

                    documents.append(
                        {
                            "text": chunk,
                            "source": pdf_path.name,
                            "page": page_number,
                            "chunk": chunk_number,
                            "domain": domain,
                            "id": hashlib.md5(
                                normalize_for_hash(chunk).encode(
                                    "utf-8"
                                )
                            ).hexdigest(),
                        }
                    )

            pdf.close()

        except Exception as error:
            print(
                f"Could not process {pdf_path.name}: {error}"
            )

    return documents


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# BUILD MULTI-RAG INDEXES
# ============================================================

@st.cache_resource(show_spinner=False)
def build_multi_rag_indexes(documents):

    model = load_embedding_model()

    domain_documents = {}

    for document in documents:

        domain = document["domain"]

        if domain not in domain_documents:
            domain_documents[domain] = []

        domain_documents[domain].append(document)

    indexes = {}

    for domain, docs in domain_documents.items():

        texts = [
            document["text"]
            for document in docs
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings)

        indexes[domain] = {
            "index": index,
            "documents": docs,
        }

    return indexes


# ============================================================
# QUERY DOMAIN ROUTING
# ============================================================

def detect_relevant_domains(query: str):

    query_lower = query.lower()

    domain_keywords = {

        "HR": [
            "leave",
            "attendance",
            "employee",
            "joining",
            "onboarding",
            "hr",
            "holiday",
            "working hours",
            "performance",
            "probation",
        ],

        "IT": [
            "email",
            "laptop",
            "computer",
            "software",
            "account",
            "login",
            "password",
            "vpn",
            "access",
            "system",
            "it",
        ],

        "Security": [
            "security",
            "mfa",
            "2fa",
            "password",
            "phishing",
            "device",
            "data",
            "privacy",
            "incident",
            "cyber",
        ],

        "Benefits": [
            "benefit",
            "insurance",
            "medical",
            "health",
            "allowance",
            "compensation",
            "bonus",
            "reimbursement",
        ],

        "Department": [
            "department",
            "team",
            "manager",
            "engineering",
            "marketing",
            "sales",
            "project",
            "developer",
            "development",
            "workflow",
        ],
    }

    matched_domains = []

    for domain, keywords in domain_keywords.items():

        if any(
            keyword in query_lower
            for keyword in keywords
        ):
            matched_domains.append(domain)

    # For broad questions, search all domains.
    if not matched_domains:
        return list(domain_keywords.keys())

    # Security + IT often overlap.
    if "Security" in matched_domains and "IT" not in matched_domains:
        matched_domains.append("IT")

    return list(dict.fromkeys(matched_domains))


# ============================================================
# MULTI-RAG RETRIEVAL
# ============================================================

def retrieve_context(
    query: str,
    indexes,
    top_k_per_domain=TOP_K_PER_DOMAIN,
):

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    relevant_domains = detect_relevant_domains(query)

    retrieved = []

    for domain in relevant_domains:

        if domain not in indexes:
            continue

        index_data = indexes[domain]

        index = index_data["index"]
        docs = index_data["documents"]

        k = min(
            top_k_per_domain,
            len(docs),
        )

        if k <= 0:
            continue

        scores, indices = index.search(
            query_embedding,
            k,
        )

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            if idx < 0 or idx >= len(docs):
                continue

            document = docs[idx].copy()

            document["score"] = float(score)

            retrieved.append(document)

    # Remove duplicates
    unique = {}

    for document in retrieved:
        unique[document["id"]] = document

    retrieved = list(unique.values())

    # Sort by semantic similarity
    retrieved.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return retrieved[:FINAL_CONTEXT_CHUNKS]


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_answer(
    query: str,
    retrieved_documents,
):

    api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY"),
    )

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    client = Groq(api_key=api_key)

    context_parts = []

    for i, document in enumerate(
        retrieved_documents,
        start=1,
    ):

        context_parts.append(
            f"""
SOURCE {i}
Domain: {document['domain']}
Document: {document['source']}
Page: {document['page']}

Content:
{document['text']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are EmployeeOnboard AI, an enterprise employee
onboarding knowledge assistant.

Your job is to answer employee questions using ONLY
the retrieved organizational context.

Rules:

1. Do not invent company policies.
2. Do not use outside knowledge when answering.
3. If the answer is not present in the retrieved context,
   clearly say that the information was not found in the
   available company knowledge base.
4. Give practical, clear and professional answers.
5. When appropriate, organize procedures as numbered steps.
6. If multiple departments provide relevant information,
   combine them into one coherent answer.
7. Preserve important conditions, exceptions and requirements.
8. Do not claim that an action is required unless the
   retrieved documents support it.
"""

    user_prompt = f"""
EMPLOYEE QUESTION:

{query}

RETRIEVED ORGANIZATIONAL CONTEXT:

{context}

Prepare a concise, professional and grounded answer.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.choices[0].message.content


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>👥 Employee<span class="accent">Onboard</span> AI</h1>
        <p>
            Multi-RAG employee knowledge assistant for
            onboarding, workplace policies, IT setup,
            security procedures and departmental guidance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

with st.spinner(
    "Initializing enterprise knowledge base..."
):

    documents = load_pdf_documents()

if not documents:

    st.warning(
        """
        No PDF knowledge documents were found.

        Please place your organizational PDF documents
        inside the `knowledge_base` folder.
        """
    )

    st.stop()


with st.spinner(
    "Building multi-domain semantic indexes..."
):

    rag_indexes = build_multi_rag_indexes(
        documents
    )


# ============================================================
# KNOWLEDGE BASE METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Knowledge Chunks",
        len(documents),
    )

with col2:
    st.metric(
        "Documents",
        len(
            set(
                d["source"]
                for d in documents
            )
        ),
    )

with col3:
    st.metric(
        "Knowledge Domains",
        len(rag_indexes),
    )

with col4:
    st.metric(
        "RAG Status",
        "Online",
    )


st.markdown("")


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    "### Ask your onboarding question"
)

st.caption(
    "Ask about HR policies, IT setup, security, "
    "benefits, workplace procedures or department guidelines."
)

query = st.text_area(
    label="Employee question",
    placeholder=(
        "Example: What should I complete during my first week?"
    ),
    height=120,
    label_visibility="collapsed",
)


ask_button = st.button(
    "🔎 Ask EmployeeOnboard AI",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESS QUERY
# ============================================================

if ask_button:

    if not query.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Searching organizational knowledge..."
        ):

            retrieved_documents = retrieve_context(
                query.strip(),
                rag_indexes,
            )

        if not retrieved_documents:

            st.info(
                """
                I could not find relevant information
                in the available organizational knowledge base.
                """
            )

        else:

            with st.spinner(
                "Generating a grounded AI response..."
            ):

                try:

                    answer = generate_answer(
                        query.strip(),
                        retrieved_documents,
                    )

                    st.markdown(
                        """
                        <div class="answer-card">
                            <div class="answer-title">
                                AI Response
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(answer)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # ----------------------------------------
                    # RETRIEVED SOURCES
                    # ----------------------------------------

                    st.markdown(
                        "### Retrieved Sources"
                    )

                    shown_sources = set()

                    for document in retrieved_documents:

                        source_key = (
                            document["source"],
                            document["page"],
                        )

                        if source_key in shown_sources:
                            continue

                        shown_sources.add(source_key)

                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-name">
                                    📄 {document['source']}
                                </div>
                                <div class="source-meta">
                                    Domain: {document['domain']}
                                    &nbsp; • &nbsp;
                                    Page: {document['page']}
                                    &nbsp; • &nbsp;
                                    Retrieval Score:
                                    {document['score']:.3f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as error:

                    st.error(
                        f"Unable to generate the AI response: {error}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        EmployeeOnboard AI · Multi-RAG Enterprise Knowledge Assistant
        <br>
        Python · Streamlit · FAISS · Sentence Transformers · Groq
    </div>
    """,
    unsafe_allow_html=True,
)
