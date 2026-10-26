'''
This file implements an intelligent, fingerprint-based caching mechanism for a RAG (Retrieval-Augmented Generation) pipeline. It ensures that no time or money is wasted in re-chunking and re-embedding a PDF unless something about its contents, chunking strategy, or embedding model changes.

Example questions:
- For PDF 1:
    - Easy: What is the maximum limit?
    - Hard: If one is earning Rs.20,000 a month, what should be their total wage (EPF) for September 2026 under Scenario B? [from page 2]
- For PDF 2:
    - Easy: Who are the authors of this book?
    - Hard: What are some issues faced during K-means clustering? [from page 400]
'''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda


os.environ["LANGSMITH_PROJECT"] = "RAG Chatbot"
load_dotenv()

PDF_PATH = "LangSmith/EPFO_Wage_Ceiling_FAQs.pdf"
# PDF_PATH = "LangSmith/An_Introduction_To_Statistical_Learning.pdf"
INDEX_ROOT = Path("LangSmith/.indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ====== STAGE 1: RETRIEVER =====

em_model_1, em_model_2 = "gemini-embedding-2-preview", "sentence-transformers/all-MiniLM-L6-v2"

model = em_model_1
embeddings = GoogleGenerativeAIEmbeddings(
    model=model
)

# model = em_model_2
# embeddings = HuggingFaceEmbeddings(
#     model_name=model,
#     encode_kwargs={"normalize_embeddings": True}
# )

# ---------- Traced setup steps ----------
@traceable(
    name="load_pdf",
    tags=["pdf"],
    metadata={"loader": "PyPDFLoader"}
)
def load_pdf():
    """
    Loads a PDF using PyPDFLoader
    """

    loader = PyPDFLoader(PDF_PATH)
    doc_list = loader.load()  # list[Document]
    print("DONE: Document Loading")

    return doc_list


@traceable(name="split_documents")
def split_documents(docs, chunk_size, chunk_overlap):
    """
    Splits a PDF into text chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    print("DONE: Chunking")

    return splits


@traceable(
    name="build_vectorstore",
    tags=["embedding", "vectorstore"],
    metadata={
        "vectorstore": "FAISS",
        "embedding-model": model
    }
)
def build_vectorstore(splits):

    vectorstore = FAISS.from_documents(splits, embeddings)
    print("Embedded and saved FAISS index to disk")

    return vectorstore


# ----------------- Cache key / fingerprint -----------------
def file_fingerprint(path: str) -> dict:
    """
    Reads the PDF file and calculates a unique SHA-256 cryptographic hash along with the file's size in bytes and its last modification time

    This function reads the entire file every time it is invoked. It opens the file in binary read mode ("rb") and loops through it in 1 MB chunks from start to finish to compute the SHA-256 hash, which guarantees that any structural change to the content of a file with the same name in a future run is caught.
    
    NOTE: For standard PDFs (a few megabytes), reading the file block-by-block takes a fraction of a second, so it does not noticeably slow down startup.
    """

    file = Path(path)
    h = hashlib.sha256()

    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    
    file_fingerprint = {
        "hash_code": h.hexdigest(),
        "size": file.stat().st_size,
        "mtime": int(file.stat().st_mtime)
    }

    return file_fingerprint


def index_key(chunk_size: int, chunk_overlap: int) -> str:
    """
    Takes the file fingerprint and bundles it into a metadata dictionary

    This function serializes the metadata dictionary created to a JSON string and hashes it via SHA-256 to create a single alphanumeric string called `index_key` that becomes the name of a unique folder inside `.indices/` (e.g., `.indices/a1b2c3d4e5f6...`).
    """

    meta = {
        "pdf_fingerprint": file_fingerprint(PDF_PATH),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": str(model)
    }

    index_key = hashlib.sha256(
        json.dumps(
            meta,
            sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    return index_key


# ----------------- Explicitly traced load/build runs -----------------
@traceable(
    name="load_index",
    tags=["index"]
)
def load_index_run(index_dir: Path):
    """
    Loads an existing FAISS index from disk
    """

    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Loaded existing FAISS index from disk")

    return vectorstore


@traceable(
    name="build_index",
    tags=["index"]
)
def build_index_run(
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int
):
    """
    Creates a new FAISS index

    STEPS:
    1. Loads the PDF.
    2. Splits it into text chunks.
    3. Generates vector embeddings.
    4. Saves the FAISS vector database locally into `.indices/<key>/`.
    5. Saves a readable `meta.json` file inside that folder for reference.
    """

    docs = load_pdf()  # child
    splits = split_documents(docs, chunk_size, chunk_overlap)  # child
    vectorstore = build_vectorstore(splits)  # child
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    (index_dir / "meta.json").write_text(
        json.dumps({
            "pdf_path": os.path.abspath(PDF_PATH),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": str(model)
        },
        indent=2
    ))

    return vectorstore


# ----------------- Dispatcher (not traced) -----------------
def load_or_build_index(
    chunk_size: int,
    chunk_overlap: int,
    force_rebuild: bool = False
):
    """
    Loads the existing index or creates a new one

    This function executes follows either of the following:

    PATH 1: Cache Miss (Index Created) - The function checks if the folder `.indices/<key>` already exists, and if it does not, it executes `build_index_run`. This path is triggered in the following scenarios:
        - The given PDF is being processed for the first time
        - The given PDF has contents and/or metadata (size or modification time) different from all the previously processed PDFs
        - The chunk size or overlap has changed from before
        - The embedding model has changed from before

    PATH 2: Cache Hit (Index Loaded, NOT Created) - If the folder `.indices/<key>` already exists, the function skips processing completely and runs `load_index_run`, instantly loading the pre-built FAISS index from disk in milliseconds.

    Example scenarios:
    - Scenario A: A file with the same name as a previous file, but different content
        - What happens: The file size, modification time, and the internal byte contents are different, which results in the computed SHA-256 hash to be completely different. This changes the final `index_key`, pointing to a new, non-existent folder path inside `.indices/`. Hence, the cache check evaluates to `False`, triggering a cache miss, thus, automatically re-loading, re-chunking, and re-embedding the new content, creating a fresh index directory.

    - Scenario B: A file with its content the same as a previous file, but a different name
        - What happens: The function never looks at the filename when calculating the file's fingerprint; it only reads the actual data bytes inside the file via `file_fingerprint`, and since, the content, size, and bytes of the file are identical, its SHA-256 hash will be identical to the old one, generating the same `_index_key`. The function will find the existing folder in `.indices/` and achieve a cache hit, loading the pre-existing index instantly without doing any embedding work.
    """

    key = index_key(chunk_size, chunk_overlap)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild

    if cache_hit:
        return load_index_run(index_dir)
    else:
        return build_index_run(index_dir, chunk_size, chunk_overlap)


# ====== STAGE 2: GENERATOR =====

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

def format_docs(docs):
    formatted_docs = "\n\n".join(d.page_content for d in docs)
    return formatted_docs


@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(
    chunk_size: int,
    chunk_overlap: int,
    force_rebuild: bool = False
):

    return load_or_build_index(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=force_rebuild
    )


@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    question: str
):

    print("\nProcessing...\n")
    chunk_size, chunk_overlap = 1000, 150
    vectorstore = setup_pipeline(chunk_size, chunk_overlap)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("DONE: Embedding and Indexing")
    print("COMPLETED: STAGE 1 (Retriever) \n")

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    print("Analyzing context...")
    chain = parallel | prompt | llm | StrOutputParser()
    print("COMPLETED: STAGE 2 (Generator) \n")

    answer = chain.invoke(
        question,
        config={
            "run_name": "pdf_rag_query",
            "tags": ["qa"],
            "metadata": {"k": 4}
        }
    )

    return answer


# ----------------- CLI -----------------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(q)
    print("\nA:", ans)