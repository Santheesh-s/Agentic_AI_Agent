"""RAG retriever: FAISS vector store over the knowledge base, with citations.

Each retrieved chunk keeps its source filename in metadata so that callers
(assistant/nodes.py) can cite exactly which document a fact came from, as
required by the course rubric ("a RAG pipeline with citations").
"""
import os


def setup_rag():
    """Builds (or returns None if no documents exist) a retriever over
    rag/knowledge_base/*.txt, falling back to data/who_factsheets.txt."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(base_dir, "knowledge_base")

    docs_to_load = []
    if os.path.isdir(kb_dir):
        for filename in sorted(os.listdir(kb_dir)):
            if filename.endswith(".txt"):
                docs_to_load.append(os.path.join(kb_dir, filename))

    if not docs_to_load:
        data_path = os.path.join(os.path.dirname(base_dir), "data", "who_factsheets.txt")
        if os.path.exists(data_path):
            docs_to_load.append(data_path)

    if not docs_to_load:
        return None

    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import CharacterTextSplitter

        try:  # the maintained package if installed, else the deprecated community shim
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        all_documents = []
        for file_path in docs_to_load:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = os.path.basename(file_path)  # used for citations
            all_documents.extend(docs)

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(all_documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except ImportError as exc:
        print(f"[RAG] Embedding dependencies not installed ({exc}); "
              f"run `pip install -r requirements.txt`. Continuing without RAG citations.")
        return None
    except Exception as exc:
        print(f"[RAG] Could not build the vector store ({exc.__class__.__name__}: {exc}). "
              f"Continuing without RAG citations.")
        return None


def format_with_citations(docs) -> str:
    """Render retrieved documents as 'text [Source: filename]' for grounded, cited answers."""
    if not docs:
        return ""
    lines = []
    for d in docs:
        source = d.metadata.get("source", "knowledge base")
        lines.append(f"{d.page_content.strip()} [Source: {source}]")
    return "\n".join(lines)
