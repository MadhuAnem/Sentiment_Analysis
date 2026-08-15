from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_community.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    """Manages document chunking, embedding generation, and FAISS vector indexing."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        """Splits raw text into semantically relevant chunks and indexes them into FAISS."""
        documents = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {"doc_id": i}
            docs = self.splitter.create_documents([text], metadatas=[meta])
            documents.extend(docs)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, top_k: int = 3) -> list[Document]:
        """Retrieves top-K document passages matching query embeddings."""
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=top_k)
