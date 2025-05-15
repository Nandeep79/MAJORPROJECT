import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_vector_db(data_dir="customs_documents", index_dir="customs_faiss_index"):
    # Load documents
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # Save FAISS index locally
    db.save_local(index_dir)
    print(f"✅ FAISS vector DB saved to '{index_dir}'")

if __name__ == "__main__":
    build_vector_db()
