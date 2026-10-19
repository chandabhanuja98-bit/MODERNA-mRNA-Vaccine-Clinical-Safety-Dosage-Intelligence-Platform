from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
import time
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_data(path):
    loaded_data = Document(path)
    text = "\n\n".join(
        paragraph.text
        for paragraph in loaded_data.paragraphs
        if paragraph.text.strip()
    )
    return text

def chucking(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

    
def create_embeddings():

    model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    return model

def store_vectors(chunks, model):

    vector_store = Chroma(
        collection_name="moderna_guidelines",
        embedding_function=model,
        persist_directory="outputs/chroma_store"
    )

    existing_data = vector_store.get()
    if existing_data["ids"]:
        vector_store.delete(
            ids=existing_data["ids"]
        )

    ids = [
        f"moderna_chunk_{i}"
        for i in range(len(chunks))
    ]

    vector_store.add_texts(
        texts=chunks,
        ids=ids
    )
    print(f"\nStored {len(chunks)} chunks in ChromaDB.")
    return vector_store

def query_guidelines(
    vectorstore,
    question: str,
    k: int = 3
):

    start = time.time()
    results = vectorstore.similarity_search(question,k=k)
    elapsed = time.time() - start

    print(
        f"\nRetrieved {len(results)} chunks "
        f"in {elapsed:.3f}s"
    )


    for rank, result in enumerate(results,start=1):

        print(f"\nResult {rank}")
        print("-" * 60)

        print(result.page_content)

    return (
        [r.page_content for r in results],
        elapsed
    )

if __name__ == "__main__":

    text = load_data("data/Moderna_mRNA_Safety_Guidelines.docx")
    print("Document loaded successfully.")
    chunks = chucking(text)
    print(f"\nTotal chunks: {len(chunks)}")
    model = create_embeddings()
    print("\nEmbedding model loaded successfully.")
    vector_store = store_vectors(chunks,model)

    test_questions = [
    "What are the storage requirements for unopened products?",
    "How should products be handled after thawing?",
    "What temperature conditions are required during transport?",
    "What contraindications and warnings should be reviewed?",
    "How should adverse events be documented?"
]
    result_table = []
    for question in test_questions :
        chunks , latency = query_guidelines(vector_store,question,k=3)
        result_table.append ({
            "question" : question,
            "Top results" : chunks[0] if chunks else "no results",
            "Latency in seconds" : round(latency,4)
        })
    data_results = pd.DataFrame(result_table)
    print(data_results)

    # Reloading the stored vectors from chromaDB
    reloaded_vector_store = Chroma(
        collection_name="moderna_guidelines",
        embedding_function=model,
        persist_directory="outputs/chroma_store"
    )
    print(" The existing chromaDB loaded successfully ")
    print(" The reloaded data : ",reloaded_vector_store)

    result = reloaded_vector_store.similarity_search("What are the storage requirements for unopened products?",k=3)
    print(" The result is : ",result)