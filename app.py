from flask import Flask, jsonify, render_template, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
import pandas as pd

app = Flask(__name__)

folder_path = "vectorDatabase"
cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching islamic docuemnts. Answer with hadith reference,book name,hadith number etc. Dont answer additonal questons. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)


@app.route("/query", methods=["POST"])
def aiPost():
    print("Post /query called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return jsonify(response_answer)

@app.route("/query_context", methods=["POST"])
def askPDFPost():
    print("Post /query_context called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 4,
            "score_threshold": 0.5,  
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        source = doc.metadata.get("source", "Unknown")  # Provide a default value
        sources.append(
            {"source": source, "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return jsonify(response_answer)

@app.route("/csv", methods=["POST"])
def csvPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "dataset/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    # Load and process CSV data
    df = pd.read_csv(save_file, usecols=["Question", "Full Answer"])
    docs = [
        Document(page_content=f"Question: {row['Question']}\nAnswer: {row['Full Answer']}")
        for _, row in df.iterrows()
    ]
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    # Save new data to the vector store
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return jsonify(response)

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
