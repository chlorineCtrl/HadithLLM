from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


app = Flask(__name__)


# llm = Ollama(model="llama3")
# response = llm.invoke("tell a cat joke")
# print(response)

def start_app():
    app.run(host="0.0.0.0",port=8080,debug=True)

if __name__ == "__main__":
    start_app()