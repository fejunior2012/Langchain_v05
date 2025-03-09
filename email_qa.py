# pip install streamlit
# pip install -U langchain langchain-community
# pip install python-dotenv
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate # exemplo prompt
from langchain.chat_models import ChatOpenAI # modelo utilizado
from langchain.chains import LLMChain # conecta tudo
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader

# Carrega a chave Open AI
load_dotenv()

loader = CSVLoader(file_path="knowledge_base.csv")
documents = loader.load()