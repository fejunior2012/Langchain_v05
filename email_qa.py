# pip install streamlit
# pip install -U langchain langchain-community
# pip install python-dotenv

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate # exemplo prompt
from langchain.chains import LLMChain # conecta tudo
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredEmailLoader, TextLoader

# Carrega a chave Open AI 
load_dotenv()


# Carrega o arquivo CSV com as perguntas e respostas
loader = CSVLoader(file_path="knowledge_base.csv")
documents = loader.load()

# Carregar mais e-maisl
text_loader = TextLoader("EMAIL00001.txt")
documents += text_loader.load()
text_loader = TextLoader("EMAIL00002.txt")
documents += text_loader.load()

# Outra opção é carregar os e-mails com extensao .eml
# email_loader = UnstructuredEmailLoader("emails_arquivo.eml")
# documents += email_loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response] if similar_response else []

# Quanto maior a temperatura, mais distante da exatidão. Valores decimais de 0 até 1.
# Defina o modelo gpt
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

template = """
    Você um assistente virtual de uma escola de programação focada na linguaguem python.
    Sua função será responder e-mails que recebemos de potenciais clientes.
    Vou lhe passar alguns e-mails antigos enviados por nosso time de vendas para que você use como modelo

    Siga todas as regras abaixo:
    1/ Você deve busca se comportar de maneira semelhante à roberta@email.com

    2/ Suas respostas devem ser bem similares ou até identicas às enviadas por ela no passado, tanto em termos de comprimento, tom de voz, argumentos lógicos e demais detalhes.

    3/ Alguns dos e-mails podem conter links e informações irrelevantes. Preste atenção apenas ao contéudo útil da mensagem.

    Arqui está uma mensagem recebida de um novo cliente.
    {message}

    Aqui está uma lista de e-mails trocados anteriormente antre outros clientes e nossa atendente.
    Este histórico de conversa servirá de base para que voce compreenda nossos produto e forma de atendimento
    {best_practice}

    Escreva a melhor resposta que eu deveria enviar para este potencial cliente:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.invoke({"message": message, "best_practice": best_practice})
    return response["text"] if "text" in response else response


if db:
    print("Base FAISS carregada com sucesso!")
else:
    print("Erro ao carregar a base FAISS.")

if llm:
    print("Modelo de IA carregado corretamente!")
else:
    print("Erro ao carregar o modelo de IA.")

# response = generate_response("Olá, eu posso parcelar o curso?")
# print(response)

# response = generate_response("""
#                                Boa tarde, tudo bem? As aulas são gravadas ou ao vivo? Posso assistir até qual prazo máximo?
#                              """)
# print(response)

def main():
    st.set_page_config(
        page_icon="E-mail manager"
    )
    st.header("E-mail manager")
    message = st.text_area("E-mail do cliente")

    if message:
        st.write("Gerando um e-mail resposta baseado nas melhores práticas...")
        result = generate_response(message)
        st.info(result)

if __name__ == '__main__':
    main()

# Execute: streamlit run .\email_qa.py