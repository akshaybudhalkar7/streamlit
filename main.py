import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import dill
from langchain.vectorstores import Chroma

os.chdir(r'D:\RANDD')

os.environ["OPENAI_API_KEY"] = ""
pdf_docs = []

# Sidebar Contents
with st.sidebar:
    st.title('LLM Resume Parser App')
    st.markdown(''' 
    ## About
    This app is LLM-powered chatbot built using
    - [Streamlit](https://streamlit.io/)
    - [OpenAI](https://openai.com/)
    - [Langchain](https://www.langchain.com/)

    ''')
    add_vertical_space(5)

    # Upload a pdf file
    pdf = st.file_uploader("Upload Resume", type='pdf')


def main(pdf):
    st.write("Chat with Resumes")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=200,
                                                       length_function=len)

        chunks = text_splitter.split_text(text=text)

        # Embeddings
        # name = pdf.name[:-4]
        # if os.path.exists(f"{name}.pkl") and os.path.getsize(f"{name}.pkl") > 0:
        #     with open(f"{name}.pkl", "rb") as f:
        #         vectorstore = pickle.load(f)
        #     st.write("Embedding loaded from the disk")
        # else:
        #     embeddings = OpenAIEmbeddings()
        #     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        #     with open(f"{name}.pkl", "wb") as f:
        #         pickle.dump(vectorstore, f)
        #     st.write("Embedding computation completed")

        # persist_directory = "./db"
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        # vectorstore = Chroma.from_documents(documents=text,
        #                                     embedding=embeddings, persist_directory=persist_directory)
        # vectorstore.persist()

        # Ask user questions/query
        query = st.text_input("Ask questions about your resumes")

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)

            # llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(OpenAI(), chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main(pdf)





