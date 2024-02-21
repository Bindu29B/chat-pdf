import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfFileReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from io import BytesIO

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made with ❤ by [Prompt Engineer](https://youtube.com/@engineerprompt)')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfFileReader(BytesIO(pdf.read()))
        if pdf_reader.numPages == 0:
            st.error("Error: Empty PDF file.")
            return
        
        text = ""
        for page in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page).extractText()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Get store name from PDF filename
        store_name = os.path.splitext(pdf.name)[0]
        st.write(store_name)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                try:
                    vectors = pickle.load(f)
                    faiss_index = pickle.load(f)
                    VectorStore = FAISS(vectors=vectors, faiss_index=faiss_index)
                except (EOFError, pickle.UnpicklingError) as e:
                    st.error(f"Error: {type(e).__name__} - {e}. Please re-upload the PDF file.")
                    return
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore.vectors, f)  # Pickle only vectors
                pickle.dump(VectorStore.faiss_index, f)  # Pickle only FAISS index

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="qa")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)

if __name__ == '__main__':
    main()
