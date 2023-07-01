import streamlit as st
from dotenv import load_dotenv
import pickle
import io
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from google.cloud import storage

#streamlit.config.server.address = '0.0.0.0'
#streamlit.config.server.port = 8080
import os
from google.cloud import storage

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the path to the key file
key_file_path = os.path.join(current_dir, "key.json")

# Set the environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file_path


# Access Google Cloud services using the credentials
storage_client = storage.Client()

# Instantiate a storage client
storage_client = storage.Client()

# Specify the name of your bucket
bucket_name = "chatpdf27june"

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Chitvan Sir IAS Classes')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ under the guidance of Yogi Ji and [Chitvan Sir](https://www.facebook.com/photo/?fbid=776380562444880&set=a.331160846966856)')

load_dotenv()

def main():
    st.header("Upload your PDF or try asking directly!! 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # Prompt Template
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer in English in a concise and academic style:"""
    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        # Get a reference to the bucket
        bucket = storage_client.bucket(bucket_name)

        # Specify the name of your blob
        blob_name = f"{store_name}.pkl"

        # Get a reference to the blob
        blob = bucket.blob(blob_name)

        # Check if blob exists
        if blob.exists():
            byte_stream = io.BytesIO()
            blob.download_to_file(byte_stream)
            byte_stream.seek(0)
            VectorStore = pickle.load(byte_stream)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            byte_stream = io.BytesIO()
            pickle.dump(VectorStore, byte_stream)
            byte_stream.seek(0)
            blob.upload_from_file(byte_stream)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=9)
            chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
            response = chain({"input_documents": docs, "question": query}, return_only_outputs=False)
            st.write(response)

if __name__ == '__main__':
    main()
