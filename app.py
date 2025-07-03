import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Google Generative AI via LangChain and native SDK
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def local_css():
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border: none;
            padding: 0.5em 1em;
            font-size: 16px;
            border-radius: 8px;
            margin-top: 0.35em;
        }
        .stButton>button:hover {
            background-color: #1565C0;
            color: white;
        }
        .block-container .stTextInput>div>div>input {
            margin-top: 0.35em;
        }
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant helping a user based on the contents of multiple PDF documents.

    First, carefully read the provided document content. Use it as your **primary source**.

    If the answer is found directly in the content, answer accurately using that information.

    If the answer is **partially related**, you may **enhance it with general knowledge**, but make sure to **note what is from the PDF** and what is from external knowledge.

    If the answer is **completely unrelated** to the content, respond with:  
    \"I couldn't find that information in the provided documents.\"

    ---------------------
    Context:\n{context}\n
    ---------------------

    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    try:
        if not os.path.exists("faiss_index"):
            st.error("Please upload and process PDF files first!")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.success("Response ready!")
        st.markdown(f"**Reply:** {response['output_text']}")

    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.error("Make sure you have uploaded and processed PDF files first.")

def main():
    st.set_page_config(page_title="PaperPal - Chat with Your PDFs", page_icon=None, layout="wide")
    local_css()

    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>PaperPal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Chat with your PDF documents effortlessly</p>", unsafe_allow_html=True)
    st.divider()

    with st.sidebar:
        st.title("Upload PDFs")
        st.markdown("Upload one or more PDF files to start interacting with them.")
        pdf_docs = st.file_uploader("Select PDF files", accept_multiple_files=True, type=['pdf'])

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Extracting and indexing text..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text.strip():
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("Done! You can now ask questions about your PDFs.")
                        else:
                            st.error("No extractable text found in the uploaded files.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Please upload at least one PDF.")

    st.markdown("## Ask a Question")
    st.markdown("Type a question below based on the uploaded PDFs.")

    with st.form(key="qa_form"):
        user_question = st.text_input("", placeholder="Enter your question here")
        ask = st.form_submit_button("Ask")

        if ask and user_question:
            with st.spinner("Processing..."):
                user_input(user_question)

    st.markdown("---")

if __name__ == "__main__":
    main()