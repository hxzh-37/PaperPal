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

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create the conversational chain for Q&A"""
    prompt_template = """
    You are an intelligent assistant helping a user based on the contents of multiple PDF documents.

    First, carefully read the provided document content. Use it as your **primary source**.

    If the answer is found directly in the content, answer accurately using that information.

    If the answer is **partially related**, you may **enhance it with general knowledge**, but make sure to **note what is from the PDF** and what is from external knowledge.

    If the answer is **completely unrelated** to the content, respond with:  
    "I couldn't find that information in the provided documents."

    ---------------------
    Context:\n {context}?\n
    ---------------------

    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Process user question and provide response"""
    try:
        # Load the vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Check if faiss_index exists
        if not os.path.exists("faiss_index"):
            st.error("Please upload and process PDF files first!")
            return
            
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        st.write("Reply: ", response["output_text"])
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.error("Make sure you have uploaded and processed PDF files first.")

def main():
    st.set_page_config(page_title="Chat with Multiple PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text.strip():
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("Done! You can now ask questions about your PDFs.")
                        else:
                            st.error("No text could be extracted from the PDF files.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()