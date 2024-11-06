


import streamlit as st
import os
from typing import List
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from termcolor import cprint
from transformers import AutoTokenizer

# CONSTANTS =====================================================
EMBED_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_NAME = "mixtral-8x7b-32768"
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 8192
DOCUMENT_DIR = "./uploaded_documents/"  # the directory where the pdf files should be placed
VECTOR_STORE_DIR = "./vectorstore/"  # the directory where the vectors are stored
COLLECTION_NAME = "collection1"




load_dotenv()


def save_uploaded_files(uploaded_files) -> List[str]:
    """Save the uploaded files to the DOCUMENT_DIR and return their paths."""
    if not os.path.exists(DOCUMENT_DIR):
        os.makedirs(DOCUMENT_DIR)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCUMENT_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    cprint("[+] Uploaded files saved.", "green")
    return file_paths


def load_documents(file_paths: List[str]) -> List[Document]:
    """Loads the pdf files from the given file paths."""
    try:
        print("[+] Loading documents...")

        documents = []
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        
        cprint(f"[+] Document loaded, total pages: {len(documents)}", "green")
        return documents
    except Exception as e:
        cprint(f"[-] Error loading documents: {e}", "red")
        return []

def chunk_document(documents: List[Document]) -> List[Document]:
    """Splits the input documents into maximum of CHUNK_SIZE chunks."""
    tokenizer = AutoTokenizer.from_pretrained(
        "jinaai/" + EMBED_MODEL_NAME, cache_dir=os.environ.get("HF_HOME")
    )
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE // 50,
    )

    print(f"[+] Splitting documents...")
    chunks = text_splitter.split_documents(documents)
    cprint(f"[+] Document splitting done, {len(chunks)} chunks total.", "green")

    return chunks

def create_and_store_embeddings(
    embedding_model: JinaEmbeddings, chunks: List[Document]
) -> Chroma:
    """Calculates the embeddings and stores them in a chroma vectorstore."""
    if not chunks:
        raise ValueError("No document chunks to embed and store.")
    
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )
    cprint("[+] Vectorstore created.", "green")

    return vectorstore

def get_vectorstore_retriever(embedding_model: JinaEmbeddings) -> VectorStoreRetriever:
    """Returns the vectorstore retriever."""
    db = Chroma(persist_directory=VECTOR_STORE_DIR)
    try:
        db.get_collection(COLLECTION_NAME)
        retriever = Chroma(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        ).as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        cprint(f"[-] Error retrieving collection: {e}", "red")
        retriever = None

    return retriever

def create_rag_chain(embedding_model: JinaEmbeddings, llm: ChatGroq, file_paths: List[str]) -> Runnable:
    """Creates the RAG chain."""
    template = """ Act as an Indian legal assistant, answer queries related only to legal related queries and say "I don't know" for other irrelevant questions. 
    {context}
    </context>

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retriever = get_vectorstore_retriever(embedding_model)
    if not retriever:
        pdf = load_documents(file_paths)
        if not pdf:
            raise ValueError("No documents loaded. Please upload valid PDF files.")
        chunks = chunk_document(pdf)
        if not chunks:
            raise ValueError("No document chunks created. Please check the documents and try again.")
        retriever = create_and_store_embeddings(embedding_model, chunks).as_retriever(
            search_kwargs={"k": 3}
        )

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain



css = '''
 <style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}

.chat-message.user {
    background-color: #8a9abd;
}

.chat-message.bot {
    background-color: #1f48a1;
}

.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
'''

# HTML templates for user and bot messages
bot_template = '''
  <div class="chat-message bot">
    <div class="message">{{MSG}}</div>
 </div>
'''

user_template = '''
  <div class="chat-message user">  
    <div class="message">{{MSG}}</div>
 </div>
'''

def main():
    st.title("Indian Legal Assistant")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.sidebar.header("Upload PDF documents")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

    if st.sidebar.button("Process PDFs") and uploaded_files:
        embedding_model = JinaEmbeddings(
            jina_api_key=os.environ.get("JINA_API_KEY"),
            model_name=EMBED_MODEL_NAME,
        )
        file_paths = save_uploaded_files(uploaded_files)
        pdf = load_documents(file_paths)
        if not pdf:
            st.sidebar.error("No documents loaded. Please upload valid PDF files.")
            return
        chunks = chunk_document(pdf)
        if not chunks:
            st.sidebar.error("No document chunks created. Please check the documents and try again.")
            return
        create_and_store_embeddings(embedding_model, chunks)
        st.sidebar.success("PDFs processed successfully")

    query = st.text_input("Enter a query:")
    if query and uploaded_files:
        embedding_model = JinaEmbeddings(
            jina_api_key=os.environ.get("JINA_API_KEY"),
            model_name=EMBED_MODEL_NAME,
        )
        llm = ChatGroq(temperature=LLM_TEMPERATURE, model_name=LLM_NAME)
        try:
            file_paths = save_uploaded_files(uploaded_files)  # Ensure file_paths is assigned here
            chain = create_rag_chain(embedding_model=embedding_model, llm=llm, file_paths=file_paths)
            response = chain.invoke({"input": query})
            # Append the question and answer to the chat_history list
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Apply custom CSS
    st.markdown(css, unsafe_allow_html=True)

    # Display all chat history using the templates
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)




if __name__ == "__main__":
    main()