import streamlit as st

# LangChain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

load_dotenv()

OLLAMA_MODEL = "mistral"


def get_llm():
    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.1,  # Lower temperature for more focused responses
        stop=[
            "\nUser:", "\nHuman:", "\nAssistant:", "\nAI:", "\nSolution:",
              "\nAnswer:", "\nSummary:", "\nResponse:", "\nBot"
        ]
    )
    return llm
    

# PDF processing functions:
def document_loader(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    return documents


def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def vector_database(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_db


# Custom QA prompt template
QA_PROMPT_TEMPLATE = """Use ONLY the following context to answer the question. Do not use any 
prior knowledge.

Context:
{context}

Question: {question}

Instructions:
- Answer ONLY based on the context provided above
- Answer briefly. Do not repeat yourself. Do not continue after answering
- If the answer is not found in the context, say "The document does not contain this information."
- Be direct and concise
- Do not start with greetings or phrases like "Based on the text"
- Do not simulate a conversation

Direct Answer:"""


def get_qa_chain(retriever_obj):
    """Create a QA chain with custom prompt."""
    llm = get_llm()
    
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def summarize_text(text):
    """Summarize text with strict instructions."""
    llm = get_llm()
    
    # Truncate text if too long (Phi has limited context)
    max_chars = 3000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    prompt = f"""Read the following document text and provide a factual summary.

Document Text:
{text}

Instructions:
- Write a summary in 3-5 sentences
- Include only facts from the document
- Do not add greetings, opinions, or conversational text
- Do not simulate a dialogue or conversation
- Start directly with the summary content

Summary:"""
    
    response = llm.invoke(prompt)
    # cleaned = clean_response(response)
    return response


# --- Streamlit UI Interface ---

st.set_page_config(page_title="PDF Assistant", layout="wide")
st.title("📚 PDF Assistant")
st.markdown("Upload a PDF and either ask questions or get a summary")

# Create Tabs similar to Gradio
tab1, tab2 = st.tabs(["📄 Q&A from PDF", "📝 Summarize PDF"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload PDF for Q&A", type="pdf", key="qa_upload")
        question = st.text_area("Ask a question about the PDF", placeholder="What is the main topic?")
        qa_button = st.button("Get Answer")

    with col2:
        if qa_button and uploaded_file and question:
            with st.spinner("Thinking..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                data = document_loader("temp.pdf")
                chunks = text_splitter(data)
                db = vector_database(chunks)
                retriever_obj = db.as_retriever(search_kwargs={"k": 4})
                
                qa_chain = get_qa_chain(retriever_obj)
                response = qa_chain.invoke({"input": question})
                
                # answer = clean_response(response["result"])
                
                st.subheader("Answer")
                st.write(response["result"])

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        sum_file = st.file_uploader("Upload PDF for summarization", type="pdf", key="sum_upload")
        sum_button = st.button("Generate Summary")

    with col2:
        if sum_button and sum_file:
            with st.spinner("Summarizing..."):
                with open("temp_sum.pdf", "wb") as f:
                    f.write(sum_file.getbuffer())
                
                docs = document_loader("temp_sum.pdf")
                # Combine first few pages
                full_text = "\n".join([d.page_content for d in docs[:5]])
                
                summary = summarize_text(full_text)
                
                st.subheader("Summary")
                st.write(summary)

st.divider()
st.info(f"Using Ollama Model: {OLLAMA_MODEL}")






