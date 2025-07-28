import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from app.utils.util import remove_think_blocks

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from app.client.prompts.arkline_chat_prompt import prompt_template

working_dir = os.path.dirname(os.path.abspath((__file__)))
file_name = os.path.join(working_dir, "resource", "NOAH_info.pdf")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# loading the embedding model
embedding = HuggingFaceEmbeddings()

# load the llm form groq
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=1000
)

class ArklineAIChat:
    def __init__(self):
        self.embedding = embedding
        self.llm = llm
        self.file_name = file_name
        self.vectorstore = None

    def process_document(self):
        if not os.path.exists(self.file_name):
            raise FileNotFoundError(f"Document {self.file_name} not found.")
        
        # Load the document
        loader = PDFPlumberLoader(self.file_name)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Create and persist the vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embedding,
            persist_directory=f"{working_dir}/doc_vectorstore"
        )
    
    def answer_question(self, user_question):
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized. Call process_document first.")
        
        retriever = self.vectorstore.as_retriever()

        # Compose LCEL chain
        rag_chain = (
            RunnableMap({
                "context": lambda x: retriever.invoke(x["question"]),
                "question": lambda x: x["question"]
            })
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Run
        result = rag_chain.invoke({"question": user_question})
        return remove_think_blocks(result)
# ========== Run Script ==========
if __name__ == "__main__":
    try:
        print("Processing document to Chroma DB...")
        chat = ArklineAIChat()
        chat.process_document()
        print("Document processed successfully.")
    except Exception as e:
        print(f"Error processing document: {e}")

    try:
        test_question = "What is NOAH?"
        print(f"Asking question: {test_question}")
        answer = chat.answer_question(test_question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error answering question: {e}")