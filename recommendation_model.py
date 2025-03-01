import os
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import configure
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('new.env')

# Configure Google Generative AI with your API key
configure(api_key=os.getenv("GEMINI_API_KEY"))

# Custom GeminiEmbeddings class inheriting from Embeddings
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key, request_timeout=60):
        self.api_key = api_key
        self.request_timeout = request_timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Replace with actual Gemini API call for document embeddings
        return [[-1.0] * 512 for _ in range(len(texts))]  # Return dummy embeddings of correct shape

    def embed_query(self, text: str) -> list[float]:
        # Replace with actual Gemini API call for query embeddings
        return [-1.0] * 512  # Return dummy embedding of correct shape

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

class GenerateLearningPathIndexEmbeddings:
    def __init__(self, csv_filename="one.csv"):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        self.data_path = os.path.join(os.getcwd(), csv_filename)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"CSV file not found at {self.data_path}")
            
        self.our_custom_data = None
        self.gemini_embeddings = None
        self.faiss_vectorstore = None

        self.load_csv_data()
        self.get_gemini_embeddings()
        self.create_faiss_vectorstore_with_csv_data_and_gemini_embeddings()

    def load_csv_data(self):
        print(' -- Started loading .csv file for chunking purposes.')
        loader = TextLoader(self.data_path)
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        self.our_custom_data = text_splitter.split_documents(document)
        print(f' -- Finished splitting text from the .csv file ({self.data_path}).')

    def get_gemini_embeddings(self):
        self.gemini_embeddings = GeminiEmbeddings(api_key=self.gemini_api_key, request_timeout=60)

    def create_faiss_vectorstore_with_csv_data_and_gemini_embeddings(self):
        faiss_vectorstore_foldername = "faiss_learning_path_index"
        csv_last_modified = datetime.fromtimestamp(os.path.getmtime(self.data_path))
        index_last_modified = None

        if os.path.exists(faiss_vectorstore_foldername):
            index_last_modified = datetime.fromtimestamp(os.path.getmtime(faiss_vectorstore_foldername))

        if not os.path.exists(faiss_vectorstore_foldername) or csv_last_modified > index_last_modified:
            print(' -- Creating a new FAISS vector store from chunked text and Gemini embeddings.')
            vectorstore = FAISS.from_documents(self.our_custom_data, self.gemini_embeddings)
            vectorstore.save_local(faiss_vectorstore_foldername)
            print(f' -- Saved the newly created FAISS vector store at "{faiss_vectorstore_foldername}".')
        else:
            print(f' -- Found existing FAISS vector store at "{faiss_vectorstore_foldername}", loading from cache.')
        
        # Try to load the FAISS index with the parameter, if it fails, try without it
        try:
            self.faiss_vectorstore = FAISS.load_local(
                faiss_vectorstore_foldername, 
                self.gemini_embeddings,
                allow_dangerous_deserialization=True
            )
        except TypeError:
            # If the above fails due to the parameter not being supported, try without it
            print(' -- Parameter allow_dangerous_deserialization not supported, loading without it.')
            self.faiss_vectorstore = FAISS.load_local(
                faiss_vectorstore_foldername, 
                self.gemini_embeddings
            )

    def get_faiss_vector_store(self):
        return self.faiss_vectorstore

class GenAILearningPathIndex:
    def __init__(self, faiss_vectorstore):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.faiss_vectorstore = faiss_vectorstore

        # Updated prompt template to include an introductory paragraph
        prompt_template = (
            """
            You are an expert education advisor. For the query below, please provide:
            
            1. First, write a comprehensive introductory paragraph that:
               - Introduces the topic/field being asked about
               - Explains why this field is important or relevant
               - Provides general guidance on how to approach learning this topic
               - Mentions any prerequisites or foundational knowledge needed
               - Offers encouragement and realistic expectations about the learning journey
               - just cover all in small 5-6 sentence paragraph
            
            2. Then, use the following template to answer the question from the Learning Path Index csv file.
               Display top 7-8 results in a tabular format and it should look like this:
               | Learning Pathway     | duration     | link    | Module |
               | --- | --- | --- | --- |
               | ... | ... | ... | ... |
               
            It must contain a link for each line of the result in a table.
            Consider the duration and Module information mentioned in the question.
            If you don't know the answer, don't make an entry in the table.
            
            {context}
            
            Question: {question}
            """
        )
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.chain_type_kwargs = {"prompt": PROMPT}
        
        # Updated to use the current Gemini model name
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",  # Updated to current model name
                temperature=1.0,
                google_api_key=self.gemini_api_key
            )
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            print("Trying fallback model...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",  # Fallback to older model name
                temperature=1.0,
                google_api_key=self.gemini_api_key
            )

    def get_response_for(self, query: str):
        try:
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.faiss_vectorstore.as_retriever(),
                chain_type_kwargs=self.chain_type_kwargs
            )
            return qa.run(query)
        except Exception as e:
            return f"Error querying the model: {str(e)}"

def generate_learning_path(query, csv_filename="one.csv"):
    try:
        faiss_vectorstore = GenerateLearningPathIndexEmbeddings(csv_filename).get_faiss_vector_store()
        genAIproject = GenAILearningPathIndex(faiss_vectorstore)
        return genAIproject.get_response_for(query)
    except Exception as e:
        return f"Error generating learning path: {str(e)}"
