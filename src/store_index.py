from src.helper import load_pdf, text_split, download_hugging_face
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Load PDF data and split into chunks
extracted_data = load_pdf(data='data/')
text_chunks = text_split(extracted_data)

# Load sentence-transformers embedding model
embeddings = download_hugging_face()

# Connect to Pinecone
index_name = "m-chat"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region='us-east-1'
        )
    )
    print(f"Pinecone index '{index_name}' created successfully.")

# Upload documents to Pinecone vector store
print("Uploading documents to Pinecone. This may take a while...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
print("Indexing complete. You can now run the main application.")
