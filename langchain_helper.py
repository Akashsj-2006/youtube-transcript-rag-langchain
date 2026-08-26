from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

video_url = "https://youtu.be/d3GkLPGPtN8"


def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)

    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)

    return db


def get_response_from_query(db, query, k):
    #text-davinci can handle 4097 tokens
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    llm = ChatOllama(model="llama3.2")


    prompt = PromptTemplate(
        input_variables = ["question", "docs"], 
        template="""
        You are a helpful Youtube Assistant that can answer questions abput videos based on this video transcript.
        Answer the questions : {question}
        By searching the following video transcript: {docs}
        Only use the factual information from transcript.
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "question": query,
        "docs": docs_page_content
    })

    return response.content, docs


db = create_vector_db_from_youtube_url(video_url)

response = get_response_from_query(
    db,
    "What is this video about?",
    3
)

print(response)