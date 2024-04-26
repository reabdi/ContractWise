
import os
import time
import torch
import tempfile
import textwrap
from langchain_community.vectorstores import Pinecone as lc_pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace

# from dotenv import load_dotenv
# load_dotenv()
# Will be removed in prod
# # Configuring the Gemini model with an API key
# genai.configure(
#     api_key=os.getenv("GOOGLE_API_KEY")
# )

def check_empty_pages(documents):
    total_pages = len(documents)
    empty_pages = sum(1 for doc in documents if doc.page_content == '')
    empty_pages_percentage = (empty_pages / total_pages) * 100
    return empty_pages_percentage

def process_uploaded_files(uploaded_files, chunk_size_val, chunk_overlap_val):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Copy the uploaded files to the temporary directory
    file_paths = []
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(temp_file_path)

    # Initialize PyPDFDirectoryLoader with the path to the temp directory
    pdf_loader = PyPDFDirectoryLoader(temp_dir)
    
    # Process the files as needed
    documents = pdf_loader.load()
    
    empty_pages_percentage = check_empty_pages(documents)

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size_val,chunk_overlap=chunk_overlap_val)
    doc=text_splitter.split_documents(documents)
    # To make sure all the vectors are loaded
    time.sleep(5)
    return empty_pages_percentage, len(documents), doc


def embedding_model_select(model_name):
    model_name = model_name
    # Source: https://huggingface.co/BAAI/bge-large-en-v1.5
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )
    return bge_embeddings

def index_embedding_generation(uploaded_files, chunk_size_val, chunk_overlap_val, 
                               index_name, model_name, dimension):
    # ======= Creating the Pinecone Index ======= #
    pc = Pinecone()

    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'

    spec = ServerlessSpec(cloud=cloud, region=region)

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=dimension,
            metric='cosine',
            spec=spec
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # ======= Definig the Embedding ======= #
    bge_embeddings = embedding_model_select(model_name)
    
    # ======= Definig the index and the base retriever ======= #
    empty_pct, files_length, documents = process_uploaded_files(uploaded_files, chunk_size_val, chunk_overlap_val)
    # Adding the model name to the metadata
    for doc in documents:
        doc.metadata["embedding_model"] = model_name
    # Creating the costume ids for the indexing
    integer_list = list(range(len(documents)))
    # Convert each integer to a string using list comprehension
    string_list = [str(num) for num in integer_list]
    index_vals = PineconeVectorStore.from_documents(documents, 
                                                    bge_embeddings, 
                                                    index_name=index_name,
                                                    ids=string_list)
    retriever = index_vals.as_retriever(search_type="mmr")
    # more on the MMR search Here: https://medium.com/tech-that-works/maximal-marginal-relevance-to-rerank-results-in-unsupervised-keyphrase-extraction-22d95015c7c5
    return empty_pct, files_length, len(documents), retriever, model_name


def index_embedding_fetch_internal(index_name):
    pc = Pinecone()
    index_remote = pc.Index(index_name)
    describe = index_remote.describe_index_stats()
    
    val = index_remote.fetch(ids=['0'])
    model_name = val['vectors']['0']['metadata']['embedding_model']
    # ======= Definig the Embedding ======= #
    bge_embeddings = embedding_model_select(model_name)
    docsearch = lc_pinecone.from_existing_index(index_name=index_name, embedding=bge_embeddings)
    retriever = docsearch.as_retriever(search_type="mmr")

    return describe['total_vector_count'], retriever, model_name


def index_embedding_fetch_external(index_name, model_name):
    pc = Pinecone()
    index_remote = pc.Index(index_name)
    describe = index_remote.describe_index_stats()
    
    # ======= Definig the Embedding ======= #
    bge_embeddings = embedding_model_select(model_name)
    docsearch = lc_pinecone.from_existing_index(index_name=index_name, embedding=bge_embeddings)
    retriever = docsearch.as_retriever(search_type="mmr")

    return describe['total_vector_count'], retriever, model_name


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def llm_response_sources(llm_response):
    resource = []
    for source in llm_response["source_documents"]:
        resource.append(source.metadata['source'])
        #resource = list(dict(set(resource)).values())
        return resource

def llm_builder(llm_model_name, llm_choice):
    if llm_choice=="Google Gemini":
        llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0)
    elif llm_choice=="OpenAI":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=llm_model_name, temperature=0)
    else:
        llm = HuggingFacePipeline.from_model_id(
        model_id=llm_model_name,
        task="text-generation",
        pipeline_kwargs={"temperature": 0.1}
    )
    return llm

def retrieval_qa_chain(llm, embedding_model_name, 
                       chunk_size_val, chunk_overlap_val, retriever, question): 
    similarity_val=0.6
    # ======= Definig the Embedding ======= #
    bge_embeddings = embedding_model_select(embedding_model_name)
    splitter = CharacterTextSplitter(chunk_size=int(chunk_size_val*0.75), 
                                     chunk_overlap=int(chunk_overlap_val*0.75))
    redundant_filter = EmbeddingsRedundantFilter(embeddings=bge_embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=bge_embeddings, 
                                       similarity_threshold=0.5)
    compressor = LLMChainExtractor.from_llm(llm)
    # making the pipeline
    pipeline_compressor = DocumentCompressorPipeline(
        #transformers=[splitter, compressor, redundant_filter]
        transformers=[compressor, redundant_filter, relevant_filter]
    )
    #print("________ALL_GOOD_FOR_NOW________\n")
    # I am going with the retriever for now. 
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, 
                                                           base_retriever=retriever)  
    # create the chain to answer questions
    #print("llm:", llm) 
    qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever,
                                    return_source_documents=True) 
    llm_response = qa_chain(question)
    #print("_____________\n", llm_response)
    answer = wrap_text_preserve_newlines(llm_response['result'])
    resources = llm_response_sources(llm_response)
    return answer, resources