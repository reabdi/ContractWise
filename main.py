
import streamlit as st
import os

from huggingface_hub import login
from app import (
    set_api_key,
    index_embedding_generation,
    index_embedding_fetch_internal,
    index_embedding_fetch_external,
    retrieval_qa_chain,
    llm_builder,
    index_embedding_fetch_external_chromadb_external
)

bge_models = {
    "BGE Large - 335M": {"model_name": "BAAI/bge-large-en-v1.5", "dimention": 1024},
    "BGE Base - 109M": {"model_name": "BAAI/bge-base-en-v1.5", "dimention": 768},
    "BGE Small - 33M": {"model_name": "BAAI/bge-small-en-v1.5", "dimention": 384},
    "Other": {"model_name": None, "dimention": None},
}

llm_models = {
    "Gemini Pro v1.0": {"model_name": "gemini-pro"},
    "Gemini Pro v1.5": {"model_name": "gemini-1.5-pro-latest"}
}

llama3_models = {
    "Llama 3-70b": {"model_name": "llama3-70b-8192"},
    "Llama 3-8b": {"model_name": "llama3-8b-8192"},
}

# Logo at the top of the application
#st.image("image/inqlect_logo.webp", width=180)

if 'embedding_model_name' not in st.session_state:
    st.session_state.embedding_model_name = None
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = None
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

if "PINECONE_API_KEY" not in st.session_state:
    st.session_state["PINECONE_API_KEY"] = None
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state["GOOGLE_API_KEY"] = None
if "HUGGING_FACE_API_KEY" not in st.session_state:
    st.session_state["HUGGING_FACE_API_KEY"] = None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None

# Title of the application
st.set_page_config(page_title="RAG-Guru")
st.markdown('<div class="title">RAG-Guru</div>', unsafe_allow_html=True)

st.markdown(
    body='GenAI-Powered Analytical & Advisory Assistant'
    """
    <style>
    .title {
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        font-weight: bold;
        color: #0077B6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Using columns to create a two-panel layout similar to the sketch
col1, col2 = st.columns(2)

with col1:
    st.subheader('Step 1. Inputs', divider='blue')
    st.subheader('Vector Database', divider='gray')

    upload_to_index = st.radio("How to setup your VectorDB",
                               ("Local (Chroma DB)",
                                "On Cloud (Pinecone)"),
                                index=None)
    if upload_to_index == "Local (Chroma DB)":
        upload_to_index_ch = st.radio("Setup your Chroma VectorDB ",
                ("Create a new Chroma DB from PDF files",
                "Use an existing Chroma DB index"),
                index=None)
        if upload_to_index_ch == "Create a new Chroma DB from PDF files":
            # the file uploader widget
            uploaded_files = st.file_uploader("Choose files:", accept_multiple_files=True)
            chunk_size =  st.number_input("Provide your chunk size:", value=500, step=1)
            chunk_overlap =  st.number_input("Provide your chunk overlap value:", value=50, step=1)
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            chromadb_index_name = st.text_input(
                "Chroma DB Index Name",
                value="ChromaDB",
                help="Enter the Chroma DB index name.",
                key="chromadb_index_name"
            )
            bge_choice = st.selectbox("Select your embedding model:", list(bge_models.keys()))
            st.markdown(
            "[Click to see more on BGE models for embedding.](https://huggingface.co/BAAI)"
            )
            if bge_choice == "Other":
                st.error("Sorry, this tool currently does not support other embedding methods.")
            params = bge_models[bge_choice]
            model_name = params["model_name"]
            dimension = params["dimention"]
            run_button_process = st.button("Process")
            if run_button_process:
                if not uploaded_files:
                    st.error("Please upload your files to continue.")
                    st.stop()
                else:
                    with st.spinner("Processing your file(s)..."):
                        try:
                            if uploaded_files and chunk_size and chunk_overlap:
                                try:
                                    empty_pct, files_length, vector_ln, retriever, embedding_model_name = index_embedding_generation(uploaded_files, 
                                                                                                            chunk_size, 
                                                                                                            chunk_overlap, 
                                                                                                            chromadb_index_name,
                                                                                                            model_name, 
                                                                                                            dimension,
                                                                                                            upload_to_index)
                                    st.session_state.embedding_model_name = embedding_model_name
                                    st.session_state.retriever = retriever
                                    st.write(f"{files_length} pages (={vector_ln} vectors) were loaded.")
                                    st.success("Indexing successful!")
                                    if empty_pct > 5:
                                        # Basically if the number of empty pages is more than 5%, rais a warning.
                                        st.warning("Warning: There might be an issue with the PDF files. Empty pages percentage is higher than 5%.")
                                except Exception as e:
                                    st.error(f"Error in indexing: {e}")
                            else:
                                st.error("Please include the reuired inpts to continue.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            st.stop()
        elif upload_to_index_ch == "Use an existing Chroma DB index":
            uploaded_directory = st.text_input("Choose the Chroma DB directory")
            bge_choice = st.selectbox("Select your embedding model:", list(bge_models.keys()))
            st.markdown(
            "[Click to see more on BGE models for embedding.](https://huggingface.co/BAAI)"
            )
            if bge_choice == "Other":
                st.error("Sorry, this tool currently does not support other embedding methods.")
            params = bge_models[bge_choice]
            model_name = params["model_name"]
            run_button_fetch = st.button('Get Index')
            if run_button_fetch:
                if not uploaded_directory:
                    st.error("Please upload your Chroma DB directory to continue.")
                    st.stop()
                else:
                    with st.spinner("Importing the vector database..."):
                        try:
                            size, retriever, embedding_model_name = index_embedding_fetch_external_chromadb_external(uploaded_directory, model_name)
                            st.session_state.embedding_model_name = embedding_model_name
                            st.session_state.retriever = retriever
                            st.write(f"Total of {size} vectors fetched from '{uploaded_directory}'.")
                            st.success("Fetching was successful!")
                        except Exception as e:
                            st.error(f"Error indexing: {e}")
                            st.stop()
    elif upload_to_index == "On Cloud (Pinecone)":
        upload_to_index_pc = st.radio("Setup your Pinecone VectorDB ",
                                ("Create a new Pinecone index from PDF files",
                                "Use an existing Pinecone index"),
                                index=None)
        if upload_to_index_pc == "Create a new Pinecone index from PDF files":
            # the file uploader widget
            uploaded_files = st.file_uploader("Choose files:", accept_multiple_files=True)
            chunk_size =  st.number_input("Provide your chunk size:", value=500, step=1)
            chunk_overlap =  st.number_input("Provide your chunk overlap value:", value=50, step=1)
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            pinecone_index_name = st.text_input(
                "Pinecone Index Name",
                help="Enter the Pinecone index name.",
                key="pinecone_index_name"
            )
            pinecone_api_key = st.text_input(
                "Pinecone API Key",
                type="password",
                help="Enter your Pinecone API key.",
                key="pinecone_api_key_create"
            )
            st.markdown(
            "[Click to see more on Pinecone Vector Database](https://docs.pinecone.io/guides/projects/understanding-projects)"
            )

            bge_choice = st.selectbox("Select your embedding model:", list(bge_models.keys()))
            st.markdown(
            "[Click to see more on BGE models for embedding.](https://huggingface.co/BAAI)"
            )
            if bge_choice == "Other":
                st.error("Sorry, this tool currently does not support other embedding methods.")
            params = bge_models[bge_choice]
            model_name = params["model_name"]
            dimension = params["dimention"]

            st.session_state["PINECONE_API_KEY"] = pinecone_api_key
            run_button_process = st.button('Process')
            if run_button_process:
                if not pinecone_api_key:
                    st.error("Please add your Pinecone API key to continue.")
                    st.stop()
                elif not pinecone_index_name:
                    st.error("Please add your Pinecone Index Name to continue.")
                    st.stop()
                else:
                    set_api_key("PINECONE_API_KEY", pinecone_api_key)
                    if "PINECONE_API_KEY" not in st.session_state:
                        st.session_state["PINECONE_API_KEY"] = pinecone_api_key
                    with st.spinner("Processing your file(s)..."):
                        try:
                            if uploaded_files and chunk_size and chunk_overlap:
                                    try:
                                        empty_pct, files_length, vector_ln, retriever, embedding_model_name = index_embedding_generation(uploaded_files,
                                                                                                                            chunk_size,
                                                                                                                            chunk_overlap,
                                                                                                                            pinecone_index_name,
                                                                                                                            model_name,
                                                                                                                            dimension
                                                                                                                            )
                                        st.session_state.embedding_model_name = embedding_model_name
                                        st.session_state.retriever = retriever
                                        st.write(f"{files_length} pages (={vector_ln} vectors) were loaded.")
                                        st.success("Indexing successful!")
                                        if empty_pct > 5:
                                            # Basically if the number of empty pages is more than 5%, rais a warning.
                                            st.warning("Warning: There might be an issue with the PDF files. Empty pages percentage is higher than 5%.")
                                    except Exception as e:
                                        st.error(f"Error in indexing: {e}")
                            else:
                                st.error("Please include the reuired inpts to continue.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            st.stop()
        if upload_to_index_pc == "Use an existing Pinecone index":
            pinecone_index_name = st.text_input(
                "Pinecone Index Name",
                help="Enter the Pinecone index name.",
                key="pinecone_index_name"
            )
            pinecone_api_key = st.text_input(
                "Pinecone API Key",
                type="password",
                help="Enter your Pinecone API key.",
                key="pinecone_api_key_fetch"
            )
            st.markdown(
            "[Click to see more on Pinecone Vector Database](https://docs.pinecone.io/guides/projects/understanding-projects)"
            )
            embedding_source = st.radio('Have you used this tool to create your PineconeDB?', ("Yes", "No"), index=None)
            if embedding_source == 'No':
                bge_choice = st.selectbox("Select your embedding model:", list(bge_models.keys()))
                st.markdown(
                "[Click to see more on BGE models for embedding.](https://huggingface.co/BAAI)"
                )
                if bge_choice == "Other":
                    st.error("Sorry, this tool currently does not support other embedding methods.")
                params = bge_models[bge_choice]
                model_name = params["model_name"]
            st.session_state["PINECONE_API_KEY"] = pinecone_api_key
            run_button_fetch = st.button('Get Index')
            if run_button_fetch:
                if not pinecone_api_key:
                    st.error("Please add your Pinecone API key to continue.")
                    st.stop()
                elif not pinecone_index_name:
                    st.error("Please add your Pinecone Index Name to continue.")
                    st.stop()
                else:
                    if "PINECONE_API_KEY" not in st.session_state:
                        st.session_state["PINECONE_API_KEY"] = pinecone_api_key
                    set_api_key("PINECONE_API_KEY", pinecone_api_key)
                    with st.spinner("Getting your index..."):
                        if embedding_source == 'Yes':
                            try:
                                size, retriever, embedding_model_name = index_embedding_fetch_internal(pinecone_index_name)
                                st.session_state.embedding_model_name = embedding_model_name
                                st.session_state.retriever = retriever
                                st.write(f"Total of {size} vectors fetched from '{pinecone_index_name}'.")
                                st.success("Fetching was successful!")
                            except Exception as e:
                                st.error(f"Error indexing: {e}")
                        elif embedding_source == 'No':
                            try:
                                size, retriever, embedding_model_name = index_embedding_fetch_external(pinecone_index_name, model_name)
                                st.session_state.embedding_model_name = embedding_model_name
                                st.session_state.retriever = retriever
                                st.write(f"Total of {size} vectors fetched from '{pinecone_index_name}'.")
                                st.success("Fetching was successful!")
                            except Exception as e:
                                st.error(f"Error indexing: {e}")
    st.subheader('LLM', divider='gray')
    llm_choice = st.selectbox("Select your LLM model:", ['Llama 3', "Hugging Face", "Google Gemini", "OpenAI"], index=None)
    if llm_choice == "Llama 3":
        llm_model_llama = st.selectbox("Select Llama3 Version:", list(llama3_models.keys()))
        groq_api_key = st.text_input(
            "GroqAI API Key",
            type="password",
            help="Enter your GroqAI API key. See more here: https://console.groq.com/",
            key="groq_api_key"
        )
        st.session_state["GROQ_API_KEY"]=groq_api_key
        params = llama3_models[llm_model_llama]
        model_name = params["model_name"]
        llm_model_name = model_name
    elif llm_choice == "Hugging Face":
        llm_model_name = st.text_input(
            "Model Name (Huffing Face Hub)",
            help="Check here for the list of models: https://huggingface.co/models?sort=trending",
            key="huffing_face_hub"
        )
        huffing_face_hub_api = st.text_input(
            "Huffing Face Hub API Key",
            type="password",
            help="Check here for the list of models: https://huggingface.co/docs/api-inference/en/quicktour",
            key="huffing_face_hub_api"
        )
        st.session_state["HUGGING_FACE_API_KEY"] = huffing_face_hub_api
    elif llm_choice == "Google Gemini":
        llm_choice_gemini = st.selectbox("Select Gemini Version:", ["Gemini Pro v1.5 - Preview"])
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google API key.",
            key="gemini_api_key"
        )
        st.session_state["GOOGLE_API_KEY"] = gemini_api_key
        llm_model_name = "gemini-1.5-pro-latest"
        # This is becaus of the problem of instruction issue with the Gemeni Pro
        # if llm_choice_gemini == "Gemini Pro v1.0":
        #     llm_model_name = "gemini-pro"
        # else:
        #     llm_model_name = "gemini-1.5-pro-latest"
    elif llm_choice == "OpenAI":
        llm_choice_openai = st.selectbox("Select OpenAI Version:", ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"])
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. See more here: https://openai.com/api/",
            key="openai_api_key"
        )
        st.session_state["OPENAI_API_KEY"]=openai_api_key
        llm_model_name = llm_choice_openai
    else:
        pass
    run_button_llm = st.button('Build LLM')
    if run_button_llm:
        if llm_choice == "Llama 3":
            if not groq_api_key:
                st.error("Please add your Groq API key correctly to continue.")
                st.stop()
            else:
                set_api_key("GROQ_API_KEY", groq_api_key)
                # Check and set API token environment variables
                if "GROQ_API_KEY" not in st.session_state:
                    st.session_state["GROQ_API_KEY"] = groq_api_key
        elif llm_choice == "Hugging Face":
            # Check and set API token environment variables
            if not huffing_face_hub_api:
                st.error("Please add your Hugging Face API key correctly to continue.")
                st.stop()
            else:
                # Check and set API token environment variables
                set_api_key("HUGGING_FACE_API_KEY", huffing_face_hub_api)
                if "HUGGING_FACE_API_KEY" not in st.session_state:
                    st.session_state["HUGGING_FACE_API_KEY"] = huffing_face_hub_api
        elif llm_choice == "Google Gemini":
            if not gemini_api_key:
                st.error("Please add your Google Gemini API key correctly to continue.")
                st.stop()
            else:
                set_api_key("GOOGLE_API_KEY", gemini_api_key)
                # Check and set API token environment variables
                if "GOOGLE_API_KEY" not in st.session_state:
                    st.session_state["GOOGLE_API_KEY"] = gemini_api_key
        else:
            if not openai_api_key:
                st.error("Please add your OpenAI API key correctly to continue.")
                st.stop()
            else:
                set_api_key("OPENAI_API_KEY", openai_api_key)
                # Check and set API token environment variables
                if "OPENAI_API_KEY" not in st.session_state:
                    st.session_state["OPENAI_API_KEY"] = openai_api_key
        # ======= Creating the LLM ======= #
        with st.spinner("Building the LLM..."):
            try:
                llm=llm_builder(llm_model_name, llm_choice)
                st.session_state.llm = llm
                st.success(f"model: '{llm_model_name}' built successfully!")
            except Exception as e:
                st.error(f"Error indexing: {e}")
    st.markdown("---")
with col2:
    st.subheader('Step 2. Retrieval Q&A', divider='blue')
    st.subheader('Question', divider='gray')
    question = st.text_area(
        "Ask your question",
        help="You can ask relevant questions about your PDF files.",
        key="question",
        height=50
    )
    #similarity_val = st.number_input("Provide the similarity value for retrieval process:", value=0.5, min_value=0., max_value=1., step=0.1,
    #                                 help="See the follwing link for more infoamtion: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/#embeddingsfilter", )
    run_button = st.button('Generate Answer')
    st.subheader('Answer', divider='gray')
    # When the run button is clicked
    if run_button:
        # Show a message while loading the response
        with st.spinner("Generating the answer..."):
            try:
                embedding_model_name = st.session_state.embedding_model_name
                if st.session_state.chunk_size is None:
                    chunk_size = 300
                else:
                    chunk_size = st.session_state.chunk_size
                if st.session_state.chunk_overlap is None:
                    chunk_overlap= 50
                else:
                    chunk_overlap = st.session_state.chunk_overlap
                retriever = st.session_state.retriever
                llm = st.session_state.llm
                answer, resources = retrieval_qa_chain(llm,
                                            embedding_model_name,
                                            chunk_size,
                                            chunk_overlap,
                                            retriever,
                                            question
                                            )
                #answer, resources = llm_response_caller(qa_chain, question)
                st.write(answer)
                with st.expander("Resources from VectorDB:"):
                    st.write(resources)
            except Exception as e:
                st.error(f"Error indexing: {e}")
