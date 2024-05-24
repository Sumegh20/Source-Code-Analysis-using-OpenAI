import streamlit as st
import shutil
import os
from langchain.schema import AIMessage, HumanMessage

from src.helper import repo_ingestion, create_document_from_repo, text_splitter, create_knowledgebase, get_output_stream
from params import database_dir, git_repo_dir, maste_dir
from constant import language_options_list

st.set_page_config("Code Analyzer")
st.header("Analyze your Code ith OpenAI")


with st.sidebar:
    st.title("Menu:")
    github_url = st.text_input('Enter the github repo link')
    language = st.selectbox("Select the programming language", options=language_options_list)

    if st.button("Submit & Process"):
        if os.path.exists(maste_dir):
            shutil.rmtree(maste_dir)

        is_repo_cloned = repo_ingestion(github_url, git_repo_dir)

        if is_repo_cloned:
            with st.spinner("Processing..."):

                documents = create_document_from_repo(repo_path=git_repo_dir, language=language)
                splitted_document = text_splitter(documents)
                create_knowledgebase(texts=splitted_document, db_path=database_dir)

                st.success("Done. Reday for answring you questions.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat_history in st.session_state.chat_history:
    with st.chat_message(chat_history.role):
        st.markdown(chat_history.content)

if query := st.chat_input("Message"):
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        stream = get_output_stream(query, st.session_state.chat_history)
        response = st.write_stream(stream)

        # For static messages
        # response = get_output(query, st.session_state.chat_history)
        # st.write(response)
        
    st.session_state.chat_history.extend([
                                    HumanMessage(role="user", content=query),
                                    AIMessage(role="assistant", content=response)# stream),
                                    ])