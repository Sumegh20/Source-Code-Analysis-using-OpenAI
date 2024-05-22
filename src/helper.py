import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from parms import database_dir

#clone any github repositories 
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)


#Loading repositories as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                       suffixes=[".py"],
                                       parser = LanguageParser(language=Language.PYTHON, parser_threshold=500)
                                        )
    documents = loader.load()
    return documents


#Creating text chunks 
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON,
                                                             chunk_size = 1000,
                                                             chunk_overlap = 200)
    
    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks


#loading embeddings model
def load_embedding():
    embeddings=OpenAIEmbeddings(disallowed_special=())
    return embeddings


#creating knowlagebase
def create_knowledgebase(texts, db_path="faiss_index"):
    embeddings = load_embedding()

    vectordb = FAISS.from_documents(texts, embeddings)
    vectordb.save_local(db_path)


#loading knowledgebase
def get_knowledge_base(db_path, embedding):
    vectordb = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)

    return vectordb


#Create Prompt
def create_prompt():
    template="""You are an assistant for question-answering tasks while prioritizing a seamless user experience.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use five sentences maximum and keep the answer concise.
            You should be able to remember and reference the last three conversations between you and the user.
            Maintain a friendly, positive, and professional tone throughout interactions.
            Question: {question}
            Context: {context}
            Chat history: {chat_history}
            Answer:
        """
    prompt=ChatPromptTemplate.from_template(template)

    return prompt


# input handeler
def input_handeler(input: dict):
    return input['question']


# create RAG chain
def create_rag_chain(llm, retriever, prompt):
    rag_chain = (
            # {"context": retriever,  "question": RunnablePassthrough()}
            RunnablePassthrough().assign(
                context = input_handeler | retriever
            )
            | prompt
            | llm
            | StrOutputParser()
        )
    
    return rag_chain


def get_llm():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    return llm

#user_input
def get_output_stream(question, chat_history):
    embedding = load_embedding()
    vectordb = get_knowledge_base(db_path=database_dir, embedding=embedding)
    llm = get_llm()
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":3})
    prompt = create_prompt()

    rag_chain = create_rag_chain(llm=llm, retriever=retriever, prompt=prompt)
    
    result = rag_chain.stream({"question": question, "chat_history":chat_history})
    return result