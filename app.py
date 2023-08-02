import streamlit as st
#import your_module  # Replace this with the name of the Python file containing your existing code

# Playing with the new Open AI API : gpt-4
#importing dependencies
import langchain
import openai
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

st.title("Rev79 Knowledge Base Assistant")
# Language selection
languages = [
    "English",
    "Mandarin Chinese",
    "Spanish",
    "Hindi",
    "Arabic",
    "Portuguese",
    "Bengali",
    "Russian",
    "Japanese",
    "French",
    "Indonesian",
    "German",
    "Korean",
    "Turkish"
]

selected_language = st.selectbox("Select language:", languages)

# Input question
question = st.text_input("Enter your question:")
 
system_template = """
    You are a helpful AI Assistant. Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Question are related to how project managemnet software works. The documentation seeks to help users naviagte and use the software.
    Do not include "https://rev79" as a source. Source will always be longer URLs, for example, 
    "https://rev79.freshdesk.com/en/support/solutions/articles/47001227676"
    Answer the question in the following language, {language}. You have to include the "SOURCES" at all times regardless of the language used.
    Translate the word "SOURCES" in the language that is being used and make sure the "SOURCES" are in a new line.
    
    Example:
 
    ```
    Question: What is Rev79?
    
    Answer: Rev79 is a project management platform named after God's promise in Revelation 7:9 of all languages communities being included in his eternal purpose
    of blessing and recreation. The platform aims to help organizations, teams, and communities move forward towards this vision by providing tools for 
    managing projects and facilitating Bible translation and integral mission in all language communities. Rev79 can be used to envision, organize, collaborate, 
    and transform projects and activities.
    
    SOURCES: 
    - https://rev79.freshdesk.com/en/support/solutions/articles/47001223622-what-is-the-rev79-app-where-did-it-come-from-
    ```
     ----------------
    {summaries}"""
    
# Submit button

if st.button("Submit"):
    # Call the function from your_module with the selected language and question
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]

    #Use OpenAI's embedding
    Embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/stsb-xlm-r-multilingual')    

    # initialize pinecone
    supabase: Client = create_client(supabase_url, supabase_key)  

    docsearch = SupabaseVectorStore(embedding=embeddings, table_name='documents', client=supabase)   

    st.markdown("Assistant is typing...")

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        #SystemMessage(content=system_message),
        #HumanMessage(content="{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages) 

    docs = docsearch.similarity_search(question)
    chain_type_kwargs = {"prompt": prompt}
    chain = load_qa_with_sources_chain(
        ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY), 
        chain_type="stuff",
        prompt=prompt
    )
    text = chain({"input_documents": docs, "question": question, "language":selected_language}, return_only_outputs=True)    

    result = text['output_text']  # Replace 'run' with the appropriate function in your module
    
    # Display a header
    st.header("Answer")
    st.write(result)    

