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

# Set a password
password = st.text_input("Enter password:", value="", type="password")
if password != "ThisIsStrong":
    st.error("Incorrect password. Please try again.")
    st.stop()

st.header("Current System Template Prompt, i.e. instructions for how the assistant should respond")
st.write("""Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Question are related to how project managemnet software works. The documentation seeks to help users naviagte and use the software.
    Do not include "https://rev79" as a source. Source will always be longer URLs, for example, 
    "https://rev79.freshdesk.com/en/support/solutions/articles/47001227676"
    Answer the question in the following language, {language}. You have to include the "SOURCES" at all times regardless of the language used.
    Translate the word "SOURCES" in the language that is being used and make sure the "SOURCES" are in a new line.
    ALWAYS return a "SOURCES" part in your answer.
    The "SOURCES" part should be a reference to the source of the document from which you got your answer.
    ```
    The answer is foo
    SOURCES: xyz
    ----------------
    {summaries}""")

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
# Custom prompt section
if st.checkbox("Create custom system template prompt"):
    system_template = st.text_input('''Enter system template instructions. In other words, give the assistant the instructions of how you'd like it to answer the questions. 
    More detailed instructions will result in a better assistant. Disregard "language" and "summaries" seen in current template :''')
    system_template=system_template + """
    Answer the question in the following language, {language}.
    ----------------
    {summaries}"""
else: 
    system_template = """Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Question are related to how project managemnet software works. The documentation seeks to help users naviagte and use the software.
    Do not include "https://rev79" as a source. Source will always be longer URLs, for example, 
    "https://rev79.freshdesk.com/en/support/solutions/articles/47001227676"
    Answer the question in the following language, {language}. You have to include the "SOURCES" at all times regardless of the language used.
    Translate the word "SOURCES" in the language that is being used and make sure the "SOURCES" are in a new line.
    ALWAYS return a "SOURCES" part in your answer.
    The "SOURCES" part should be a reference to the source of the document from which you got your answer.
    ```
    The answer is foo
    SOURCES: xyz
     ----------------
    {summaries}"""
    
# Submit button

if st.button("Submit"):
    # Call the function from your_module with the selected language and question
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"] 

    #Use OpenAI's embedding
    Embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)    

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    index_name = "rev79"  

    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=Embeddings)   

    st.markdown("Assistant is typing...")

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        #SystemMessage(content=system_message),
        #HumanMessage(content="{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages) 

    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=Embeddings)
    docs = docsearch.similarity_search(question, include_metadata=True)
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

