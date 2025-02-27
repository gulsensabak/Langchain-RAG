# History aware RAG implementation

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# get the api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create embeddings
embeddings = OpenAIEmbeddings(api_key = OPENAI_API_KEY)

# create model
llm = ChatOpenAI(model = "gpt-4o", api_key = OPENAI_API_KEY)

# get the document
document = TextLoader("product-data.txt").load()

# create splitter
text_splitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)

# create chunks
chunks = text_splitters.split_documents(document)

# create vector store
vector_store = Chroma.from_documents(chunks, embeddings)

# create retriever
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for answering questions.
         Use the provided context to respond. If the answer
         isn't clear, acknowledge that you don't know.
         Limit your response to three concise sentences.
         {context}
         """ ),
         MessagesPlaceholder(variable_name="chat_history"),
         ("human", "{input}")
    ]
)


history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)

# The below 2 functions work together to complete the RAG process
# create_stuff_documents_chain is responsible for stuffing whatever data we get from the vector store that is relevant to this prompt, and then sending to the llm
qa_chain = create_stuff_documents_chain(llm, prompt_template)

# retriever retrieves the relevant data for the prompt that came in from the vector store and passes the data to the create_stuff_documents_chain
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key= "input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


st.write("Chat with Document")
question = st.text_input("Your question: ")

if question:
    response = chain_with_history.invoke({"input": question, "context": ""}, {"configurable": {"session_id": "abc123"}})
    st.write(response['answer'])