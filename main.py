# _*_ coding : utf-8 _*_
# @Time : 2024/4/27 20:15
# @Author : Mingmmm
# @File : main
# @Project : train
import datetime
import glob
import os
import openai
import streamlit as st
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter,TokenTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# 设置页面
import random
import time
from PyPDF2 import PdfReader
st.set_page_config(page_title="Welcome to Audit Law",layout = "wide")
st.title("Welcome to Audit Law📖")
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"

#设置Prompt
Prompt_template = """
假如你是一名精通审计法的审计师，请使用以下的上下文回答最后的问题。如果你不知道答案，那就说你不知道，不要试图编造答案，但关于审计的常识你要回答。必须使用中文来回答以下问题，回答的内容尽可能详细。
上下文：{context}
问题: {question}
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=Prompt_template,)

## 设置API输入窗口
def Open_api_key():
    user_api_key = st.sidebar.text_input(
        label="请输入OpenAI API后使用 ",
        placeholder="Paste your openAI API key",
        type="password")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        openai.api_key = user_api_key

# pdf文档embedding and split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
def load_db(pdf, chain_type, k):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    embeddings = OpenAIEmbeddings()
    chunks = text_splitter.split_text(text)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    retriever = knowledge_base.as_retriever(search_type="similarity", search_kwargs={"k": k})
    print(retriever)
    # create a chatbot chain. Memory is managed externally.
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa
chat_history = []

# 设置pdf传输窗口
def main():
    st.sidebar.title("请选择PDF文件")
    pdf_list = st.sidebar.file_uploader("请选择一个PDF文件", type="pdf", accept_multiple_files=False)
    if pdf_list is not None:
        st.sidebar.write("文件载入成功，现在可以进行文档问答")
        st.header("Chat with AL🤔")
    print(pdf_list)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

#设置输入、输出
    # Accept user input
    if prompt := st.chat_input("Type something..."):
        if pdf_list is not None:
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                qa = load_db(pdf_list, "stuff", 4)
                result = qa({"query": prompt, "chat_history": chat_history})
                chat_history.append((prompt, result["result"]))
                assistant_response =result["result"]
                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                #print(st.session_state.messages)
        else:
            with st.container():
                st.warning("Please upload your PDF file in the settings page.")
if __name__ == '__main__':
    Open_api_key()
    main()

