from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

load_dotenv()

api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper)

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200).split_documents(docs)

vector_db = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector_db.as_retriever()

retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search for information about langsmith. For any questions about LangSmith, use this tool")

arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

tools = [wiki, arxiv, retriever_tool]

llm = ChatOpenAI(model = "gpt-4", temperature=0.3)
prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

# agent_executor.invoke(
#     {
#         "input": "Tell me something about Vision Transformers (ViT)."
#     }
# )
st.title("RAG Application")
st.write("Enter your question")

user_input = st.text_input("Query", "")

if st.button("submit"):
    if user_input:
        response = agent_executor.invoke({"input": user_input})
        st.write(response["output"])
    
    else:
        st.write("Please enter a query")

if st.button("quit"):
    st.write("Thank you for using this app")




