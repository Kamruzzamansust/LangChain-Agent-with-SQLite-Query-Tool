# streamlit_app.py

import streamlit as st
from langchain.agents import AgentExecutor, create_openai_functions_agent, initialize_agent, AgentType
from langchain_groq import ChatGroq
import sqlite3
from langchain.tools import Tool
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
import os

# Streamlit app setup
st.title("LangChain Agent with SQLite Query Tool")
st.write("Use the sidebar to configure the API key and database query.")

# Sidebar for API Key input
groq_api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your GROQ API Key to proceed.")
    st.stop()

# Initialize the ChatGroq language model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192",
    temperature=0.5
)

# Connect to the SQLite database
conn = sqlite3.connect("db.sqlite")

def run_sqlite_query(query):
    """Execute a SQLite query and return the results."""
    c = conn.cursor()
    c.execute(query)
    return c.fetchall()

# Define the tool for running SQLite queries
run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a SQLite query.",
    func=run_sqlite_query
)

# Define the chat prompt template
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

# Initialize the agent with the LLM, prompt, and tools
agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=[run_query_tool]
)

# Initialize the agent executor
agent_executor = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    tools=[run_query_tool],
    verbose=True
)

# Main app functionality
query = st.text_area("Enter your query:", "How many users are in the database?")

if st.button("Run Query"):
    with st.spinner("Running the query..."):
        response = agent_executor.invoke({"input": query})
        st.write(response['output'])
