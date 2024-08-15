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
from langchain.schema import SystemMessage
from pydantic import BaseModel
from report import write_report_tool
from langchain.memory import ConversationBufferMemory
from chat_model_handlers import timing_handler
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
    model_name="gemma2-9b-It",
    temperature=0.5
)
#Llama3-8b-8192
# Connect to the SQLite database
conn = sqlite3.connect("db.sqlite")


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join([row[0] for row in rows])

tables = list_tables()

def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'"   + table  + "'" for table in table_names) 
    rows = c.execute(f"SELECT sql from sqlite_master where type = 'table' and name IN({tables});")
    return  '\n'.join(row[0] for row in rows if row[0] is not None)

class DescribeTablesArgsSchema(BaseModel):
    table_names: list[str]

describe_tables_tool = Tool.from_function(
    name = "describe",
    description= "Given a list of table names , returns the schema of those tables",
    func = describe_tables,
    args_schema= DescribeTablesArgsSchema

)  
 




def run_sqlite_query(query):
    """Execute a SQLite query and return the results."""
    c = conn.cursor()
    c.execute(query)
    return c.fetchall()

class RunQueryArgsSchema(BaseModel):
    query: str




# Define the tool for running SQLite queries
run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a SQLite query.",
    func=run_sqlite_query,
    args_schema= RunQueryArgsSchema
)

# Define the chat prompt template
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content = (" You are an AI that has access to a SQLite Database.\n"
                                 f"The databse has tables of {tables}\n"
                                 "Do not make any assumptions about what tables exist "
                                 "or what columns exist.instead,use the 'describe_tables' function"
                                 "Remind it first undertsand the schema . that where and what tables are in there . see the column inside the table")),
        MessagesPlaceholder(variable_name='chat_history'),                        
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)


memory = ConversationBufferMemory(memory_key = "chat_history",return_messages = True)

tools = [
    run_query_tool,
    describe_tables_tool,
    write_report_tool
]

# Initialize the agent with the LLM, prompt, and tools
agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=[run_query_tool,describe_tables_tool]
)

# Initialize the agent executor
agent_executor = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    tools=[run_query_tool],
    verbose=True,
    memory = memory,
    callbacks=[timing_handler] 
)

# Main app functionality
query = st.text_area("Enter your query:","How many orders are there ?")
if st.button("Run Query"):
    with st.spinner("Running the query..."):
        response = agent_executor.invoke({"input": query})
        st.write(response['output'])
