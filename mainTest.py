import os
import psycopg2
import ast
import json
import warnings
import sys
import unidecode
from datetime import datetime
from sqlalchemy import create_engine, inspect
import pandas as pd
import plotly.express as px

dbname = user = "postgres"
password = os.getenv("POSTGRES_PASSWORD", "Podamaire.123")
host = "localhost"
port = 5432

COLUMNS_TO_NORMALIZE = [
    "phone_number",
    "condition",
    "footnote",
    "location",
]

COLUMNS_TO_KEEP = [
    "index",
    "provider_id",
    "hospital_name",
    "address",
    "city",
    "state",
    "zip_code",
    "county_name",
    "measure_id",
    "measure_name",
    "score",
    "sample",
    "measure_start_date",
    "measure_end_date",
]
DB_FIELDS = COLUMNS_TO_NORMALIZE + COLUMNS_TO_KEEP

COLUMNS_DESCRIPTIONS = {
    "index": "primary key of the database and unique identifier in the database.",
    "provider_id": "the unique identification number for health care providers",
    "hospital_name": "string containing name of hospital",
    "address": "a string containing street address only of hospital",
    "city": "the city where the hospital is located",
    "state": "the state where the hospital is located",
    "zip_code": "the zip code of hospital",
    "county_name": "the county where the hospital is located",
    "phone_number": "the phone number of the hospital",
    "condition": "the condition being measured and categorizes types of tests/treatments being monitored",
    "measure_id": "the id of the measure being used",
    "measure_name": "contains all related measure under respective condition that was studied as part of the respective condition category",
    "score": "the score of the measure used. Score grades how well the measure did compared to expectations or goals",
    "sample": "the number of patients in the particular study",
    "footnote": "contains additional information about the measure",
    "measure_start_date": "the date when all test students were allowed entry into their respective test groups, when the measure started",
    "measure_end_date": "the date when no more data entries can be made, when the measure ends",
    "location": "the full address of the hospital containing the street address, city, state, and zip code",
}


from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentType
from langchain.agents import create_sql_agent
from langchain.agents import create_openai_functions_agent
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.sql_database import SQLDatabase
from langchain.tools import tool, Tool
import streamlit as st
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
TABLE_NAME = "hospital_care_data"

db = SQLDatabase.from_uri(
    url,
    include_tables=[TABLE_NAME],
    sample_rows_in_table_info=1,
)

CUSTOM_SUFFIX = """Begin!

Relevant pieces of previous conversation:
{history}
(Note: Only reference this information if it is relevant to the current query.)

Question: {input}
Thought Process: It is imperative that I do not fabricate information not present in the database or engage in hallucination; 
maintaining trustworthiness is crucial. If the user specifies a category, I should attempt to align it with the categories in the `condition` or 'footnote' 
or columns of the `hospital_care_data` table, utilizing the `get_categories` tool with an empty string as the argument. 
Next, I will acquire the schema of the `hospital_care_data` table using the `sql_db_schema` tool. 
Utilizing the `get_columns_descriptions` tool is highly advisable for a deeper understanding of the `hospital_care_data` columns, except for straightforward tasks. 
When provided with a hospital's city, I will search in the `city` column; for a hospital's address, in the `address` column; for a hospital's state, in the 'state' column; for a hospital's phone number in the phone number column.. 
The `get_today_date` tool, requiring an empty string as an argument, will provide today's date. 
In SQL queries involving string or TEXT comparisons when provided hospital name, city, state, and address, I must use the `UPPER()` function for case-insensitive comparisons and the `LIKE` operator for fuzzy matching. 
My final response must be delivered in the language of the user's query.

{agent_scratchpad}
"""

langchain_chat_kwargs = {
    "temperature": 0,
    "max_tokens": 4000,
    "verbose": True,
}
chat_openai_model_kwargs = {
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": -1,
}

# Database constants
hospital_care_data = [
    "index",
    "provider_id",
    "hospital_name",
    "address",
    "city",
    "state",
    "zip_code",
    "county_name",
    "phone_number",
    "condition",
    "measure_id",
    "measure_name",
    "score",
    "sample",
    "footnote",
    "measure_start_date",
    "measure_end_date",
    "location",
]


def get_chat_openai(model_name):
    """ return instance of the ChatOpenAI class intialized with the specifed model name.
        args: model_name (str): name of model to use
        returns:
        ChatOpenAI: instance of the ChatOpenAI class """
    llm = ChatOpenAI(
        openai_api_key="enter_your_api_key_here",
        model_name=model_name,
        model_kwargs=chat_openai_model_kwargs,
        **langchain_chat_kwargs
    )
    return llm


def get_sql_toolkit(tool_llm_name: str):
    """ get sql toolkit for given llm name.
        parameters: tool_llm_name (str): the name of the llm tool.
        returns:
            SQLDatabaseToolkit: The SQL toolkit object """
    llm_tool = get_chat_openai(model_name=tool_llm_name)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm_tool)
    return toolkit


def get_agent_llm(agent_llm_name: str):
    """retrieves llm agent with the specified name.
        parameters: agent_llm_name (str): name of the LLM agent.
        returns:
        llm_agent: the llm agent object. """
    llm_agent = get_chat_openai(model_name=agent_llm_name)
    return llm_agent


def create_agent(
        tool_llm_name: str = "gpt-4-1106-preview",
        agent_llm_name: str = "gpt-4-1106-preview",
        memory: ConversationBufferMemory=None,):
    """ creates sql agent using the specified tool and agent LLM names.
        args:
            tool_llm_name (str, optional): name of the SQL toolkit LLM, defaults to "gpt-4-1106-preview".
            agent_llm_name (str, optional): name of the agent LLM, defaults to "gpt-4-1106-preview".
        returns:
            agent: the created SQL agent. """
    agent_tools = sql_agent_tools()
    llm_agent = get_agent_llm(agent_llm_name)
    toolkit = get_sql_toolkit(tool_llm_name)
    memory = memory or ConversationBufferMemory(memory_key="history", input_key="input")

    agent = create_sql_agent(
        llm=llm_agent,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        toolkit=toolkit,
        input_variables=["input", "agent_scratchpad", "history"],
        suffix=CUSTOM_SUFFIX,
        agent_executor_kwargs={"memory": memory},
        extra_tools=agent_tools,
        verbose=True,
    )
    return agent

# Define a compliant prompt template including agent_scratchpad and memory
DEFAULT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def _clean_memory_if_invalid(memory):
    if hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
        if not all(isinstance(m, BaseMessage) for m in memory.chat_memory.messages):
            memory.chat_memory.messages = []



def create_metadata_agent(model_name="gpt-4-1106-preview", memory=None):
    llm = get_chat_openai(model_name)
    tools = sql_agent_tools()[:2]
    memory = memory or ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)
    _clean_memory_if_invalid(memory)
    agent_core = create_openai_functions_agent(llm=llm, tools=tools, prompt=DEFAULT_PROMPT)
    return AgentExecutor(agent=agent_core, tools=tools, memory=memory, verbose=True)


def create_general_qa_agent(model_name="gpt-4-1106-preview"):
    llm = get_chat_openai(model_name)
    prompt = PromptTemplate.from_template("Explain the following SQL result in plain English:\n{input}")
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain


def pipeline_agent(question: str):
    sql_agent = create_agent()
    sql_result = sql_agent.invoke({"input": question})["output"]
    print("[DEBUG] SQL Result:", sql_result)

    explainer_agent = create_general_qa_agent()
    final_response = explainer_agent.run(f"SQL result: {sql_result}")

    return final_response




# Create a wrapper Tool so this chain can be called from other agents too
pipeline_tool = Tool.from_function(
    func=pipeline_agent,
    name="multi_agent_chain",
    description="Combines metadata understanding, SQL querying, and result explanation. Use for complex or ambiguous queries."
)


def create_combined_agent(model_name="gpt-4-1106-preview", memory=None):
    llm = get_chat_openai(model_name)
    tools = sql_agent_tools() + [pipeline_tool]
    memory = memory or ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)
    _clean_memory_if_invalid(memory)
    agent_core = create_openai_functions_agent(llm=llm, tools=tools, prompt=DEFAULT_PROMPT)
    return AgentExecutor(agent=agent_core, tools=tools, memory=memory, verbose=True)


def show_table_preview():
    st.sidebar.markdown("---")
    st.sidebar.subheader("View Table Data")

    dbname = "postgres"
    user = "postgres"
    password = os.getenv("POSTGRES_PASSWORD", "Podamaire.123")
    host = "localhost"
    port = 5432

    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(conn_str)
    inspector = inspect(engine)
    table_list = inspector.get_table_names()

    selected_table = st.sidebar.selectbox("Choose a table", table_list)

    if "show_data" not in st.session_state:
        st.session_state.show_data = False

    if st.sidebar.button("Show Data"):
        st.session_state.show_data = True

    if st.session_state.show_data:
        df = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 50", con=engine)
        st.session_state.table_df = df

    if "table_df" in st.session_state:
        st.subheader(f"Preview: {selected_table}")
        df = st.session_state.table_df
        st.dataframe(df)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        #df["sample"] = pd.to_numeric(df["sample"], errors="coerce")

        if "score" in df.columns and "condition" in df.columns:
            fig = px.box(df, x="condition", y="score", template="plotly_dark", points="all")
            st.plotly_chart(fig, use_container_width=True)


"""get_categories tool helps agent fetch list of distinct items from categorical columns as there is lower amount of values 
    which does does not make it the optimal retrieval tool"""


def run_query_save_results(dbx, queryx):
    res = dbx.run(queryx)
    res = [el for sub in ast.literal_eval(res) for el in sub]
    return res


def get_categories(queryx: str) -> str:
    # for categories/subcategories. a json is returned where the key is category/subcategory and value is list of unique items for both
    cat1 = run_query_save_results(
        db, "SELECT DISTINCT condition from hospital_care_data"
    )
    cat2 = run_query_save_results(
        db, "SELECT DISTINCT footnote from hospital_care_data"
    )
    cat1_str = (
            "List of unique values in condition column : \n"
            + json.dumps(cat1, ensure_ascii=False)
    )
    cat2_str = (
            "\n List of unique values in footnote column: \n"
            + json.dumps(cat2, ensure_ascii=False)
    )

    return cat1_str + cat2_str


"""get_columns_descriptions tool returns short descriptions for every ambiguous column"""


def get_columns_descriptions(queryx: str) -> str:
    """useful for retrieving desscription of the columns in hospital_care_data table"""
    return json.dumps(COLUMNS_DESCRIPTIONS)


"""get_today_date tool allows retrieval of today's date using python datetime library and useful for agent
    when asked about temporality"""


def get_today_date(queryx: str) -> str:
    """useful to get date of today."""
    today_date_string = datetime.now().strftime("%Y-%m-%d")
    return today_date_string


def sql_agent_tools():
    def get_categories(_: str) -> str:
        cat1 = run_query_save_results(db, "SELECT DISTINCT condition from hospital_care_data")
        cat2 = run_query_save_results(db, "SELECT DISTINCT footnote from hospital_care_data")
        return (
            "List of unique values in condition column:\n" + json.dumps(cat1, ensure_ascii=False) +
            "\nList of unique values in footnote column:\n" + json.dumps(cat2, ensure_ascii=False)
        )

    def get_columns_descriptions(_: str) -> str:
        return json.dumps(COLUMNS_DESCRIPTIONS)

    def get_today_date(_: str) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    return [
        Tool.from_function(func=get_categories, name="get_categories", description="Returns unique condition and footnote values as categories."),
        Tool.from_function(func=get_columns_descriptions, name="get_columns_descriptions", description="Provides descriptions of columns in hospital_care_data."),
        Tool.from_function(func=get_today_date, name="get_today_date", description="Provides today's date."),
    ]


connection = psycopg2.connect(
    dbname=dbname,
    user=user,
    password=password,
    host=host,
    port=port
)
cursor = connection.cursor()
table_name = "hospital_care_data"
query = f"SELECT hospital_name FROM {table_name} WHERE city = 'Wedowee';"

# Execute the query
cursor.execute(query)

# Fetch all rows from the result
rows = cursor.fetchall()

for row in rows:
    print(row)