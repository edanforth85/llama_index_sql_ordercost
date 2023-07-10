import os
import streamlit as st
import ploty.express as px
from collections import defaultdict
import numpy as np
import re
import openai
from io import StringIO
import sys
import traceback
from contextlib import redirect_stdout
import ast

from streamlit_chat import message as st_message
from sqlalchemy import create_engine

from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

from llama_index import GPTSQLStructStoreIndex, LLMPredictor, ServiceContext
from llama_index import SQLDatabase as llama_SQLDatabase
from llama_index.indices.struct_store import SQLContextContainerBuilder

from constants import (
    DEFAULT_SQL_PATH,
    DEFAULT_ORDER_TABLE_DESCRP,
    DEFAULT_COST_TABLE_DESCRP,
    DEFAULT_LC_TOOL_DESCRP,
)
from utils import get_sql_index_tool, get_llm


@st.cache_resource
def initialize_index(
    llm_name, model_temperature, table_context_dict, api_key, sql_path=DEFAULT_SQL_PATH
):
    """Create the GPTSQLStructStoreIndex object."""
    llm = get_llm(llm_name, model_temperature, api_key)

    engine = create_engine(sql_path)
    sql_database = llama_SQLDatabase(engine)

    context_container = None
    if table_context_dict is not None:
        context_builder = SQLContextContainerBuilder(
            sql_database, context_dict=table_context_dict
        )
        context_container = context_builder.build_context_container()

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm))
    index = GPTSQLStructStoreIndex(
        [],
        sql_database=sql_database,
        sql_context_container=context_container,
        service_context=service_context,
    )

    return index


@st.cache_resource
def initialize_chain(llm_name, model_temperature, lc_descrp, api_key, _sql_index):
    """Create a (rather hacky) custom agent and sql_index tool."""
    sql_tool = Tool(
        name="SQL Index",
        func=get_sql_index_tool(
            _sql_index, _sql_index.sql_context_container.context_dict
        ),
        description=lc_descrp,
    )

    llm = get_llm(llm_name, model_temperature, api_key=api_key)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent_chain = initialize_agent(
        [sql_tool],
        llm,
        agent="chat-conversational-react-description",
        verbose=True,
        memory=memory,
    )

    return agent_chain


st.title("Order Cost Natural Language Charts with ChatGPT, LangChain and Llama Index 🦙")
st.markdown(
    (
        "This sandbox uses a sqlite database by default, powered by [Llama Index](https://gpt-index.readthedocs.io/en/latest/index.html) ChatGPT, and LangChain.\n\n"
        "The database contains information on order cost for a fictional business."
        "This data is spread across two tables - orders - containing order master data and cost - containing cost by month per order.\n\n"
        "Using the setup page, you can adjust LLM settings, change the context for the SQL tables, and change the tool description for Langchain."
        "The other tabs will perform chatbot and text2sql operations.\n\n"
        "Read more about LlamaIndexes structured data support [here!](https://gpt-index.readthedocs.io/en/latest/guides/tutorials/sql_guide.html)"
    )
)


setup_tab, llama_tab, lc_tab, chart_tab = st.tabs(
    ["Setup", "Llama Index", "Langchain+Llama Index", "Charts"]
)

with setup_tab:
    st.subheader("LLM Setup")
    api_key = st.text_input("Enter your OpenAI API key here", type="password")
    llm_name = st.selectbox(
        "Which LLM?", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
    )
    model_temperature = st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0, step=0.1
    )

    st.subheader("Table Setup")
    order_table_descrp = st.text_area(
        "Order table description", value=DEFAULT_ORDER_TABLE_DESCRP
    )
    cost_table_descrp = st.text_area(
        "Cost table description", value=DEFAULT_COST_TABLE_DESCRP
    )

    table_context_dict = {
        "order_table": order_table_descrp,
        "cost_table": cost_table_descrp,
    }

    use_table_descrp = st.checkbox("Use table descriptions?", value=True)
    lc_descrp = st.text_area("LangChain Tool Description", value=DEFAULT_LC_TOOL_DESCRP)

if 'global_response' not in st.session_state:
    st.session_state['global_response'] = ""
if 'query_str' not in st.session_state:
    st.session_state['query_str'] = ""
if 'response_table' not in st.session_state:
    st.session_state['response_table'] = ""
if 'chart_response' not in st.session_state:
    st.session_state['chart_response'] = ""    
if 'global_python' not in st.session_state:
    st.session_state['global_python'] = ""   

with llama_tab:
    st.subheader("Text2SQL with Llama Index")
    if st.button("Initialize Index", key="init_index_1"):
        st.session_state["llama_index"] = initialize_index(
            llm_name,
            model_temperature,
            table_context_dict if use_table_descrp else None,
            api_key,
        )

    if "llama_index" in st.session_state:
        query_text = st.text_input(
            "Query:", value="Show me cost for each Line_of_Business (look up via order using order_table but don't display the order) for each month filter to actual"
        )
        use_nl = st.checkbox("Return natural language response?")
        if st.button("Run Query") and query_text:
            with st.spinner("Getting response..."):
                try:
                    response = st.session_state["llama_index"].as_query_engine(synthesize_response=use_nl).query(query_text)
                    response_text = str(response)
                    response_sql = response #.extra_info["sql_query"]
                    st.session_state['global_response'] = str(response)
                    st.session_state['query_str'] = query_text
                    st.session_state['response_table'] = ast.literal_eval(response_text)

                except Exception as e:
                    response_text = "Error running SQL Query."
                    response_sql = str(e)

            col1, col2 = st.columns(2)
            with col1:
                st.text("SQL Result:")
                st.markdown(response_text)

            with col2:
                st.text("SQL Query:")
                st.markdown(response_sql) #response_sql

with lc_tab:
    st.subheader("Langchain + Llama Index SQL Demo")

    if st.button("Initialize Agent"):
        st.session_state["llama_index"] = initialize_index(
            llm_name,
            model_temperature,
            table_context_dict if use_table_descrp else None,
            api_key,
        )
        st.session_state["lc_agent"] = initialize_chain(
            llm_name,
            model_temperature,
            lc_descrp,
            api_key,
            st.session_state["llama_index"],
        )
        st.session_state["chat_history"] = []

    model_input = st.text_input(
        "Message:", value="How much was actual spend in June?"
    )
    if "lc_agent" in st.session_state and st.button("Send"):
        model_input = "User: " + model_input
        st.session_state["chat_history"].append(model_input)
        with st.spinner("Getting response..."):
            response = st.session_state["lc_agent"].run(input=model_input)
        st.session_state["chat_history"].append(response)

    if "chat_history" in st.session_state:
        for msg in st.session_state["chat_history"]:
            st_message(msg.split("User: ")[-1], is_user="User: " in msg)

with chart_tab:
    st.subheader("Chart time!")

    chart_input = st.text_input(
        "Response:", value=st.session_state['global_response']
    )
    if st.button("Chart it!", key="chart_button"):
        openai.api_key = api_key
        fig = ""
        prompt = (
            f"Can you show me how to create a chart of the following data in streamlit using plotly express (stacked bar chart by LOB)? {st.session_state['global_response']} " \
            f"Just give me the python code with no pip installs and no comments or natural language instructions but do display it as a python block. Don't take any short cuts - "\
            f"you have access to the data as st.session_state['response_table'] (always write it out completely with the prefix st.session_state." \
            f"it's a python list, don't declare it in the code you return or show the data in your response). " \
            f"Here is the question it is intended to answer: {st.session_state['query_str']}" \
            f"Show the fig with st.plotly_chart(fig, theme='streamlit', use_container_width=True) instead of plt.show(). "\
            f"Here is a good example response: \n" +
            """'''python\nimport plotly.express as px\nimport pandas as pd\n\n# Create a DataFrame\ndf = pd.DataFrame(st.session_state['response_table'], 
            columns=['LOB', 'Month', 'Value'])\n\n# Sort the months\nmonths_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
            'September', 'October', 'November', 'December']\ndf['Month'] = pd.Categorical(df['Month'], categories=months_order, ordered=True)\n\n
            # Create the stacked bar chart\nfig = px.bar(df, x='Month', y='Value', color='LOB', barmode='stack')\n\nst.plotly_chart(fig, theme=\"streamlit\", use_container_width=True)\n'''\n"""
            f"again! display the code as a python block, not regular text so your response will start as python\nimport plotly.express"
        )

        # st.write(f"prompt: {prompt}")
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo", # gpt-4-0613
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        st.session_state['chart_response'] = f"response: {str(response)}"

        code_blocks = re.findall(r'```(?:python)?(?:\n|\s)(.*?)(?:\n|\s)```', response.choices[0].message['content'], re.DOTALL)
        python_code = "\n".join(block for block in code_blocks if not block.startswith("Output:"))

        st.session_state['global_python'] = python_code
        st.write(f"Here is your chart!\n" ) 

        try:
            exec(st.session_state['global_python'])
        except Exception as e:
            error = traceback.format_exc()
    if st.button("Show response", key="response_button"):
        st.write(st.session_state['chart_response'])
