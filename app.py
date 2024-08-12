import streamlit as st
import openai
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import plotly.express as px  # Import Plotly Express
import plotly.graph_objs as go  # Import Plotly Graph Objects

# Load API keys
with open('key/key.txt') as f:
    openai.api_key = f.read().strip()

google_api_key = 'AIzaSyB1ENyockmC2sWDs0XprPpBeFCdTwWY6dw'  # Replace with a secure way to load this key

# Streamlit UI
st.title("Data Visualization Generator using Langchain agent")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    if df.empty:
        st.error("The uploaded CSV file is empty. Please upload a valid CSV file with data.")
    else:
        st.write("Data Preview:")
        st.write(df.head())

        user_input = st.text_input("Enter your Query")

        if st.button("Submit"):
            # Create the CSV agent using the uploaded file
            agent = create_csv_agent(
                ChatGoogleGenerativeAI(google_api_key=google_api_key, model='gemini-1.5-pro-latest'),
                'D:\langchai_agent\supermarket_sales - Sheet1.csv',  # Use the uploaded file directly
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True
            )
            result = agent.invoke(user_input)
            st.write(result['output'])

            # Generate code using GPT-3.5-turbo
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a helpful AI assistant. The data has been loaded into a DataFrame named 'df'. 
                    The columns are: 'Invoice ID', 'Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 
                    'Total', 'Date', 'Time', 'Payment', 'cogs', 'gross margin percentage', 'gross income', 'Rating'. 
                    Please generate Python visualization code using plotly with different colors and the DataFrame 'df' without any syntax errors. 
                    Don't generate any docstring or triple quotes in the code. try to add som different colors in my visualization"""},
                    {"role": "user", "content": user_input}
                ]
            )

            if response.choices:
                plotly_code = response.choices[0].message['content']
                #st.write("Generated Plotly Code:")
                #st.code(plotly_code)

                # Prepare a context for the execution of the generated code
                exec_context = {
                    "df": df,
                    "pd": pd,  # Include pandas as pd in the context
                    "px": px,
                    "go": go,
                    "st": st
                }

                # Execute the generated Plotly code
                try:
                    exec(plotly_code, exec_context)
                except Exception as e:
                    st.error(f"Error in executing Plotly code: {e}")
