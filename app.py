# Import necessary libraries and modules
import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Define a function to generate a summarized response for the given text
def generate_response(txt):
    # # Initialize the OpenAI model with the provided API key, a temperature and latest model type
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
    # Initialize a CharacterTextSplitter to split the input text
    text_splitter = CharacterTextSplitter()
    # Split the input text
    texts = text_splitter.split_text(txt)
    # Create a list of Document objects, each containing a part of the split text
    docs = [Document(page_content=t) for t in texts]
    # Load a summarization chain for the OpenAI model
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    # Run the summarization chain on the list of Document objects and return the result
    return chain.run(docs)

# Set the title of the Streamlit app
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

# Create a text area in the Streamlit app where the user can enter text
txt_input = st.text_area('Enter your text', '', height=200)

# Initialize a list to store the summarized result
result = []
# Create a form in the Streamlit app to accept user's text input for summarization
with st.form('summarize_form', clear_on_submit=True):
    # Create a text input field in the form where the user can enter their OpenAI API key
    openai_api_key = st.text_input('OpenAI API Key', type = 'password', disabled=not txt_input)
    # Create a submit button in the form
    submitted = st.form_submit_button('Submit')
    # If the form is submitted and the OpenAI API key is valid, generate a summarized response
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            # Generate a summarized response for the entered text
            response = generate_response(txt_input)
             # Append the summarized response to the result list
            result.append(response)
            # Delete the OpenAI API key after use
            del openai_api_key

# If there is a summarized result, display it in an info box in the Streamlit app
if len(result):
    st.info(response)