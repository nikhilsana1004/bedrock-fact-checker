import streamlit as st
import boto3
import json
import pandas as pd
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# AWS client setup
try:
    session = boto3.Session(region_name=AWS_REGION)
    bedrock_runtime = session.client('bedrock-runtime')
    comprehend = session.client('comprehend')
except (NoCredentialsError, PartialCredentialsError) as e:
    st.error("AWS credentials not found. Please configure your environment correctly.")

def call_bedrock_titan_model(prompt):
    """Call Amazon Bedrock Titan model for text generation."""
    try:
        input_body = {"inputText": prompt}
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-text-lite-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(input_body)
        )
        output_body = json.loads(response["body"].read().decode())
        return output_body["results"][0]["outputText"], output_body
    except Exception as e:
        st.error(f"Error invoking model: {e}")
        return None, None

def check_toxicity_with_comprehend(text):
    """Check text toxicity using Amazon Comprehend."""
    try:
        response = comprehend.detect_toxic_content(
            TextSegments=[{"Text": text}],
            LanguageCode='en'
        )

        # Remove the "GRAPHIC" label from the response
        for result in response.get("ResultList", []):
            result["Labels"] = [label for label in result["Labels"] if label["Name"] != "GRAPHIC"]

        toxicity_detected = any(item['Toxicity'] > 0.5 for item in response['ResultList'])
        return toxicity_detected, response
    except Exception as e:
        st.error(f"Error checking toxicity: {e}")
        return False, None

def json_to_table(toxicity_json):
    """Convert toxicity JSON to DataFrame for display."""
    table_data = []
    for result in toxicity_json.get("ResultList", []):
        for label in result.get("Labels", []):
            table_data.append({
                "Name": label["Name"],
                "Score": label["Score"]
            })
    if table_data:
        df = pd.DataFrame(table_data)
        return df
    else:
        return None

def search_google(query, api_key, cse_id):
    """Search Google using Custom Search API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": 5
    }
    response = requests.get(url, params=params)
    return response.json()

def summarize_snippets(snippets):
    """Summarize search snippets using Claude."""
    combined_text = " ".join(snippets)
    prompt = f"Human: Summarize the following text:\n\n{combined_text}\n\nSummary:\nAssistant:"

    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-v2',
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 200
        })
    )

    output_body = json.loads(response["body"].read().decode())
    summary = output_body["completion"]

    return summary.strip()

def fact_check(statement, summary):
    """Fact-check statement using Claude."""
    prompt = f"Human: Please fact-check the following statement: '{statement}' based on the following summary: {summary}\nAssistant:"

    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-v2',
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 200
        })
    )

    result = json.loads(response["body"].read().decode())
    fact_check_result = result.get('completion', 'No fact-checking result found.')

    return fact_check_result.strip()

def generate_fact_check(statement, api_key, cse_id):
    """Generate comprehensive fact-check with sources."""
    search_results = search_google(statement, api_key, cse_id)
    snippets = [item['snippet'] for item in search_results.get('items', [])]

    if not snippets:
        return "No relevant results found for the given statement."

    summary = summarize_snippets(snippets)
    sources = "\n".join([f"- [{item['title']}]({item['link']})" for item in search_results.get('items', [])])

    fact_check_result = fact_check(statement, summary)

    return f"Fact-Check Summary:\n{fact_check_result}\n\nSources:\n{sources}"

# Suggested questions
suggested_questions = [
    "Is Mumbai the most populated city in the world?",
    "What are the best travel destinations?",
    "How do I find the cheapest flights?",
    "How can I stay safe while traveling abroad?",
    "What are the top 10 travel tips for first-time travelers?",
    "How do I find eco-friendly travel options?",
    "What are the best ways to book hotels?",
    "How do I plan a budget-friendly trip?"
]

def main():
    """Main application function."""
    st.title(" Amazon Bedrock Fact Checker")
    st.caption("AI-powered fact-checking with toxicity detection")

    selected_question = st.selectbox(
        "Choose a question to explore:",
        suggested_questions
    )

    user_input = st.text_input("Or enter your own query", selected_question if selected_question else "")

    # Initialize session state
    if 'response_text' not in st.session_state:
        st.session_state.response_text = None
    if 'toxicity_details' not in st.session_state:
        st.session_state.toxicity_details = None

    # Button: Submit Query
    if st.button("Submit Query"):
        response_text, output_body = call_bedrock_titan_model(user_input)
        if response_text:
            st.session_state.response_text = response_text
            toxicity_detected, toxicity_details = check_toxicity_with_comprehend(response_text)
            st.session_state.toxicity_details = toxicity_details
            st.session_state.toxicity_detected = toxicity_detected

    # Display stored response
    if st.session_state.response_text:
        st.write("Model Response:")
        st.write(st.session_state.response_text)

        # Toxicity details
        if st.session_state.toxicity_details:
            df_toxicity = json_to_table(st.session_state.toxicity_details)
            if df_toxicity is not None:
                st.write("Toxicity Details:")
                st.table(df_toxicity)

    # Button: Fact-Check Response
    if st.session_state.response_text and st.button("Fact-Check Response"):
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            st.error("Google API credentials not configured. Please check your .env file.")
        else:
            st.write("Fact-checking the response...")
            fact_check_summary = generate_fact_check(st.session_state.response_text, GOOGLE_API_KEY, GOOGLE_CSE_ID)
            st.write("Fact-Check Summary:")
            st.write(fact_check_summary)

if __name__ == "__main__":
    main()
