import pandas as pd
import asyncio
import openai
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import streamlit as st

# Set up OpenAI API key (if using OpenAI)
openai.api_key = "your-api-key-here"  # Replace with your actual OpenAI key

# Function to load a dataset
def load_dataset(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

# Function to generate search URLs
def generate_search_urls(entities, query_template):
    base_url = "https://www.google.com/search?q="
    search_urls = []
    for entity in entities:
        query = query_template.replace("{entity}", str(entity))
        search_urls.append(base_url + query.replace(" ", "+"))
    return search_urls

# Async function to perform web search and parse results
async def search_and_parse_async(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        await page.goto(url)
        content = await page.content()
        await browser.close()
    
    soup = BeautifulSoup(content, 'html.parser')
    results = []
    for result in soup.select("h3"):
        title = result.get_text()
        link = result.find_parent("a")['href']
        results.append({"title": title, "link": link})
    return results

# Function to parse the search results with an LLM (using OpenAI as an example)
def parse_with_llm(search_results, entity):
    prompt = f"Summarize the following search results for the entity '{entity}':\n\n"
    for result in search_results:
        prompt += f"Title: {result['title']}\nLink: {result['link']}\n\n"

    # Request to OpenAI GPT-3/4 for summarization
    response = openai.Completion.create(
        model="gpt-4",  # or "gpt-3.5-turbo" depending on the model you want to use
        prompt=prompt,
        max_tokens=200,  # You can adjust the token length based on the response size you expect
        temperature=0.5  # Control creativity, lower means more deterministic
    )
    
    summary = response.choices[0].text.strip()
    return {"entity": entity, "summary": summary}

# Async function to process entities
async def process_entities(entities, search_urls):
    structured_results = []
    for entity, url in zip(entities, search_urls):
        st.write(f"Searching for {entity}...")
        search_results = await search_and_parse_async(url)
        st.write(f"Parsing results with LLM for {entity}...")
        structured_result = parse_with_llm(search_results, entity)
        structured_results.append({"entity": entity, "result": structured_result})
    return structured_results

# Main function for processing the dataset
def process_file(file, query_template):
    # Load the dataset
    df = load_dataset(file)
    
    # Ask user to choose a column for entities
    column = st.selectbox(f"Available columns: {list(df.columns)}", list(df.columns))
    entities = df[column].dropna().tolist()
    
    # Generate search URLs
    search_urls = generate_search_urls(entities, query_template)
    
    # Process entities using asyncio.run
    structured_results = asyncio.run(process_entities(entities, search_urls))
    
    # Display results in a table
    results_df = pd.DataFrame(structured_results)
    st.write("Structured Results:", results_df)

    # Optionally, save the results to a CSV file and allow the user to download it
    output_file = "structured_output.csv"
    results_df.to_csv(output_file, index=False)
    st.download_button(
        label="Download Results",
        data=results_df.to_csv(index=False),
        file_name=output_file,
        mime="text/csv"
    )

# Streamlit UI
def main():
    st.title("Web Search and Summarization with LLM")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        query_template = st.text_input("Enter the query template (use '{entity}' for placeholder):")

        if query_template:
            st.write("Processing...")
            process_file(uploaded_file, query_template)
        else:
            st.write("Please enter a query template.")

if __name__ == "__main__":
    main()
