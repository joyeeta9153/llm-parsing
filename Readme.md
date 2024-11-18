# Web Search and Summarization with LLM (Streamlit App)

## Overview

This project is a Streamlit application that allows users to upload a dataset (CSV or Excel file), define a search query template, and then perform web searches for each entity in a chosen column of the dataset. The search results are parsed and summarized using OpenAI's language models (GPT-3 or GPT-4). The structured results are then displayed in the app and can be downloaded as a CSV file.

## Features

- **File Upload**: Users can upload a CSV or Excel file containing a list of entities.
- **Query Template**: Users can define a query template with a placeholder (`{entity}`) to search for information related to the entities.
- **Web Search**: For each entity in the chosen column, the app generates search URLs using the query template and fetches results from Google.
- **Result Summarization**: The search results are summarized using OpenAI's language models (GPT-3 or GPT-4).
- **Download Results**: The processed and summarized results are displayed in a table, and users can download the output as a CSV file.

## Requirements

- **Python 3.7+**
- Required Python packages:
  - `streamlit`
  - `openai`
  - `playwright`
  - `beautifulsoup4`
  - `pandas`
  - 
Loom video link:- https://www.loom.com/share/554081a3590144d2a51ce215d7a17346?sid=99bdf823-f3bd-4892-bfd7-21d22e0e37be

You can install the necessary dependencies using:

```bash
pip install streamlit openai playwright beautifulsoup4 pandas

