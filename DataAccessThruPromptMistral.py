import sqlite3
import json
import boto3
import os
from dotenv import load_dotenv

# === LOAD ENV VARS ===
load_dotenv()

# AWS credentials & region must be in .env
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
    raise ValueError("AWS credentials not found in .env")

# Bedrock Mistral Model ID
MODEL_ID = "mistral.mixtral-8x7b-instruct-v0:1"

# === CONFIG ===
DB_FILE = "nbfc_core_banking.db"
DATA_DICTIONARY_FILE = "nbfc_data_dictionary.json"

# Create Bedrock Runtime client
bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

# === LOAD DATA DICTIONARY ===
try:
    with open(DATA_DICTIONARY_FILE, "r", encoding="utf-8") as f:
        DATA_DICTIONARY = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"{DATA_DICTIONARY_FILE} not found in project folder.")

# === FUNCTIONS ===
def call_mistral(prompt: str) -> str:
    """Send prompt to Mistral on Bedrock and return text output."""
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 1024,
            "temperature": 0.1,
            "topP": 0.9
        }
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result.get("results", [{}])[0].get("outputText", "").strip()

def prompt_to_sql(user_prompt):
    """Convert user request to SQL query using Mistral with schema context."""
    base_prompt = f"""
You are a SQL generation engine for a reporting tool.

Task: Convert the following request into a valid **SQLite** SQL query.
The database schema is provided below in JSON format.
Use this schema to ensure correct joins, column usage, and filtering.
Do not invent tables or columns that are not in the schema.

Database Schema (JSON):
{json.dumps(DATA_DICTIONARY, indent=2)}

Output format must be ONLY JSON:
{{
  "sql": "<SQL query here>"
}}

User request: {user_prompt}
"""
    print("\n[DEBUG] Prompt sent to Mistral for SQL generation:\n", base_prompt)

    raw_text = call_mistral(base_prompt)

    cleaned_text = (
        raw_text.replace("```json", "")
                .replace("```sql", "")
                .replace("```", "")
                .strip()
    )

    try:
        parsed = json.loads(cleaned_text)
        sql_query = parsed.get("sql", "").strip()
    except json.JSONDecodeError:
        sql_query = cleaned_text

    print("\n[DEBUG] SQL Query Generated:\n", sql_query)
    return sql_query


def run_sql_query(sql_query):
    """Run SQL query on SQLite DB."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    data = cursor.fetchall()
    conn.close()
    print("\n[DEBUG] Data returned from database:\n", data)
    return data


def generate_final_answer(user_prompt, query_result):
    """Generate final natural language answer using Mistral."""
    if len(query_result) == 1 and len(query_result[0]) == 1:
        result_str = str(query_result[0][0])
    else:
        result_str = json.dumps(query_result)

    answer_prompt = f"""
You are a helpful NBFC banking assistant working with a *synthetic demo dataset*.
This is fictional and safe to share.

The user asked: "{user_prompt}"
The database query returned: {result_str}

Please give a clear, concise, and helpful answer in natural language.
"""
    print("\n[DEBUG] Prompt sent to Mistral for final answer:\n", answer_prompt)

    return call_mistral(answer_prompt)


# === MAIN LOOP ===
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a question about the NBFC data (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            sql = prompt_to_sql(user_input)
            data = run_sql_query(sql)
            answer = generate_final_answer(user_input, data)
            print("\n[ANSWER]:", answer)
        except Exception as e:
            print("Error:", e)
