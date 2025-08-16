import sqlite3
import json
import google.generativeai as genai
import os

# === CONFIG ===
DB_FILE = "nbfc_core_banking.db"
DATA_DICTIONARY_FILE = "nbfc_data_dictionary.json"  # must be in project folder
MODEL_ID = "gemini-2.5-pro"

# Gemini API key is taken from Dockerfile ENV
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment or Dockerfile.")
genai.configure(api_key=API_KEY)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# === LOAD DATA DICTIONARY ===
try:
    with open(DATA_DICTIONARY_FILE, "r", encoding="utf-8") as f:
        DATA_DICTIONARY = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"{DATA_DICTIONARY_FILE} not found in project folder.")

# === FUNCTIONS ===
def prompt_to_sql(user_prompt):
    """Convert user request to SQL query using Gemini with schema context."""
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

    print("\n[DEBUG] Prompt sent to Gemini for SQL generation:\n", base_prompt)

    model = genai.GenerativeModel(MODEL_ID)
    response = model.generate_content(base_prompt, safety_settings=SAFETY_SETTINGS)

    if not response.candidates or not response.candidates[0].content.parts:
        print("[WARN] Gemini blocked SQL generation — retrying...")
        retry_prompt = base_prompt.replace("largest loan", "highest loan amount")
        response = model.generate_content(retry_prompt, safety_settings=SAFETY_SETTINGS)

    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("No valid SQL from Gemini after retry.")

    raw_text = response.candidates[0].content.parts[0].text
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
    """Generate final natural language answer using Gemini."""
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

    print("\n[DEBUG] Prompt sent to Gemini for final answer:\n", answer_prompt)

    model = genai.GenerativeModel(MODEL_ID)
    response = model.generate_content(answer_prompt, safety_settings=SAFETY_SETTINGS)

    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("No valid answer from Gemini.")

    return response.candidates[0].content.parts[0].text


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
