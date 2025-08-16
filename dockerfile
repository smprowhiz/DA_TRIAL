FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nbfc_core_banking.db .
COPY nbfc_data_dictionary.json .  
COPY DataAccessThruPromptMistral.py .

# Set Gemini API key directly here (replace with your real key)
# ENV GEMINI_API_KEY="AIzaSyC2T-p9BqQmptoN-PnnRulu-8rAJMTNaMQ"

CMD ["python", "DataAccessThruPromptMistral.py"]
