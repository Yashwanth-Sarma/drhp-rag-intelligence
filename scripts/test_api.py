python -c "
from dotenv import load_dotenv
import os
load_dotenv()
keys = ['GROQ_API_KEY', 'CEREBRAS_API_KEY', 'GEMINI_API_KEY']
for k in keys:
    val = os.getenv(k)
    print(f'{k}: {\"SET (\" + str(len(val)) + \" chars)\" if val else \"MISSING\"}')"