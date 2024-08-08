# import os
#
# the_key=os.getenv("OPENAI_API_KEY")
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

the_key = os.getenv('OPENAI_API_KEY')
da_pass = os.getenv('DA_PASSWORD')
