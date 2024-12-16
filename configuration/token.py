""" Instructions on how to savely handle secrets.

Instructions:
- Read https://dev.to/emma_donery/python-dotenv-keep-your-secrets-safe-4ocn to understand how python-dotenv works.
- Create a .env file from the .env.template file.
- Make sure to never add the .env file to version control. It is also in the .gitignore.
"""

from dotenv import load_dotenv
import os

load_dotenv()


# Note: Do not hardcode these values.
IBM_TOKEN = os.environ['IBM_TOKEN']
HUB = os.environ['HUB']
GROUP = os.environ['GROUP']
PROJECT = os.environ['PROJECT']
DEVICE_NAME = os.environ['DEVICE_NAME']
