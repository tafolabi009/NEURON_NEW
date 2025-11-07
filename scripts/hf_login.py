#!/usr/bin/env python3
"""
Login to HuggingFace and save token
"""
from huggingface_hub import login

TOKEN = ""

login(token=TOKEN)
print("✅ HuggingFace login successful!")
print("Token saved for future use")
