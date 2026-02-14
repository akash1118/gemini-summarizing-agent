import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.schemas import SummarySchema

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0.1
    )


def get_structured_llm():
    llm = get_llm()
    return llm.with_structured_output(SummarySchema)