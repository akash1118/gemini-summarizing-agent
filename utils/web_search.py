from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

def search_web(query: str) -> str:
    return search_tool.run(query)

