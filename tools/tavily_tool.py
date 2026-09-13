from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def tavily_search(query):
    response = tavily_client.search(query, max_results=5)
    results = []

    for i, r in enumerate(response["results"], 1):
        title = r.get("title", "unknown")
        url = r.get("url", "unknown")
        snippet = r.get("content", "unknown")
        if(len(snippet) > 300):
            snippet = snippet[:300].rsplit(' ', 1)[0] + "..."

        results.append(f"{i}. **{title}**\n URL: {url}\nSnippet: {snippet}\n")
    return "\n\n".join(results)
