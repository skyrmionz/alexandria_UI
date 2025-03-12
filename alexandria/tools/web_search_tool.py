# tools/web_search_tool.py

import os
from typing import Optional
from duckduckgo_search import DDGS

def web_search_tool(query: str, num_results: int = 3) -> str:
    """
    Search the web for information using DuckDuckGo.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 3)
        
    Returns:
        A string containing the search results
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            
            if not results:
                return "No results found for your query."
            
            formatted_results = []
            for i, r in enumerate(results, 1):
                title = r.get('title', 'No title')
                body = r.get('body', 'No content')
                href = r.get('href', 'No link')
                formatted_results.append(f"{i}. {title}\n   {body}\n   Source: {href}\n")
            
            return "Web Search Results:\n\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"
