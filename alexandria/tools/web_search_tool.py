# tools/web_search_tool.py

from duckduckgo_search import DDGS

def web_search_tool(query: str) -> str:
    """
    Perform a web search using DuckDuckGo via the DDGS class.

    Args:
        query (str): The search query.

    Returns:
        str: A formatted string with the top search results.
    """
    try:
        # Use the DDGS context manager as shown in the Medium article.
        with DDGS() as ddgs:
            # Perform a text search; you can customize parameters as needed.
            results = ddgs.text(query, max_results=3)
        
        if results:
            # Format each result with its title and URL.
            formatted_results = "\n".join(
                f"{result.get('title', 'No title')}: {result.get('href', 'No URL')}"
                for result in results
            )
            return formatted_results
        else:
            return "No results found."
    except Exception as e:
        return f"An error occurred while searching: {str(e)}"
