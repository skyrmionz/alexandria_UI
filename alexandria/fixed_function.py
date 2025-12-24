@lru_cache(maxsize=1000)
def _cached_tool_execution(tool_name: str, tool_args_str: str):
    """Cache tool execution results to avoid redundant calls."""
    try:
        tools_list = get_tools(current_agent_type)
        tools_dict = {t.name: t for t in tools_list}
        
        if tool_name not in tools_dict:
            return f"<tool_result>Error: Tool '{tool_name}' not found</tool_result>"
        
        tool = tools_dict[tool_name]
        tool_args = json.loads(tool_args_str)
        
        # Use the preferred invoke method if available
        if hasattr(tool, "invoke"):
            # New method: use invoke
            if "query" in tool_args and len(tool_args) == 1:
                result = tool.invoke(tool_args["query"])
            else:
                # For tools with multiple parameters or different parameter names
                result = tool.invoke(tool_args)
        else:
            # Fallback to run method if available
            if hasattr(tool, "run"):
                if "query" in tool_args and len(tool_args) == 1:
                    result = tool.run(tool_args["query"])
                else:
                    result = tool.run(**tool_args)
            else:
                # Last resort - direct calling (deprecated)
                result = tool(**tool_args)
            
        return f"<tool_result>{result}</tool_result>"
    except Exception as e:
        logger.exception("Error in cached tool execution", error=str(e))
        return f"<tool_result>Error: {str(e)}</tool_result>"
