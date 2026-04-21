from langchain_core.tools import Tool
from pathlib import Path
import requests
from typing import Optional
import json
from rich.console import Console
from rich.markup import escape
from setup import workspace

_console = Console()
_CORAL   = "#C8603A"
_BULLET  = f"[{_CORAL}]⬤[/{_CORAL}]"
_NEST    = "[dim]  ⎿[/dim]"


def read_file(file_path: str) -> str:
    """
    Read and return the contents of a file.
    
    Args:
        file_path: Path to the file (relative to workspace)
    
    Returns:
        The file contents as a string
    """
    full_path = Path(workspace) / file_path
    
    if not full_path.exists():
        _console.print()
        _console.print(f"{_BULLET} [bold red]Error[/bold red]")
        _console.print(f"{_NEST} File not found - {file_path}")
        _console.print()
        return f"Error: File not found - {file_path}"
    
    if not full_path.is_file():
        _console.print()
        _console.print(f"{_BULLET} [bold red]Error[/bold red]")
        _console.print(f"{_NEST} {file_path} is not a file")
        _console.print()
        return f"Error: {file_path} is not a file"
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        _console.print()
        _console.print(f"{_BULLET} [bold cyan]Reading File[/bold cyan]")
        _console.print(f"{_NEST} [dim]{file_path}[/dim]")
        _console.print(f"{_NEST} [green]✓ Done[/green]")
        _console.print()
        
        return content
    
    except Exception as e:
        _console.print()
        _console.print(f"{_BULLET} [bold red]Error[/bold red]")
        _console.print(f"{_NEST} {str(e)}")
        _console.print()
        return f"Error reading file: {str(e)}"
 
 
read_file_tool = Tool(
    name="read_file",
    func=read_file,
    description="Read the contents of a file. Input: file path (relative to workspace). Returns: file contents as text."
)




def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        Search results as formatted text
    """
    try:
        from ddgs import DDGS
        
        _console.print()
        _console.print(f"{_BULLET} [bold cyan]Searching Web[/bold cyan]")
        _console.print(f"{_NEST} [dim]{query}[/dim]")
        
        results = []
        with DDGS() as ddgs_search:
            search_results = ddgs_search.text(query, max_results=num_results)
            
            if not search_results:
                _console.print(f"{_NEST} [yellow]No results found[/yellow]")
                _console.print()
                return "No results found for the query."
            
            _console.print(f"{_NEST} [green]✓ Found {len(search_results)} results[/green]")
            
            for i, result in enumerate(search_results, 1):
                results.append(f"{i}. {result['title']}\n   {result['body']}\n   URL: {result['href']}\n")
        
        _console.print()
        return "\n".join(results)
    
    except ImportError:
        _console.print()
        _console.print(f"{_BULLET} [bold red]Error[/bold red]")
        _console.print(f"{_NEST} ddgs package not installed")
        _console.print()
        return "Error: ddgs package not installed. Install it with: pip install ddgs"
    except Exception as e:
        _console.print()
        _console.print(f"{_BULLET} [bold red]Error[/bold red]")
        _console.print(f"{_NEST} {str(e)}")
        _console.print()
        return f"Error during search: {str(e)}"
 
 
web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Search the web for information. Input: search query (and optionally num_results). Returns: list of search results with titles, snippets, and URLs."
)



def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file (relative to workspace)
        content: Content to write to the file
    
    Returns:
        Success or error message
    """
    try:
        full_path = Path(workspace) / file_path
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        _console.print()
        _console.print(f"{_BULLET} [bold cyan]Writing File[/bold cyan]")
        _console.print(f"{_NEST} [dim]{file_path}[/dim]")
        _console.print(f"{_NEST} [green]✓ Success[/green]")
        _console.print()
        
        return f"✓ File written successfully: {file_path}"
    
    except Exception as e:
        _console.print()
        _console.print(f"{_BULLET} [bold red]Error[/bold red]")
        _console.print(f"{_NEST} {str(e)}")
        _console.print()
        return f"Error writing file: {str(e)}"
 
 
write_file_tool = Tool(
    name="write_file",
    func=write_file,
    description="Write content to a file. Input: JSON with 'file_path' (relative to workspace) and 'content' (text to write). Creates parent directories if needed. Returns: success or error message."
)
 
 
 
def list_directory(dir_path: str = "") -> str:
    """
    List contents of a directory.
    
    Args:
        dir_path: Path to directory (relative to workspace). Empty string = root workspace
    
    Returns:
        Formatted list of files and directories
    """
    try:
        if dir_path == "":
            full_path = Path(workspace)
        else:
            full_path = Path(workspace) / dir_path
        
        if not full_path.exists():
            _console.print()
            _console.print(f"{_BULLET} [bold red]Error[/bold red]")
            _console.print(f"{_NEST} Directory not found - {dir_path}")
            _console.print()
            return f"Error: Directory not found - {dir_path}"
        
        if not full_path.is_dir():
            _console.print()
            _console.print(f"{_BULLET} [bold red]Error[/bold red]")
            _console.print(f"{_NEST} {dir_path} is not a directory")
            _console.print()
            return f"Error: {dir_path} is not a directory"
        
        items = sorted(full_path.iterdir())
        
        _console.print()
        _console.print(f"{_BULLET} [bold cyan]Listing Directory[/bold cyan]")
        _console.print(f"{_NEST} [dim]{dir_path if dir_path else 'workspace root'}[/dim]")
        
        if not items:
            _console.print(f"{_NEST} [dim]Empty directory[/dim]")
            _console.print()
            return f"Directory is empty: {dir_path}"
        
        results = []
        for item in items:
            if item.is_dir():
                _console.print(f"{_NEST}  📁 [dim]{item.name}/[/dim]")
                results.append(f"📁 {item.name}/")
            else:
                _console.print(f"{_NEST}  📄 [dim]{item.name}[/dim]")
                results.append(f"📄 {item.name}")
        
        _console.print()
        
        return "\n".join(results)
    
    except Exception as e:
        _console.print()
        _console.print(f"{_BULLET} [bold red]Error[/bold red]")
        _console.print(f"{_NEST} {str(e)}")
        _console.print()
        return f"Error listing directory: {str(e)}"
 
 
list_directory_tool = Tool(
    name="list_directory",
    func=list_directory,
    description="List files and directories in a folder. Input: directory path (relative to workspace, empty string for root). Returns: formatted list of contents."
)

from str_replace_tool       import  str_replace_tool
from persistant_shell_tool  import  bash_tool
from view_image             import view_image_tool


base_tools = [web_search, str_replace_tool, list_directory, bash_tool, read_file, write_file, view_image_tool]