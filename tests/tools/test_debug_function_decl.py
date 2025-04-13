"""Debug script to examine the structure of function declarations."""

from src.cli_code.tools.test_runner import TestRunnerTool
import json

def main():
    """Print the structure of function declarations."""
    tool = TestRunnerTool()
    function_decl = tool.get_function_declaration()
    
    print("Function Declaration Properties:")
    print(f"Name: {function_decl.name}")
    print(f"Description: {function_decl.description}")
    
    print("\nParameters Type:", type(function_decl.parameters))
    print("Parameters Dir:", dir(function_decl.parameters))
    
    # Check the type_ value
    print("\nType_ value:", function_decl.parameters.type_)
    print("Type_ repr:", repr(function_decl.parameters.type_))
    print("Type_ type:", type(function_decl.parameters.type_))
    print("Type_ str:", str(function_decl.parameters.type_))
    
    # Check the properties attribute
    print("\nProperties type:", type(function_decl.parameters.properties))
    if hasattr(function_decl.parameters, 'properties'):
        print("Properties dir:", dir(function_decl.parameters.properties))
        print("Properties keys:", function_decl.parameters.properties.keys())
        
        # Iterate through property items
        for key, value in function_decl.parameters.properties.items():
            print(f"\nProperty '{key}':")
            print(f"  Value type: {type(value)}")
            print(f"  Value dir: {dir(value)}")
            print(f"  Value.type_: {value.type_}")
            print(f"  Value.type_ repr: {repr(value.type_)}")
            print(f"  Value.type_ type: {type(value.type_)}")
            print(f"  Value.description: {value.description}")
            
    # Try __repr__ of the entire object
    print("\nFunction declaration repr:", repr(function_decl))

if __name__ == "__main__":
    main() 