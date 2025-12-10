from langchain_core.prompts import ChatPromptTemplate
import os as os

def get_prompt_txt(filename):
    base_dir = os.path.dirname(__file__)  
    filepath = os.path.join(base_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
    
def get_prompt(filename):
    prompt_text = get_prompt_txt(filename)
    return ChatPromptTemplate.from_template(prompt_text)