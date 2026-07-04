
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os

load_dotenv()
# Just for checking LLM connectivity

def main():

    print("Hello from rag_pipeline!")
    
    # Testing LLM
    llm_openai = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    llm_anthropic = ChatAnthropic(
        model="claude-haiku-4-5",
        temperature=0
    )
    # Test
    response_openai = llm_openai.invoke("Hello, how are you?")
    response_anthropic = llm_anthropic.invoke("Hello, how are you?")
    print(response_openai.content)
    print(response_anthropic.content)


if __name__ == "__main__":
    main()
