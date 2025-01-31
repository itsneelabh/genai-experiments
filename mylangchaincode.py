# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM

# template = """You are a travel planner. Plan a travel for the location {question}.
# """
#
# prompt = ChatPromptTemplate.from_template(template)
#
# model = OllamaLLM(model="llama3.1:8b")
#
# chain = prompt | model
#
# print(chain.invoke({"question": "Paris"}))

from langchain_ollama import OllamaLLM

def main():
    # Initialize the Ollama LLM
    model_name = "llama3.2"  # Replace with the locally available model name

    try:
        # Create an Ollama LLM instance
        llm = OllamaLLM(model=model_name)

        # Prompt for the LLaMA model
        prompt = "Explain the significance of LangChain in building AI applications."

        # Get the response from the LLM
        response = llm.invoke(prompt)

        # Print the response
        print("Response from LLaMA 3.2:")
        print(response)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()