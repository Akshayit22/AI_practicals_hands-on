import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.messages import HumanMessage
from langchain_community.tools import tool
from rich import print

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
'''
search_tool = TavilySearchResults(max_result = 5)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into 5 clear bullet points

{news}
"""
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("Latest AI news of 2026")

result = chain.invoke({"news":news_result})

print(result)
'''
# ----------------------- other tool ---------------------

@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}

llm_with_tool = llm.bind_tools([get_text_length])

# input_tokens in llm_with_tool will be more in that the normal , because tool is going with prompt
# result = llm_with_tool.invoke("hello")

message = []
prompt = input("your input : ")
query = HumanMessage( "what is the lengeth of given test : "+ prompt)
message.append(query)

result = llm_with_tool.invoke(message)
message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
   

result = llm_with_tool.invoke(message)
print(result.content)

'''
Output: 
HumanMessage(content='what is the lengeth of given test : how are you?', additional_kwargs={}, response_metadata={})
AIMessage(
    content='',
    additional_kwargs={
        'tool_calls': [{'id': 'ky99k9n39', 'function': {'arguments': '{"text":"how are you?"}', 'name': 'get_text_length'}, 'type': 'function'}]
    },
    response_metadata={
        'token_usage': {
            'completion_tokens': 17,
            'prompt_tokens': 234,
            'total_tokens': 251,
            'completion_time': 0.052964214,
            'completion_tokens_details': None,
            'prompt_time': 0.012399385,
            'prompt_tokens_details': None,
            'queue_time': 0.056211414,
            'total_time': 0.065363599
        },
        'model_name': 'llama-3.3-70b-versatile',
        'system_fingerprint': 'fp_dae98b5ecb',
        'service_tier': 'on_demand',
        'finish_reason': 'tool_calls',
        'logprobs': None
    },
    id='run--204f7643-4fe5-4cb8-bbcf-2a88b04a88fc-0',
    tool_calls=[{'name': 'get_text_length', 'args': {'text': 'how are you?'}, 'id': 'ky99k9n39', 'type': 'tool_call'}],
    usage_metadata={'input_tokens': 234, 'output_tokens': 17, 'total_tokens': 251}
)
ToolMessage(content='12', name='get_text_length', tool_call_id='ky99k9n39')
The length of the given text "how are you?" is 12.

'''

