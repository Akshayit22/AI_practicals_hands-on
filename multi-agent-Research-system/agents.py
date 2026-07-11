from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY  = os.environ["AZURE_AI_API_KEY"]
BASE_URL = os.environ["AZURE_AI_BASE_URL"]
MODEL    = os.getenv("AZURE_AI_MODEL", "grok-4.3")

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.7,
    max_tokens=1000,
    timeout=60,
    default_headers={"api-key": API_KEY},  # Foundry accepts this; harmless otherwise
)

# creating Agents
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )
    
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
