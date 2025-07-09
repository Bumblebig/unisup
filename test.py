import os
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain_openai import OpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import openai
import asyncio

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define Pinecone database index
index_name = "comsis"
index = pc.Index(index_name)

# Store memory for each session
user_memories = {}

def pinecone_search(query):
    """Search the Pinecone vector database with the query."""
    # Generate embeddings for query
    embedding = embeddings.embed_query(query)
    
    # Perform similarity search
    result = index.query(vector=embedding, top_k=5, include_metadata=True)
    
    # Extract relevant information from the result
    matches = [match['metadata']['text'] for match in result['matches']]
    return "\n".join(matches)

def openai_conversation_responder(query):
    """Handle general conversation using GPT-3.5."""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

# Define the tools
tools = [
    Tool(
        name="OpenAI Conversation Responder",
        func=openai_conversation_responder,
        description="useful for responding to general conversation in a friendly manner"
    ),
    Tool(
        name="Pinecone Vector Search",
        func=pinecone_search,
        description="useful for retrieving relevant information from the Pinecone vector database"
    )
]

# Set up the base template
template = """Answer the following questions with as much detail as possible, providing a step-by-step guide whenever appropriate. If the question asked is not related to academic, campus, or student support issues, respond by saying that it isn't within the scope of Unilorin Student Support and that you cannot help with that query. Remember: your name is Unilorin Student Support, developed by Abdurrahman Abdulsalam, a 400L student in the Department of Information Technology, 2024 set. Use the following tools:

{tools}

Use the following format:

Question: the input question you must answer  
Thought: your internal thought process on how to answer  
Action: the action to take, which should be one of [{tool_names}]  
Action Input: the input to the action  
Observation: the result of the action  
... (this Thought/Action/Action Input/Observation can repeat N times)  
Thought: I now know the final answer  
Final Answer: the final answer to the original input question, with complete explanations and detailed steps

Begin! Remember to answer as a helpful student support assistant. If the question asked is not related to academic or campus issues, respond by saying that it isn't within the scope of Unilorin Student Support and that you cannot help with that query.

Answer the following questions with as much detail as possible. Remember: your name is Unilorin Student Support, developed by Abdurrahman Abdulsalam, a 400L student in the Department of Information Technology, 2024 set. Provide a step-by-step guide and practical scenarios or examples when appropriate. Ensure your final answer is comprehensive, with complete sentences and all necessary steps, so that even without access to your thought process, the explanation is clear and detailed.

Use the following tools:

{tools}

Use the following format:

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}

Begin! Remember to answer as a helpful student support assistant. If the question asked is not related to academic or campus issues, respond by saying that it isn't within the scope of Unilorin Student Support and that you cannot help with that query.

Ensure your Final Answer includes all relevant steps from the observations, providing a complete summary with clear explanations. Make sure your sentences are always complete and coherent. If the question asked is not related to academic or campus issues, simply state that it isn't within the scope of Unilorin Student Support and you cannot help with it.

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        history = kwargs.pop("history", "")
        kwargs["history"] = history
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # If no match, return an AgentFinish with the current thought
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

llm = OpenAI(temperature=0.9)
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

def get_user_memory(user_id):
    """Retrieve or initialize memory for a user."""
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=2)
    return user_memories[user_id]

async def process_query(user_id, query):
    """Process a user query with the agent."""
    user_memory = get_user_memory(user_id)
    
    # Create a temporary AgentExecutor with the user's memory
    user_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=user_memory
    )
    
    try:
        output = await user_agent_executor.arun({"input": query})
        return {"output": output}
    except Exception as e:
        print(f"Error processing query: {e}")
        return {"error": "An error occurred while processing your request."}

# Simple command-line interface for testing
async def main():
    print("Welcome to the Unilorin Student Support Chatbot!")
    user_id = "test_user"  # For testing purposes
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        result = await process_query(user_id, user_input)
        if "error" in result:
            print(f"\nComsis: {result['error']}")
        else:
            print(f"\nComsis: {result['output']}")

if __name__ == "__main__":
    asyncio.run(main())