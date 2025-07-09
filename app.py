import os
from pinecone import Pinecone
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
# For handling conversation history
from langchain.memory import ConversationBufferWindowMemory
# from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import openai
import asyncio

# Load api key
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialise pinecone and flask
pc = Pinecone(api_key=PINECONE_API_KEY)
app = Flask(__name__)
CORS(app)

# define embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)


# from langchain_community.vectorstores import Pinecone 
# define pinecone database index
index_name = "comsis"
index = pc.Index(index_name)

# Store memory for each session
user_memories = {}
user_memory = ''

# Define the tool
   

def pinecone_search(query):
    # generates embeddings for query
    embedding = embeddings.embed_query(query)
    
    # Perform similarity search
    result = index.query(vector=embedding, top_k=5, include_metadata=True)
    
    # Extract relevant information from the result
    matches = [match['metadata']['text'] for match in result['matches']]
    return "\n".join(matches)


# Define the function to handle general conversation using GPT-4
def openai_conversation_responder(query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a friendly assistant."},
                  {"role": "user", "content": query}]
    )

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
template = """I am Unilorin Student Support, developed by Abdurrahman Abdulsalam, a 400L student in the Department of Information Technology, 2024 set. 

SCOPE CHECK: I only assist with academic, campus, and student support issues including:
- Course registration, academic schedules, grades
- Campus facilities, services, and policies  
- Student accommodation, financial aid, counseling
- University procedures and requirements
- Campus life and student activities

If a question is outside this scope, I must respond: "I'm sorry, but that question isn't within my scope as Unilorin Student Support. I can only help with academic, campus, and student support matters. Is there something university-related I can assist you with instead?"

Available tools:
{tools}

Previous conversation history:
{history}

Question: {input}

I will follow this format:
Thought: First, I'll check if this question is within my scope (academic/campus/student support). If yes, I'll determine the best approach to help.
Action: {tool_names}
Action Input: the specific input for the tool
Observation: the result I receive
... (I may repeat this process as needed)
Thought: Now I have enough information to provide a comprehensive, helpful response in first person
Final Answer: My complete response speaking directly to the student, incorporating insights from my observations into a clear, detailed answer

{agent_scratchpad}

Remember: I speak in first person ("I can help you...", "I recommend...", "In my experience..."), stay within scope, and ensure my Final Answer is more comprehensive than my final Thought by incorporating all relevant details from my observations."""

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

memory = ConversationBufferWindowMemory(k=2)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

def get_user_memory(user_id):
    """Retrieve or initialize memory for a user."""
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=2)
    return user_memories[user_id]

async def async_run_llm_agent(query):
    try:
        output = await agent_executor.arun({"input": query})
        return {"output": output}
    except Exception as e:
        print(f"Error in async_run_llm_agent: {e}")
        return {"output": "An error occurred while processing your request."}


@app.route('/')
def home():
    return "Hi! I'm Unilorin Student Support. How can I help you today?"

@app.route('/api/chat', methods=['POST'])
async def chat():
    data = request.json
    user_query = data.get("message", "")
    user_id = data.get("user_id")
    
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    user_memory = get_user_memory(user_id)  # Get or initialize memory for the user
    
    # Create a temporary AgentExecutor with the user's memory
    user_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=user_memory
    )
    
    try:
        output = await user_agent_executor.arun({"input": user_query})
        return jsonify({"output": output})
    except Exception as e:
        print(f"Error in chat route: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)