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
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are UNILORIN Student Support, an IT support assistant for University of Ilorin students. Provide helpful, professional responses about university matters."},
                {"role": "user", "content": query}
            ],
            max_tokens=300,  # Limit response length
            temperature=0.7  # Lower temperature for more consistent responses
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "I'm having trouble accessing my knowledge base right now. Please try again shortly."

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

# Improve the template to be more direct and less prone to loops
template = """I am UNILORIN Student Support, developed by Abdurrahman Abdulsalam from IT Department.

IDENTITY: UNILORIN Student Support - IT and student guidance assistant

RULES:
1. For capability/identity questions: Answer directly without tools
2. For specific questions: Use ONE appropriate tool
3. Always start Final Answer with "I am UNILORIN Student Support"
4. Keep responses concise and helpful

Available tools: {tools}
History: {history}
Question: {input}

Think step by step:
Thought: [Quick analysis - is this capability question or needs tool?]
Action: [ONE tool name from: {tool_names}] OR [Skip if capability question]
Action Input: [Specific query]
Observation: [Tool result]
Final Answer: I am UNILORIN Student Support. [Your helpful response]

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

memory = ConversationBufferWindowMemory(k=2)

# Add timeout and iteration limits to prevent infinite loops
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    max_iterations=3,  # Limit to 3 iterations
    max_execution_time=25,  # 25 second timeout
    early_stopping_method="generate"
)

def get_user_memory(user_id):
    """Retrieve or initialize memory for a user."""
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=2)
    return user_memories[user_id]

# Remove async - Flask doesn't handle async well without additional setup
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get("message", "")
    user_id = data.get("user_id")
    
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    if not user_query or len(user_query.strip()) == 0:
        return jsonify({"error": "Message cannot be empty"}), 400
    
    # Handle capability questions directly to avoid agent loops
    capability_keywords = [
        "what can you do", "who are you", "what are you", 
        "tell me about yourself", "your capabilities", "your name",
        "what is your name", "introduce yourself"
    ]
    
    if any(keyword in user_query.lower() for keyword in capability_keywords):
        response = "I am UNILORIN Student Support. I can help you with IT issues, academic procedures, campus information, student services, and general university guidance. What specific area would you like assistance with?"
        return jsonify({"output": response})
    
    user_memory = get_user_memory(user_id)
    
    # Create agent executor with strict limits
    user_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=user_memory,
        max_iterations=3,
        max_execution_time=20,  # 20 seconds max
        early_stopping_method="generate"
    )
    
    try:
        # Use synchronous run instead of arun
        output = user_agent_executor.run(input=user_query)
        
        # Validate output before returning
        if not output or output.strip() == "":
            raise ValueError("Empty response generated")
        
        if "Agent stopped due to iteration limit" in output:
            raise TimeoutError("Agent exceeded iteration limit")
        
        if "Agent stopped due to time limit" in output:
            raise TimeoutError("Agent exceeded time limit")
        
        return jsonify({"output": output})
        
    except TimeoutError as e:
        print(f"Timeout error: {e}")
        # Don't save to memory, throw proper error
        return jsonify({
            "error": "Request timeout - please try asking a more specific question",
            "error_type": "timeout"
        }), 408
        
    except ValueError as e:
        print(f"Value error: {e}")
        return jsonify({
            "error": "Unable to generate a valid response - please rephrase your question",
            "error_type": "invalid_response"
        }), 422
        
    except Exception as e:
        print(f"Unexpected error in chat route: {e}")
        # For any other errors, don't save to memory
        return jsonify({
            "error": "I'm experiencing technical difficulties. Please try again in a moment.",
            "error_type": "system_error"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)