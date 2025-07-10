import os
from pinecone import Pinecone
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
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

# define pinecone database index
index_name = "comsis"
index = pc.Index(index_name)

# Store memory for each session
user_memories = {}

def pinecone_search(query):
    try:
        # generates embeddings for query
        embedding = embeddings.embed_query(query)
        
        # Perform similarity search
        result = index.query(vector=embedding, top_k=5, include_metadata=True)
        
        # Extract relevant information from the result
        matches = [match['metadata']['text'] for match in result['matches']]
        
        if not matches:
            return "I couldn't find specific information about that in our database. Please contact the university directly for accurate information."
        
        return "\n".join(matches)
    except Exception as e:
        return "I'm having trouble accessing the database right now. Please try again later."

def openai_conversation_responder(query):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Unilorin Student Support. Provide helpful, friendly responses about university matters. Keep responses concise and direct."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return "I'm having trouble processing your request right now. Please try again."

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

# CLEAN TEMPLATE - No exposed reasoning
template = """You are Unilorin Student Support, developed by Abdurrahman Abdulsalam.

You help students with academic, campus, and student support matters including:
- Course registration, academic schedules, grades, school fees
- Campus facilities, services, and policies  
- Student accommodation, financial aid, counseling
- University procedures and requirements
- Campus life and student activities

For questions outside this scope, respond: "I can only help with academic, campus, and student support matters. Is there something university-related I can assist you with instead?"

Available tools:
{tools}

Conversation history:
{history}

Student question: {input}

Provide a helpful, direct response to the student's question. Use the available tools when needed to get accurate information.

{agent_scratchpad}"""

# FIXED: Clean prompt template without "Thought:" injection
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\n"
            thoughts += f"Action Input: {action.tool_input}\n"
            thoughts += f"Observation: {observation}\n"
        
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

# ROBUST OUTPUT PARSER
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Always prioritize Final Answer
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            clean_answer = self.clean_response(final_answer)
            return AgentFinish(
                return_values={"output": clean_answer},
                log=llm_output,
            )
        
        # Look for action pattern
        action_match = re.search(r"Action:\s*(.*?)(?=\n|$)", llm_output, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.*?)(?=\n|$)", llm_output, re.DOTALL)
        
        if action_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip() if input_match else ""
            
            # Validate tool name
            valid_tools = ["OpenAI Conversation Responder", "Pinecone Vector Search"]
            if action in valid_tools:
                return AgentAction(tool=action, tool_input=action_input, log=llm_output)
        
        # If no clear action, provide direct response
        clean_response = self.clean_response(llm_output)
        if len(clean_response) < 10:  # Too short, provide default
            clean_response = "I'm here to help with your university-related questions. How can I assist you?"
        
        return AgentFinish(
            return_values={"output": clean_response},
            log=llm_output,
        )
    
    def clean_response(self, text):
        """Remove internal reasoning from response"""
        lines = text.split('\n')
        clean_lines = []
        
        skip_keywords = [
            'Thought:', 'Action:', 'Action Input:', 'Observation:', 
            'I need to', 'First,', 'Let me think', 'I should',
            'The question is', 'This is about'
        ]
        
        for line in lines:
            line = line.strip()
            if line and not any(line.startswith(keyword) for keyword in skip_keywords):
                clean_lines.append(line)
        
        result = ' '.join(clean_lines).strip()
        return result

output_parser = CustomOutputParser()

llm = OpenAI(temperature=0.5)  # Lower temperature for more consistent responses
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
    
    if not user_query.strip():
        return jsonify({"error": "Please provide a message"}), 400
    
    user_memory = get_user_memory(user_id)
    
    # Create a temporary AgentExecutor with the user's memory
    user_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=False,  # CRITICAL: Must be False for clean responses
        memory=user_memory,
        max_iterations=3  # Limit iterations to prevent loops
    )
    
    try:
        output = await user_agent_executor.arun({"input": user_query})
        return jsonify({"output": output})
    except Exception as e:
        print(f"Error in chat route: {e}")
        return jsonify({"output": "I'm having trouble processing your request right now. Please try again or contact university support directly."}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)