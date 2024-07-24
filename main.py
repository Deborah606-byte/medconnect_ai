import csv
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import os
from googleapiclient.discovery import build

load_dotenv()

app = FastAPI()

# Configure CORS
def configure_cors(app: FastAPI):
    origins = [
        "http://localhost",  # Allow local frontend
        "http://localhost:3000", 
        "https://medconnect-knb2.onrender.com",
        "https://medconnect-gh.vercel.app" # Allow local frontend running on port 3000
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # Allow specific origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
        allow_headers=["*"],  # Allow all headers
    )

# Configure CORS
configure_cors(app)

MONGO_URI = os.getenv("ATLAS_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    chats_collection = db["chats"]
    messages_collection = db["messages"]
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise

def load_hospitals() -> List[Dict[str, str]]:
    hospitals = []
    try:
        with open('cleaned_hospitals.csv', 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                hospitals.append(row)
    except FileNotFoundError:
        print("Error: 'cleaned_hospitals.csv' file not found.")
    return hospitals

hospitals_data = load_hospitals()

def find_nearby_hospitals(location: str) -> List[str]:
    nearby = []
    for hospital in hospitals_data:
        if (location.lower() in hospital['Region'].lower() or 
            location.lower() in hospital['District'].lower() or 
            location.lower() in hospital['Town'].lower()):
            nearby.append(f"{hospital['FacilityName']} ({hospital['Type']}) in {hospital['Town']}, {hospital['District']}, {hospital['Region']}")
    return nearby

# Custom search wrapper
def google_search(query, api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id).execute()
    return res.get('items', [])  # Return an empty list if no items are found

def google_wrapper(input_text):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    # Perform UpToDate search
    uptodate_results = google_search(f"site:uptodate.com {input_text}", api_key, cse_id)
    
    # Perform Drugs.com search
    drugs_results = google_search(f"site:drugs.com {input_text}", api_key, cse_id)
    
    formatted_results = "UpToDate Results:\n" + format_search_results(uptodate_results)
    formatted_results += "\n\nDrugs.com Results:\n" + format_search_results(drugs_results)
    
    # Extract location from input
    location = input_text.split("Location:")[-1].split("\n")[0].strip() if "Location:" in input_text else ""
    
    if location:
        nearby_hospitals = find_nearby_hospitals(location)
        if nearby_hospitals:
            formatted_results += "\n\nNearby healthcare facilities for further diagnosis or treatment:\n" + "\n".join(nearby_hospitals[:5])  # Limit to top 5 for brevity
        else:
            formatted_results += "\n\nNo specific healthcare facilities found in the exact location, please consult with a local healthcare provider."
    
    # Add medication suggestions
    medication_results = google_search(f"site:drugs.com {input_text} medications", api_key, cse_id)
    formatted_results += "\n\nSuggested Medications:\n" + format_search_results(medication_results)
    
    return formatted_results

def format_search_results(results):
    formatted = ""
    for result in results[:3]:  # Limit to top 3 results
        title = result.get('title', 'No title')
        snippet = result.get('snippet', 'No snippet')
        link = result.get('link', 'No link')
        formatted += f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n\n"
    return formatted

def google_search(query, api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id).execute()
    return res.get('items', [])  # Return an empty list if no items are found

# Define tools
tools = [
    Tool(
        name="CombinedMedicalSearch",
        func=google_wrapper,
        description="performs a single search query on both UpToDate and Drugs.com for medical and pharmacological information"
    )
]

# Custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = self.tools[0].name  # Only use the name of the single tool
        return self.template.format(**kwargs)

# Custom output parser
outputs = []
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        # Split the llm_output into blocks based on "Thought:" and "Observation:"
        blocks = re.split(r"(Thought:|Observation:)", llm_output)
        
        for i in range(1, len(blocks), 2):
            if blocks[i] == "Thought:":
                thought = blocks[i] + blocks[i+1].split("Action:")[0].strip()
                outputs.append(thought)
            elif blocks[i] == "Observation:":
                observation = blocks[i] + blocks[i+1].strip()
                outputs.append(observation)

        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

# Set up the agent
template_with_history = """Answer the following questions as best you can, speaking as a compassionate medical professional. If you think there is need for diagnosis or treatment, provide further questions to be asked to the patient and recommend nearby healthcare facilities if available. You have access to the following tool:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, must be exactly "CombinedMedicalSearch"
Action Input: the input to the action (your search query)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, including nearby healthcare facility recommendations and suggested medications if available, and a list of questions to ask the patient for better diagnosis.

Begin! Remember to use only the exact tool name "CombinedMedicalSearch" for all searches. Do not try to use separate tools for UpToDate or Drugs.com.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

output_parser = CustomOutputParser()
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

# Pydantic models
class PatientInfo(BaseModel):
    name: str
    age: int
    gender: str
    location: str

class Question(BaseModel):
    text: str

class ChatRequest(BaseModel):
    patient: PatientInfo
    question: Question
    userId: str

class ChatResponse(BaseModel):
    observation: List[str]
    answer: str
    chatObjectId: str
    id: str
    question: str

class Chat(BaseModel):
    id: str
    patient: PatientInfo
    title: str
    chatId: str

class MessageResponse(BaseModel):
    id: str
    chatObjectId: str
    question: str
    observation: List[str]
    answer: str

class MessageRequest(BaseModel):
    question: Question  
    userId: str

class Question(BaseModel):
    text: str
    
# Helper function to generate chatId
def generate_chat_id() -> str:
    latest_chat = chats_collection.find_one(sort=[("chatId", -1)])
    if latest_chat:
        latest_id = latest_chat["chatId"]
        print(f"Latest chatId: {latest_id}")  # Debugging statement
        number = int(latest_id.replace("MDCKC", "")) + 1
        new_chat_id = f"MDCKC{number:03d}"
    else:
        new_chat_id = "MDCKC001"
    print(f"Generated chatId: {new_chat_id}")  # Debugging statement
    return new_chat_id


# Helper function to generate chat title
def generate_chat_title(question: str) -> str:
    # Extract the last few significant words from the question
    words = question.split()
    title = " ".join(words[-2:])  # Adjust the number of words as needed
    
    # Ensure the title is not too long
    if len(title) > 30:
        title = title[:27] + "..."
    
    return title


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global outputs
    outputs = []  # Clear previous outputs

    def process_response(response: str):
        chat_title_index = response.rfind("ChatTitle:")
        message, chat_title = "", ""
        if chat_title_index != -1:
            # Extract the message (everything before "ChatTitle:")
            message = response[:chat_title_index].strip()
            
            # Extract the chat title (everything after "ChatTitle:")
            chat_title = response[chat_title_index + len("ChatTitle:"):].strip()
        
        return message, chat_title

    def save_message(chat_object_id, chat_id, question, observation, answer):
        message_data = {
            "chatObjectId": str(chat_object_id),  # Convert ObjectId to string
            "chatId": chat_id,
            "question": question,
            "observation": observation,
            "answer": answer
        }
        message_insert_result = messages_collection.insert_one(message_data)
        return message_insert_result.inserted_id

    def handle_chat_and_save_message(request, message, chat_title):
        chat_id = generate_chat_id()

        # Save chat data to MongoDB and get the ObjectId
        chat_data = {
            "patient": request.patient.dict(),
            "title": chat_title,
            "chatId": chat_id,
            "userId": str(request.userId)
        }
        chat_insert_result = chats_collection.insert_one(chat_data)
        chat_object_id = chat_insert_result.inserted_id

        # Save message data to MongoDB with reference to the chat's ObjectId and chatId
        message_object_id = save_message(chat_object_id, chat_id, request.question.text, outputs, message)

        return chat_object_id, message_object_id

    try:
        # Generate chat data
        patient_context = f"Patient: {request.patient.name}, Age: {request.patient.age}, Gender: {request.patient.gender}, Location: {request.patient.location}"
        full_question = f"{patient_context}\n\nQuestion: {request.question.text}\n\nPlease include nearby healthcare facility recommendations, suggested medications, and a list of questions to ask the patient for better diagnosis in your answer if available. End your answer with ChatTitle:(the title of the chat in 2-3 words)"
        
        response = agent_executor.run(full_question)
        message, chat_title = process_response(response)

        chat_object_id, message_object_id = handle_chat_and_save_message(request, message, chat_title)

        return ChatResponse(observation=outputs, answer=message, chatObjectId=str(chat_object_id), id=str(message_object_id), question=request.question.text)
    
    except Exception as e:
        error_message = str(e)
        if "Could not parse LLM output:" in error_message:
            response = error_message.split("Could not parse LLM output:")[1].strip()
            message, chat_title = process_response(response)
            
            chat_object_id, message_object_id = handle_chat_and_save_message(request, message, chat_title)
            
            return ChatResponse(observation=outputs, answer=message, chatObjectId=str(chat_object_id), id=str(message_object_id), question=request.question.text)
        
        raise HTTPException(status_code=500, detail=error_message)


#FastAPI endpoint to add an existing message to a chat
@app.post("/message/{chatObjectId}", response_model=ChatResponse)
async def add_message_to_chat(chatObjectId: str, request: MessageRequest):
    global outputs
    outputs = []  # Clear previous outputs

    def process_response(response: str):
        chat_title_index = response.rfind("ChatTitle:")
        message = ""
        if chat_title_index != -1:
            # Extract the message (everything before "ChatTitle:")
            message = response[:chat_title_index].strip()
        
        return message
    
    def save_message(chat_object_id, chat_id, question, observation, answer):
        message_data = {
            "chatObjectId": chat_object_id,
            "chatId": chat_id,
            "question": question,
            "observation": observation,
            "answer": answer
        }
        message_insert_result = messages_collection.insert_one(message_data)
        return message_insert_result.inserted_id

    try:
        chat = chats_collection.find_one({"_id": ObjectId(chatObjectId)})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_history = list(messages_collection.find({"chatObjectId": chatObjectId}).sort("_id", 1))
        
        context = f"This is a follow-up question for an existing chat. Chat ID: {chat['chatId']}\n"
        context += f"Previous messages:\n"
        for msg in chat_history[-2:]:
            context += f"Q: {msg['question']}\nA: {msg['answer']}\n\n"
        
        full_question = f"{context}New question: {request.question.text}\n\nPlease answer the new question in the context of the previous conversation."

        response = agent_executor.run(full_question)
        message = process_response(response)

        message_object_id = save_message(chatObjectId, chat["chatId"], request.question.text, outputs, message)

        return ChatResponse(
            observation=outputs,
            answer=message,
            chatObjectId=chatObjectId,
            id=str(message_object_id),
            question=request.question.text
        )

    except Exception as e:
        error_message = str(e)
        print("EXCEPTION")
        if "Could not parse LLM output:" in error_message:
            response = error_message.split("Could not parse LLM output:")[1].strip()
            message = process_response(response)
            
            message_object_id = save_message(chatObjectId, chat["chatId"], request.question.text, outputs, message)
            
            return ChatResponse(
                observation=outputs,
                answer=message,
                chatObjectId=chatObjectId,
                id=str(message_object_id),
                question=request.question.text
            )
        
        raise HTTPException(status_code=500, detail=error_message)

# FastAPI endpoint to get all chats
@app.get("/chats", response_model=List[Chat])
async def get_all_chats():
    try:
        chats = list(chats_collection.find())
        for chat in chats:
            chat["id"] = str(chat["_id"])
            chat["chatId"] = chat.get("chatId", "Unknown")  # Add chatId if missing
            del chat["_id"]
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

## FastAPI endpoint to get all chats by userId
@app.get("/chats/{userId}", response_model=List[Chat])
async def get_chats_by_user_id(userId: str):
    try:
        chats = list(chats_collection.find({"userId": userId}))
        for chat in chats:
            chat["id"] = str(chat["_id"])
            chat["chatId"] = chat.get("chatId", "Unknown")
            del chat["_id"]
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint to get messages by chatObjectId
@app.get("/messages/{chatObjectId}", response_model=List[MessageResponse])
async def get_messages_by_chat_object_id(chatObjectId: str):
    try:
        print(f"Fetching messages for chatObjectId: {chatObjectId}")  # Debugging statement
        messages = list(messages_collection.find({"chatObjectId": chatObjectId}))  # Query using string chatObjectId
        print(f"Messages found: {len(messages)}")  # Debugging statement
        for message in messages:
            message["id"] = str(message["_id"])
            message["chatObjectId"] = str(message["chatObjectId"])  # Ensure chatObjectId is a string
            del message["_id"]
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# FastAPI endpoint to get chats by a userId
@app.get("/chats/user/{userId}", response_model=List[Chat])
async def get_chats_by_user_id(userId: str):
    try:
        # Use the correct field for userId
        chats = list(chats_collection.find({"userId": userId})) 
        print(f"Chats found: {len(chats)}")  # Query by userId field
        for chat in chats:
            chat["id"] = str(chat["_id"])
            chat["chatId"] = chat.get("chatId", "Unknown")
            del chat["_id"]
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
