import csv
from typing import List, Dict
from fastapi import FastAPI, HTTPException
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

load_dotenv()

app = FastAPI()

def load_hospitals() -> List[Dict[str, str]]:
    hospitals = []
    with open('cleaned_hospitals.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            hospitals.append(row)
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
def duck_wrapper(input_text):
    search = DuckDuckGoSearchRun()
    search_results = search.run(f"site:uptodate.com {input_text}")
    
    # Extract location from input
    location = input_text.split("Location:")[-1].split("\n")[0].strip() if "Location:" in input_text else ""
    
    if location:
        nearby_hospitals = find_nearby_hospitals(location)
        if nearby_hospitals:
            search_results += "\n\nNearby healthcare facilities for further diagnosis or treatment:\n" + "\n".join(nearby_hospitals[:5])  # Limit to top 5 for brevity
        else:
            search_results += "\n\nNo specific healthcare facilities found in the exact location, please consult with a local healthcare provider."
    
    # Add medication suggestions
    medication_results = search.run(f"site:drugs.com {input_text} medications")
    search_results += "\n\nSuggested Medications:\n" + "\n".join(medication_results[:5])  # Limit to top 5 for brevity
    
    return search_results

# Define tools
tools = [
    Tool(
        name="Search UpToDate and Drugs.com",
        func=duck_wrapper,
        description="useful for when you need to answer medical and pharmacological questions"
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
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Custom output parser
outputs = []
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        # Split the llm_output into blocks based on "Thought:" and "Observation:"
        blocks = re.split(r"Observation:", llm_output)
        
        for block in blocks:
            if "Action:" in block and "Action Input:" in block:
                observation = block.split("Action:")[0].strip()
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
template_with_history = """Answer the following questions as best you can, speaking as a compassionate medical professional. If you think there is need for diagnosis or treatment, provide further questions to be asked to the patient and recommend nearby healthcare facilities if available. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action                       
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the result of the action(observations), the final answer to the original input question, nearby healthcare facility recommendations if available, suggested medications if available, and a list of questions to ask the patient for better diagnosis.

Begin! Remember to provide all the questions to ask, if any, and always include nearby healthcare facility recommendations and suggested medications in your final answer if they are available.

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

class ChatResponse(BaseModel):
    observation: List[str]
    answer: str

# FastAPI endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global outputs
    outputs = []  # Clear previous outputs
    try:
        patient_context = f"Patient: {request.patient.name}, Age: {request.patient.age}, Gender: {request.patient.gender}, Location: {request.patient.location}"
        full_question = f"{patient_context}\n\nQuestion: {request.question.text}\n\nPlease include nearby healthcare facility recommendations, suggested medications, and a list of questions to ask the patient for better diagnosis in your answer if available."
        
        response = agent_executor.run(full_question)
        
        return ChatResponse(observation=outputs, answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
