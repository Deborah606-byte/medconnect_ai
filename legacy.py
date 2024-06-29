from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os

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

# Custom search wrapper
def duck_wrapper(input_text):
    search = DuckDuckGoSearchRun()
    search_results = search.run(f"site:webmd.com {input_text}")
    return search_results

# Define tools
tools = [
    Tool(
        name="Search WebMD",
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
template_with_history = """Answer the following questions as best you can, but speaking as a compassionate medical professional.  If you think there is need for diagnosis, provide further questions to be asked to the patient. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action                       
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the result of the action(observations) and the final answer to the original input question.

Begin! Remember to provide me with all the questions to ask, if any . If the condition is serious advise a referral to the hospital.

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
    try:
        patient_context = f"Patient: {request.patient.name}, Age: {request.patient.age}, Gender: {request.patient.gender}"
        full_question = f"{patient_context}\n\nQuestion: {request.question.text}"
        
        response = agent_executor.run(full_question)
        return ChatResponse(observation=outputs, answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)