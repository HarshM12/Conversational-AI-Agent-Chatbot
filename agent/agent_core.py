from pydantic import BaseModel
from pydantic.v1 import BaseConfig
import os
from dotenv import load_dotenv
from typing import Any, Callable
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.tools import StructuredTool
from .tools import lookup_faq, collect_feedback,track_order




class CustomAgentExecutor(AgentExecutor):
    class Config(BaseConfig):
        arbitrary_types_allowed = True


load_dotenv()

def _wrap_as_structured(tool_func: Callable[..., Any], *, name: str | None = None, description: str | None = None):
    try:
        return StructuredTool.from_function(
            func=tool_func,
            name=name or getattr(tool_func, "__name__", "unnamed_tool"),
            description=description or (tool_func.__doc__ or ""),
            infer_schema=True,
        )
    except Exception:
    
        return tool_func

tools = [
    _wrap_as_structured(
        lookup_faq,
        name="lookup_faq",
        description="Lookup a frequently asked question by id or question text. Returns the FAQ answer string.",
    ),
    _wrap_as_structured(
        collect_feedback,
        name="collect_feedback",
        description="Collect feedback from the user and store/acknowledge it. Returns a confirmation string.",
    ),
    _wrap_as_structured(
        track_order,
        name="track_order",
        description="Returns order status for a given order ID.",
    ),
]


_agent_executor: AgentExecutor | None = None

def get_agent_executor():
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")

    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    chat_model = ChatGroq(
        api_key=groq_api_key,
        model=groq_model,
        temperature=0.6,   
        max_tokens=700,
    )

    format_instructions = (
    "STRICT OUTPUT FORMAT REQUIRED:\n"
    "Thought: <your reasoning>\n"
    "Action: <tool_name> OR Final Answer\n"
    "Action Input: <JSON arguments if a tool was chosen> (omit if Final Answer)\n"
    "Observation: <result of the tool> (only if a tool was used)\n"
    "Final Answer: <your final response to the user>\n\n"
    "IMPORTANT: Never output Action: None. Use Action: Final Answer when no tool is needed."
)

    prompt = PromptTemplate.from_template(
f"""You are a helpful customer service AI agent. 
Always check the FAQ tool before answering on your own. 

If a tool returns a clear and complete answer (is_final: True), immediately stop and return it as your Final Answer.

STRICT RULES:
- If the user wants to track an order but does not provide an order ID:
  Action: Final Answer
  Final Answer: Please provide your order ID so I can track it.

- Only call the track_order tool after receiving a valid order ID.

- If a tool is needed and all arguments are present, use:
  Action: <tool_name>
  Action Input: <JSON arguments>

- If the user input does NOT require a tool (e.g., greetings, thanks, casual conversation), use:
  Action: Final Answer
  Final Answer: <your response>

- Never use Action: None.
- Always pass tool arguments as proper JSON values.
- Do NOT wrap numeric IDs in extra quotes or strings unnecessarily.

Available tools:
{{tools}}

Available tool names:
{{tool_names}}

Chat History:
{{chat_history}}

User Question: {{input}}

{{agent_scratchpad}}"""
)

    agent = create_react_agent(
        llm=chat_model,
        tools=tools,
        prompt=prompt,
    )

    _agent_executor = CustomAgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    return _agent_executor
