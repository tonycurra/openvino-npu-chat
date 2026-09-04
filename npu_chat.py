#!/usr/bin/env python3
"""
Minimal OpenVINO NPU chatbot with LangChain and conversation memory.
Uses local Phi-3.5 model on NPU with LangChain's Runnable interface.

Requirements:
    pip install openvino-genai langchain langchain-core

Usage:
    python npu_chat.py
    (type 'quit' or 'exit' to end, 'clear' to reset)
"""

import os
import sys
from typing import List, Dict, Any

import openvino_genai as ov_genai
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# --- Configuration ---
MODEL_PATH = "models/npu/Phi-3.5-mini-instruct-int4-gq-ov"
DEVICE = "NPU"
MAX_LENGTH = 200
MAX_HISTORY = 4


# --- Load Model ---
print("Loading OpenVINO model...", file=sys.stderr)
pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)
print("Model loaded.", file=sys.stderr)


# --- Chat History ---
class ChatHistory:
    def __init__(self):
        self.pairs: List[tuple] = []
    
    def add(self, user: str, assistant: str):
        self.pairs.append((user, assistant))
    
    def get_context_messages(self) -> List:
        recent = self.pairs[-MAX_HISTORY:] if self.pairs else []
        messages = []
        for user, ai in recent:
            messages.append(HumanMessage(content=user))
            messages.append(AIMessage(content=ai))
        return messages
    
    def clear(self):
        self.pairs = []


# --- LangChain Chain ---
def format_prompt(messages: List) -> str:
    text = ""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            text += f"System: {msg.content}\n"
        elif isinstance(msg, HumanMessage):
            text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            text += f"Assistant: {msg.content}\n"
    return text + "Assistant:"


def generate_with_history(input_dict: Dict[str, Any]) -> str:
    history: ChatHistory = input_dict.get("config", {}).get("history")
    user_input: str = input_dict["input"]
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Be concise."),
    ] + history.get_context_messages() + [
        HumanMessage(content=user_input)
    ]
    
    prompt = format_prompt(messages)
    return pipe.generate(prompt, max_new_tokens=MAX_LENGTH)


# Create chain
chain = RunnableLambda(generate_with_history)


# --- Chat Loop ---
def main():
    print("=" * 60)
    print("OpenVINO NPU Chat (Phi-3.5-mini)")
    print("=" * 60)
    print("Type 'quit'/'exit' to end, 'clear' to reset\n")
    
    history = ChatHistory()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                history.clear()
                print("Conversation cleared.")
                continue
            
            if not user_input:
                continue
            
            response = chain.invoke(
                {"input": user_input, "config": {"history": history}}
            )
            
            response_text = str(response).strip()
            history.add(user_input, response_text)
            
            print(f"Assistant: {response_text}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Download: optimum-cli export openvino --model OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov --export-dir models")
        sys.exit(1)
    
    main()
