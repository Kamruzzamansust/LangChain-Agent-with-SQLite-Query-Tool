from typing import Any, Dict, List
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
import time 

# class ChatModelStartHandler(BaseCallbackHandler):
#     def on_chat_model_start(self, serialized , messsages , **kwargs):
    


class TimingCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.start_time = time.time()
        print("Chain started.")

    def on_chain_end(self, outputs, **kwargs):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Chain ended. Time taken: {elapsed_time:.2f} seconds")

# Initialize your handler
timing_handler = TimingCallbackHandler()