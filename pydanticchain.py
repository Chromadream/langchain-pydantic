from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra, BaseModel, ValidationError

from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from langchain.prompts.chat import (
    ChatPromptValue
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate

class PydanticChain(Chain):
    """
    Pydantic-validated chain for chat model, like `gpt-3.5-turbo`
    """

    prompt: BasePromptTemplate
    """Business logic prompt object to use."""
    llm: BaseChatModel
    model_class: Any
    """Model to use"""
    output_key: str = "text"  #: :meta private:
    max_retries: int = 2

    @property
    def system_message(self):
        prompt = f"Please respond ONLY with valid json that conforms to this pydantic json_schema: {self.model_class.schema_json()}. Do not include additional text other than the object json as we will load this object with json.loads() and pydantic."
        return SystemMessage(content=prompt)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format(**inputs)
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        prompts = [HumanMessage(content=prompt_value), self.system_message]
        response = self.llm.generate_prompt(
            prompts=[ChatPromptValue(messages=prompts)],
            callbacks=run_manager.get_child() if run_manager else None
        )
        return {self.output_key: self.validate(response.generations[0][0].text, prompts)}
    
    def validate(self, output: str, prompts: List[BaseMessage], run_manager: Optional[CallbackManagerForChainRun] = None):
        last_exception = None
        for i in range(self.max_retries+1):
            if i >= 1:
                output = self.llm.generate_prompt(prompts=[ChatPromptValue(messages=prompts)],callbacks=run_manager.get_child() if run_manager else None).generations[0][0].text
            import json
            try:
                json_content = json.loads(output)
            except Exception as e:
                last_exception = e
                error_msg = f"json.loads exception: {e}"
                prompts.append(SystemMessage(content=error_msg))
                continue
            try:
                return self.model_class(**json_content)
            except ValidationError as e:
                last_exception = e
                error_msg = f"pydantic exception: {e}"
                prompts.append(SystemMessage(content=error_msg))
        raise last_exception

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "rail_chain"