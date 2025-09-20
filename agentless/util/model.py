from abc import ABC, abstractmethod
from typing import List
import logging
from agentless.util.api_requests import create_chatgpt_config, request_chatgpt_engine, query_sgl_answer, create_rm_query, query_q7bc, format_candidates_for_prompt
from agentless.util.preprocess_data import correct_file_paths
logger1 = logging.getLogger(__name__)

BATCH = 30

qa_scoring_prompt = """
You are tasked with selecting the single best answer from a list of potential solutions provided for a specific code-related task.

You will be given two main sections:

1.  Task Definition & Context
  This section contains all the necessary information to understand the task you need to evaluate responses for. 
  This includes the original prompt that specifies the requirements and expected output format, the problem description you are addressing, and any relevant context like repository structure or file contents.
2.  Candidate Answers
  This section contains a list of potential solutions that were generated based on the information in the first section.

Your goal is to carefully read and understand everything presented in the Task Definition & Context section. 
This is crucial for knowing what constitutes a correct and well-formatted answer. 
Then, you must evaluate each candidate answer presented in the Candidate Answers section. 
Select the *single* candidate answer that you believe most accurately and effectively solves the problem *and* strictly adheres to the specific instructions and formatting requested within the Task Definition & Context.

---

TASK DEFINITION & CONTEXT:
{task_definition_and_context}

---

CANDIDATE ANSWERS:
{candidate_answers}

---

**Selection Instruction:**

Based on your comprehensive evaluation of the **Task Definition & Context** and the **Candidate Answers**, select the single best candidate answer.

Provide your selection in the following **strict JSON format**. Do not include any other text, comments, or explanations outside the JSON object. The value for the "answer" key must be the exact unique identifier (e.g., "Candidate 1", "Candidate 5", "Candidate 28") corresponding to your selected best answer from the **Candidate Answers** list.

```json
{{
    "answer": "candidate X"
}}
```
"""

def _parse_model_return_lines(content: str) -> list[str]:
        if content:
            return content.strip().split("\n")

def get_best_response(messages: list, scores: list) -> str:
    """
    从消息列表中找出得分最高的那一个
    
    参数:
        messages: 消息内容列表
        scores: 对应的分数列表
    
    返回:
        分数最高的消息内容
    """
    if not messages or not scores or len(messages) != len(scores):
        raise ValueError("消息列表和分数列表必须非空且长度相同")
    
    best_idx = scores.index(max(scores))
    return messages[best_idx]

def get_top_responses(messages: list, scores: list, num_samples: int) -> list:
    """
    从消息列表中找出得分最高的num_samples个
    
    参数:
        messages: 消息内容列表
        scores: 对应的分数列表
        num_samples: 需要返回的消息数量
    
    返回:
        分数最高的num_samples个消息内容
    """
    if not messages or not scores or len(messages) != len(scores):
        raise ValueError("消息列表和分数列表必须非空且长度相同")
    
    # 获取分数最高的num_samples个索引
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_samples]
    return [messages[i] for i in top_indices]

class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        logger,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> None:
        logger.info("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.logger = logger
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = request_chatgpt_engine(config, self.logger)
        if ret:
            responses = [choice.message.content for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        # The nice thing is, when we generate multiple samples from the same input (message),
        # the input tokens are only charged once according to openai API.
        # Therefore, we assume the request cost is only counted for the first sample.
        # More specifically, the `prompt_tokens` is for one input message,
        # and the `completion_tokens` is the sum of all returned completions.
        # Therefore, for the second and later samples, the cost is zero.
        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    "response": response,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                }
            )
        return trajs
    
    def filegen(self, message: str, num_samples: int = 1, file_list: list = None, pred_list: list = None) -> List[dict]:

        youxiao_response = pred_list
        
        if youxiao_response:
            final_response = youxiao_response[:int(num_samples)]
            rm_query = create_rm_query(qa_scoring_prompt, message, final_response)
            scores_list = query_sgl_answer(rm_query, self.logger)
            responses = [get_best_response(final_response, scores_list)]
            completion_tokens = 0
            prompt_tokens = 0
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        return trajs

    def locgen(self, message: str, num_samples: int = 1, pred_list: list = None) -> List[dict]:

        prompt = pred_list["prompt"]
        all_responses = pred_list["response"][:num_samples]
        
        formatted_candidates_text = format_candidates_for_prompt(all_responses)
        final_message = qa_scoring_prompt.format(
            task_definition_and_context=prompt, 
            candidate_answers=formatted_candidates_text
        )
        
        self.logger.info(f"### PROMPT ###:\n {final_message}")

        answer = query_q7bc(final_message, self.logger)
        
        if answer:
            if answer > num_samples:
                responses = [""]
                completion_tokens = 0
                prompt_tokens = 0
            else:
                responses = [all_responses[answer - 1]]
                completion_tokens = 0
                prompt_tokens = 0
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        return trajs
    
    def linegen(self, message: str, num_samples: int = 1) -> List[dict]:
        batch_size = BATCH

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=1.0,
            batch_size=batch_size,
            model=self.name,
        )

        ret = request_chatgpt_engine(config, self.logger)
        if ret:
            inter_responses = [choice.message.content for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
            rm_query = create_rm_query(qa_scoring_prompt, message, inter_responses)
            scores_list = query_sgl_answer(rm_query, self.logger)
            responses = get_top_responses(inter_responses, scores_list, num_samples)
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    "response": response,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                }
            )
        return trajs

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config, self.logger, base_url="https://api.deepseek.com"
            )
            if ret:
                trajs.append(
                    {
                        "response": ret.choices[0].message.content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens,
                            "prompt_tokens": ret.usage.prompt_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    backend: str,
    logger,
    batch_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "deepseek":
        return DeepSeekChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise NotImplementedError
