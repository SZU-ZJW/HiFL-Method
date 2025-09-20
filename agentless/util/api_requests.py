import time
import re
import openai
import tiktoken
import requests
from typing import Dict, Union, List
from transformers import AutoTokenizer
import sglang as sgl
from sglang import system, user, assistant, gen
import numpy as np
import json
from openai import OpenAI


SAMPLE = 1

def convert_to_native_float(value):
    """将NumPy浮点数转换为Python原生浮点数"""
    if isinstance(value, (np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, (list, tuple)):
        return [convert_to_native_float(item) for item in value]
    return value


def extract_scores_regex(answers):
    """
    从LLM的输出中提取评分结果。
    
    Args:
        answers (List[str]): 包含多个LLM响应的列表
        
    Returns:
        List[float]: 提取出的final_score列表
    """
    scores = []
    for answer in answers:
        try:
            # 尝试找到JSON格式的字符串
            json_match = re.search(r'\{[^}]+\}', answer)
            if json_match:
                json_str = json_match.group(0)
                # 解析JSON
                score_dict = json.loads(json_str)
                # 提取final_score
                if 'final_score' in score_dict:
                    scores.append(score_dict['final_score'])
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        except (json.JSONDecodeError, AttributeError):
            # 如果解析失败，添加0分
            scores.append(0.0)
    
    return scores


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
            "top_p": 0.9
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
            "top_p": 0.9
        }
    return config

@sgl.function
def text_qa(s, question, num_answers=SAMPLE):
    # 初始系统提示，只设置一次
    s += system(f"You are a helpful assistant.")
    s += user(question)
    
    # 为每个回答存储一个单独的变量
    for i in range(num_answers):
        var_name = f"answer_{i}"
        s += assistant(gen(var_name, max_tokens=1024, temperature=0.6, stop=["USER:", "SYSTEM:"]))


def query_sgl_answer(questions, logger, num_answers=SAMPLE):
    sgl.set_default_backend(sgl.RuntimeEndpoint(" "))
    states = text_qa.run_batch(
        [{"question": q, "num_answers": num_answers} for q in questions],
        progress_bar=False,
    )
    
    all_answers = []
    all_scores = []
    avg_scores = []
    
    for i in range(len(states)):
        question_answers = []
        for j in range(num_answers):
            answer_text = states[i][f"answer_{j}"]
            logger.info(f"Question {i+1}, Answer variant {j+1}: \n{answer_text}")
            question_answers.append(answer_text)
        
        # 提取每个答案的分数
        scores = extract_scores_regex(question_answers)
        
        # 计算平均分数
        avg_score = round(float(np.mean(scores)), 2) if scores else 0.0
        
        all_answers.append(question_answers)
        all_scores.append(scores)
        avg_scores.append(avg_score)
    
    # 转换为原生Python类型
    all_scores = convert_to_native_float(all_scores)
    avg_scores = convert_to_native_float(avg_scores)
    
    # 日志记录所有分数
    logger.info(f"All scores (native type): \n{all_scores}")
    logger.info(f"Average scores (native type): \n{avg_scores}")
    
    # 返回每个问题的所有分数和平均分数
    return avg_scores

def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(base_url=" ", api_key="empty")

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret

def create_anthropic_config(
    message: str,
    prefill_message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": prefill_message},
            ],
        }
    return config


def request_anthropic_engine(client, config, logger, max_retries=40, timeout=100):
    ret = None
    retries = 0

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10)
        retries += 1

    return ret

def create_rm_query(qa_scoring_prompt, message: Union[str, list], response: list): 
    info_requests = []
    for res in response:
        if not isinstance(message, list):
            message_content = message
        else:
            message_content = message[0]["content"]
        
        qa_scoring = qa_scoring_prompt.format(
            query=message_content, 
            answer=res
        )
        info_requests.append(qa_scoring)
    return info_requests

def extract_candidate_number(output_text):
    # 尝试直接解析为JSON
    try:
        # 检查是否是格式2（纯JSON）
        data = json.loads(output_text.strip())
        if "answer" in data:
            match = re.search(r'candidate (\d+)', data["answer"])
            if match:
                return int(match.group(1))
    except json.JSONDecodeError:
        pass
    
    # 尝试提取代码块中的JSON（格式1）
    pattern = r'```json\s*({.*?})\s*```'
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        try:
            json_str = match.group(1).strip()
            data = json.loads(json_str)
            if "answer" in data:
                match = re.search(r'candidate (\d+)', data["answer"])
                if match:
                    return int(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 如果以上都失败，直接用正则表达式查找
    match = re.search(r'candidate (\d+)', output_text)
    if match:
        return int(match.group(1))
        
    return None

def query_q7bc(message, logger):
    client = OpenAI(
    base_url=" ",
    api_key=" ",
)
    response = client.chat.completions.create(
        model="GRM4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        temperature=0.6,
    ) 
    logger.info(f"### PROMPT:\n {message}")
    answer = response.choices[0].message.content
    logger.info(f"Q7BC response: {answer}")
    number = extract_candidate_number(answer)
    logger.info(f"Q7BC number: {number}")
    return number

def format_candidates_for_prompt(candidate_list: List[str]) -> str:
    formatted_string = ""
    for i, candidate in enumerate(candidate_list):
        # Use 1-based indexing for the label
        identifier = f"candidate {i + 1}:"

        # Append the identifier and a newline
        formatted_string += identifier + "\n"
        # Append the candidate's content
        formatted_string += candidate

        # Add a separator (double newline) between candidates, but not after the last one
        if i < len(candidate_list) - 1:
            formatted_string += "\n\n"

    return formatted_string
