# util.py

import requests
import dashscope
import zhipuai
import openai
import random
from http import HTTPStatus

P = "Here are the different parts of a story. Please splice these parts to make them more smooth and clear in format. When stitching, please strictly adhere to the content of each part, do not add too much additional information."

def unified_query(api_key, messages, model_type):
    # GPT-4, GPT-3.5 Query Function
    if model_type in ["gpt-4", "gpt-3.5-turbo"]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "model": model_type,
            "messages": [{"role": "user", "content": f"{messages}"}],
            "temperature": 1,
            "max_tokens": 2048
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    # Qwen-max Query Function
    elif model_type == "qwen-max":
        dashscope.api_key = api_key
        messages = [{"role": "user", "content": f"{messages}"}]
        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_max,
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message',
        )
        if response.status_code == HTTPStatus.OK:
            return response['output']['choices'][0]['message']['content']
        else:
            raise Exception(f"Error {response.status_code}: {response.message}")

    # Qwen-turbo Query Function
    elif model_type == "qwen-turbo":
        dashscope.api_key = api_key
        messages = [{"role": "user", "content": f"{messages}"}]
        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_turbo,
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message',
        )
        if response.status_code == HTTPStatus.OK:
            return response['output']['choices'][0]['message']['content']
        else:
            raise Exception('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code, response.code, response.message))

    # ChatGLM-turbo Query Function
    elif model_type == "ChatGLM-turbo":
        zhipuai.api_key = api_key
        raw_response = zhipuai.model_api.invoke(
            model="chatglm_turbo",
            prompt={"role": "user", "content": messages}
        )
        print(raw_response)
        return raw_response['data']['choices'][0]['content']

    else:
        raise Exception("Unsupported model type")



def concat(api_key, assembler_outputs, model_type):
    concatenated_result = '\n'.join(assembler_outputs.values())
    final_result = unified_query(api_key, P + f"{concatenated_result}", model_type)
    return final_result



def draw_picture_dalle3(api_key, prompt):
    print("Draw Picture")
    client = openai.OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    try:
        response = client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", quality="standard", n=1)
        image_url = response.data[0].url
        return image_url
    except openai.BadRequestError as e:
        print("Error encountered:", e)
        return None