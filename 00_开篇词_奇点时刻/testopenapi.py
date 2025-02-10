import openai
import os

# openai.api_key = os.getenv("OPENAI_API_KEY")
# try:
#     response = openai.Completion.create(
#         model="gpt-3.5-turbo-instruct",
#         prompt="Hello",
#         max_tokens=5
#     )
#     print("API 测试成功")
# except Exception as e:
#     print(f"API 测试失败: {str(e)}")




# 测试网络连接
import requests

os.environ['http_proxy'] = '127.0.0.1:8119'
os.environ['https_proxy'] = '127.0.0.1:8119'

try:
    # response = requests.get("https://api.openai.com", timeout=5)
    response = requests.get("https://baidu.com", timeout=10)
    print(f"OpenAI API 可访问，状态码: {response.status_code}")
except Exception as e:
    print(f"网络连接问题: {str(e)}")