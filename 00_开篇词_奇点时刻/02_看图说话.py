'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

#---- Part 0 导入所需要的类
import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
# from langchain import OpenAI
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from typing import Optional
from urllib.parse import urlparse
# from langchain.agents import create_react_agent # 未来需要改成下面的设计
import logging
logging.basicConfig(level=logging.DEBUG)

os.environ['http_proxy'] = '127.0.0.1:8119'
os.environ['https_proxy'] = '127.0.0.1:8119'


# 直接检查 API key 是否设置
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")


#---- Part I 初始化图像字幕生成模型
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-large"

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
local_model_path ='/Volumes/G/huggingface/Salesforce:blip-image-captioning-large'
processor = BlipProcessor.from_pretrained(local_model_path)
# 然后我们初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(local_model_path)

#---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool):
   
    name: str = "Image captioner"  # 添加类型注解
    description: str = "为图片创作说明文案."  # 添加类型注解

    def _run(self, image_path: str) -> str:   # 添加返回类型注解

        try:
            # 检查是否是 URL 还是本地文件路径
            parsed = urlparse(image_path)

            if parsed.scheme in ['http', 'https']:
                # 如果是网络图片
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw).convert('RGB')
            else:
                # 如果是本地文件
                if not os.path.exists(image_path):
                    return f"错误：找不到图片文件 {image_path}"
                image = Image.open(image_path).convert('RGB')

            # 预处理图像
            inputs = processor(image, return_tensors="pt")
            # 生成字幕
            out = model.generate(inputs, max_new_tokens=20)
            # 获取字幕
            caption = processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"处理图片时候=出错：{set(e)}"
    
    def _arun(self, query: str) -> None:
        raise NotImplementedError("This tool does not support async")

#---- PartIII 初始化并运行LangChain智能体
# 设置OpenAI的API密钥并初始化大语言模型（OpenAI的Text模型）
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
llm = OpenAI(
    temperature=0.2,
    model_name="gpt-3.5-turbo-instruct"  # 明确指定模型
)

# 使用工具初始化智能体并运行
tools = [ImageCapTool()]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# 未来需要改成下面的设计
# agent = create_react_agent(
#     tools=tools,
#     llm=llm,
# )
# image_path = 'https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg'
local_image_path='/Volumes/G/projecttest/langchain-in-action/附件/flower.jpg'
result=agent.invoke(input=f"{local_image_path}\n请基于图片内容，创作一段富有创意的中文推广文案")
print(result)

# agent.run(input=f"{img_url}\n请创作合适的中文推广文案")
# agent.invoke(input=f"{img_url}\n请创作合适的中文推广文案")