from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv
import os

load_dotenv()
# zhipuai 模型需要安装 pip install --upgrade httpx httpx-sse PyJWT
zhipuai_api_key = os.getenv('ZHIPUAI_API_KEY')

zhipu_model = ChatZhipuAI(
    temperature=0.8,
    api_key=zhipuai_api_key,
    model="GLM-4-Flash"
)

