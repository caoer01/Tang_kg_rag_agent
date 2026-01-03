"""Qwen LLM客户端，通过vllm部署的OpenAI兼容接口调用"""
from openai import OpenAI
from typing import Optional, Dict, Any, List
import json
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class QwenClient:
    """Qwen LLM客户端"""

    def __init__(self):
        self.client = OpenAI(
            api_key="EMPTY",  # vllm不需要API key
            base_url=settings.LLM_API_BASE
        )
        self.model_name = settings.LLM_MODEL_NAME

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = None,
            max_tokens: int = None,
            json_mode: bool = False
    ) -> str:
        """生成回复

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
            json_mode: 是否启用JSON模式

        Returns:
            生成的文本
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature or settings.LLM_TEMPERATURE,
                "max_tokens": max_tokens or settings.LLM_MAX_TOKENS
            }

            # 如果需要JSON输出，在prompt中明确要求
            if json_mode and "JSON" not in prompt.upper():
                messages[-1]["content"] += "\n\n请以JSON格式输出。"

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

    def generate_json(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = None
    ) -> Dict[Any, Any]:
        """生成JSON格式回复

        Args:
            prompt: 提示词
            system_prompt: 系统提示词
            temperature: 温度参数

        Returns:
            解析后的JSON字典
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=True
        )

        # 提取JSON内容（可能包含在```json```代码块中）
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取代码块中的JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                logger.error(f"无法解析JSON: {response}")
                raise ValueError(f"无法解析LLM返回的JSON: {response}")

    def batch_generate(
            self,
            prompts: List[str],
            system_prompt: Optional[str] = None,
            temperature: float = None
    ) -> List[str]:
        """批量生成

        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词
            temperature: 温度参数

        Returns:
            生成的文本列表
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature
            )
            results.append(result)
        return results

    def generate_with_images(
            self,
            prompt: str,
            image_paths: List[str],
            system_prompt: Optional[str] = None
    ) -> str:
        """支持图像输入的生成（用于图表描述）

        Args:
            prompt: 提示词
            image_paths: 图像路径列表
            system_prompt: 系统提示词

        Returns:
            生成的文本
        """
        # 注意：这需要vllm支持多模态模型
        # 如果使用的Qwen不支持视觉，需要使用OCR等方式预处理
        import base64

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 构建包含图像的消息
            content = [{"type": "text", "text": prompt}]
            for img_path in image_paths:
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })

            messages.append({"role": "user", "content": content})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"多模态LLM调用失败: {e}")
            # 降级处理：使用OCR
            logger.warning("降级使用OCR处理图像")
            return self._ocr_fallback(image_paths[0], prompt)

    def _ocr_fallback(self, image_path: str, prompt: str) -> str:
        """OCR降级处理"""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')

            fallback_prompt = f"{prompt}\n\n图像OCR提取的文本:\n{text}"
            return self.generate(fallback_prompt)
        except Exception as e:
            logger.error(f"OCR处理失败: {e}")
            return "图像处理失败，无法提取内容。"


# 全局实例
llm_client = QwenClient()