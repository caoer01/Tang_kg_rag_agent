"""图像处理模块 - 处理PDF中的图表"""
from PIL import Image
import pytesseract
from typing import Dict, Any, Optional
import io
import logging
from src.llm.qwen_client import llm_client
from config.prompts import IMAGE_DESCRIPTION_PROMPT

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理器"""

    def __init__(self):
        self.supported_formats = ['PNG', 'JPEG', 'JPG', 'BMP', 'TIFF']

    def process_image(
            self,
            image: Image.Image,
            use_llm: bool = True,
            use_ocr_fallback: bool = True
    ) -> Dict[str, Any]:
        """处理单张图像

        Args:
            image: PIL Image对象
            use_llm: 是否使用LLM生成描述
            use_ocr_fallback: LLM失败时是否使用OCR降级

        Returns:
            处理结果字典
        """
        result = {
            "description": "",
            "ocr_text": "",
            "metadata": {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode
            },
            "method": ""
        }

        # 图像预处理
        processed_image = self._preprocess_image(image)

        # 尝试使用LLM生成描述
        if use_llm:
            try:
                description = self._generate_llm_description(processed_image)
                result["description"] = description
                result["method"] = "llm"
                logger.info("使用LLM生成图像描述")
                return result
            except Exception as e:
                logger.warning(f"LLM处理图像失败: {e}")

        # 降级到OCR
        if use_ocr_fallback:
            try:
                ocr_text = self._extract_text_ocr(processed_image)
                result["ocr_text"] = ocr_text
                result["description"] = self._format_ocr_description(ocr_text)
                result["method"] = "ocr"
                logger.info("使用OCR提取图像文本")
                return result
            except Exception as e:
                logger.error(f"OCR处理失败: {e}")

        # 如果都失败，返回基础描述
        result["description"] = self._generate_basic_description(image)
        result["method"] = "basic"

        return result

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """图像预处理

        Args:
            image: 原始图像

        Returns:
            预处理后的图像
        """
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 调整大小（如果图像太大）
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"图像缩放至: {new_size}")

        # 增强对比度（用于OCR）
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        return image

    def _generate_llm_description(self, image: Image.Image) -> str:
        """使用LLM生成图像描述

        Args:
            image: 图像对象

        Returns:
            图像描述文本
        """
        # 保存临时图像
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name, format='JPEG', quality=85)
            tmp_path = tmp.name

        try:
            # 调用LLM的多模态接口
            description = llm_client.generate_with_images(
                prompt=IMAGE_DESCRIPTION_PROMPT,
                image_paths=[tmp_path]
            )
            return description
        finally:
            # 清理临时文件
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _extract_text_ocr(self, image: Image.Image) -> str:
        """使用OCR提取文本

        Args:
            image: 图像对象

        Returns:
            提取的文本
        """
        # 配置OCR参数（中英文）
        config = '--oem 3 --psm 6'

        try:
            # 尝试中英文混合识别
            text = pytesseract.image_to_string(
                image,
                lang='chi_sim+eng',
                config=config
            )
        except Exception as e:
            logger.warning(f"中文OCR失败，尝试英文: {e}")
            # 降级到仅英文
            text = pytesseract.image_to_string(
                image,
                lang='eng',
                config=config
            )

        # 清理文本
        text = self._clean_ocr_text(text)

        return text

    def _clean_ocr_text(self, text: str) -> str:
        """清理OCR文本

        Args:
            text: 原始OCR文本

        Returns:
            清理后的文本
        """
        # 移除多余空白
        import re
        text = re.sub(r'\s+', ' ', text)

        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()（）、。，；：！？-]', '', text)

        return text.strip()

    def _format_ocr_description(self, ocr_text: str) -> str:
        """格式化OCR文本为描述

        Args:
            ocr_text: OCR提取的文本

        Returns:
            格式化后的描述
        """
        if not ocr_text or len(ocr_text) < 10:
            return "该图表包含少量文字信息。"

        # 添加描述前缀
        description = f"图表中包含以下信息：{ocr_text}"

        # 限制长度
        if len(description) > 500:
            description = description[:497] + "..."

        return description

    def _generate_basic_description(self, image: Image.Image) -> str:
        """生成基础图像描述

        Args:
            image: 图像对象

        Returns:
            基础描述
        """
        width, height = image.size
        aspect_ratio = width / height

        # 判断图像类型
        if aspect_ratio > 1.5:
            img_type = "横向图表"
        elif aspect_ratio < 0.67:
            img_type = "纵向图表"
        else:
            img_type = "方形图表"

        # 简单的颜色分析
        try:
            # 转换为RGB
            rgb_image = image.convert('RGB')
            # 缩略图用于快速分析
            rgb_image.thumbnail((100, 100))
            pixels = list(rgb_image.getdata())

            # 计算平均颜色
            avg_r = sum(p[0] for p in pixels) / len(pixels)
            avg_g = sum(p[1] for p in pixels) / len(pixels)
            avg_b = sum(p[2] for p in pixels) / len(pixels)

            # 判断主色调
            if avg_r > avg_g and avg_r > avg_b:
                color_tone = "偏红色"
            elif avg_g > avg_r and avg_g > avg_b:
                color_tone = "偏绿色"
            elif avg_b > avg_r and avg_b > avg_g:
                color_tone = "偏蓝色"
            else:
                color_tone = "多色"
        except:
            color_tone = "彩色"

        description = (
            f"这是一张{img_type}（{width}x{height}像素），"
            f"颜色为{color_tone}。该图表可能包含医疗相关的数据或图示。"
        )

        return description

    def is_chart_or_diagram(self, image: Image.Image) -> bool:
        """判断图像是否为图表或示意图

        Args:
            image: 图像对象

        Returns:
            是否为图表
        """
        try:
            # 简单的启发式判断
            # 1. 图表通常有较高的对比度
            import numpy as np
            img_array = np.array(image.convert('L'))
            contrast = img_array.std()

            # 2. 图表通常有明显的边缘
            from PIL import ImageFilter
            edges = image.filter(ImageFilter.FIND_EDGES)
            edge_array = np.array(edges.convert('L'))
            edge_density = (edge_array > 50).sum() / edge_array.size

            # 判断标准
            is_chart = contrast > 40 and edge_density > 0.1

            return is_chart
        except Exception as e:
            logger.warning(f"图表判断失败: {e}")
            return True  # 默认认为是图表

    def extract_chart_data(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """尝试提取图表数据（高级功能）

        Args:
            image: 图像对象

        Returns:
            提取的数据字典，失败返回None
        """
        # 这里可以集成专门的图表识别库，如 ChartOCR
        # 目前返回None，表示不支持
        logger.info("图表数据提取功能待实现")
        return None

    def save_processed_image(
            self,
            image: Image.Image,
            output_path: str,
            format: str = 'JPEG',
            quality: int = 85
    ):
        """保存处理后的图像

        Args:
            image: 图像对象
            output_path: 输出路径
            format: 图像格式
            quality: 质量（1-95）
        """
        try:
            image.save(output_path, format=format, quality=quality)
            logger.info(f"图像已保存: {output_path}")
        except Exception as e:
            logger.error(f"保存图像失败: {e}")
            raise


# 全局实例
image_processor = ImageProcessor()