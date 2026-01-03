"""PDF文档加载和处理"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import io
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class PDFLoader:
    """PDF文档加载器"""

    def __init__(self):
        self.pdf_path = Path(settings.PDF_DATA_PATH)

    def load_pdf(self, pdf_file: str) -> List[Dict[str, Any]]:
        """加载单个PDF文件

        Args:
            pdf_file: PDF文件路径

        Returns:
            页面列表，每页包含文本和图像
        """
        doc = fitz.open(pdf_file)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # 提取文本
            text = page.get_text()

            # 提取图像
            images = self._extract_images(page, page_num)

            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "images": images,
                "source": pdf_file
            })

        doc.close()
        logger.info(f"加载PDF: {pdf_file}, 共{len(pages)}页")
        return pages

    def _extract_images(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """从页面提取图像

        Args:
            page: PDF页面对象
            page_num: 页码

        Returns:
            图像信息列表
        """
        images = []
        image_list = page.get_images()

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                # 提取图像数据
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # 转换为PIL Image
                image = Image.open(io.BytesIO(image_bytes))

                # 保存图像
                img_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"

                images.append({
                    "index": img_index,
                    "filename": img_filename,
                    "image": image,
                    "format": image_ext,
                    "size": image.size
                })
            except Exception as e:
                logger.warning(f"提取图像失败 page={page_num + 1}, img={img_index}: {e}")

        return images

    def load_all_pdfs(self) -> List[Dict[str, Any]]:
        """加载所有PDF文件

        Returns:
            所有文档的页面列表
        """
        all_pages = []
        pdf_files = list(self.pdf_path.glob("*.pdf"))

        logger.info(f"找到 {len(pdf_files)} 个PDF文件")

        for pdf_file in pdf_files:
            try:
                pages = self.load_pdf(str(pdf_file))
                all_pages.extend(pages)
            except Exception as e:
                logger.error(f"加载PDF失败 {pdf_file}: {e}")

        return all_pages

    def split_text_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        import re
        # 中英文句子分割
        sentences = re.split(r'[。!?;]\s*|[.!?;]\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# 全局实例
pdf_loader = PDFLoader()