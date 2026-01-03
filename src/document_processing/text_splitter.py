"""文本分割模块 - 智能分割文档为chunks"""
import re
from typing import List, Dict, Any, Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class TextSplitter:
    """文本分割器"""

    def __init__(
            self,
            chunk_size: int = None,
            chunk_overlap: int = None,
            separators: List[str] = None
    ):
        """初始化文本分割器

        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 重叠大小
            separators: 分隔符列表（优先级从高到低）
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # 默认分隔符（中英文友好）
        self.separators = separators or [
            "\n\n\n",  # 多个空行
            "\n\n",  # 段落
            "\n",  # 换行
            "。",  # 中文句号
            "！",  # 中文感叹号
            "？",  # 中文问号
            ".",  # 英文句号
            "!",  # 英文感叹号
            "?",  # 英文问号
            "；",  # 中文分号
            ";",  # 英文分号
            "，",  # 中文逗号
            ",",  # 英文逗号
            " ",  # 空格
            ""  # 字符
        ]

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """分割文本

        Args:
            text: 输入文本
            metadata: 元数据（会附加到每个chunk）

        Returns:
            chunk列表，每个chunk包含text和metadata
        """
        if not text or not text.strip():
            return []

        # 清理文本
        text = self._clean_text(text)

        # 执行分割
        chunks = self._split_text_recursive(text, self.separators)

        # 构建结果
        results = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            })

            results.append({
                "text": chunk,
                "metadata": chunk_metadata
            })

        logger.info(f"文本分割完成: {len(results)} 个chunks")
        return results

    def _clean_text(self, text: str) -> str:
        """清理文本

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # 移除多余的空白
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # 多个空行变两个
        text = re.sub(r'[ \t]+', ' ', text)  # 多个空格变一个

        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

        return text.strip()

    def _split_text_recursive(
            self,
            text: str,
            separators: List[str]
    ) -> List[str]:
        """递归分割文本

        Args:
            text: 要分割的文本
            separators: 分隔符列表

        Returns:
            文本块列表
        """
        final_chunks = []

        # 选择当前分隔符
        separator = separators[0] if separators else ""

        # 使用当前分隔符分割
        if separator:
            splits = text.split(separator)
        else:
            # 最后一级：按字符分割
            splits = list(text)

        # 处理分割结果
        current_chunk = ""
        for split in splits:
            # 如果split本身就太长，需要进一步分割
            if len(split) > self.chunk_size:
                # 保存当前chunk
                if current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = ""

                # 递归分割长文本
                if len(separators) > 1:
                    sub_chunks = self._split_text_recursive(
                        split,
                        separators[1:]
                    )
                    final_chunks.extend(sub_chunks)
                else:
                    # 没有更多分隔符，强制分割
                    final_chunks.extend(self._force_split(split))

                continue

            # 尝试添加到当前chunk
            if not current_chunk:
                current_chunk = split
            else:
                test_chunk = current_chunk + separator + split

                if len(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    # 当前chunk已满，保存并开始新chunk
                    final_chunks.append(current_chunk)
                    current_chunk = split

        # 保存最后一个chunk
        if current_chunk:
            final_chunks.append(current_chunk)

        # 处理重叠
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._add_overlap(final_chunks)

        return [chunk.strip() for chunk in final_chunks if chunk.strip()]

    def _force_split(self, text: str) -> List[str]:
        """强制分割超长文本

        Args:
            text: 超长文本

        Returns:
            分割后的块
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加chunk之间的重叠

        Args:
            chunks: 原始chunk列表

        Returns:
            带重叠的chunk列表
        """
        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue

            # 从前一个chunk取重叠部分
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk

            # 合并
            new_chunk = overlap_text + chunk
            overlapped_chunks.append(new_chunk)

        return overlapped_chunks

    def split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 中英文句子分割
        sentences = re.split(
            r'([。！？!?;；]+[""]?)',
            text
        )

        # 合并句子和标点
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            combined = (sentence + punctuation).strip()
            if combined:
                result.append(combined)

        # 处理最后一个句子（如果没有标点）
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

        return result

    def split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本

        Args:
            text: 输入文本

        Returns:
            段落列表
        """
        # 按双换行符分割
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def smart_split(
            self,
            text: str,
            preserve_structure: bool = True
    ) -> List[Dict[str, Any]]:
        """智能分割：尝试保持语义完整性

        Args:
            text: 输入文本
            preserve_structure: 是否保持结构（标题、段落等）

        Returns:
            chunk列表
        """
        chunks = []

        if preserve_structure:
            # 检测标题和段落结构
            sections = self._detect_sections(text)

            for section in sections:
                if len(section["text"]) <= self.chunk_size:
                    # 整个section作为一个chunk
                    chunks.append({
                        "text": section["text"],
                        "metadata": {
                            "type": section["type"],
                            "level": section.get("level")
                        }
                    })
                else:
                    # section太长，需要分割
                    sub_chunks = self.split_text(
                        section["text"],
                        metadata={
                            "type": section["type"],
                            "level": section.get("level")
                        }
                    )
                    chunks.extend(sub_chunks)
        else:
            # 普通分割
            chunks = self.split_text(text)

        return chunks

    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """检测文本中的章节结构

        Args:
            text: 输入文本

        Returns:
            章节列表
        """
        sections = []

        # 简单的章节检测（基于标题模式）
        lines = text.split('\n')
        current_section = {"type": "paragraph", "text": "", "level": 0}

        for line in lines:
            line = line.strip()

            # 检测标题
            heading_match = re.match(r'^(#+)\s+(.+)$', line)  # Markdown标题
            number_heading = re.match(r'^(\d+\.)+\s+(.+)$', line)  # 数字标题

            if heading_match:
                # 保存前一个section
                if current_section["text"]:
                    sections.append(current_section)

                # 开始新section
                level = len(heading_match.group(1))
                current_section = {
                    "type": "heading",
                    "level": level,
                    "text": heading_match.group(2)
                }
                sections.append(current_section)
                current_section = {"type": "paragraph", "text": "", "level": level}

            elif number_heading:
                # 保存前一个section
                if current_section["text"]:
                    sections.append(current_section)

                # 开始新section
                current_section = {
                    "type": "heading",
                    "level": 1,
                    "text": number_heading.group(2)
                }
                sections.append(current_section)
                current_section = {"type": "paragraph", "text": "", "level": 1}

            else:
                # 普通文本
                if current_section["text"]:
                    current_section["text"] += "\n" + line
                else:
                    current_section["text"] = line

        # 保存最后一个section
        if current_section["text"]:
            sections.append(current_section)

        return sections

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取分割统计信息

        Args:
            chunks: chunk列表

        Returns:
            统计信息
        """
        if not chunks:
            return {"count": 0}

        sizes = [len(chunk["text"]) for chunk in chunks]

        return {
            "count": len(chunks),
            "total_characters": sum(sizes),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "chunk_size_config": self.chunk_size,
            "overlap_config": self.chunk_overlap
        }


# 全局实例
text_splitter = TextSplitter()