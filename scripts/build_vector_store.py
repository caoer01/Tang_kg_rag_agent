"""构建向量数据库脚本"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.document_processing.pdf_loader import pdf_loader
from src.vector_store.embeddings import embedding_model
from src.vector_store.milvus_client import milvus_client
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.llm.qwen_client import llm_client
from config.prompts import IMAGE_DESCRIPTION_PROMPT
from config.settings import settings
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """分割文本为块

    Args:
        text: 输入文本
        chunk_size: 块大小（字符数）
        overlap: 重叠大小

    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # 尝试在句子边界分割
        if end < text_len:
            last_period = chunk.rfind('。')
            last_exclaim = chunk.rfind('!')
            last_question = chunk.rfind('?')
            boundary = max(last_period, last_exclaim, last_question)

            if boundary > chunk_size * 0.5:  # 至少保留一半内容
                chunk = chunk[:boundary + 1]
                end = start + boundary + 1

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


def process_images(pages: list) -> list:
    """处理图像，生成描述

    Args:
        pages: 页面列表

    Returns:
        图像描述列表
    """
    image_descriptions = []

    for page in tqdm(pages, desc="处理图像"):
        if not page.get("images"):
            continue

        for img_info in page["images"]:
            try:
                # 使用LLM生成图像描述
                # 注意：需要多模态模型支持
                description = llm_client.generate(
                    IMAGE_DESCRIPTION_PROMPT,
                    temperature=0.3
                )

                # 或者使用OCR作为降级方案
                # description = extract_text_from_image(img_info["image"])

                image_descriptions.append({
                    "description": description,
                    "metadata": {
                        "source": page["source"],
                        "page_num": page["page_num"],
                        "image_index": img_info["index"],
                        "type": "image"
                    }
                })
            except Exception as e:
                logger.error(f"处理图像失败: {e}")

    return image_descriptions


def build_vector_store():
    """构建向量数据库"""
    logger.info("=" * 50)
    logger.info("开始构建向量数据库")
    logger.info("=" * 50)

    # 1. 加载PDF文档
    logger.info("步骤1: 加载PDF文档")
    pages = pdf_loader.load_all_pdfs()
    logger.info(f"加载了 {len(pages)} 页文档")

    # 2. 分割文本
    logger.info("步骤2: 分割文本")
    text_chunks = []
    chunk_metadatas = []

    for page in tqdm(pages, desc="分割文本"):
        if not page.get("text"):
            continue

        chunks = split_text_into_chunks(
            page["text"],
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )

        for i, chunk in enumerate(chunks):
            text_chunks.append(chunk)
            chunk_metadatas.append({
                "source": page["source"],
                "page_num": page["page_num"],
                "chunk_index": i,
                "type": "text"
            })

    logger.info(f"分割为 {len(text_chunks)} 个文本块")

    # 3. 向量化文本
    logger.info("步骤3: 向量化文本")
    text_embeddings = embedding_model.embed_texts(text_chunks)
    logger.info(f"生成了 {len(text_embeddings)} 个向量")

    # 4. 插入Milvus
    logger.info("步骤4: 插入向量数据库")
    batch_size = 100
    for i in tqdm(range(0, len(text_chunks), batch_size), desc="插入Milvus"):
        batch_texts = text_chunks[i:i + batch_size]
        batch_embeddings = text_embeddings[i:i + batch_size].tolist()
        batch_metadatas = chunk_metadatas[i:i + batch_size]

        milvus_client.insert_texts(
            texts=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            collection_type="text"
        )

    # 5. 处理图像（如果有）
    logger.info("步骤5: 处理图像")
    image_data = process_images(pages)

    if image_data:
        image_texts = [item["description"] for item in image_data]
        image_metadatas = [item["metadata"] for item in image_data]
        image_embeddings = embedding_model.embed_texts(image_texts)

        logger.info(f"处理了 {len(image_data)} 张图像")

        # 插入图像描述
        for i in range(0, len(image_texts), batch_size):
            batch_texts = image_texts[i:i + batch_size]
            batch_embeddings = image_embeddings[i:i + batch_size].tolist()
            batch_metadatas = image_metadatas[i:i + batch_size]

            milvus_client.insert_texts(
                texts=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                collection_type="image"
            )

    # 6. 构建BM25索引
    logger.info("步骤6: 构建BM25索引")
    documents = [
        {"text": text, "metadata": meta}
        for text, meta in zip(text_chunks, chunk_metadatas)
    ]
    hybrid_retriever.build_bm25_index(documents)

    # 7. 统计信息
    logger.info("=" * 50)
    logger.info("向量库构建完成!")
    logger.info("=" * 50)
    text_stats = milvus_client.get_collection_stats("text")
    logger.info(f"文本集合: {text_stats}")

    if image_data:
        image_stats = milvus_client.get_collection_stats("image")
        logger.info(f"图像集合: {image_stats}")


if __name__ == "__main__":
    try:
        build_vector_store()
    except Exception as e:
        logger.error(f"构建向量库失败: {e}", exc_info=True)
        sys.exit(1)