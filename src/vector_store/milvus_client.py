"""Milvus向量数据库客户端"""
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, utility
)
from typing import List, Dict, Any, Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus客户端"""

    def __init__(self):
        """连接Milvus"""
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
        logger.info(f"已连接Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

        # 初始化集合
        self.text_collection = self._init_collection(
            settings.MILVUS_COLLECTION_TEXT,
            "文档文本集合"
        )
        self.image_collection = self._init_collection(
            settings.MILVUS_COLLECTION_IMAGE,
            "图表描述集合"
        )

    def _init_collection(self, collection_name: str, description: str) -> Collection:
        """初始化或获取集合

        Args:
            collection_name: 集合名称
            description: 集合描述

        Returns:
            Collection对象
        """
        if utility.has_collection(collection_name):
            logger.info(f"集合 {collection_name} 已存在")
            return Collection(collection_name)

        # 定义Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.VECTOR_DIM),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields=fields, description=description)

        # 创建集合
        collection = Collection(name=collection_name, schema=schema)

        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        logger.info(f"创建集合 {collection_name}")
        return collection

    def insert_texts(
            self,
            texts: List[str],
            embeddings: List[List[float]],
            metadatas: List[Dict[str, Any]],
            collection_type: str = "text"
    ) -> List[int]:
        """插入文本数据

        Args:
            texts: 文本列表
            embeddings: 向量列表
            metadatas: 元数据列表
            collection_type: 集合类型 ("text" 或 "image")

        Returns:
            插入的ID列表
        """
        collection = (
            self.text_collection if collection_type == "text"
            else self.image_collection
        )

        entities = [texts, embeddings, metadatas]
        insert_result = collection.insert(entities)
        collection.flush()

        logger.info(f"插入 {len(texts)} 条数据到 {collection.name}")
        return insert_result.primary_keys

    def search(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            collection_type: str = "text",
            filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """向量搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回top-k结果
            collection_type: 集合类型
            filter_expr: 过滤表达式

        Returns:
            搜索结果列表
        """
        collection = (
            self.text_collection if collection_type == "text"
            else self.image_collection
        )

        # 加载集合到内存
        collection.load()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["text", "metadata"]
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata")
                })

        return formatted_results

    def delete_collection(self, collection_name: str):
        """删除集合"""
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"删除集合 {collection_name}")

    def get_collection_stats(self, collection_type: str = "text") -> Dict[str, Any]:
        """获取集合统计信息

        Args:
            collection_type: 集合类型

        Returns:
            统计信息字典
        """
        collection = (
            self.text_collection if collection_type == "text"
            else self.image_collection
        )

        collection.load()
        return {
            "name": collection.name,
            "num_entities": collection.num_entities,
            "description": collection.description
        }


# 全局实例
milvus_client = MilvusClient()