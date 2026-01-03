"""混合检索器：结合BM25和向量检索"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import jieba
import logging
from config.settings import settings
from src.vector_store.embeddings import embedding_model
from src.vector_store.milvus_client import milvus_client

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器"""

    def __init__(self):
        self.bm25_index = None
        self.documents = []
        self.bm25_weight = settings.BM25_WEIGHT
        self.vector_weight = settings.VECTOR_WEIGHT

    def build_bm25_index(self, documents: List[Dict[str, Any]]):
        """构建BM25索引

        Args:
            documents: 文档列表，每个文档包含text和metadata
        """
        self.documents = documents

        # 分词
        tokenized_docs = [list(jieba.cut(doc["text"])) for doc in documents]

        # 构建BM25索引
        self.bm25_index = BM25Okapi(tokenized_docs)
        logger.info(f"构建BM25索引，文档数: {len(documents)}")

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25关键词检索

        Args:
            query: 查询文本
            top_k: 返回top-k结果

        Returns:
            检索结果列表
        """
        if not self.bm25_index:
            logger.warning("BM25索引未构建")
            return []

        # 查询分词
        tokenized_query = list(jieba.cut(query))

        # BM25评分
        scores = self.bm25_index.get_scores(tokenized_query)

        # 获取top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(scores[idx]),
                "source": "bm25"
            })

        return results

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """向量相似度检索

        Args:
            query: 查询文本
            top_k: 返回top-k结果

        Returns:
            检索结果列表
        """
        # 查询向量化
        query_embedding = embedding_model.embed_query(query)

        # Milvus检索
        results = milvus_client.search(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
            collection_type="text"
        )

        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result["text"],
                "metadata": result["metadata"],
                "score": result["score"],
                "source": "vector"
            })

        return formatted_results

    def hybrid_search(
            self,
            query: str,
            top_k: int = 10,
            use_bm25: bool = True,
            use_vector: bool = True
    ) -> List[Dict[str, Any]]:
        """混合检索

        Args:
            query: 查询文本
            top_k: 返回top-k结果
            use_bm25: 是否使用BM25
            use_vector: 是否使用向量检索

        Returns:
            融合后的检索结果
        """
        all_results = {}

        # BM25检索
        if use_bm25 and self.bm25_index:
            bm25_results = self.bm25_search(query, top_k=top_k * 2)
            for result in bm25_results:
                key = result["text"]
                if key not in all_results:
                    all_results[key] = {
                        "text": result["text"],
                        "metadata": result["metadata"],
                        "bm25_score": 0.0,
                        "vector_score": 0.0
                    }
                all_results[key]["bm25_score"] = result["score"]

        # 向量检索
        if use_vector:
            vector_results = self.vector_search(query, top_k=top_k * 2)
            for result in vector_results:
                key = result["text"]
                if key not in all_results:
                    all_results[key] = {
                        "text": result["text"],
                        "metadata": result["metadata"],
                        "bm25_score": 0.0,
                        "vector_score": 0.0
                    }
                all_results[key]["vector_score"] = result["score"]

        # 归一化并融合分数
        if all_results:
            # 归一化BM25分数
            bm25_scores = [r["bm25_score"] for r in all_results.values()]
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

            # 向量分数已经是余弦相似度(0-1)，无需归一化

            # 计算融合分数
            for result in all_results.values():
                normalized_bm25 = result["bm25_score"] / max_bm25
                result["final_score"] = (
                        self.bm25_weight * normalized_bm25 +
                        self.vector_weight * result["vector_score"]
                )

        # 排序并返回top-k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]

        return sorted_results

    def search_with_filter(
            self,
            query: str,
            top_k: int = 10,
            metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """带元数据过滤的检索

        Args:
            query: 查询文本
            top_k: 返回top-k结果
            metadata_filter: 元数据过滤条件，如{"source": "某文档.pdf"}

        Returns:
            过滤后的检索结果
        """
        results = self.hybrid_search(query, top_k=top_k * 3)

        if metadata_filter:
            filtered_results = []
            for result in results:
                metadata = result["metadata"]
                match = all(
                    metadata.get(key) == value
                    for key, value in metadata_filter.items()
                )
                if match:
                    filtered_results.append(result)

                if len(filtered_results) >= top_k:
                    break

            return filtered_results

        return results[:top_k]


# 全局实例
hybrid_retriever = HybridRetriever()