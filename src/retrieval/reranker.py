"""检索结果重排序模块"""
from typing import List, Dict, Any
import logging
from src.llm.qwen_client import llm_client

logger = logging.getLogger(__name__)


class Reranker:
    """检索结果重排序器"""

    def __init__(self):
        """初始化重排序器"""
        self.rerank_methods = ["llm", "cross_encoder", "bm25_boost"]

    def rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            method: str = "llm",
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            method: 重排序方法
            top_k: 返回top-k

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        if method == "llm":
            return self._rerank_with_llm(query, documents, top_k)
        elif method == "cross_encoder":
            return self._rerank_with_cross_encoder(query, documents, top_k)
        elif method == "bm25_boost":
            return self._rerank_with_bm25_boost(query, documents, top_k)
        else:
            logger.warning(f"未知的重排序方法: {method}")
            return documents[:top_k]

    def _rerank_with_llm(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_k: int
    ) -> List[Dict[str, Any]]:
        """使用LLM进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量

        Returns:
            重排序后的文档
        """
        try:
            # 限制输入LLM的文档数量
            docs_to_rank = documents[:min(len(documents), 10)]

            # 构建prompt
            prompt = f"""请对以下检索结果按照与问题的相关性进行评分（0-1分）。

问题: {query}

检索结果:
"""
            for i, doc in enumerate(docs_to_rank, 1):
                text = doc.get("text", "")[:200]  # 截断过长文本
                prompt += f"\n{i}. {text}..."

            prompt += "\n\n请以JSON格式返回每条结果的评分，例如: {\"scores\": [0.9, 0.7, 0.5, ...]}"

            # 调用LLM
            result = llm_client.generate_json(prompt)
            scores = result.get("scores", [])

            # 应用评分
            for i, score in enumerate(scores[:len(docs_to_rank)]):
                docs_to_rank[i]["rerank_score"] = float(score)

            # 排序
            ranked = sorted(
                docs_to_rank,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )

            return ranked[:top_k]

        except Exception as e:
            logger.error(f"LLM重排序失败: {e}")
            return documents[:top_k]

    def _rerank_with_cross_encoder(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_k: int
    ) -> List[Dict[str, Any]]:
        """使用Cross-Encoder模型重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量

        Returns:
            重排序后的文档
        """
        try:
            # 这里可以集成CrossEncoder模型
            # from sentence_transformers import CrossEncoder
            # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            # 由于没有安装CrossEncoder，这里使用简化版本
            logger.warning("CrossEncoder未实现，使用原始评分")

            # 简单地使用原始分数
            scored_docs = documents[:min(len(documents), 20)]

            # 这里应该计算query和每个doc的cross-encoder分数
            # 现在只是保持原有分数
            for doc in scored_docs:
                if "rerank_score" not in doc:
                    doc["rerank_score"] = doc.get("score", 0)

            ranked = sorted(
                scored_docs,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )

            return ranked[:top_k]

        except Exception as e:
            logger.error(f"CrossEncoder重排序失败: {e}")
            return documents[:top_k]

    def _rerank_with_bm25_boost(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_k: int
    ) -> List[Dict[str, Any]]:
        """使用BM25提升重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量

        Returns:
            重排序后的文档
        """
        import jieba
        from collections import Counter

        # 查询词频
        query_tokens = list(jieba.cut(query))
        query_counter = Counter(query_tokens)

        # 计算每个文档的BM25提升分数
        for doc in documents:
            text = doc.get("text", "")
            doc_tokens = list(jieba.cut(text))
            doc_counter = Counter(doc_tokens)

            # 计算词重叠
            overlap_score = sum(
                min(query_counter[term], doc_counter[term])
                for term in query_counter
            ) / max(len(query_tokens), 1)

            # 结合原始分数
            original_score = doc.get("score", 0)
            doc["rerank_score"] = 0.7 * original_score + 0.3 * overlap_score

        # 排序
        ranked = sorted(
            documents,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True
        )

        return ranked[:top_k]

    def diversified_rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_k: int = 5,
            diversity_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """多样性重排序（MMR算法）

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量
            diversity_weight: 多样性权重（0-1）

        Returns:
            多样化后的文档列表
        """
        if not documents or len(documents) <= top_k:
            return documents

        from src.vector_store.embeddings import embedding_model
        import numpy as np

        try:
            # 向量化
            query_vec = embedding_model.embed_query(query)
            doc_texts = [d.get("text", "") for d in documents]
            doc_vecs = embedding_model.embed_texts(doc_texts)

            # 计算相似度
            def cosine_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            query_sims = [cosine_sim(query_vec, doc_vec) for doc_vec in doc_vecs]

            # MMR选择
            selected_indices = []
            remaining_indices = list(range(len(documents)))

            # 选择第一个（最相关的）
            best_idx = max(remaining_indices, key=lambda i: query_sims[i])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            # 迭代选择
            while len(selected_indices) < top_k and remaining_indices:
                mmr_scores = []

                for idx in remaining_indices:
                    # 相关性分数
                    relevance = query_sims[idx]

                    # 多样性分数（与已选文档的最大相似度）
                    max_sim = max(
                        cosine_sim(doc_vecs[idx], doc_vecs[sel_idx])
                        for sel_idx in selected_indices
                    )

                    # MMR分数
                    mmr = (1 - diversity_weight) * relevance - diversity_weight * max_sim
                    mmr_scores.append((idx, mmr))

                # 选择MMR分数最高的
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

            # 返回选中的文档
            return [documents[i] for i in selected_indices]

        except Exception as e:
            logger.error(f"多样性重排序失败: {e}")
            return documents[:top_k]

    def explain_ranking(
            self,
            query: str,
            document: Dict[str, Any]
    ) -> str:
        """解释为什么文档被排在当前位置

        Args:
            query: 查询文本
            document: 文档

        Returns:
            解释文本
        """
        explanation_parts = []

        # 原始分数
        if "score" in document:
            explanation_parts.append(
                f"原始检索分数: {document['score']:.3f}"
            )

        # 重排序分数
        if "rerank_score" in document:
            explanation_parts.append(
                f"重排序分数: {document['rerank_score']:.3f}"
            )

        # 来源
        if "source" in document:
            explanation_parts.append(
                f"来源: {document['source']}"
            )

        # 元数据
        metadata = document.get("metadata", {})
        if "source" in metadata:
            explanation_parts.append(
                f"文档: {metadata['source']}"
            )

        return " | ".join(explanation_parts)


# 全局实例
reranker = Reranker()
