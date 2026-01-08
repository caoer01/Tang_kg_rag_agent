"""文本向量化模块 - 完整实现"""
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional, Dict, Any
import numpy as np
import logging
import torch
from pathlib import Path
import pickle
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """向量化模型 - 支持多种embedding方法"""

    def __init__(self, model_name: str = None, device: str = None, cache_dir: str = None):
        """初始化向量化模型

        Args:
            model_name: 模型名称，默认使用配置文件中的设置
            device: 设备 ('cuda', 'cpu', 'mps')，自动检测
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL

        # 自动检测设备
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"加载Embedding模型: {self.model_name}")
        logger.info(f"使用设备: {self.device}")

        # 加载模型
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=cache_dir
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"模型加载成功，向量维度: {self.dimension}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

        # 向量缓存（可选）
        self.cache_enabled = False
        self.cache = {}
        self.cache_file = Path("./cache/embeddings.pkl")

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """单个文本向量化

        Args:
            text: 输入文本
            normalize: 是否归一化向量

        Returns:
            向量数组 (dimension,)
        """
        if not text or not text.strip():
            logger.warning("输入文本为空，返回零向量")
            return np.zeros(self.dimension)

        # 检查缓存
        if self.cache_enabled and text in self.cache:
            return self.cache[text]

        try:
            # 向量化
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )

            # 保存到缓存
            if self.cache_enabled:
                self.cache[text] = embedding

            return embedding

        except Exception as e:
            logger.error(f"文本向量化失败: {e}")
            return np.zeros(self.dimension)

    def embed_texts(
            self,
            texts: List[str],
            batch_size: int = None,
            normalize: bool = True,
            show_progress: bool = True
    ) -> np.ndarray:
        """批量文本向量化

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否归一化向量
            show_progress: 是否显示进度条

        Returns:
            向量矩阵 (len(texts), dimension)
        """
        if not texts:
            logger.warning("输入文本列表为空")
            return np.array([])

        # 过滤空文本
        valid_texts = [t if t and t.strip() else " " for t in texts]

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                device=self.device
            )

            logger.info(f"批量向量化完成: {len(texts)} 个文本")
            return embeddings

        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            return np.zeros((len(texts), self.dimension))

    def embed_query(self, query: str, add_instruction: bool = False) -> np.ndarray:
        """查询向量化（可以添加特殊处理）

        Args:
            query: 查询文本
            add_instruction: 是否添加查询指令前缀

        Returns:
            向量数组
        """
        # 为查询添加特殊标记（某些模型需要）
        if add_instruction:
            query = f"查询: {query}"

        return self.embed_text(query)

    def embed_documents_with_metadata(
            self,
            documents: List[Dict[str, Any]],
            text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        """向量化带元数据的文档

        Args:
            documents: 文档列表，每个文档包含text和metadata
            text_key: 文本字段的键名

        Returns:
            带向量的文档列表
        """
        texts = [doc.get(text_key, "") for doc in documents]
        embeddings = self.embed_texts(texts)

        # 将向量添加到文档中
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding

        return documents

    def compute_similarity(
            self,
            text1: Union[str, np.ndarray],
            text2: Union[str, np.ndarray],
            metric: str = "cosine"
    ) -> float:
        """计算两个文本的相似度

        Args:
            text1: 文本1或向量1
            text2: 文本2或向量2
            metric: 相似度度量 ('cosine', 'dot', 'euclidean')

        Returns:
            相似度分数
        """
        # 获取向量
        vec1 = text1 if isinstance(text1, np.ndarray) else self.embed_text(text1)
        vec2 = text2 if isinstance(text2, np.ndarray) else self.embed_text(text2)

        if metric == "cosine":
            # 余弦相似度
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        elif metric == "dot":
            # 点积
            similarity = np.dot(vec1, vec2)
        elif metric == "euclidean":
            # 欧氏距离（转换为相似度）
            distance = np.linalg.norm(vec1 - vec2)
            similarity = 1 / (1 + distance)
        else:
            raise ValueError(f"未知的相似度度量: {metric}")

        return float(similarity)

    def compute_similarity_matrix(
            self,
            texts: List[str],
            metric: str = "cosine"
    ) -> np.ndarray:
        """计算文本列表的相似度矩阵

        Args:
            texts: 文本列表
            metric: 相似度度量

        Returns:
            相似度矩阵 (len(texts), len(texts))
        """
        embeddings = self.embed_texts(texts)
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = self.compute_similarity(embeddings[i], embeddings[j], metric)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix

    def find_most_similar(
            self,
            query: str,
            candidates: List[str],
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """查找最相似的文本

        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回top-k结果

        Returns:
            相似文本列表
        """
        query_vec = self.embed_query(query)
        candidate_vecs = self.embed_texts(candidates)

        # 计算相似度
        similarities = [
            self.compute_similarity(query_vec, cand_vec)
            for cand_vec in candidate_vecs
        ]

        # 排序
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            {
                "text": candidates[idx],
                "similarity": similarities[idx],
                "rank": rank + 1
            }
            for rank, idx in enumerate(ranked_indices)
        ]

        return results

    def enable_cache(self, cache_file: str = None):
        """启用向量缓存

        Args:
            cache_file: 缓存文件路径
        """
        self.cache_enabled = True
        if cache_file:
            self.cache_file = Path(cache_file)

        # 加载已有缓存
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"加载缓存: {len(self.cache)} 条记录")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")

    def save_cache(self):
        """保存缓存到文件"""
        if not self.cache_enabled:
            return

        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"缓存已保存: {len(self.cache)} 条记录")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("缓存已清空")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache)
        }

    def benchmark(self, test_texts: List[str] = None) -> Dict[str, Any]:
        """性能基准测试

        Args:
            test_texts: 测试文本列表

        Returns:
            基准测试结果
        """
        import time

        if test_texts is None:
            test_texts = [
                             "糖尿病是一种慢性代谢疾病",
                             "胰岛素是降低血糖的重要激素",
                             "定期检测血糖对糖尿病管理很重要"
                         ] * 10

        # 单个文本测试
        start = time.time()
        _ = self.embed_text(test_texts[0])
        single_time = time.time() - start

        # 批量测试
        start = time.time()
        _ = self.embed_texts(test_texts)
        batch_time = time.time() - start

        return {
            "num_texts": len(test_texts),
            "single_text_time": f"{single_time:.4f}s",
            "batch_time": f"{batch_time:.4f}s",
            "avg_time_per_text": f"{batch_time / len(test_texts):.4f}s",
            "throughput": f"{len(test_texts) / batch_time:.2f} texts/s"
        }

    def __del__(self):
        """析构函数：保存缓存"""
        if self.cache_enabled and self.cache:
            self.save_cache()

# 全局实例
embedding_model = EmbeddingModel()


# 便捷函数
def embed_text(text: str) -> np.ndarray:
    """快捷函数：向量化单个文本"""
    return embedding_model.embed_text(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """快捷函数：向量化多个文本"""
    return embedding_model.embed_texts(texts)


def compute_similarity(text1: str, text2: str) -> float:
    """快捷函数：计算文本相似度"""
    return embedding_model.compute_similarity(text1, text2)
