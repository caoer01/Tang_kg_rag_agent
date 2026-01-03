"""查询改写模块 - 优化检索查询"""
from typing import List, Dict, Any
import logging
from src.llm.qwen_client import llm_client
from config.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询改写器"""

    def __init__(self):
        """初始化查询改写器"""
        # 医疗领域同义词词典
        self.synonyms = self._load_synonyms()

        # 缩写词典
        self.abbreviations = self._load_abbreviations()

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """加载同义词词典

        Returns:
            同义词字典
        """
        return {
            "糖尿病": ["高血糖症", "消渴症", "DM", "diabetes"],
            "血糖": ["血糖浓度", "血糖值", "葡萄糖"],
            "胰岛素": ["insulin", "INS"],
            "并发症": ["并发病", "合并症", "伴随疾病"],
            "治疗": ["诊治", "医治", "疗法"],
            "症状": ["临床表现", "表现", "体征"],
            "药物": ["药品", "用药", "medication"],
            "检查": ["检验", "化验", "test"],
        }

    def _load_abbreviations(self) -> Dict[str, str]:
        """加载缩写词典

        Returns:
            缩写-全称字典
        """
        return {
            "DM": "糖尿病",
            "T1DM": "1型糖尿病",
            "T2DM": "2型糖尿病",
            "GDM": "妊娠期糖尿病",
            "HbA1c": "糖化血红蛋白",
            "FPG": "空腹血糖",
            "PPG": "餐后血糖",
            "OGTT": "口服葡萄糖耐量试验",
            "BMI": "体重指数",
            "BP": "血压",
        }

    def rewrite_query(
            self,
            query: str,
            num_rewrites: int = 3,
            previous_rewrites: List[str] = None
    ) -> List[str]:
        """改写查询

        Args:
            query: 原始查询
            num_rewrites: 生成改写数量
            previous_rewrites: 之前的改写（避免重复）

        Returns:
            改写后的查询列表
        """
        rewrites = []

        # 方法1: LLM改写
        llm_rewrites = self._rewrite_with_llm(
            query,
            num_rewrites,
            previous_rewrites or []
        )
        rewrites.extend(llm_rewrites)

        # 方法2: 基于规则的改写
        rule_rewrites = self._rewrite_with_rules(query)
        rewrites.extend(rule_rewrites)

        # 去重
        unique_rewrites = []
        seen = set([query] + (previous_rewrites or []))

        for rewrite in rewrites:
            if rewrite not in seen:
                unique_rewrites.append(rewrite)
                seen.add(rewrite)

        return unique_rewrites[:num_rewrites]

    def _rewrite_with_llm(
            self,
            query: str,
            num_rewrites: int,
            previous_rewrites: List[str]
    ) -> List[str]:
        """使用LLM改写查询

        Args:
            query: 原始查询
            num_rewrites: 改写数量
            previous_rewrites: 已有改写

        Returns:
            改写列表
        """
        try:
            prompt = QUERY_REWRITE_PROMPT.format(
                question=query,
                previous_rewrites=previous_rewrites,
                num_rewrites=num_rewrites
            )

            result = llm_client.generate_json(prompt)
            rewrites = result.get("rewrites", [])

            return rewrites

        except Exception as e:
            logger.error(f"LLM查询改写失败: {e}")
            return []

    def _rewrite_with_rules(self, query: str) -> List[str]:
        """基于规则改写查询

        Args:
            query: 原始查询

        Returns:
            改写列表
        """
        rewrites = []

        # 规则1: 扩展同义词
        synonym_query = self._expand_synonyms(query)
        if synonym_query != query:
            rewrites.append(synonym_query)

        # 规则2: 展开缩写
        expanded_query = self._expand_abbreviations(query)
        if expanded_query != query:
            rewrites.append(expanded_query)

        # 规则3: 添加医学术语
        medical_query = self._add_medical_terms(query)
        if medical_query != query:
            rewrites.append(medical_query)

        # 规则4: 改变疑问方式
        question_variants = self._vary_question_form(query)
        rewrites.extend(question_variants)

        return rewrites

    def _expand_synonyms(self, query: str) -> str:
        """扩展同义词

        Args:
            query: 原始查询

        Returns:
            扩展后的查询
        """
        expanded_terms = []

        for term, synonyms in self.synonyms.items():
            if term in query:
                # 添加同义词
                expanded_terms.extend([term] + synonyms[:2])

        if expanded_terms:
            # 去重
            unique_terms = list(dict.fromkeys(expanded_terms))
            return " ".join(unique_terms)

        return query

    def _expand_abbreviations(self, query: str) -> str:
        """展开缩写

        Args:
            query: 原始查询

        Returns:
            展开后的查询
        """
        expanded = query

        for abbr, full_term in self.abbreviations.items():
            if abbr in query:
                # 同时保留缩写和全称
                expanded = expanded.replace(abbr, f"{abbr} {full_term}")

        return expanded

    def _add_medical_terms(self, query: str) -> str:
        """添加相关医学术语

        Args:
            query: 原始查询

        Returns:
            增强后的查询
        """
        # 关键词到相关术语的映射
        term_mapping = {
            "糖尿病": ["血糖控制", "胰岛素抵抗", "代谢"],
            "并发症": ["微血管", "大血管", "神经病变"],
            "治疗": ["药物", "饮食", "运动"],
            "症状": ["多饮", "多尿", "多食"],
        }

        added_terms = []
        for keyword, related_terms in term_mapping.items():
            if keyword in query:
                added_terms.extend(related_terms[:2])

        if added_terms:
            return f"{query} {' '.join(added_terms)}"

        return query

    def _vary_question_form(self, query: str) -> List[str]:
        """变换疑问形式

        Args:
            query: 原始查询

        Returns:
            变换后的查询列表
        """
        variants = []

        # 检测问题类型
        if "什么是" in query:
            # "什么是X" → "X的定义", "X是什么"
            term = query.replace("什么是", "").strip("?？")
            variants.append(f"{term}的定义")
            variants.append(f"{term}是什么")

        elif "如何" in query or "怎么" in query:
            # "如何X" → "X的方法", "X怎么做"
            action = query.replace("如何", "").replace("怎么", "").strip("?？")
            variants.append(f"{action}的方法")
            variants.append(f"怎样{action}")

        elif "为什么" in query:
            # "为什么X" → "X的原因", "X的机制"
            phenomenon = query.replace("为什么", "").strip("?？")
            variants.append(f"{phenomenon}的原因")
            variants.append(f"{phenomenon}的机制")

        elif "哪些" in query:
            # "有哪些X" → "X的种类", "X的分类"
            term = query.replace("有哪些", "").replace("哪些", "").strip("?？")
            variants.append(f"{term}的种类")
            variants.append(f"{term}的类型")

        return [v for v in variants if v != query]

    def simplify_query(self, query: str) -> str:
        """简化查询（提取核心关键词）

        Args:
            query: 原始查询

        Returns:
            简化后的查询
        """
        import jieba.analyse

        # 提取关键词
        keywords = jieba.analyse.extract_tags(
            query,
            topK=5,
            withWeight=False
        )

        return " ".join(keywords)

    def expand_query(self, query: str, expansion_terms: int = 3) -> str:
        """扩展查询（添加相关术语）

        Args:
            query: 原始查询
            expansion_terms: 扩展术语数量

        Returns:
            扩展后的查询
        """
        import jieba

        # 分词
        tokens = list(jieba.cut(query))

        # 为每个关键词查找同义词
        expanded = tokens.copy()

        for token in tokens:
            if token in self.synonyms:
                synonyms = self.synonyms[token][:expansion_terms]
                expanded.extend(synonyms)

        return " ".join(expanded)

    def decompose_query(self, query: str) -> List[str]:
        """分解复杂查询为多个子查询

        Args:
            query: 复杂查询

        Returns:
            子查询列表
        """
        sub_queries = []

        # 检测复合问题
        if "和" in query or "与" in query:
            # 分割并列问题
            parts = query.replace("和", "，").replace("与", "，").split("，")
            sub_queries.extend([p.strip() for p in parts if p.strip()])

        elif "；" in query or ";" in query:
            # 分号分割
            parts = query.replace("；", ";").split(";")
            sub_queries.extend([p.strip() for p in parts if p.strip()])

        # 如果没有分解，返回原查询
        if not sub_queries:
            sub_queries = [query]

        return sub_queries

    def contextualize_query(
            self,
            query: str,
            conversation_history: List[str]
    ) -> str:
        """根据对话历史添加上下文

        Args:
            query: 当前查询
            conversation_history: 对话历史

        Returns:
            上下文化的查询
        """
        if not conversation_history:
            return query

        # 检测代词
        pronouns = ["它", "这个", "那个", "他", "她"]
        has_pronoun = any(p in query for p in pronouns)

        if has_pronoun and conversation_history:
            # 从历史中提取主题
            recent_context = " ".join(conversation_history[-2:])

            # 简单替换（实际应该用共指消解）
            contextualized = f"{recent_context} {query}"
            return contextualized

        return query

    def get_rewrite_statistics(
            self,
            original_query: str,
            rewrites: List[str]
    ) -> Dict[str, Any]:
        """获取改写统计

        Args:
            original_query: 原始查询
            rewrites: 改写列表

        Returns:
            统计信息
        """
        import jieba

        original_tokens = set(jieba.cut(original_query))

        stats = {
            "original_length": len(original_query),
            "num_rewrites": len(rewrites),
            "rewrite_stats": []
        }

        for rewrite in rewrites:
            rewrite_tokens = set(jieba.cut(rewrite))

            overlap = original_tokens & rewrite_tokens
            new_terms = rewrite_tokens - original_tokens

            stats["rewrite_stats"].append({
                "text": rewrite,
                "length": len(rewrite),
                "overlap_ratio": len(overlap) / len(original_tokens) if original_tokens else 0,
                "new_terms": list(new_terms)
            })

        return stats


# 全局实例
query_rewriter = QueryRewriter()