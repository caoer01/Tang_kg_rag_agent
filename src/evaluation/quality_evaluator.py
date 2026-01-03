"""答案质量评估模块"""
from typing import Dict, Any, List
import logging
from src.llm.qwen_client import llm_client
from config.prompts import QUALITY_EVALUATION_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """质量评估器"""

    def __init__(self):
        """初始化评估器"""
        # 评估维度及其权重
        self.dimensions = {
            "relevance": 0.25,  # 相关性
            "accuracy": 0.25,  # 准确性
            "completeness": 0.20,  # 完整性
            "professionalism": 0.15,  # 专业性
            "readability": 0.15  # 可读性
        }

        self.quality_threshold = settings.QUALITY_THRESHOLD

    def evaluate(
            self,
            question: str,
            answer: str,
            context: str,
            method: str = "llm"
    ) -> Dict[str, Any]:
        """评估答案质量

        Args:
            question: 用户问题
            answer: 生成的答案
            context: 参考上下文
            method: 评估方法 ("llm", "heuristic", "hybrid")

        Returns:
            评估结果字典
        """
        if method == "llm":
            return self._evaluate_with_llm(question, answer, context)
        elif method == "heuristic":
            return self._evaluate_with_heuristics(question, answer, context)
        elif method == "hybrid":
            # 结合两种方法
            llm_eval = self._evaluate_with_llm(question, answer, context)
            heuristic_eval = self._evaluate_with_heuristics(question, answer, context)
            return self._merge_evaluations(llm_eval, heuristic_eval)
        else:
            logger.warning(f"未知的评估方法: {method}")
            return self._evaluate_with_heuristics(question, answer, context)

    def _evaluate_with_llm(
            self,
            question: str,
            answer: str,
            context: str
    ) -> Dict[str, Any]:
        """使用LLM评估

        Args:
            question: 问题
            answer: 答案
            context: 上下文

        Returns:
            评估结果
        """
        try:
            prompt = QUALITY_EVALUATION_PROMPT.format(
                question=question,
                answer=answer,
                context=context
            )

            result = llm_client.generate_json(prompt, temperature=0.1)

            # 验证结果格式
            required_keys = [
                "overall_score", "relevance", "accuracy",
                "completeness", "professionalism", "readability"
            ]

            for key in required_keys:
                if key not in result:
                    result[key] = 0.5  # 默认值

            # 确保所有分数在0-1范围内
            for key in required_keys:
                result[key] = max(0.0, min(1.0, float(result[key])))

            return result

        except Exception as e:
            logger.error(f"LLM评估失败: {e}")
            # 返回默认评分
            return {
                "overall_score": 0.5,
                "relevance": 0.5,
                "accuracy": 0.5,
                "completeness": 0.5,
                "professionalism": 0.5,
                "readability": 0.5,
                "issues": [f"评估失败: {str(e)}"],
                "suggestions": "无法自动评估"
            }

    def _evaluate_with_heuristics(
            self,
            question: str,
            answer: str,
            context: str
    ) -> Dict[str, Any]:
        """使用启发式规则评估

        Args:
            question: 问题
            answer: 答案
            context: 上下文

        Returns:
            评估结果
        """
        scores = {}
        issues = []

        # 1. 相关性评估
        relevance = self._check_relevance(question, answer)
        scores["relevance"] = relevance
        if relevance < 0.6:
            issues.append("答案与问题相关性较低")

        # 2. 准确性评估
        accuracy = self._check_accuracy(answer, context)
        scores["accuracy"] = accuracy
        if accuracy < 0.6:
            issues.append("答案可能包含未基于上下文的内容")

        # 3. 完整性评估
        completeness = self._check_completeness(question, answer)
        scores["completeness"] = completeness
        if completeness < 0.6:
            issues.append("答案不够完整")

        # 4. 专业性评估
        professionalism = self._check_professionalism(answer)
        scores["professionalism"] = professionalism
        if professionalism < 0.6:
            issues.append("医学术语使用不够专业")

        # 5. 可读性评估
        readability = self._check_readability(answer)
        scores["readability"] = readability
        if readability < 0.6:
            issues.append("答案可读性较差")

        # 计算总分
        overall = sum(
            scores[dim] * weight
            for dim, weight in self.dimensions.items()
        )

        return {
            "overall_score": overall,
            "relevance": scores["relevance"],
            "accuracy": scores["accuracy"],
            "completeness": scores["completeness"],
            "professionalism": scores["professionalism"],
            "readability": scores["readability"],
            "issues": issues,
            "suggestions": self._generate_suggestions(scores, issues)
        }

    def _check_relevance(self, question: str, answer: str) -> float:
        """检查相关性

        Args:
            question: 问题
            answer: 答案

        Returns:
            相关性分数 (0-1)
        """
        import jieba

        # 提取关键词
        question_tokens = set(jieba.cut(question))
        answer_tokens = set(jieba.cut(answer))

        # 停用词
        stopwords = {"的", "是", "在", "有", "和", "了", "吗", "呢", "吧", "啊"}
        question_tokens -= stopwords
        answer_tokens -= stopwords

        # 计算重叠
        if not question_tokens:
            return 0.5

        overlap = question_tokens & answer_tokens
        relevance = len(overlap) / len(question_tokens)

        # 检查是否直接回答
        question_keywords = ["什么", "如何", "为什么", "哪些", "怎么"]
        has_question_word = any(kw in question for kw in question_keywords)

        # 如果是疑问句但答案太短，降低分数
        if has_question_word and len(answer) < 20:
            relevance *= 0.5

        return min(1.0, relevance)

    def _check_accuracy(self, answer: str, context: str) -> float:
        """检查准确性（是否基于上下文）

        Args:
            answer: 答案
            context: 上下文

        Returns:
            准确性分数 (0-1)
        """
        import jieba

        if not context:
            return 0.5  # 无上下文时无法判断

        # 提取答案中的关键信息
        answer_tokens = set(jieba.cut(answer))
        context_tokens = set(jieba.cut(context))

        # 停用词
        stopwords = {"的", "是", "在", "有", "和", "了", "等", "及"}
        answer_tokens -= stopwords
        context_tokens -= stopwords

        # 计算答案中有多少内容来自上下文
        if not answer_tokens:
            return 0.5

        grounded = answer_tokens & context_tokens
        accuracy = len(grounded) / len(answer_tokens)

        # 检查幻觉指标
        hallucination_patterns = [
            "我认为", "可能是", "应该是", "据我所知",
            "一般来说", "通常情况下"
        ]

        has_hedging = sum(1 for p in hallucination_patterns if p in answer)

        # 如果有太多不确定表达，降低分数
        if has_hedging > 2:
            accuracy *= 0.8

        return min(1.0, accuracy)

    def _check_completeness(self, question: str, answer: str) -> float:
        """检查完整性

        Args:
            question: 问题
            answer: 答案

        Returns:
            完整性分数 (0-1)
        """
        # 基于长度的简单启发
        min_length = 30
        ideal_length = 150

        answer_length = len(answer)

        if answer_length < min_length:
            return 0.3
        elif answer_length < ideal_length:
            return 0.5 + (answer_length - min_length) / (ideal_length - min_length) * 0.3
        else:
            return 0.8

    def _check_professionalism(self, answer: str) -> float:
        """检查专业性

        Args:
            answer: 答案

        Returns:
            专业性分数 (0-1)
        """
        # 医学术语列表（示例）
        medical_terms = [
            "糖尿病", "血糖", "胰岛素", "并发症", "治疗",
            "症状", "诊断", "检查", "药物", "血压",
            "代谢", "激素", "器官", "疾病", "患者"
        ]

        # 统计医学术语出现次数
        term_count = sum(1 for term in medical_terms if term in answer)

        # 基于术语密度评分
        words = len(answer)
        if words == 0:
            return 0.0

        term_density = term_count / (words / 10)  # 每10字的术语数

        score = min(1.0, term_density / 2)  # 归一化

        # 检查非专业表达
        informal_patterns = ["哈哈", "呵呵", "嘻嘻", "哎呀"]
        has_informal = any(p in answer for p in informal_patterns)

        if has_informal:
            score *= 0.5

        return score

    def _check_readability(self, answer: str) -> float:
        """检查可读性

        Args:
            answer: 答案

        Returns:
            可读性分数 (0-1)
        """
        # 句子数量
        import re
        sentences = re.split(r'[。!?！？]', answer)
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # 平均句子长度
        avg_sentence_length = len(answer) / len(sentences)

        # 理想句子长度：20-50字
        if 20 <= avg_sentence_length <= 50:
            length_score = 1.0
        elif avg_sentence_length < 20:
            length_score = avg_sentence_length / 20
        else:
            length_score = max(0.3, 1.0 - (avg_sentence_length - 50) / 100)

        # 段落结构
        paragraphs = answer.split('\n')
        has_structure = len(paragraphs) > 1
        structure_score = 1.0 if has_structure else 0.7

        # 综合评分
        readability = 0.6 * length_score + 0.4 * structure_score

        return readability

    def _merge_evaluations(
            self,
            eval1: Dict[str, Any],
            eval2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """合并两个评估结果

        Args:
            eval1: 第一个评估结果
            eval2: 第二个评估结果

        Returns:
            合并后的评估结果
        """
        merged = {}

        # 对每个维度取平均
        for dim in self.dimensions.keys():
            score1 = eval1.get(dim, 0.5)
            score2 = eval2.get(dim, 0.5)
            merged[dim] = (score1 + score2) / 2

        # 总分
        merged["overall_score"] = (
                                          eval1.get("overall_score", 0.5) +
                                          eval2.get("overall_score", 0.5)
                                  ) / 2

        # 合并问题和建议
        issues1 = eval1.get("issues", [])
        issues2 = eval2.get("issues", [])
        merged["issues"] = list(set(issues1 + issues2))

        merged["suggestions"] = eval1.get("suggestions", "") or eval2.get("suggestions", "")

        return merged

    def _generate_suggestions(
            self,
            scores: Dict[str, float],
            issues: List[str]
    ) -> str:
        """生成改进建议

        Args:
            scores: 各维度分数
            issues: 问题列表

        Returns:
            改进建议
        """
        suggestions = []

        if scores.get("relevance", 1) < 0.6:
            suggestions.append("建议更直接地回答问题核心")

        if scores.get("accuracy", 1) < 0.6:
            suggestions.append("建议更多地基于提供的上下文信息")

        if scores.get("completeness", 1) < 0.6:
            suggestions.append("建议补充更多细节和说明")

        if scores.get("professionalism", 1) < 0.6:
            suggestions.append("建议使用更专业的医学术语")

        if scores.get("readability", 1) < 0.6:
            suggestions.append("建议优化句子结构，提高可读性")

        return "；".join(suggestions) if suggestions else "答案质量良好"

    def is_acceptable(self, evaluation: Dict[str, Any]) -> bool:
        """判断答案是否可接受

        Args:
            evaluation: 评估结果

        Returns:
            是否可接受
        """
        overall_score = evaluation.get("overall_score", 0)
        return overall_score >= self.quality_threshold

    def get_improvement_priority(
            self,
            evaluation: Dict[str, Any]
    ) -> List[str]:
        """获取改进优先级

        Args:
            evaluation: 评估结果

        Returns:
            按优先级排序的维度列表
        """
        # 按分数从低到高排序
        dimension_scores = [
            (dim, evaluation.get(dim, 1.0))
            for dim in self.dimensions.keys()
        ]

        dimension_scores.sort(key=lambda x: x[1])

        return [dim for dim, score in dimension_scores if score < 0.7]


# 全局实例
quality_evaluator = QualityEvaluator()