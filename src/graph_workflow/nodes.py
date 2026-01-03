"""LangGraph工作流节点定义"""
from typing import Dict, Any
import logging
from config.prompts import *
from config.settings import settings
from src.llm.qwen_client import llm_client
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.knowledge_graph.neo4j_client import neo4j_client
from src.graph_workflow.state import DiabetesQAState

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """工作流节点集合"""

    @staticmethod
    def analyze_question_complexity(state: DiabetesQAState) -> DiabetesQAState:
        """节点1: 分析问题复杂度"""
        logger.info(f"分析问题复杂度: {state['question']}")

        try:
            prompt = COMPLEXITY_ANALYSIS_PROMPT.format(question=state["question"])
            result = llm_client.generate_json(prompt)

            state["question_analysis"] = result
            state["is_complex_question"] = result["is_complex"]
            state["step_logs"].append(f"复杂度分析: {result['reasoning']}")

            logger.info(f"问题分类: {'复杂' if result['is_complex'] else '简单'}")
        except Exception as e:
            logger.error(f"复杂度分析失败: {e}")
            state["error"] = str(e)
            state["is_complex_question"] = False

        return state

    @staticmethod
    def simple_rag_retrieval(state: DiabetesQAState) -> DiabetesQAState:
        """节点2: 简单问题RAG检索"""
        logger.info("执行简单RAG检索")

        try:
            question = state["question"]

            # 混合检索
            results = hybrid_retriever.hybrid_search(
                query=question,
                top_k=settings.TOP_K_SIMPLE
            )

            state["retrieval_results"] = results
            state["step_logs"].append(f"检索到 {len(results)} 条相关文档")

        except Exception as e:
            logger.error(f"简单RAG检索失败: {e}")
            state["error"] = str(e)
            state["retrieval_results"] = []

        return state

    @staticmethod
    def rerank_results(state: DiabetesQAState) -> DiabetesQAState:
        """节点3: 重排序优化"""
        logger.info("执行检索结果重排序")

        try:
            question = state["question"]
            results = state["retrieval_results"]

            if not results:
                return state

            # 使用LLM进行重排序
            rerank_prompt = f"""请对以下检索结果按照与问题的相关性进行评分(0-1)。

问题: {question}

检索结果:
"""
            for i, result in enumerate(results[:10]):  # 最多重排10条
                rerank_prompt += f"\n{i + 1}. {result['text'][:200]}..."

            rerank_prompt += "\n\n请以JSON格式返回每条结果的评分: {\"scores\": [score1, score2, ...]}"

            rerank_result = llm_client.generate_json(rerank_prompt)
            scores = rerank_result.get("scores", [])

            # 更新分数
            for i, score in enumerate(scores[:len(results)]):
                results[i]["rerank_score"] = score

            # 重新排序
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            state["retrieval_results"] = results[:settings.RERANK_TOP_K]

            state["step_logs"].append(f"重排序后保留 {len(state['retrieval_results'])} 条结果")

        except Exception as e:
            logger.warning(f"重排序失败，使用原始结果: {e}")

        return state

    @staticmethod
    def extract_entities(state: DiabetesQAState) -> DiabetesQAState:
        """节点4: 提取实体（用于图谱查询）"""
        logger.info("提取问题中的实体")

        try:
            question = state["question"]
            prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
            result = llm_client.generate_json(prompt)

            entities = result.get("entities", [])
            if state["question_analysis"]:
                state["question_analysis"]["identified_entities"] = [e["text"] for e in entities]

            state["step_logs"].append(f"识别实体: {[e['text'] for e in entities]}")

        except Exception as e:
            logger.error(f"实体提取失败: {e}")

        return state

    @staticmethod
    def query_knowledge_graph(state: DiabetesQAState) -> DiabetesQAState:
        """节点5: 查询知识图谱"""
        logger.info("查询知识图谱")

        try:
            entities = state["question_analysis"]["identified_entities"]
            question = state["question"]

            if not entities:
                state["step_logs"].append("未识别到实体，跳过图谱查询")
                return state

            # 生成Cypher查询
            cypher_prompt = CYPHER_GENERATION_PROMPT.format(
                entities=entities,
                question=question
            )
            cypher_result = llm_client.generate_json(cypher_prompt)
            cypher_query = cypher_result.get("cypher", "")

            # 执行查询
            if cypher_query:
                graph_data = neo4j_client.execute_query(cypher_query)

                # 总结图谱结果
                summary_prompt = GRAPH_SUMMARY_PROMPT.format(
                    question=question,
                    graph_results=graph_data
                )
                summary = llm_client.generate(summary_prompt)

                state["graph_results"] = {
                    "cypher": cypher_query,
                    "raw_results": graph_data,
                    "summary": summary
                }

                state["step_logs"].append(f"图谱查询: 找到 {len(graph_data)} 条结果")

        except Exception as e:
            logger.error(f"图谱查询失败: {e}")
            state["graph_results"] = None

        return state

    @staticmethod
    def complex_rag_retrieval(state: DiabetesQAState) -> DiabetesQAState:
        """节点6: 复杂问题RAG检索"""
        logger.info("执行复杂RAG检索")

        try:
            question = state["question"]

            # 检索更多文档
            results = hybrid_retriever.hybrid_search(
                query=question,
                top_k=settings.TOP_K_COMPLEX
            )

            state["retrieval_results"] = results
            state["step_logs"].append(f"复杂检索: {len(results)} 条文档")

        except Exception as e:
            logger.error(f"复杂RAG检索失败: {e}")
            state["retrieval_results"] = []

        return state

    @staticmethod
    def generate_answer(state: DiabetesQAState) -> DiabetesQAState:
        """节点7: 生成答案"""
        logger.info("生成答案")

        try:
            question = state["question"]

            # 构建上下文
            context_parts = []

            # 添加文档检索结果
            if state["retrieval_results"]:
                context_parts.append("【文档检索结果】")
                for i, result in enumerate(state["retrieval_results"][:5], 1):
                    context_parts.append(f"{i}. {result['text']}")

            # 添加图谱查询结果
            if state.get("graph_results"):
                context_parts.append("\n【知识图谱信息】")
                context_parts.append(state["graph_results"]["summary"])

            context = "\n".join(context_parts)
            state["context"] = context

            # 生成答案
            prompt = ANSWER_GENERATION_PROMPT.format(
                question=question,
                context=context
            )
            answer = llm_client.generate(prompt)

            state["answer"] = answer
            state["step_logs"].append("答案生成完成")

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            state["answer"] = "抱歉，生成答案时出现错误。"
            state["error"] = str(e)

        return state

    @staticmethod
    def evaluate_quality(state: DiabetesQAState) -> DiabetesQAState:
        """节点8: 评估答案质量"""
        logger.info("评估答案质量")

        try:
            question = state["question"]
            answer = state["answer"]
            context = state["context"]

            prompt = QUALITY_EVALUATION_PROMPT.format(
                question=question,
                answer=answer,
                context=context
            )

            eval_result = llm_client.generate_json(prompt)
            state["quality_eval"] = eval_result

            # 判断是否需要重试
            overall_score = eval_result.get("overall_score", 0)
            state["should_retry"] = (
                    overall_score < settings.QUALITY_THRESHOLD and
                    state["retry_count"] < settings.MAX_RETRY
            )

            state["step_logs"].append(f"质量评分: {overall_score:.2f}")

            if state["should_retry"]:
                state["retry_count"] += 1
                logger.info(f"质量不达标，准备重试 (第{state['retry_count']}次)")

        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            state["should_retry"] = False

        return state

    @staticmethod
    def rewrite_query(state: DiabetesQAState) -> DiabetesQAState:
        """节点9: 查询改写（用于重试）"""
        logger.info("改写查询")

        try:
            question = state["question"]
            previous_rewrites = state.get("rewritten_queries", [])

            prompt = QUERY_REWRITE_PROMPT.format(
                question=question,
                previous_rewrites=previous_rewrites,
                num_rewrites=2
            )

            result = llm_client.generate_json(prompt)
            new_rewrites = result.get("rewrites", [])

            state["rewritten_queries"].extend(new_rewrites)

            # 使用第一个改写版本重新检索
            if new_rewrites:
                state["question"] = new_rewrites[0]
                state["step_logs"].append(f"查询改写: {new_rewrites[0]}")

        except Exception as e:
            logger.error(f"查询改写失败: {e}")

        return state

    @staticmethod
    def external_search(state: DiabetesQAState) -> DiabetesQAState:
        """节点10: 外部搜索（兜底方案）"""
        logger.info("执行外部搜索")

        # 这里可以集成外部搜索API
        # 示例：调用搜索引擎API
        state["answer"] = "抱歉，我无法从现有知识库中找到满意的答案。建议您咨询专业医生。"
        state["step_logs"].append("触发外部搜索兜底")

        return state


# 导出节点实例
nodes = WorkflowNodes()