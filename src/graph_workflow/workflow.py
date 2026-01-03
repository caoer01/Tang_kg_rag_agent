"""LangGraph工作流定义"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.graph_workflow.state import DiabetesQAState
from src.graph_workflow.nodes import nodes
import logging

logger = logging.getLogger(__name__)


def create_workflow():
    """创建糖尿病问答工作流"""

    # 创建状态图
    workflow = StateGraph(DiabetesQAState)

    # 添加节点
    workflow.add_node("analyze_complexity", nodes.analyze_question_complexity)
    workflow.add_node("simple_retrieval", nodes.simple_rag_retrieval)
    workflow.add_node("rerank", nodes.rerank_results)
    workflow.add_node("extract_entities", nodes.extract_entities)
    workflow.add_node("graph_query", nodes.query_knowledge_graph)
    workflow.add_node("complex_retrieval", nodes.complex_rag_retrieval)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.add_node("evaluate_quality", nodes.evaluate_quality)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("external_search", nodes.external_search)

    # 设置入口点
    workflow.set_entry_point("analyze_complexity")

    # 定义条件路由函数
    def route_by_complexity(state: DiabetesQAState) -> str:
        """根据问题复杂度路由"""
        if state.get("error"):
            return END

        if state["is_complex_question"]:
            return "extract_entities"
        else:
            return "simple_retrieval"

    def route_after_simple_retrieval(state: DiabetesQAState) -> str:
        """简单检索后的路由"""
        if not state["retrieval_results"]:
            # 如果没有检索到结果，升级为复杂问题
            logger.info("简单检索无结果，升级为复杂问题")
            state["is_complex_question"] = True
            return "extract_entities"
        return "rerank"

    def route_after_quality_eval(state: DiabetesQAState) -> str:
        """质量评估后的路由"""
        if state["should_retry"]:
            if state["is_complex_question"]:
                return "rewrite_query"
            else:
                # 简单问题质量不达标，升级为复杂问题
                logger.info("简单问题质量不达标，升级为复杂问题")
                state["is_complex_question"] = True
                return "extract_entities"

        # 如果重试次数超过限制且质量仍不达标
        if (state["retry_count"] >= 3 and
                state["quality_eval"]["overall_score"] < 0.5):
            return "external_search"

        return END

    def route_after_rewrite(state: DiabetesQAState) -> str:
        """查询改写后的路由"""
        if state["is_complex_question"]:
            return "complex_retrieval"
        else:
            return "simple_retrieval"

    # 添加边（简单问题流程）
    workflow.add_conditional_edges(
        "analyze_complexity",
        route_by_complexity,
        {
            "simple_retrieval": "simple_retrieval",
            "extract_entities": "extract_entities",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "simple_retrieval",
        route_after_simple_retrieval,
        {
            "rerank": "rerank",
            "extract_entities": "extract_entities"
        }
    )

    workflow.add_edge("rerank", "generate_answer")

    # 添加边（复杂问题流程）
    workflow.add_edge("extract_entities", "graph_query")
    workflow.add_edge("graph_query", "complex_retrieval")
    workflow.add_edge("complex_retrieval", "generate_answer")

    # 添加边（质量评估和重试）
    workflow.add_conditional_edges(
        "evaluate_quality",
        route_after_quality_eval,
        {
            "rewrite_query": "rewrite_query",
            "extract_entities": "extract_entities",
            "external_search": "external_search",
            END: END
        }
    )

    workflow.add_edge("generate_answer", "evaluate_quality")

    workflow.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "simple_retrieval": "simple_retrieval",
            "complex_retrieval": "complex_retrieval"
        }
    )

    workflow.add_edge("external_search", END)

    return workflow


def compile_workflow(checkpointer=None):
    """编译工作流

    Args:
        checkpointer: 检查点保存器，用于持久化对话状态

    Returns:
        编译后的工作流
    """
    workflow = create_workflow()

    # 如果没有提供checkpointer，使用内存保存器
    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = workflow.compile(checkpointer=checkpointer)

    logger.info("工作流编译完成")
    return compiled


# 可视化工作流（可选）
def visualize_workflow(output_path: str = "workflow.png"):
    """生成工作流可视化图

    Args:
        output_path: 输出文件路径
    """
    try:
        workflow = create_workflow()

        # 生成Mermaid图
        mermaid_code = workflow.get_graph().draw_mermaid()

        # 保存为文本文件
        with open(output_path.replace('.png', '.mmd'), 'w', encoding='utf-8') as f:
            f.write(mermaid_code)

        logger.info(f"工作流Mermaid图已保存: {output_path.replace('.png', '.mmd')}")

        # 如果安装了graphviz，可以生成PNG
        try:
            from langgraph.graph import Graph
            img = workflow.get_graph().draw_mermaid_png()
            with open(output_path, 'wb') as f:
                f.write(img)
            logger.info(f"工作流PNG图已保存: {output_path}")
        except Exception as e:
            logger.warning(f"无法生成PNG图，请安装graphviz: {e}")

    except Exception as e:
        logger.error(f"可视化工作流失败: {e}")