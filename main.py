"""糖尿病智能问答系统主程序"""
from src.graph_workflow.workflow import compile_workflow
from src.graph_workflow.state import DiabetesQAState
from langgraph.checkpoint.sqlite import SqliteSaver
from config.settings import settings
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diabetes_qa.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DiabetesQASystem:
    """糖尿病问答系统"""

    def __init__(self):
        """初始化系统"""
        logger.info("初始化糖尿病问答系统")

        # 创建检查点保存器（使用SQLite持久化）
        checkpoint_path = Path(settings.CHECKPOINT_DIR) / "checkpoints.db"
        self.checkpointer = SqliteSaver.from_conn_string(str(checkpoint_path))

        # 编译工作流
        self.workflow = compile_workflow(self.checkpointer)

        logger.info("系统初始化完成")

    def ask(
            self,
            question: str,
            user_id: str = "default_user",
            thread_id: str = "default_thread"
    ) -> dict:
        """提问

        Args:
            question: 用户问题
            user_id: 用户ID
            thread_id: 对话线程ID

        Returns:
            回答结果字典
        """
        logger.info(f"收到问题: {question}")
        logger.info(f"用户ID: {user_id}, 线程ID: {thread_id}")

        # 初始化状态
        initial_state = {
            "question": question,
            "user_id": user_id,
            "messages": [],
            "retrieval_results": [],
            "rewritten_queries": [],
            "retry_count": 0,
            "is_complex_question": False,
            "should_retry": False,
            "should_use_external_search": False,
            "step_logs": [],
            "answer": "",
            "context": ""
        }

        # 配置（包含thread_id用于持久化）
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }

        try:
            # 执行工作流
            result = self.workflow.invoke(initial_state, config)

            # 格式化返回结果
            response = {
                "answer": result.get("answer", ""),
                "question": question,
                "is_complex": result.get("is_complex_question", False),
                "quality_score": result.get("quality_eval", {}).get("overall_score", 0),
                "retry_count": result.get("retry_count", 0),
                "step_logs": result.get("step_logs", []),
                "sources": self._extract_sources(result),
                "success": True
            }

            logger.info(f"回答完成，质量评分: {response['quality_score']:.2f}")
            return response

        except Exception as e:
            logger.error(f"处理问题失败: {e}", exc_info=True)
            return {
                "answer": "抱歉，处理您的问题时出现错误。",
                "question": question,
                "error": str(e),
                "success": False
            }

    def _extract_sources(self, result: dict) -> list:
        """提取引用来源"""
        sources = []

        # 从检索结果中提取
        for res in result.get("retrieval_results", [])[:3]:
            metadata = res.get("metadata", {})
            sources.append({
                "type": metadata.get("type", "document"),
                "source": metadata.get("source", ""),
                "page": metadata.get("page_num")
            })

        # 从图谱结果中提取
        if result.get("graph_results"):
            sources.append({
                "type": "knowledge_graph",
                "description": "知识图谱查询"
            })

        return sources

    def get_conversation_history(self, thread_id: str) -> list:
        """获取对话历史

        Args:
            thread_id: 对话线程ID

        Returns:
            对话历史列表
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.workflow.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []

    def clear_conversation(self, thread_id: str):
        """清除对话历史

        Args:
            thread_id: 对话线程ID
        """
        try:
            # LangGraph会自动管理状态，这里可以选择性实现清除逻辑
            logger.info(f"清除对话历史: {thread_id}")
        except Exception as e:
            logger.error(f"清除对话历史失败: {e}")


def interactive_mode():
    """交互模式"""
    print("=" * 60)
    print("糖尿病智能问答系统")
    print("=" * 60)
    print("输入问题开始对话，输入 'quit' 退出")
    print("输入 'history' 查看对话历史")
    print("输入 'clear' 清除对话历史")
    print("=" * 60)

    system = DiabetesQASystem()
    thread_id = "interactive_session"

    while True:
        try:
            question = input("\n您的问题: ").strip()

            if not question:
                continue

            if question.lower() == 'quit':
                print("再见!")
                break

            if question.lower() == 'history':
                history = system.get_conversation_history(thread_id)
                print("\n对话历史:")
                for msg in history:
                    print(f"  {msg}")
                continue

            if question.lower() == 'clear':
                system.clear_conversation(thread_id)
                print("对话历史已清除")
                continue

            # 提问
            print("\n处理中...")
            response = system.ask(question, thread_id=thread_id)

            if response["success"]:
                print(f"\n回答: {response['answer']}")
                print(f"\n问题类型: {'复杂' if response['is_complex'] else '简单'}")
                print(f"质量评分: {response['quality_score']:.2f}")
                print(f"重试次数: {response['retry_count']}")

                if response.get("sources"):
                    print("\n参考来源:")
                    for source in response["sources"]:
                        print(f"  - {source}")
            else:
                print(f"\n错误: {response.get('error', '未知错误')}")

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            logger.error(f"交互模式错误: {e}", exc_info=True)


def batch_mode(questions_file: str, output_file: str):
    """批量处理模式

    Args:
        questions_file: 问题文件路径（每行一个问题）
        output_file: 输出文件路径
    """
    import json

    logger.info(f"批量处理模式: {questions_file} -> {output_file}")

    system = DiabetesQASystem()

    # 读取问题
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]

    logger.info(f"共 {len(questions)} 个问题")

    # 处理问题
    results = []
    for i, question in enumerate(questions, 1):
        logger.info(f"处理问题 {i}/{len(questions)}: {question}")
        response = system.ask(question, thread_id=f"batch_{i}")
        results.append(response)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="糖尿病智能问答系统")
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="运行模式"
    )
    parser.add_argument(
        "--input",
        help="批量模式输入文件"
    )
    parser.add_argument(
        "--output",
        help="批量模式输出文件"
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        interactive_mode()
    elif args.mode == "batch":
        if not args.input or not args.output:
            print("批量模式需要指定 --input 和 --output")
            exit(1)
        batch_mode(args.input, args.output)