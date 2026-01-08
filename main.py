"""糖尿病智能问答系统主程序 - 完整记忆功能版本"""
from src.graph_workflow.workflow import compile_workflow
from src.graph_workflow.state import DiabetesQAState
from langgraph.checkpoint.sqlite import SqliteSaver
from src.memory.checkpointer import conversation_checkpointer
from src.memory.user_store import user_store
from config.settings import settings
import logging
from pathlib import Path
from datetime import datetime

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
    """糖尿病问答系统 - 完整记忆功能版"""

    def __init__(self):
        """初始化系统"""
        logger.info("初始化糖尿病问答系统（完整记忆版）")

        # 第1层：工作流状态记忆 (LangGraph)
        checkpoint_path = Path(settings.CHECKPOINT_DIR) / "checkpoints.db"
        self.langgraph_checkpointer = SqliteSaver.from_conn_string(str(checkpoint_path))

        # 第2层：对话历史记忆 (自定义)
        self.conversation_checkpointer = conversation_checkpointer

        # 第3层：用户画像记忆
        self.user_store = user_store

        # 编译工作流
        self.workflow = compile_workflow(self.langgraph_checkpointer)

        logger.info("三层记忆系统初始化完成")
        logger.info(f"  - 工作流状态: {checkpoint_path}")
        logger.info(f"  - 对话历史: {self.conversation_checkpointer.db_path}")
        logger.info(f"  - 用户画像: {self.user_store.db_path}")

    def ask(
            self,
            question: str,
            user_id: str = "default_user",
            thread_id: str = "default_thread",
            use_history: bool = True,
            use_profile: bool = True
    ) -> dict:
        """提问（集成三层记忆）

        Args:
            question: 用户问题
            user_id: 用户ID
            thread_id: 对话线程ID
            use_history: 是否使用对话历史
            use_profile: 是否使用用户画像

        Returns:
            回答结果字典
        """
        logger.info(f"收到问题: {question}")
        logger.info(f"用户ID: {user_id}, 线程ID: {thread_id}")

        try:
            # ===== 第1步：加载记忆 =====

            # 加载对话历史（第2层）
            conversation_history = []
            if use_history:
                history_messages = self.conversation_checkpointer.get_conversation_history(
                    thread_id,
                    limit=10  # 最近10轮对话
                )
                conversation_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in history_messages
                ]
                logger.info(f"加载对话历史: {len(conversation_history)} 条消息")

            # 加载用户画像（第3层）
            user_profile = None
            if use_profile:
                user_profile = self.user_store.get_user(user_id)
                if user_profile:
                    logger.info(f"加载用户画像: {user_profile['diabetes_type']}")
                else:
                    # 创建新用户
                    self.user_store.create_or_update_user(user_id)
                    logger.info(f"创建新用户: {user_id}")

            # ===== 第2步：优化查询（利用历史和画像） =====

            optimized_question = self._optimize_question_with_context(
                question,
                conversation_history,
                user_profile
            )

            # ===== 第3步：初始化工作流状态 =====

            initial_state = {
                "question": optimized_question,
                "original_question": question,  # 保留原始问题
                "user_id": user_id,
                "messages": conversation_history,  # 注入对话历史
                "user_profile": user_profile,  # 注入用户画像
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

            # ===== 第4步：执行工作流 =====

            result = self.workflow.invoke(initial_state, config)

            # ===== 第5步：保存到三层记忆 =====

            # 第1层：LangGraph 自动保存工作流状态 ✅

            # 第2层：保存对话消息
            self.conversation_checkpointer.add_message(
                thread_id=thread_id,
                role="user",
                content=question,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                }
            )

            self.conversation_checkpointer.add_message(
                thread_id=thread_id,
                role="assistant",
                content=result.get("answer", ""),
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "quality_score": result.get("quality_eval", {}).get("overall_score", 0),
                    "is_complex": result.get("is_complex_question", False)
                }
            )

            # 第3层：更新用户画像
            # 记录查询
            self.user_store.record_query(
                user_id=user_id,
                query=question,
                category=self._categorize_question(question)
            )

            # 提取并更新主题兴趣
            topics = self._extract_topics(question, result)
            if topics:
                self.user_store.update_topic_interest(user_id, topics)

            # ===== 第6步：格式化返回结果 =====

            response = {
                "answer": result.get("answer", ""),
                "question": question,
                "optimized_question": optimized_question if optimized_question != question else None,
                "is_complex": result.get("is_complex_question", False),
                "quality_score": result.get("quality_eval", {}).get("overall_score", 0),
                "retry_count": result.get("retry_count", 0),
                "step_logs": result.get("step_logs", []),
                "sources": self._extract_sources(result),
                "conversation_context": len(conversation_history) > 0,
                "user_profile_used": user_profile is not None,
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

    def _optimize_question_with_context(
            self,
            question: str,
            conversation_history: list,
            user_profile: dict
    ) -> str:
        """利用对话历史和用户画像优化问题

        Args:
            question: 原始问题
            conversation_history: 对话历史
            user_profile: 用户画像

        Returns:
            优化后的问题
        """
        # 1. 处理代词引用（利用对话历史）
        if conversation_history and self._has_pronoun(question):
            # 从历史中提取主题
            recent_context = self._extract_context_from_history(conversation_history[-4:])
            if recent_context:
                optimized = f"{recent_context} {question}"
                logger.info(f"问题上下文化: {question} -> {optimized}")
                return optimized

        # 2. 添加用户相关信息（利用用户画像）
        if user_profile:
            diabetes_type = user_profile.get("diabetes_type")
            if diabetes_type and diabetes_type not in question:
                # 如果问题没有指定类型，添加用户的糖尿病类型
                logger.info(f"添加用户画像信息: {diabetes_type}")
                return f"{question}（针对{diabetes_type}患者）"

        return question

    def _has_pronoun(self, text: str) -> bool:
        """检测文本中是否有代词"""
        pronouns = ["它", "这个", "那个", "这些", "那些", "他", "她"]
        return any(p in text for p in pronouns)

    def _extract_context_from_history(self, history: list) -> str:
        """从对话历史中提取上下文"""
        # 提取最近提到的实体
        import jieba

        entities = []
        for msg in history:
            if msg["role"] == "user":
                # 简单的实体提取
                tokens = jieba.cut(msg["content"])
                # 假设名词是实体
                entities.extend([t for t in tokens if len(t) > 1])

        # 返回最常见的实体
        if entities:
            from collections import Counter
            most_common = Counter(entities).most_common(1)[0][0]
            return most_common

        return ""

    def _categorize_question(self, question: str) -> str:
        """问题分类（用于用户画像）"""
        categories = {
            "症状": ["症状", "表现", "感觉"],
            "治疗": ["治疗", "药物", "胰岛素", "吃药"],
            "饮食": ["饮食", "吃", "食物", "营养"],
            "运动": ["运动", "锻炼", "活动"],
            "检查": ["检查", "检测", "化验", "血糖"],
            "并发症": ["并发症", "并发", "合并症"],
        }

        for category, keywords in categories.items():
            if any(kw in question for kw in keywords):
                return category

        return "其他"

    def _extract_topics(self, question: str, result: dict) -> list:
        """提取问题主题（用于用户画像）"""
        # 从问题和答案中提取主题
        topics = []

        # 简单的关键词提取
        import jieba.analyse

        # 从问题提取
        question_keywords = jieba.analyse.extract_tags(question, topK=3)
        topics.extend(question_keywords)

        # 从实体识别结果提取
        if result.get("question_analysis"):
            entities = result["question_analysis"].get("identified_entities", [])
            topics.extend(entities)

        # 去重
        return list(set(topics))[:5]

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
        """获取对话历史（从第2层记忆）

        Args:
            thread_id: 对话线程ID

        Returns:
            对话历史列表
        """
        return self.conversation_checkpointer.get_conversation_history(thread_id)

    def get_user_profile(self, user_id: str) -> dict:
        """获取用户画像（从第3层记忆）

        Args:
            user_id: 用户ID

        Returns:
            用户画像字典
        """
        return self.user_store.get_user(user_id) or {}

    def get_user_behavior_analysis(self, user_id: str) -> dict:
        """获取用户行为分析

        Args:
            user_id: 用户ID

        Returns:
            行为分析结果
        """
        return self.user_store.analyze_user_behavior(user_id)

    def get_personalized_recommendations(self, user_id: str) -> list:
        """获取个性化推荐

        Args:
            user_id: 用户ID

        Returns:
            推荐内容列表
        """
        return self.user_store.get_personalized_recommendations(user_id)

    def update_user_info(
            self,
            user_id: str,
            diabetes_type: str = None,
            preferences: dict = None,
            profile: dict = None
    ):
        """更新用户信息

        Args:
            user_id: 用户ID
            diabetes_type: 糖尿病类型
            preferences: 用户偏好
            profile: 用户档案
        """
        self.user_store.create_or_update_user(
            user_id=user_id,
            diabetes_type=diabetes_type,
            preferences=preferences,
            profile=profile
        )
        logger.info(f"用户信息已更新: {user_id}")

    def clear_conversation(self, thread_id: str):
        """清除对话历史

        Args:
            thread_id: 对话线程ID
        """
        self.conversation_checkpointer.delete_conversation(thread_id)
        logger.info(f"对话历史已清除: {thread_id}")

    def get_system_statistics(self) -> dict:
        """获取系统统计信息

        Returns:
            统计信息字典
        """
        return {
            "conversations": self.conversation_checkpointer.get_statistics(),
            "users": self.user_store.get_statistics()
        }


def interactive_mode():
    """交互模式（完整记忆版）"""
    print("=" * 60)
    print("糖尿病智能问答系统（完整记忆版）")
    print("=" * 60)
    print("功能说明:")
    print("  - 自动记忆对话历史")
    print("  - 理解上下文和代词引用")
    print("  - 基于用户画像个性化回答")
    print("\n命令:")
    print("  quit     - 退出系统")
    print("  history  - 查看对话历史")
    print("  profile  - 查看用户画像")
    print("  analyze  - 查看行为分析")
    print("  recommend- 获取个性化推荐")
    print("  clear    - 清除对话历史")
    print("  settype  - 设置糖尿病类型")
    print("=" * 60)

    system = DiabetesQASystem()

    # 用户信息
    user_id = input("\n请输入用户ID (按回车使用默认): ").strip() or "demo_user"
    thread_id = f"session_{user_id}"

    # 检查是否是新用户
    user_profile = system.get_user_profile(user_id)
    if not user_profile:
        print(f"\n欢迎新用户: {user_id}")
        diabetes_type = input("请选择糖尿病类型 (1型/2型/妊娠期，按回车跳过): ").strip()
        if diabetes_type:
            system.update_user_info(user_id, diabetes_type=diabetes_type)
            print(f"✓ 已设置为: {diabetes_type}")
    else:
        print(f"\n欢迎回来: {user_id}")
        if user_profile.get("diabetes_type"):
            print(f"糖尿病类型: {user_profile['diabetes_type']}")
        print(f"总交互次数: {user_profile['interaction_count']}")

    while True:
        try:
            question = input("\n您的问题: ").strip()

            if not question:
                continue

            # 处理命令
            if question.lower() == 'quit':
                print("再见!")
                break

            elif question.lower() == 'history':
                history = system.get_conversation_history(thread_id)
                print(f"\n对话历史 (共{len(history)}条):")
                for msg in history[-10:]:  # 显示最近10条
                    role = "用户" if msg["role"] == "user" else "助手"
                    print(f"  [{role}] {msg['content'][:50]}...")
                continue

            elif question.lower() == 'profile':
                profile = system.get_user_profile(user_id)
                print("\n用户画像:")
                for key, value in profile.items():
                    print(f"  {key}: {value}")
                continue

            elif question.lower() == 'analyze':
                analysis = system.get_user_behavior_analysis(user_id)
                print("\n行为分析:")
                print(f"  总交互次数: {analysis.get('total_interactions', 0)}")
                print(f"  常问主题: {', '.join(analysis.get('frequent_topics', []))}")
                print(f"  活跃时间段: {analysis.get('active_hours', [])}")
                continue

            elif question.lower() == 'recommend':
                recommendations = system.get_personalized_recommendations(user_id)
                print("\n个性化推荐:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
                continue

            elif question.lower() == 'clear':
                system.clear_conversation(thread_id)
                print("✓ 对话历史已清除")
                continue

            elif question.lower() == 'settype':
                diabetes_type = input("请输入糖尿病类型 (1型/2型/妊娠期): ").strip()
                if diabetes_type:
                    system.update_user_info(user_id, diabetes_type=diabetes_type)
                    print(f"✓ 已更新为: {diabetes_type}")
                continue

            # 提问
            print("\n处理中...")
            response = system.ask(
                question,
                user_id=user_id,
                thread_id=thread_id,
                use_history=True,
                use_profile=True
            )

            if response["success"]:
                print(f"\n回答: {response['answer']}")
                print(f"\n类型: {'复杂' if response['is_complex'] else '简单'} | "
                      f"质量: {response['quality_score']:.2f} | "
                      f"重试: {response['retry_count']}")

                if response.get("optimized_question"):
                    print(f"优化后的问题: {response['optimized_question']}")

                if response.get("conversation_context"):
                    print("✓ 已利用对话历史")

                if response.get("user_profile_used"):
                    print("✓ 已使用用户画像")
            else:
                print(f"\n错误: {response.get('error', '未知错误')}")

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            logger.error(f"交互模式错误: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="糖尿病智能问答系统（完整记忆版）")
    parser.add_argument("--mode", choices=["interactive"], default="interactive")

    args = parser.parse_args()

    if args.mode == "interactive":
        interactive_mode()
