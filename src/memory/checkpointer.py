"""对话检查点管理 - 持久化对话状态"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class ConversationCheckpointer:
    """对话检查点管理器"""

    def __init__(self, db_path: str = None):
        """初始化检查点管理器

        Args:
            db_path: 数据库路径
        """
        if db_path is None:
            db_path = str(Path(settings.CHECKPOINT_DIR) / "conversations.db")

        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建对话表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS conversations
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           thread_id
                           TEXT
                           NOT
                           NULL,
                           user_id
                           TEXT
                           NOT
                           NULL,
                           timestamp
                           DATETIME
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           state
                           TEXT
                           NOT
                           NULL,
                           UNIQUE
                       (
                           thread_id
                       )
                           )
                       """)

        # 创建消息表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS messages
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           thread_id
                           TEXT
                           NOT
                           NULL,
                           role
                           TEXT
                           NOT
                           NULL,
                           content
                           TEXT
                           NOT
                           NULL,
                           metadata
                           TEXT,
                           timestamp
                           DATETIME
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           thread_id
                       ) REFERENCES conversations
                       (
                           thread_id
                       )
                           )
                       """)

        # 创建索引
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_thread_id
                           ON conversations(thread_id)
                       """)

        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_messages_thread
                           ON messages(thread_id)
                       """)

        conn.commit()
        conn.close()

        logger.info(f"检查点数据库初始化完成: {self.db_path}")

    def save_checkpoint(
            self,
            thread_id: str,
            user_id: str,
            state: Dict[str, Any]
    ):
        """保存检查点

        Args:
            thread_id: 对话线程ID
            user_id: 用户ID
            state: 对话状态
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 序列化状态
            state_json = json.dumps(state, ensure_ascii=False)

            # 插入或更新
            cursor.execute("""
                           INSERT INTO conversations (thread_id, user_id, state, timestamp)
                           VALUES (?, ?, ?, ?) ON CONFLICT(thread_id) 
            DO
                           UPDATE SET
                               state = excluded.state,
                               timestamp = excluded.timestamp
                           """, (thread_id, user_id, state_json, datetime.now()))

            conn.commit()
            conn.close()

            logger.debug(f"检查点已保存: {thread_id}")

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")

    def load_checkpoint(
            self,
            thread_id: str
    ) -> Optional[Dict[str, Any]]:
        """加载检查点

        Args:
            thread_id: 对话线程ID

        Returns:
            对话状态，如果不存在返回None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                           SELECT state
                           FROM conversations
                           WHERE thread_id = ?
                           """, (thread_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                state = json.loads(result[0])
                logger.debug(f"检查点已加载: {thread_id}")
                return state

            return None

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None

    def add_message(
            self,
            thread_id: str,
            role: str,
            content: str,
            metadata: Dict[str, Any] = None
    ):
        """添加消息到对话历史

        Args:
            thread_id: 对话线程ID
            role: 角色 ("user", "assistant")
            content: 消息内容
            metadata: 元数据
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

            cursor.execute("""
                           INSERT INTO messages (thread_id, role, content, metadata)
                           VALUES (?, ?, ?, ?)
                           """, (thread_id, role, content, metadata_json))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"添加消息失败: {e}")

    def get_conversation_history(
            self,
            thread_id: str,
            limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取对话历史

        Args:
            thread_id: 对话线程ID
            limit: 返回消息数量

        Returns:
            消息列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                           SELECT role, content, metadata, timestamp
                           FROM messages
                           WHERE thread_id = ?
                           ORDER BY timestamp DESC
                               LIMIT ?
                           """, (thread_id, limit))

            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "role": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "timestamp": row[3]
                })

            conn.close()

            # 反转顺序（最早的在前）
            messages.reverse()
            return messages

        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []

    def delete_conversation(self, thread_id: str):
        """删除对话

        Args:
            thread_id: 对话线程ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))

            conn.commit()
            conn.close()

            logger.info(f"对话已删除: {thread_id}")

        except Exception as e:
            logger.error(f"删除对话失败: {e}")

    def get_user_conversations(
            self,
            user_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取用户的所有对话

        Args:
            user_id: 用户ID
            limit: 返回数量

        Returns:
            对话列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                           SELECT thread_id, timestamp
                           FROM conversations
                           WHERE user_id = ?
                           ORDER BY timestamp DESC
                               LIMIT ?
                           """, (user_id, limit))

            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    "thread_id": row[0],
                    "timestamp": row[1]
                })

            conn.close()
            return conversations

        except Exception as e:
            logger.error(f"获取用户对话失败: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 总对话数
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]

            # 总消息数
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]

            # 活跃用户数
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
            active_users = cursor.fetchone()[0]

            conn.close()

            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "active_users": active_users
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# 全局实例
conversation_checkpointer = ConversationCheckpointer()