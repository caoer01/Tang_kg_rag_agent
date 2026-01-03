"""用户信息存储 - 管理用户画像和偏好"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import Counter
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class UserStore:
    """用户信息存储管理器"""

    def __init__(self, db_path: str = None):
        """初始化用户存储

        Args:
            db_path: 数据库路径
        """
        if db_path is None:
            db_path = str(Path(settings.USER_STORE_DIR) / "users.db")

        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 用户表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS users
                       (
                           user_id
                           TEXT
                           PRIMARY
                           KEY,
                           diabetes_type
                           TEXT,
                           created_at
                           DATETIME
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           last_interaction
                           DATETIME,
                           interaction_count
                           INTEGER
                           DEFAULT
                           0,
                           preferences
                           TEXT,
                           profile
                           TEXT
                       )
                       """)

        # 用户查询历史表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS user_queries
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           TEXT
                           NOT
                           NULL,
                           query
                           TEXT
                           NOT
                           NULL,
                           category
                           TEXT,
                           timestamp
                           DATETIME
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           user_id
                       )
                           )
                       """)

        # 用户主题兴趣表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS user_topics
                       (
                           user_id
                           TEXT
                           NOT
                           NULL,
                           topic
                           TEXT
                           NOT
                           NULL,
                           count
                           INTEGER
                           DEFAULT
                           1,
                           last_seen
                           DATETIME
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           PRIMARY
                           KEY
                       (
                           user_id,
                           topic
                       ),
                           FOREIGN KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           user_id
                       )
                           )
                       """)

        conn.commit()
        conn.close()

        logger.info(f"用户存储数据库初始化完成: {self.db_path}")

    def create_or_update_user(
            self,
            user_id: str,
            diabetes_type: str = None,
            preferences: Dict[str, Any] = None,
            profile: Dict[str, Any] = None
    ):
        """创建或更新用户信息

        Args:
            user_id: 用户ID
            diabetes_type: 糖尿病类型
            preferences: 用户偏好
            profile: 用户档案
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            preferences_json = json.dumps(preferences, ensure_ascii=False) if preferences else None
            profile_json = json.dumps(profile, ensure_ascii=False) if profile else None

            cursor.execute("""
                           INSERT INTO users (user_id, diabetes_type, preferences, profile, last_interaction)
                           VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id) 
            DO
                           UPDATE SET
                               diabetes_type = COALESCE (excluded.diabetes_type, diabetes_type),
                               preferences = COALESCE (excluded.preferences, preferences),
                               profile = COALESCE (excluded.profile, profile),
                               last_interaction = excluded.last_interaction,
                               interaction_count = interaction_count + 1
                           """, (user_id, diabetes_type, preferences_json, profile_json, datetime.now()))

            conn.commit()
            conn.close()

            logger.debug(f"用户信息已更新: {user_id}")

        except Exception as e:
            logger.error(f"更新用户信息失败: {e}")

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息

        Args:
            user_id: 用户ID

        Returns:
            用户信息字典，如果不存在返回None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                           SELECT diabetes_type,
                                  created_at,
                                  last_interaction,
                                  interaction_count,
                                  preferences,
                                  profile
                           FROM users
                           WHERE user_id = ?
                           """, (user_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    "user_id": user_id,
                    "diabetes_type": result[0],
                    "created_at": result[1],
                    "last_interaction": result[2],
                    "interaction_count": result[3],
                    "preferences": json.loads(result[4]) if result[4] else {},
                    "profile": json.loads(result[5]) if result[5] else {}
                }

            return None

        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            return None

    def record_query(
            self,
            user_id: str,
            query: str,
            category: str = None
    ):
        """记录用户查询

        Args:
            user_id: 用户ID
            query: 查询文本
            category: 查询分类
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 记录查询
            cursor.execute("""
                           INSERT INTO user_queries (user_id, query, category)
                           VALUES (?, ?, ?)
                           """, (user_id, query, category))

            # 更新最后交互时间
            cursor.execute("""
                           UPDATE users
                           SET last_interaction  = ?,
                               interaction_count = interaction_count + 1
                           WHERE user_id = ?
                           """, (datetime.now(), user_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"记录查询失败: {e}")

    def update_topic_interest(
            self,
            user_id: str,
            topics: List[str]
    ):
        """更新用户主题兴趣

        Args:
            user_id: 用户ID
            topics: 主题列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for topic in topics:
                cursor.execute("""
                               INSERT INTO user_topics (user_id, topic, count, last_seen)
                               VALUES (?, ?, 1, ?) ON CONFLICT(user_id, topic)
                DO
                               UPDATE SET
                                   count = count + 1,
                                   last_seen = excluded.last_seen
                               """, (user_id, topic, datetime.now()))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"更新主题兴趣失败: {e}")

    def get_user_topics(
            self,
            user_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取用户感兴趣的主题

        Args:
            user_id: 用户ID
            limit: 返回数量

        Returns:
            主题列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                           SELECT topic, count, last_seen
                           FROM user_topics
                           WHERE user_id = ?
                           ORDER BY count DESC, last_seen DESC LIMIT ?
                           """, (user_id, limit))

            topics = []
            for row in cursor.fetchall():
                topics.append({
                    "topic": row[0],
                    "count": row[1],
                    "last_seen": row[2]
                })

            conn.close()
            return topics

        except Exception as e:
            logger.error(f"获取用户主题失败: {e}")
            return []

    def get_user_query_history(
            self,
            user_id: str,
            limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取用户查询历史

        Args:
            user_id: 用户ID
            limit: 返回数量

        Returns:
            查询历史列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                           SELECT query, category, timestamp
                           FROM user_queries
                           WHERE user_id = ?
                           ORDER BY timestamp DESC
                               LIMIT ?
                           """, (user_id, limit))

            queries = []
            for row in cursor.fetchall():
                queries.append({
                    "query": row[0],
                    "category": row[1],
                    "timestamp": row[2]
                })

            conn.close()
            return queries

        except Exception as e:
            logger.error(f"获取查询历史失败: {e}")
            return []

    def analyze_user_behavior(self, user_id: str) -> Dict[str, Any]:
        """分析用户行为模式

        Args:
            user_id: 用户ID

        Returns:
            行为分析结果
        """
        try:
            # 获取用户信息
            user = self.get_user(user_id)
            if not user:
                return {}

            # 获取查询历史
            queries = self.get_user_query_history(user_id, limit=100)

            # 获取主题兴趣
            topics = self.get_user_topics(user_id)

            # 分析查询频率
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 按时间段统计
            cursor.execute("""
                           SELECT strftime('%H', timestamp) as hour,
                COUNT(*) as count
                           FROM user_queries
                           WHERE user_id = ?
                           GROUP BY hour
                           ORDER BY count DESC
                               LIMIT 5
                           """, (user_id,))

            active_hours = [{"hour": row[0], "count": row[1]} for row in cursor.fetchall()]

            # 查询类别分布
            cursor.execute("""
                           SELECT category, COUNT(*) as count
                           FROM user_queries
                           WHERE user_id = ? AND category IS NOT NULL
                           GROUP BY category
                           ORDER BY count DESC
                           """, (user_id,))

            category_dist = [{"category": row[0], "count": row[1]} for row in cursor.fetchall()]

            conn.close()

            return {
                "user_id": user_id,
                "diabetes_type": user.get("diabetes_type"),
                "total_interactions": user.get("interaction_count", 0),
                "frequent_topics": [t["topic"] for t in topics[:5]],
                "active_hours": active_hours,
                "category_distribution": category_dist,
                "recent_queries": [q["query"] for q in queries[:5]]
            }

        except Exception as e:
            logger.error(f"用户行为分析失败: {e}")
            return {}

    def get_personalized_recommendations(
            self,
            user_id: str
    ) -> List[str]:
        """获取个性化推荐

        Args:
            user_id: 用户ID

        Returns:
            推荐主题列表
        """
        analysis = self.analyze_user_behavior(user_id)

        recommendations = []

        # 基于糖尿病类型推荐
        diabetes_type = analysis.get("diabetes_type")
        if diabetes_type == "1型糖尿病":
            recommendations.extend([
                "胰岛素使用技巧",
                "低血糖预防",
                "1型糖尿病运动指南"
            ])
        elif diabetes_type == "2型糖尿病":
            recommendations.extend([
                "饮食控制方法",
                "口服降糖药使用",
                "体重管理"
            ])

        # 基于兴趣主题推荐相关内容
        topics = analysis.get("frequent_topics", [])
        if "并发症" in topics:
            recommendations.append("糖尿病并发症早期预防")
        if "饮食" in topics or "营养" in topics:
            recommendations.append("糖尿病饮食搭配建议")

        return recommendations[:5]

    def delete_user(self, user_id: str):
        """删除用户数据

        Args:
            user_id: 用户ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM user_topics WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM user_queries WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

            conn.commit()
            conn.close()

            logger.info(f"用户数据已删除: {user_id}")

        except Exception as e:
            logger.error(f"删除用户数据失败: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 总用户数
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]

            # 糖尿病类型分布
            cursor.execute("""
                           SELECT diabetes_type, COUNT(*) as count
                           FROM users
                           WHERE diabetes_type IS NOT NULL
                           GROUP BY diabetes_type
                           """)
            diabetes_dist = [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]

            # 活跃用户（最近7天）
            cursor.execute("""
                           SELECT COUNT(*)
                           FROM users
                           WHERE last_interaction >= datetime('now', '-7 days')
                           """)
            active_users_7d = cursor.fetchone()[0]

            conn.close()

            return {
                "total_users": total_users,
                "diabetes_distribution": diabetes_dist,
                "active_users_7d": active_users_7d
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# 全局实例
user_store = UserStore()