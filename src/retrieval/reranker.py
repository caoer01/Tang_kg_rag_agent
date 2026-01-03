"""知识图谱查询模块 - 高级查询功能"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from src.knowledge_graph.neo4j_client import neo4j_client
from src.llm.qwen_client import llm_client
from config.prompts import CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class GraphQueryEngine:
    """图谱查询引擎"""

    def __init__(self):
        """初始化查询引擎"""
        self.max_path_length = 5  # 最大路径长度
        self.max_results = 50  # 最大返回结果数

    def query_by_entities(
            self,
            entities: List[str],
            max_hops: int = 2
    ) -> Dict[str, Any]:
        """根据实体查询相关子图

        Args:
            entities: 实体名称列表
            max_hops: 最大跳数

        Returns:
            查询结果
        """
        if not entities:
            return {"nodes": [], "relationships": [], "summary": ""}

        try:
            # 查询子图
            subgraph = neo4j_client.get_subgraph(
                entity_names=entities,
                max_depth=max_hops
            )

            # 生成总结
            summary = self._summarize_subgraph(entities, subgraph)

            return {
                "nodes": subgraph.get("nodes", []),
                "relationships": subgraph.get("relationships", []),
                "summary": summary,
                "entity_count": len(subgraph.get("nodes", [])),
                "relation_count": len(subgraph.get("relationships", []))
            }

        except Exception as e:
            logger.error(f"实体查询失败: {e}")
            return {"nodes": [], "relationships": [], "summary": "查询失败"}

    def query_shortest_path(
            self,
            source: str,
            target: str,
            relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """查询两个实体间的最短路径

        Args:
            source: 源实体名称
            target: 目标实体名称
            relation_types: 关系类型限制

        Returns:
            路径列表
        """
        try:
            # 构建Cypher查询
            if relation_types:
                rel_filter = "|".join(relation_types)
                rel_pattern = f"[r:{rel_filter}]"
            else:
                rel_pattern = "[r]"

            cypher = f"""
            MATCH (source {{name: $source}}), (target {{name: $target}})
            MATCH path = shortestPath((source)-{rel_pattern}*..{self.max_path_length}-(target))
            RETURN 
                [node in nodes(path) | node.name] as node_names,
                [node in nodes(path) | labels(node)[0]] as node_types,
                [rel in relationships(path) | type(rel)] as relation_types,
                length(path) as path_length
            LIMIT 5
            """

            results = neo4j_client.execute_query(cypher, {
                "source": source,
                "target": target
            })

            return results

        except Exception as e:
            logger.error(f"最短路径查询失败: {e}")
            return []

    def query_multi_hop_relations(
            self,
            entity: str,
            hops: int = 2,
            direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """查询多跳关系

        Args:
            entity: 实体名称
            hops: 跳数
            direction: 方向 ("outgoing", "incoming", "both")

        Returns:
            关系路径列表
        """
        try:
            results = neo4j_client.get_entity_relations(
                entity_name=entity,
                direction=direction,
                max_depth=hops
            )

            # 按路径长度排序
            results.sort(key=lambda x: x.get("path_length", 0))

            return results[:self.max_results]

        except Exception as e:
            logger.error(f"多跳关系查询失败: {e}")
            return []

    def query_by_pattern(
            self,
            pattern: str,
            parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """使用自定义模式查询

        Args:
            pattern: Cypher模式
            parameters: 查询参数

        Returns:
            查询结果
        """
        try:
            return neo4j_client.execute_query(pattern, parameters)
        except Exception as e:
            logger.error(f"模式查询失败: {e}")
            return []

    def query_neighborhood(
            self,
            entity: str,
            radius: int = 1,
            node_limit: int = 20
    ) -> Dict[str, Any]:
        """查询实体的邻域

        Args:
            entity: 实体名称
            radius: 邻域半径
            node_limit: 节点数量限制

        Returns:
            邻域子图
        """
        try:
            cypher = f"""
            MATCH path = (center {{name: $entity}})-[*1..{radius}]-(neighbor)
            WITH center, neighbor, path
            LIMIT {node_limit}
            RETURN 
                collect(DISTINCT neighbor {{.*, id: id(neighbor)}}) as neighbors,
                collect(DISTINCT {{
                    source: startNode(relationships(path)[0]).name,
                    target: endNode(relationships(path)[0]).name,
                    type: type(relationships(path)[0])
                }}) as relationships
            """

            results = neo4j_client.execute_query(cypher, {"entity": entity})

            if results:
                return {
                    "center": entity,
                    "neighbors": results[0].get("neighbors", []),
                    "relationships": results[0].get("relationships", [])
                }

            return {"center": entity, "neighbors": [], "relationships": []}

        except Exception as e:
            logger.error(f"邻域查询失败: {e}")
            return {"center": entity, "neighbors": [], "relationships": []}

    def query_similar_entities(
            self,
            entity: str,
            entity_type: Optional[str] = None,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """查询相似实体（基于结构相似性）

        Args:
            entity: 实体名称
            entity_type: 实体类型
            top_k: 返回top-k

        Returns:
            相似实体列表
        """
        try:
            # 获取目标实体的邻居
            target_neighbors = self._get_neighbors(entity)

            # 查找有相似邻居的实体
            type_filter = f":{entity_type}" if entity_type else ""

            cypher = f"""
            MATCH (target {{name: $entity}})
            MATCH (target)-[]-(neighbor)
            MATCH (similar{type_filter})-[]-(neighbor)
            WHERE similar.name <> $entity
            WITH similar, count(DISTINCT neighbor) as common_neighbors
            ORDER BY common_neighbors DESC
            LIMIT {top_k}
            RETURN 
                similar.name as name,
                labels(similar)[0] as type,
                common_neighbors
            """

            results = neo4j_client.execute_query(cypher, {"entity": entity})
            return results

        except Exception as e:
            logger.error(f"相似实体查询失败: {e}")
            return []

    def _get_neighbors(self, entity: str) -> List[str]:
        """获取实体的直接邻居

        Args:
            entity: 实体名称

        Returns:
            邻居名称列表
        """
        cypher = """
        MATCH (e {name: $entity})-[]-(neighbor)
        RETURN DISTINCT neighbor.name as name
        """

        results = neo4j_client.execute_query(cypher, {"entity": entity})
        return [r["name"] for r in results]

    def aggregate_query(
            self,
            entity_type: str,
            aggregation: str = "count",
            group_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """聚合查询

        Args:
            entity_type: 实体类型
            aggregation: 聚合函数 ("count", "sum", "avg")
            group_by: 分组字段

        Returns:
            聚合结果
        """
        try:
            if group_by:
                cypher = f"""
                MATCH (n:{entity_type})
                RETURN n.{group_by} as group_key, count(n) as count
                ORDER BY count DESC
                """
            else:
                cypher = f"""
                MATCH (n:{entity_type})
                RETURN count(n) as total
                """

            return neo4j_client.execute_query(cypher)

        except Exception as e:
            logger.error(f"聚合查询失败: {e}")
            return []

    def query_relation_chain(
            self,
            start_entity: str,
            relation_chain: List[str]
    ) -> List[Dict[str, Any]]:
        """查询特定关系链

        Args:
            start_entity: 起始实体
            relation_chain: 关系类型链，如 ["CAUSES", "HAS_SYMPTOM"]

        Returns:
            路径结果
        """
        try:
            # 构建关系链模式
            patterns = []
            for i, rel_type in enumerate(relation_chain):
                patterns.append(f"-[r{i}:{rel_type}]->")

            pattern_str = "".join(patterns)

            cypher = f"""
            MATCH path = (start {{name: $entity}}){pattern_str}(end)
            RETURN 
                [node in nodes(path) | {{name: node.name, type: labels(node)[0]}}] as nodes,
                [rel in relationships(path) | type(rel)] as relations
            LIMIT 20
            """

            return neo4j_client.execute_query(cypher, {"entity": start_entity})

        except Exception as e:
            logger.error(f"关系链查询失败: {e}")
            return []

    def generate_cypher_from_nl(
            self,
            question: str,
            entities: List[str]
    ) -> Tuple[str, str]:
        """从自然语言生成Cypher查询

        Args:
            question: 自然语言问题
            entities: 识别的实体

        Returns:
            (cypher查询, 解释)
        """
        try:
            prompt = CYPHER_GENERATION_PROMPT.format(
                entities=entities,
                question=question
            )

            result = llm_client.generate_json(prompt)

            cypher = result.get("cypher", "")
            explanation = result.get("explanation", "")

            # 验证Cypher语法
            if cypher and self._validate_cypher(cypher):
                return cypher, explanation
            else:
                logger.warning("生成的Cypher无效")
                return "", "Cypher生成失败"

        except Exception as e:
            logger.error(f"Cypher生成失败: {e}")
            return "", str(e)

    def _validate_cypher(self, cypher: str) -> bool:
        """验证Cypher查询语法

        Args:
            cypher: Cypher查询

        Returns:
            是否有效
        """
        # 基本语法检查
        required_keywords = ["MATCH", "RETURN"]
        cypher_upper = cypher.upper()

        for keyword in required_keywords:
            if keyword not in cypher_upper:
                return False

        # 检查是否有危险操作
        dangerous_keywords = ["DELETE", "DROP", "CREATE INDEX", "DETACH DELETE"]
        for keyword in dangerous_keywords:
            if keyword in cypher_upper:
                logger.warning(f"检测到危险操作: {keyword}")
                return False

        return True

    def _summarize_subgraph(
            self,
            entities: List[str],
            subgraph: Dict[str, Any]
    ) -> str:
        """总结子图结果

        Args:
            entities: 查询的实体
            subgraph: 子图数据

        Returns:
            总结文本
        """
        nodes = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])

        if not nodes:
            return f"未找到关于 {', '.join(entities)} 的相关信息。"

        # 统计节点类型
        from collections import Counter
        node_types = Counter(node.get("type") for node in nodes)
        rel_types = Counter(rel.get("type") for rel in relationships)

        summary_parts = [
            f"查询 {', '.join(entities)} 相关信息:",
            f"找到 {len(nodes)} 个相关实体",
        ]

        # 添加类型分布
        if node_types:
            type_str = ", ".join(
                f"{count}个{type_}"
                for type_, count in node_types.most_common(3)
            )
            summary_parts.append(f"包括: {type_str}")

        # 添加关系信息
        if rel_types:
            rel_str = ", ".join(
                f"{count}个{type_}"
                for type_, count in rel_types.most_common(3)
            )
            summary_parts.append(f"关系类型: {rel_str}")

        return "。".join(summary_parts) + "。"

    def explain_query_result(
            self,
            question: str,
            result: Dict[str, Any]
    ) -> str:
        """解释查询结果

        Args:
            question: 原始问题
            result: 查询结果

        Returns:
            自然语言解释
        """
        from config.prompts import GRAPH_SUMMARY_PROMPT

        try:
            prompt = GRAPH_SUMMARY_PROMPT.format(
                question=question,
                graph_results=result
            )

            explanation = llm_client.generate(prompt)
            return explanation

        except Exception as e:
            logger.error(f"结果解释失败: {e}")
            return result.get("summary", "")


# 全局实例
graph_query_engine = GraphQueryEngine()