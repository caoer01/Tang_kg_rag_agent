"""Neo4j知识图谱客户端"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j客户端"""

    def __init__(self):
        """连接Neo4j"""
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        logger.info(f"已连接Neo4j: {settings.NEO4J_URI}")

    def close(self):
        """关闭连接"""
        self.driver.close()

    def execute_query(self, cypher: str, parameters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """执行Cypher查询

        Args:
            cypher: Cypher查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def find_entity(self, entity_name: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """查找实体节点

        Args:
            entity_name: 实体名称
            entity_type: 实体类型 (Disease, Drug, Symptom等)

        Returns:
            匹配的实体列表
        """
        if entity_type:
            cypher = f"""
            MATCH (n:{entity_type})
            WHERE n.name CONTAINS $name
            RETURN n, labels(n) as types
            """
        else:
            cypher = """
            MATCH (n)
            WHERE n.name CONTAINS $name
            RETURN n, labels(n) as types
            """

        return self.execute_query(cypher, {"name": entity_name})

    def get_entity_relations(
            self,
            entity_name: str,
            relation_type: Optional[str] = None,
            direction: str = "both",
            max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """获取实体的关系

        Args:
            entity_name: 实体名称
            relation_type: 关系类型 (CAUSES, TREATS等)
            direction: 方向 ("outgoing", "incoming", "both")
            max_depth: 最大跳数

        Returns:
            关系列表
        """
        relation_pattern = f"[r:{relation_type}]" if relation_type else "[r]"

        if direction == "outgoing":
            path_pattern = f"-{relation_pattern}->"
        elif direction == "incoming":
            path_pattern = f"<-{relation_pattern}-"
        else:
            path_pattern = f"-{relation_pattern}-"

        cypher = f"""
        MATCH path = (source {{name: $name}}){path_pattern * 1..{max_depth} }(target)
        RETURN 
            source.name as source_name,
            labels(source)[0] as source_type,
            [rel in relationships(path) | type(rel)] as relation_types,
            target.name as target_name,
            labels(target)[0] as target_type,
            length(path) as path_length
        LIMIT 50
        """

        return self.execute_query(cypher, {"name": entity_name})

    def multi_entity_query(
            self,
            entity_names: List[str],
            relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """多实体关系查询

        Args:
            entity_names: 实体名称列表
            relation_types: 关系类型列表

        Returns:
            实体间的关系路径
        """
        if len(entity_names) < 2:
            return []

        # 构建关系模式
        if relation_types:
            rel_pattern = "|".join(relation_types)
            relation_pattern = f"[r:{rel_pattern}]"
        else:
            relation_pattern = "[r]"

        cypher = f"""
        MATCH (e1), (e2)
        WHERE e1.name IN $names AND e2.name IN $names AND e1 <> e2
        MATCH path = shortestPath((e1)-{relation_pattern}*..5-(e2))
        RETURN 
            e1.name as entity1,
            e2.name as entity2,
            [rel in relationships(path) | type(rel)] as relations,
            [node in nodes(path) | node.name] as path_nodes,
            length(path) as distance
        ORDER BY distance
        LIMIT 20
        """

        return self.execute_query(cypher, {"names": entity_names})

    def get_subgraph(
            self,
            entity_names: List[str],
            max_depth: int = 2
    ) -> Dict[str, Any]:
        """获取实体相关的子图

        Args:
            entity_names: 中心实体名称列表
            max_depth: 扩展深度

        Returns:
            子图数据 {nodes: [...], relationships: [...]}
        """
        cypher = f"""
        MATCH (start)
        WHERE start.name IN $names
        CALL apoc.path.subgraphAll(start, {{
            maxLevel: {max_depth},
            relationshipFilter: null
        }})
        YIELD nodes, relationships
        RETURN 
            [node in nodes | {{id: id(node), name: node.name, type: labels(node)[0]}}] as nodes,
            [rel in relationships | {{
                source: startNode(rel).name,
                target: endNode(rel).name,
                type: type(rel)
            }}] as relationships
        """

        results = self.execute_query(cypher, {"names": entity_names})
        if results:
            return results[0]
        return {"nodes": [], "relationships": []}

    def search_by_property(
            self,
            property_name: str,
            property_value: Any,
            node_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """按属性搜索节点

        Args:
            property_name: 属性名
            property_value: 属性值
            node_label: 节点标签

        Returns:
            匹配的节点列表
        """
        label_pattern = f":{node_label}" if node_label else ""
        cypher = f"""
        MATCH (n{label_pattern})
        WHERE n.{property_name} = $value
        RETURN n, labels(n) as types
        """

        return self.execute_query(cypher, {"value": property_value})

    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息

        Returns:
            统计信息字典
        """
        stats = {}

        # 节点统计
        node_count_query = "MATCH (n) RETURN count(n) as count"
        stats["total_nodes"] = self.execute_query(node_count_query)[0]["count"]

        # 关系统计
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        stats["total_relationships"] = self.execute_query(rel_count_query)[0]["count"]

        # 节点类型统计
        node_types_query = """
        MATCH (n)
        RETURN labels(n)[0] as type, count(n) as count
        ORDER BY count DESC
        """
        stats["node_types"] = self.execute_query(node_types_query)

        # 关系类型统计
        rel_types_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
        """
        stats["relationship_types"] = self.execute_query(rel_types_query)

        return stats


# 全局实例
neo4j_client = Neo4jClient()