"""加载知识图谱脚本"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_graph.neo4j_client import neo4j_client
from config.settings import settings
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_entities(file_path: str):
    """加载实体数据

    CSV格式示例:
    entity_id,name,type,properties
    1,糖尿病,Disease,{"description": "慢性代谢疾病"}
    """
    logger.info(f"加载实体文件: {file_path}")
    df = pd.read_csv(file_path)

    # 清空现有数据（可选）
    # neo4j_client.execute_query("MATCH (n) DETACH DELETE n")

    # 按类型分组创建节点
    for entity_type in df['type'].unique():
        type_df = df[df['type'] == entity_type]

        logger.info(f"创建 {entity_type} 节点: {len(type_df)} 个")

        for _, row in tqdm(type_df.iterrows(), total=len(type_df), desc=f"加载{entity_type}"):
            # 解析properties JSON
            properties = {}
            if pd.notna(row.get('properties')):
                import json
                try:
                    properties = json.loads(row['properties'])
                except:
                    pass

            # 创建节点
            cypher = f"""
            CREATE (n:{entity_type} {{
                entity_id: $entity_id,
                name: $name
            }})
            """

            # 添加额外属性
            if properties:
                prop_strs = [f"n.{k} = ${k}" for k in properties.keys()]
                cypher += "SET " + ", ".join(prop_strs)

            params = {
                "entity_id": str(row['entity_id']),
                "name": row['name'],
                **properties
            }

            neo4j_client.execute_query(cypher, params)

    # 创建索引
    logger.info("创建索引")
    for entity_type in df['type'].unique():
        try:
            neo4j_client.execute_query(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{entity_type}) ON (n.name)"
            )
            neo4j_client.execute_query(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{entity_type}) ON (n.entity_id)"
            )
        except Exception as e:
            logger.warning(f"创建索引失败: {e}")


def load_relationships(file_path: str):
    """加载关系数据

    CSV格式示例:
    source_id,target_id,relation_type,properties
    1,2,CAUSES,{"confidence": 0.95}
    """
    logger.info(f"加载关系文件: {file_path}")
    df = pd.read_csv(file_path)

    logger.info(f"创建关系: {len(df)} 条")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="加载关系"):
        # 解析properties
        properties = {}
        if pd.notna(row.get('properties')):
            import json
            try:
                properties = json.loads(row['properties'])
            except:
                pass

        # 创建关系
        cypher = f"""
        MATCH (source {{entity_id: $source_id}})
        MATCH (target {{entity_id: $target_id}})
        CREATE (source)-[r:{row['relation_type']}]->(target)
        """

        # 添加关系属性
        if properties:
            prop_strs = [f"r.{k} = ${k}" for k in properties.keys()]
            cypher += "SET " + ", ".join(prop_strs)

        params = {
            "source_id": str(row['source_id']),
            "target_id": str(row['target_id']),
            **properties
        }

        try:
            neo4j_client.execute_query(cypher, params)
        except Exception as e:
            logger.error(f"创建关系失败: {e}, 数据: {row.to_dict()}")


def load_knowledge_graph():
    """加载知识图谱"""
    logger.info("=" * 50)
    logger.info("开始加载知识图谱")
    logger.info("=" * 50)

    data_path = Path(settings.NEO4J_DATA_PATH)

    # 1. 加载实体
    entity_file = data_path / "entities.csv"
    if entity_file.exists():
        load_entities(str(entity_file))
    else:
        logger.error(f"实体文件不存在: {entity_file}")
        return

    # 2. 加载关系
    relation_file = data_path / "relationships.csv"
    if relation_file.exists():
        load_relationships(str(relation_file))
    else:
        logger.warning(f"关系文件不存在: {relation_file}")

    # 3. 统计信息
    logger.info("=" * 50)
    logger.info("知识图谱加载完成!")
    logger.info("=" * 50)

    stats = neo4j_client.get_statistics()
    logger.info(f"总节点数: {stats['total_nodes']}")
    logger.info(f"总关系数: {stats['total_relationships']}")
    logger.info("\n节点类型分布:")
    for node_type in stats['node_types']:
        logger.info(f"  {node_type['type']}: {node_type['count']}")
    logger.info("\n关系类型分布:")
    for rel_type in stats['relationship_types']:
        logger.info(f"  {rel_type['type']}: {rel_type['count']}")


if __name__ == "__main__":
    try:
        load_knowledge_graph()
    except Exception as e:
        logger.error(f"加载知识图谱失败: {e}", exc_info=True)
        sys.exit(1)