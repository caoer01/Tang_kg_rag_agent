"""实体提取模块 - 从文本中识别医疗实体"""
import re
from typing import List, Dict, Any, Set, Tuple
import logging
from src.llm.qwen_client import llm_client
from config.prompts import ENTITY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class EntityExtractor:
    """实体提取器"""

    def __init__(self):
        """初始化实体提取器"""
        # 医疗实体类型
        self.entity_types = [
            "Disease",  # 疾病
            "Drug",  # 药物
            "Symptom",  # 症状
            "Test",  # 检查
            "Treatment",  # 治疗
            "Organ",  # 器官
            "Indicator"  # 指标
        ]

        # 加载医疗词典（示例）
        self.medical_dict = self._load_medical_dictionary()

    def _load_medical_dictionary(self) -> Dict[str, List[str]]:
        """加载医疗词典

        Returns:
            按类型分类的医疗术语词典
        """
        # 这里可以从文件加载，现在使用示例数据
        return {
            "Disease": [
                "糖尿病", "1型糖尿病", "2型糖尿病", "妊娠糖尿病",
                "糖尿病肾病", "糖尿病视网膜病变", "糖尿病神经病变",
                "高血压", "冠心病", "脑卒中", "肥胖症"
            ],
            "Drug": [
                "二甲双胍", "胰岛素", "格列美脲", "阿卡波糖",
                "吡格列酮", "西格列汀", "利拉鲁肽", "恩格列净"
            ],
            "Symptom": [
                "多饮", "多尿", "多食", "消瘦", "乏力",
                "视力模糊", "伤口愈合慢", "皮肤瘙痒", "肢体麻木"
            ],
            "Test": [
                "空腹血糖", "餐后血糖", "糖化血红蛋白", "口服葡萄糖耐量试验",
                "尿微量白蛋白", "肾功能", "血脂", "眼底检查"
            ],
            "Treatment": [
                "饮食控制", "运动疗法", "药物治疗", "胰岛素治疗",
                "血糖监测", "自我管理"
            ],
            "Organ": [
                "胰腺", "肝脏", "肾脏", "心脏", "眼睛",
                "神经", "血管", "足部"
            ],
            "Indicator": [
                "血糖", "血压", "血脂", "体重指数", "糖化血红蛋白",
                "空腹血糖", "餐后血糖", "尿蛋白"
            ]
        }

    def extract_entities(
            self,
            text: str,
            use_llm: bool = True,
            use_dict: bool = True
    ) -> List[Dict[str, Any]]:
        """提取实体

        Args:
            text: 输入文本
            use_llm: 是否使用LLM提取
            use_dict: 是否使用词典匹配

        Returns:
            实体列表
        """
        entities = []

        # 方法1: LLM提取
        if use_llm:
            llm_entities = self._extract_with_llm(text)
            entities.extend(llm_entities)

        # 方法2: 词典匹配
        if use_dict:
            dict_entities = self._extract_with_dictionary(text)
            entities.extend(dict_entities)

        # 去重和合并
        entities = self._merge_entities(entities)

        logger.info(f"提取到 {len(entities)} 个实体")
        return entities

    def _extract_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """使用LLM提取实体

        Args:
            text: 输入文本

        Returns:
            实体列表
        """
        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(question=text)
            result = llm_client.generate_json(prompt)

            entities = result.get("entities", [])

            # 添加来源标记
            for entity in entities:
                entity["source"] = "llm"
                entity["confidence"] = 0.8  # LLM提取的默认置信度

            return entities

        except Exception as e:
            logger.error(f"LLM实体提取失败: {e}")
            return []

    def _extract_with_dictionary(self, text: str) -> List[Dict[str, Any]]:
        """使用词典匹配提取实体

        Args:
            text: 输入文本

        Returns:
            实体列表
        """
        entities = []

        for entity_type, terms in self.medical_dict.items():
            for term in terms:
                # 查找所有出现位置
                positions = [m.start() for m in re.finditer(re.escape(term), text)]

                for pos in positions:
                    entities.append({
                        "text": term,
                        "type": entity_type,
                        "source": "dict",
                        "confidence": 1.0,  # 词典匹配的置信度
                        "position": pos
                    })

        return entities

    def _merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并重复实体

        Args:
            entities: 原始实体列表

        Returns:
            去重后的实体列表
        """
        # 使用字典去重（基于text和type）
        merged = {}

        for entity in entities:
            key = (entity["text"], entity["type"])

            if key not in merged:
                merged[key] = entity
            else:
                # 如果已存在，保留置信度更高的
                if entity["confidence"] > merged[key]["confidence"]:
                    merged[key] = entity
                # 或者合并来源
                elif entity["source"] != merged[key]["source"]:
                    merged[key]["source"] = "both"
                    merged[key]["confidence"] = max(
                        entity["confidence"],
                        merged[key]["confidence"]
                    )

        return list(merged.values())

    def extract_entity_relations(
            self,
            text: str,
            entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """提取实体间的关系

        Args:
            text: 输入文本
            entities: 已识别的实体列表

        Returns:
            关系列表
        """
        relations = []

        # 提取共现关系
        cooccurrence_relations = self._extract_cooccurrence(text, entities)
        relations.extend(cooccurrence_relations)

        # 使用模式匹配提取关系
        pattern_relations = self._extract_with_patterns(text, entities)
        relations.extend(pattern_relations)

        logger.info(f"提取到 {len(relations)} 个关系")
        return relations

    def _extract_cooccurrence(
            self,
            text: str,
            entities: List[Dict[str, Any]],
            window_size: int = 50
    ) -> List[Dict[str, Any]]:
        """提取共现关系

        Args:
            text: 输入文本
            entities: 实体列表
            window_size: 窗口大小（字符数）

        Returns:
            共现关系列表
        """
        relations = []

        # 按位置排序实体
        sorted_entities = sorted(
            entities,
            key=lambda e: e.get("position", 0)
        )

        # 检查窗口内的实体对
        for i, entity1 in enumerate(sorted_entities):
            pos1 = entity1.get("position", 0)

            for entity2 in sorted_entities[i + 1:]:
                pos2 = entity2.get("position", 0)

                # 如果在窗口内
                if abs(pos2 - pos1) <= window_size:
                    relations.append({
                        "source": entity1["text"],
                        "source_type": entity1["type"],
                        "target": entity2["text"],
                        "target_type": entity2["type"],
                        "relation_type": "CO_OCCUR",
                        "confidence": 0.5,
                        "distance": abs(pos2 - pos1)
                    })
                else:
                    break  # 超出窗口，不再检查后续实体

        return relations

    def _extract_with_patterns(
            self,
            text: str,
            entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用模式匹配提取关系

        Args:
            text: 输入文本
            entities: 实体列表

        Returns:
            关系列表
        """
        relations = []

        # 定义关系模式
        patterns = [
            # 因果关系
            (r'(.+?)(导致|引起|造成)(.+)', "CAUSES"),
            # 治疗关系
            (r'(.+?)(治疗|用于|控制)(.+)', "TREATS"),
            # 症状关系
            (r'(.+?)(表现为|出现|伴有)(.+)', "HAS_SYMPTOM"),
            # 检查关系
            (r'(.+?)(检测|监测|评估)(.+)', "DIAGNOSES"),
            # 影响关系
            (r'(.+?)(影响|损害|累及)(.+)', "AFFECTS"),
        ]

        # 创建实体文本到实体的映射
        entity_map = {e["text"]: e for e in entities}

        # 按句子处理
        sentences = re.split(r'[。！？.!?]', text)

        for sentence in sentences:
            for pattern, relation_type in patterns:
                matches = re.finditer(pattern, sentence)

                for match in matches:
                    source_text = match.group(1).strip()
                    target_text = match.group(3).strip()

                    # 检查是否匹配已知实体
                    source_entity = self._find_matching_entity(
                        source_text, entity_map
                    )
                    target_entity = self._find_matching_entity(
                        target_text, entity_map
                    )

                    if source_entity and target_entity:
                        relations.append({
                            "source": source_entity["text"],
                            "source_type": source_entity["type"],
                            "target": target_entity["text"],
                            "target_type": target_entity["type"],
                            "relation_type": relation_type,
                            "confidence": 0.7,
                            "context": sentence
                        })

        return relations

    def _find_matching_entity(
            self,
            text: str,
            entity_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """在文本中查找匹配的实体

        Args:
            text: 文本片段
            entity_map: 实体映射

        Returns:
            匹配的实体，如果没有返回None
        """
        # 精确匹配
        if text in entity_map:
            return entity_map[text]

        # 部分匹配
        for entity_text, entity in entity_map.items():
            if entity_text in text or text in entity_text:
                return entity

        return None

    def normalize_entity(self, entity_text: str, entity_type: str) -> str:
        """实体标准化

        Args:
            entity_text: 实体文本
            entity_type: 实体类型

        Returns:
            标准化后的实体文本
        """
        # 移除多余空格
        normalized = entity_text.strip()

        # 类型特定的标准化
        if entity_type == "Disease":
            # 疾病名称标准化
            normalized = normalized.replace("糖尿病", "糖尿病")  # 示例

        elif entity_type == "Drug":
            # 药物名称标准化（通用名）
            pass

        return normalized

    def get_entity_statistics(
            self,
            entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """获取实体统计信息

        Args:
            entities: 实体列表

        Returns:
            统计信息
        """
        from collections import Counter

        type_counts = Counter(e["type"] for e in entities)
        source_counts = Counter(e["source"] for e in entities)

        return {
            "total": len(entities),
            "by_type": dict(type_counts),
            "by_source": dict(source_counts),
            "unique_entities": len(set(e["text"] for e in entities))
        }


# 全局实例
entity_extractor = EntityExtractor()