"""提示词模板"""

# 问题复杂度分析提示词
COMPLEXITY_ANALYSIS_PROMPT = """你是一个医疗问答系统的问题分析专家。请分析以下用户问题的复杂度。

复杂问题特征：
1. 涉及多个实体(>2个)，如"糖尿病、高血压、肾病"
2. 需要推理链(因果关系、对比分析)，如"为什么糖尿病会导致视网膜病变"
3. 需要聚合信息(统计、排序、归纳)，如"列举所有降糖药的副作用"
4. 时间序列分析，如"糖尿病并发症的发展过程"
5. 多跳关系查询，如"二甲双胍影响哪些器官，这些器官又与哪些疾病相关"
6. 需要背景知识推理，如"为什么糖尿病患者需要控制碳水化合物"

简单问题特征：
1. 单一实体查询，如"什么是糖尿病"
2. 直接事实查询，如"正常血糖范围是多少"
3. 定义类问题，如"什么是胰岛素抵抗"
4. 无需推理的描述性问题，如"糖尿病有哪些症状"

用户问题: {question}

请以JSON格式输出分析结果：
{{
    "is_complex": true/false,
    "complexity_score": 0.0-1.0,
    "reasoning": "判断理由",
    "identified_entities": ["实体1", "实体2"],
    "required_reasoning_type": "推理类型"
}}
"""

# 实体识别提示词
ENTITY_EXTRACTION_PROMPT = """从以下医疗问题中提取关键实体。

实体类型包括：
- 疾病(Disease): 糖尿病、高血压等
- 药物(Drug): 二甲双胍、胰岛素等
- 症状(Symptom): 多饮、多尿等
- 检查(Test): 血糖检测、糖化血红蛋白等
- 治疗(Treatment): 饮食控制、运动疗法等
- 器官(Organ): 胰腺、肾脏等

问题: {question}

请以JSON格式返回：
{{
    "entities": [
        {{"text": "实体名", "type": "实体类型"}},
        ...
    ]
}}
"""

# 查询改写提示词
QUERY_REWRITE_PROMPT = """你是一个医疗信息检索专家。请将用户问题改写为更适合检索的形式。

改写策略：
1. 扩展医学术语(如"糖尿病"→"糖尿病 高血糖 胰岛素")
2. 添加相关概念
3. 使用同义词
4. 分解复杂问题

原问题: {question}

已尝试的改写版本: {previous_rewrites}

请提供{num_rewrites}个不同的改写版本，以JSON格式返回：
{{
    "rewrites": ["改写1", "改写2", ...]
}}
"""

# 答案生成提示词
ANSWER_GENERATION_PROMPT = """你是一个专业的糖尿病健康顾问。请基于以下检索到的上下文信息，准确回答用户的问题。

用户问题: {question}

检索到的上下文:
{context}

要求：
1. 答案必须基于提供的上下文
2. 如果上下文不足以回答问题，明确说明
3. 使用专业但易懂的语言
4. 对于医疗建议，务必提醒用户咨询医生
5. 结构清晰，重点突出

请回答：
"""

# 图表描述生成提示词
IMAGE_DESCRIPTION_PROMPT = """请详细描述这张医疗图表的内容。

描述要求：
1. 图表类型(折线图、柱状图、表格等)
2. 主要数据和趋势
3. 关键信息点
4. 数值范围和单位
5. 医学含义

请提供详细描述：
"""

# 质量评估提示词
QUALITY_EVALUATION_PROMPT = """你是一个医疗问答质量评估专家。请评估以下答案的质量。

用户问题: {question}

生成的答案: {answer}

参考上下文: {context}

评估维度：
1. 相关性(0-1): 答案是否直接回答了问题
2. 准确性(0-1): 答案是否基于上下文，无编造内容
3. 完整性(0-1): 答案是否全面覆盖问题要点
4. 专业性(0-1): 医学术语使用是否准确
5. 可读性(0-1): 表达是否清晰易懂

请以JSON格式返回评估结果：
{{
    "overall_score": 0.0-1.0,
    "relevance": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "professionalism": 0.0-1.0,
    "readability": 0.0-1.0,
    "issues": ["问题1", "问题2"],
    "suggestions": "改进建议"
}}
"""

# 图谱查询生成提示词
CYPHER_GENERATION_PROMPT = """你是一个Neo4j Cypher查询专家。请根据用户问题生成Cypher查询语句。

知识图谱结构：
- 节点类型: Disease(疾病), Drug(药物), Symptom(症状), Treatment(治疗), Organ(器官)
- 关系类型: CAUSES(导致), TREATS(治疗), HAS_SYMPTOM(有症状), AFFECTS(影响), RELATED_TO(相关)

识别的实体: {entities}

用户问题: {question}

请生成Cypher查询，以JSON格式返回：
{{
    "cypher": "MATCH查询语句",
    "explanation": "查询逻辑说明"
}}

示例：
问题: "糖尿病会导致哪些并发症"
{{
    "cypher": "MATCH (d:Disease {{name: '糖尿病'}})-[:CAUSES]->(c:Disease) RETURN c.name as complication",
    "explanation": "查询糖尿病节点通过CAUSES关系指向的所有疾病节点"
}}
"""

# 知识图谱结果总结提示词
GRAPH_SUMMARY_PROMPT = """请将以下知识图谱查询结果整理成易于理解的文本描述。

查询问题: {question}

图谱结果: {graph_results}

要求：
1. 用自然语言描述关系和路径
2. 突出关键信息
3. 保持结构清晰

请提供总结：
"""