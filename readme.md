# 糖尿病智能问答系统

基于LangGraph的智能医疗问答系统，支持简单/复杂问题自适应处理、多模态RAG、知识图谱查询和质量评估。
注：目前可视化前端界面还没实现，data的数据分为图谱和普通pdf文档。
实体和关系数据下载地址：https://github.com/luluforever/Diabetes-knowledge-graph
该数据来源于天池竞赛平台，链接地址：https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.12281978.0.0.7592412fAyjFC6&dataId=22288 要获取原始数据请到天池平台下载。
pdf的文档我是直接在网上找的论文。

## 系统架构

```
用户问题 → 复杂度分析 → 分流处理
                          ↓
                   ┌──────┴──────┐
                   ↓              ↓
              简单问题         复杂问题
                   ↓              ↓
            混合RAG检索    实体识别+图谱查询
                   ↓              ↓
              重排序优化    双重RAG融合
                   ↓              ↓
                   └──────┬──────┘
                          ↓
                     答案生成
                          ↓
                     质量评估
                          ↓
                  ┌───────┴────────┐
                  ↓                ↓
            达标(返回)      不达标(简单问题升级为复杂问题，复杂问题进行重写，三次重写还不行调用外部搜索工具)
```

## 核心特性

### 1. 自适应问题分类
- **简单问题**: 单实体、直接事实、定义类查询
- **复杂问题**: 多实体、推理链、时间序列、多跳关系

### 2. 双重RAG架构
- **文档RAG**: BM25 + 向量检索混合，重排序优化
- **图谱RAG**: Neo4j知识图谱，支持多跳关系查询

### 3. 多模态支持
- PDF文档解析（文字+图表）
- 图表LLM描述生成
- 独立图表向量库

### 4. 质量保障
- 6维度评估：相关性、准确性、完整性、专业性、可读性
- 自动重试机制（最多3次）
- 简单问题质量不达标自动升级为复杂问题
- 兜底外部搜索

### 5. 对话记忆
- SQLite持久化检查点
- 跨会话对话历史
- 用户画像存储

## 技术栈

| 组件 | 技术 |
|------|------|
| 工作流引擎 | LangGraph |
| LLM | Qwen3-8B (vllm部署) |
| 向量数据库 | Milvus |
| 知识图谱 | Neo4j |
| Embedding | sentence-transformers |
| 关键词检索 | BM25 |
| 文档处理 | PyMuPDF |

## 安装部署

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n your_env_name python=3.12
conda activate your_env_name

# 安装依赖
pip install -r requirements.txt
```

### 2. 服务部署[可以去docker hub上拉取]

#### Milvus (向量数据库)
```bash
# Docker方式
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

#### Neo4j (知识图谱)
```bash
# Docker方式
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

#### Qwen3-8B (vllm)
```bash
# 已通过vllm容器部署，确保服务运行在 http://localhost:8000
```

### 3. 配置

创建 `.env` 文件：

```env
# LLM配置
LLM_API_BASE=http://localhost:8000/v1
LLM_MODEL_NAME=Qwen/Qwen3-8B-Instruct【自行替换自己部署的模型】

# Milvus配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Neo4j配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# 数据路径
PDF_DATA_PATH=/data/pdf
NEO4J_DATA_PATH=/data/neo4j
```

## 数据准备

### 1. PDF文档

将PDF文档放入 `/data/pdf/` 目录。系统会自动：
- 提取文本内容
- 识别和描述图表
- 分块并向量化

### 2. 知识图谱数据

在 `/data/neo4j/` 目录准备两个CSV文件：

#### entities.csv
```csv
entity_id,name,type,properties
1,糖尿病,Disease,"{\"description\": \"慢性代谢疾病\"}"
2,二甲双胍,Drug,"{\"usage\": \"降糖药\"}"
3,多饮多尿,Symptom,"{\"severity\": \"常见\"}"
```

#### relationships.csv
```csv
source_id,target_id,relation_type,properties
1,3,HAS_SYMPTOM,"{\"confidence\": 0.95}"
2,1,TREATS,"{\"efficacy\": \"高\"}"
```

### 3. 构建数据库

```bash
# 构建向量库
python scripts/build_vector_store.py

# 加载知识图谱
python scripts/load_knowledge_graph.py
```

## 使用方法

### 交互模式

```bash
python main.py --mode interactive
```

示例对话：
```
您的问题: 什么是糖尿病？

回答: 糖尿病是一种慢性代谢性疾病，主要特征是血糖水平持续升高...

问题类型: 简单
质量评分: 0.89
重试次数: 0
```

### 批量处理模式

准备问题文件 `questions.txt`：
```
什么是糖尿病？
糖尿病有哪些并发症？
二甲双胍和胰岛素有什么区别？
```

运行：
```bash
python main.py --mode batch --input questions.txt --output results.json
```

### API集成

```python
from main import DiabetesQASystem

# 初始化系统
qa_system = DiabetesQASystem()

# 提问
response = qa_system.ask(
    question="糖尿病会导致哪些并发症？",
    user_id="user123",
    thread_id="conversation_001"
)

print(response["answer"])
print(f"质量评分: {response['quality_score']}")
```

## 工作流详解

### 简单问题流程

1. **复杂度分析**: 判定为简单问题
2. **混合检索**: BM25 + 向量检索
3. **重排序**: LLM评分优化排序
4. **答案生成**: 基于top-3文档生成
5. **质量评估**: 
   - 达标 → 返回答案
   - 不达标 → 升级为复杂问题

### 复杂问题流程

1. **复杂度分析**: 判定为复杂问题
2. **实体识别**: 提取关键实体
3. **图谱查询**: 
   - 生成Cypher查询
   - 执行图谱检索
   - 总结图谱结果
4. **文档检索**: 扩大检索范围(top-10)
5. **答案生成**: 融合图谱+文档信息
6. **质量评估**:
   - 达标 → 返回答案
   - 不达标 → 查询改写重试(最多3次)
   - 仍不达标 → 外部搜索兜底

## 质量评估标准

| 维度 | 评分标准 |
|------|---------|
| 相关性 | 是否直接回答问题 |
| 准确性 | 是否基于上下文，无编造 |
| 完整性 | 是否全面覆盖要点 |
| 专业性 | 医学术语使用准确性 |
| 可读性 | 表达清晰易懂 |

**总分 ≥ 0.7** 视为合格

## 项目结构

```
糖尿病智能问答系统/
├── config/                    # 配置文件
│   ├── settings.py           # 系统配置
│   └── prompts.py            # 提示词模板
├── src/
│   ├── document_processing/  # 文档处理
│   │   └── pdf_loader.py
│   ├── vector_store/         # 向量存储
│   │   ├── milvus_client.py
│   │   └── embeddings.py
│   ├── knowledge_graph/      # 知识图谱
│   │   └── neo4j_client.py
│   ├── retrieval/            # 检索模块
│   │   └── hybrid_retriever.py
│   ├── graph_workflow/       # LangGraph工作流
│   │   ├── state.py          # 状态定义
│   │   ├── nodes.py          # 节点定义
│   │   └── workflow.py       # 工作流编排
│   └── llm/                  # LLM客户端
│       └── qwen_client.py
├── scripts/                   # 工具脚本
│   ├── build_vector_store.py
│   └── load_knowledge_graph.py
├── main.py                    # 主程序
└── requirements.txt           # 依赖
```

## 性能指标

| 指标 | 目标值 |
|------|--------|
| 简单问题响应时间 | < 3秒 |
| 复杂问题响应时间 | < 10秒 |
| 答案质量评分 | > 0.7 |
| 首次成功率 | > 80% |


**注意**: 该项目主要用于大模型应用开发学习。本系统仅供参考，不能替代专业医疗建议，用户应咨询医生获取专业诊疗意见。同时，也不要拿此项目用作任何商业行为！！！【包括去咸鱼、小红书等平台卖！！！】

