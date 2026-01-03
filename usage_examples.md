# 糖尿病智能问答系统 - 使用示例

## 目录
1. [快速开始](#快速开始)
2. [基础用法](#基础用法)
3. [高级功能](#高级功能)
4. [API接口](#api接口)
5. [自定义配置](#自定义配置)
6. [故障排除](#故障排除)

---

## 快速开始

### 1. 安装和配置

```bash
# 1. 克隆项目
git clone <repository-url>
cd diabetes-qa-system

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置你的配置
```

### 2. 启动必要服务

```bash
# Milvus (向量数据库)
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Neo4j (知识图谱)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:latest

# 确保 vllm 服务已启动在 http://localhost:8000
```

### 3. 准备数据

```bash
# 生成示例知识图谱数据
python scripts/generate_sample_data.py

# 将PDF文档放入 /data/pdf/ 目录

# 构建向量库
python scripts/build_vector_store.py

# 加载知识图谱
python scripts/load_knowledge_graph.py
```

### 4. 运行系统

```bash
# 交互模式
python main.py --mode interactive

# 批量处理
python main.py --mode batch --input questions.txt --output results.json
```

---

## 基础用法

### 交互式问答

```bash
$ python main.py --mode interactive

============================================================
糖尿病智能问答系统
============================================================
输入问题开始对话，输入 'quit' 退出
输入 'history' 查看对话历史
输入 'clear' 清除对话历史
============================================================

您的问题: 什么是糖尿病？

处理中...

回答: 糖尿病是一种慢性代谢性疾病，主要特征是血糖水平持续升高。
这是由于胰腺不能产生足够的胰岛素（1型糖尿病），或者身体无法有效
利用所产生的胰岛素（2型糖尿病）。主要症状包括多饮、多尿、多食和
体重下降。长期高血糖可能导致严重的并发症，如心血管疾病、肾病、
视网膜病变等。

问题类型: 简单
质量评分: 0.89
重试次数: 0

参考来源:
  - {'type': 'document', 'source': 'diabetes_guide.pdf', 'page': 5}
  - {'type': 'knowledge_graph', 'description': '知识图谱查询'}
```

### Python API 调用

```python
from main import DiabetesQASystem

# 初始化系统
qa_system = DiabetesQASystem()

# 简单问答
response = qa_system.ask(
    question="什么是糖尿病？",
    user_id="user_001",
    thread_id="session_001"
)

print(f"答案: {response['answer']}")
print(f"质量评分: {response['quality_score']:.2f}")
print(f"问题类型: {'复杂' if response['is_complex'] else '简单'}")
```

---

## 高级功能

### 1. 复杂问题查询

```python
# 多实体、多跳关系查询
response = qa_system.ask(
    question="糖尿病会导致哪些并发症，这些并发症又会影响哪些器官？",
    user_id="user_001",
    thread_id="session_001"
)

# 系统会：
# 1. 识别多个实体（糖尿病、并发症、器官）
# 2. 在知识图谱中查询多跳关系
# 3. 结合文档检索
# 4. 生成综合答案
```

### 2. 对话历史管理

```python
# 获取对话历史
history = qa_system.get_conversation_history(thread_id="session_001")

for msg in history:
    print(f"{msg['role']}: {msg['content']}")

# 清除对话
qa_system.clear_conversation(thread_id="session_001")
```

### 3. 用户画像管理

```python
from src.memory.user_store import user_store

# 创建/更新用户信息
user_store.create_or_update_user(
    user_id="user_001",
    diabetes_type="2型糖尿病",
    preferences={
        "language": "zh",
        "detail_level": "detailed"
    },
    profile={
        "age": 45,
        "duration": "5年"
    }
)

# 获取用户信息
user = user_store.get_user("user_001")
print(user)

# 分析用户行为
behavior = user_store.analyze_user_behavior("user_001")
print(f"常问主题: {behavior['frequent_topics']}")

# 获取个性化推荐
recommendations = user_store.get_personalized_recommendations("user_001")
print(f"推荐内容: {recommendations}")
```

### 4. 直接使用检索模块

```python
from src.retrieval.hybrid_retriever import hybrid_retriever

# 混合检索
results = hybrid_retriever.hybrid_search(
    query="糖尿病的治疗方法",
    top_k=5,
    use_bm25=True,
    use_vector=True
)

for result in results:
    print(f"相关度: {result['final_score']:.3f}")
    print(f"内容: {result['text'][:100]}...")
    print()
```

### 5. 直接使用知识图谱

```python
from src.knowledge_graph.graph_query import graph_query_engine

# 查询实体关系
result = graph_query_engine.query_by_entities(
    entities=["糖尿病", "高血压"],
    max_hops=2
)

print(f"找到 {result['entity_count']} 个相关实体")
print(result['summary'])

# 查询最短路径
paths = graph_query_engine.query_shortest_path(
    source="二甲双胍",
    target="糖尿病",
    relation_types=["TREATS"]
)

for path in paths:
    print(f"路径长度: {path['path_length']}")
    print(f"节点: {' -> '.join(path['node_names'])}")
```

### 6. 质量评估

```python
from src.evaluation.quality_evaluator import quality_evaluator

# 评估答案质量
evaluation = quality_evaluator.evaluate(
    question="什么是糖尿病？",
    answer="糖尿病是一种慢性疾病...",
    context="从文档检索到的上下文...",
    method="hybrid"  # llm, heuristic, hybrid
)

print(f"总分: {evaluation['overall_score']:.2f}")
print(f"相关性: {evaluation['relevance']:.2f}")
print(f"准确性: {evaluation['accuracy']:.2f}")
print(f"问题: {evaluation['issues']}")
print(f"建议: {evaluation['suggestions']}")

# 判断是否可接受
if quality_evaluator.is_acceptable(evaluation):
    print("答案质量合格")
else:
    print("需要改进")
```

---

## API 接口

### REST API 示例

如果需要提供 REST API 服务，可以使用 FastAPI：

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import DiabetesQASystem

app = FastAPI(title="糖尿病智能问答API")
qa_system = DiabetesQASystem()

class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"
    thread_id: str = "default"

class AnswerResponse(BaseModel):
    answer: str
    quality_score: float
    is_complex: bool
    sources: list

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        response = qa_system.ask(
            question=request.question,
            user_id=request.user_id,
            thread_id=request.thread_id
        )
        
        if not response["success"]:
            raise HTTPException(status_code=500, detail=response.get("error"))
        
        return AnswerResponse(
            answer=response["answer"],
            quality_score=response["quality_score"],
            is_complex=response["is_complex"],
            sources=response["sources"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    history = qa_system.get_conversation_history(thread_id)
    return {"history": history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

启动 API 服务：
```bash
python api_server.py
```

调用 API：
```bash
curl -X POST "http://localhost:8080/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是糖尿病？",
    "user_id": "user_001",
    "thread_id": "session_001"
  }'
```

---

## 自定义配置

### 1. 修改配置文件

编辑 `config/settings.py` 或 `.env` 文件：

```python
# .env 文件示例
LLM_API_BASE=http://localhost:8000/v1
LLM_MODEL_NAME=Qwen/Qwen2.5-8B-Instruct
LLM_TEMPERATURE=0.1

MILVUS_HOST=localhost
MILVUS_PORT=19530

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# 检索参数
TOP_K_SIMPLE=5
TOP_K_COMPLEX=10
RERANK_TOP_K=3

# 质量阈值
QUALITY_THRESHOLD=0.7
MAX_RETRY=3

# 检索权重
BM25_WEIGHT=0.3
VECTOR_WEIGHT=0.7
```

### 2. 自定义提示词

编辑 `config/prompts.py`：

```python
# 修改答案生成提示词
ANSWER_GENERATION_PROMPT = """你是一个专业的糖尿病健康顾问。

【自定义说明】
- 使用通俗易懂的语言
- 结构化输出
- 提供实用建议

用户问题: {question}

检索到的上下文:
{context}

请回答：
"""
```

### 3. 自定义工作流节点

添加新的处理节点：

```python
# src/graph_workflow/nodes.py

@staticmethod
def custom_processing_node(state: DiabetesQAState) -> DiabetesQAState:
    """自定义处理节点"""
    # 你的自定义逻辑
    state["step_logs"].append("执行自定义处理")
    return state
```

在工作流中添加：

```python
# src/graph_workflow/workflow.py

workflow.add_node("custom_node", nodes.custom_processing_node)
workflow.add_edge("analyze_complexity", "custom_node")
```

---

## 故障排除

### 问题 1: Milvus 连接失败

```bash
# 检查 Milvus 是否运行
docker ps | grep milvus

# 重启 Milvus
docker restart milvus

# 查看日志
docker logs milvus
```

### 问题 2: Neo4j 查询超时

```python
# 在 config/settings.py 中增加超时时间
NEO4J_TIMEOUT = 30  # 秒

# 或者在查询中添加 LIMIT
cypher = "MATCH (n) RETURN n LIMIT 100"
```

### 问题 3: LLM 响应慢

```python
# 减少 max_tokens
LLM_MAX_TOKENS = 1000

# 使用更小的模型
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# 调整温度参数
LLM_TEMPERATURE = 0.05  # 更确定性的输出
```

### 问题 4: 内存不足

```bash
# 减少批处理大小
EMBEDDING_BATCH_SIZE = 16  # 默认32

# 减少检索数量
TOP_K_SIMPLE = 3
TOP_K_COMPLEX = 5

# 分批处理文档
python scripts/build_vector_store.py --batch-size 50
```

### 问题 5: 检索结果质量差

```python
# 调整检索权重
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

# 启用重排序
reranker.rerank(query, documents, method="llm", top_k=3)

# 扩展查询
from src.retrieval.query_rewriter import query_rewriter
rewrites = query_rewriter.rewrite_query(query, num_rewrites=3)
```

---

## 性能优化建议

### 1. 向量检索优化

```python
# 使用 GPU 加速 embedding
# 在 embeddings.py 中
model = SentenceTransformer(model_name, device='cuda')

# 批量处理
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
```

### 2. 缓存机制

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(query: str):
    return qa_system.ask(query)
```

### 3. 异步处理

```python
import asyncio

async def ask_multiple_questions(questions):
    tasks = [
        asyncio.to_thread(qa_system.ask, q)
        for q in questions
    ]
    return await asyncio.gather(*tasks)
```

---

## 扩展开发

### 添加新的数据源

```python
# src/document_processing/custom_loader.py

class CustomDataLoader:
    def load_data(self, source):
        # 实现你的数据加载逻辑
        pass
```

### 集成外部API

```python
# src/retrieval/external_search.py

class ExternalSearchAPI:
    def search(self, query):
        # 调用外部搜索API
        response = requests.get(f"https://api.example.com/search?q={query}")
        return response.json()
```

---

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

MIT License - 详见 LICENSE 文件

---

## 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 文档: [Wiki]