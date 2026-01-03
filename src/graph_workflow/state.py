"""LangGraph工作流状态定义"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import MessagesState


class QuestionAnalysis(TypedDict):
    """问题分析结果"""
    is_complex: bool
    complexity_score: float
    reasoning: str
    identified_entities: List[str]
    required_reasoning_type: str


class RetrievalResult(TypedDict):
    """检索结果"""
    text: str
    metadata: Dict[str, Any]
    score: float
    source: str


class GraphQueryResult(TypedDict):
    """图谱查询结果"""
    cypher: str
    raw_results: List[Dict[str, Any]]
    summary: str


class QualityEvaluation(TypedDict):
    """质量评估结果"""
    overall_score: float
    relevance: float
    accuracy: float
    completeness: float
    professionalism: float
    readability: float
    issues: List[str]
    suggestions: str


class DiabetesQAState(MessagesState):
    """糖尿病问答系统状态

    继承自MessagesState以支持对话历史
    """
    # 输入
    question: str
    user_id: Optional[str]

    # 问题分析
    question_analysis: Optional[QuestionAnalysis]

    # 检索相关
    retrieval_results: List[RetrievalResult]
    graph_results: Optional[GraphQueryResult]
    rewritten_queries: List[str]

    # 生成相关
    answer: str
    context: str

    # 质量评估
    quality_eval: Optional[QualityEvaluation]
    retry_count: int

    # 流程控制
    is_complex_question: bool
    should_retry: bool
    should_use_external_search: bool

    # 错误信息
    error: Optional[str]

    # 中间步骤记录（用于调试）
    step_logs: List[str]


class UserProfile(TypedDict):
    """用户画像"""
    user_id: str
    diabetes_type: Optional[str]  # 1型/2型/妊娠期等
    frequent_topics: List[str]
    interaction_count: int
    last_interaction: str
    preferences: Dict[str, Any]