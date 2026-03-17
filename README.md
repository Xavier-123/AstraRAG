# AstraRAG

**企业级 Agentic RAG + Self-Improving RAG 对话系统框架**

AstraRAG 是一套完善的企业级 RAG 对话系统框架，由 **Agentic RAG** 与 **Self-Improving RAG** 两大核心模块构成，主体框架基于 [LangGraph](https://github.com/langchain-ai/langgraph) 搭建，首选 OpenAI 生态作为大模型服务提供方。

---

## 架构概览

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     Agentic RAG Layer                       │
│                                                             │
│  QueryUnderstanding → TaskPlanning → RetrievalPlanning      │
│         │                                    │              │
│         │                      ┌─────────────┘              │
│         │                      ▼                            │
│         │            MultiRetriever (并行检索)               │
│         │                      │                            │
│         │                   Rerank                          │
│         │                      │                            │
│         │            ContextEngineering                     │
│         │                      │                            │
│         └──────────────► Reasoning ◄─────────┐             │
│                              │                │             │
│                         (multi-hop loop)──────┘             │
│                              │                              │
│                          Reflection                         │
│                              │                              │
│              ┌───────────────┴──────────────┐               │
│         quality ok?                    quality low?          │
│              │                              │               │
│             END                    re-generate (loop)       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      Memory Layer                           │
│  ConversationMemory │ KnowledgeMemory │ EpisodicMemory      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                  Self-Improving Layer                       │
│  EvaluationLayer → FeedbackLearning → SystemOptimization   │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心设计原则

| 原则 | 说明 |
|------|------|
| **解耦模块** | 每个 Agent 作为独立模块，通过 `GraphState` 通信，互不直接调用 |
| **抽象优先** | 所有 Agent 继承 `BaseAgent` 抽象类，支持灵活扩展和替换 |
| **多智能协同** | LangGraph 编排多 Agent 并行/串行协作，支持动态路由 |
| **自我改进** | 评估 → 反馈 → 优化 闭环，持续提升系统质量 |
| **企业级可靠性** | Tenacity 重试、结构化日志、Pydantic 数据校验 |

---

## 目录结构

```
AstraRAG/
├── requirements.txt               # 项目依赖
├── pyproject.toml                 # 项目配置
├── examples/
│   └── basic_usage.py             # 完整使用示例
└── astra_rag/
    ├── __init__.py                # 主入口，导出 AstraRAGSystem
    ├── core/
    │   ├── base_agent.py          # 所有 Agent 的抽象基类
    │   ├── state.py               # LangGraph GraphState 定义
    │   └── config.py              # 系统配置（支持环境变量）
    ├── agents/
    │   ├── query_understanding.py # 查询理解 Agent
    │   ├── task_planning.py       # 任务规划 Agent
    │   ├── retrieval_planning.py  # 检索策略规划 Agent
    │   ├── multi_retriever.py     # 多检索器并行执行 Agent
    │   ├── rerank.py              # 重排序 Agent
    │   ├── context_engineering.py # 上下文工程 Agent
    │   ├── reasoning.py           # 推理 Agent
    │   └── reflection.py          # 反思/质量检查 Agent
    ├── memory/
    │   ├── conversation_memory.py # 对话历史（Redis/内存）
    │   ├── knowledge_memory.py    # 知识库（Chroma/FAISS）
    │   └── episodic_memory.py     # 情景记忆（策略经验）
    ├── self_improving/
    │   ├── evaluation.py          # 自动评估层（RAGAS 风格）
    │   ├── feedback_learning.py   # 反馈学习
    │   └── system_optimization.py # 系统自动优化
    ├── workflow/
    │   └── graph.py               # LangGraph 主工作流定义
    └── utils/
        └── llm.py                 # LLM 工具函数
```

---

## 四大核心层

### 1. Query Layer
用户原始查询入口。

### 2. Agentic RAG Layer

| Agent | 功能 |
|-------|------|
| `QueryUnderstandingAgent` | 查询重写、意图识别、复杂度估计、查询分解 |
| `TaskPlanningAgent` | 任务规划，决定执行步骤和调用哪些组件 |
| `RetrievalPlanningAgent` | 检索策略规划（向量搜索/GraphRAG/Web搜索/混合/跳过） |
| `MultiRetrieverAgent` | 并行执行多个检索器，聚合去重结果 |
| `RerankAgent` | 精排，基于相关性评分过滤文档 |
| `ContextEngineeringAgent` | 去重、压缩、过滤、摘要、上下文排序 |
| `ReasoningAgent` | Chain-of-Thought、多跳推理、工具使用 |
| `ReflectionAgent` | 答案验证、幻觉检测、置信度评分 |

### 3. Memory Layer

| 组件 | 功能 |
|------|------|
| `ConversationMemory` | 对话历史，支持 Redis 或内存后端 |
| `KnowledgeMemory` | 知识库，支持 Chroma、FAISS 等向量数据库 |
| `EpisodicMemory` | 情景记忆，存储哪些检索策略有效的经验 |

### 4. Self-Improving Layer

| 组件 | 功能 |
|------|------|
| `EvaluationLayer` | RAGAS 风格自动评估（faithfulness、relevancy 等） |
| `FeedbackLearning` | 根据评估结果生成优化建议，学习有效策略 |
| `SystemOptimization` | Prompt 优化、检索参数调整、策略选择优化 |

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 OpenAI API Key
OPENAI_API_KEY=sk-...
```

### 基础使用

```python
import asyncio
from astra_rag import AstraRAGSystem
from astra_rag.core.config import SystemConfig

async def main():
    # 初始化系统
    config = SystemConfig()
    system = AstraRAGSystem(config=config)

    # 执行查询
    result = await system.arun(
        query="什么是 RAG？它有哪些主要组成部分？",
        session_id="demo-session"
    )

    print("答案:", result["answer"])
    print("置信度:", result["confidence_score"])
    print("推理链:", result["reasoning_chain"])

asyncio.run(main())
```

更多示例请参考 [`examples/basic_usage.py`](examples/basic_usage.py)。

---

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `OPENAI_API_KEY` | — | OpenAI API 密钥（必填） |
| `OPENAI_BASE_URL` | OpenAI 官方 | 自定义 API 地址（兼容其他 OpenAI 生态服务） |
| `ASTRA_LLM_MODEL` | `gpt-4o-mini` | 默认 LLM 模型 |
| `ASTRA_TEMPERATURE` | `0.0` | LLM 温度 |
| `ASTRA_TOP_K` | `5` | 检索 Top-K 数量 |
| `ASTRA_REDIS_URL` | `redis://localhost:6379` | Redis 连接地址 |
| `ASTRA_MAX_ITERATIONS` | `3` | 最大反思迭代次数 |
| `ASTRA_DEBUG` | `false` | 开启调试日志 |

---

## 扩展开发

### 自定义 Agent

```python
from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.state import GraphState

class MyCustomAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "my_custom_agent"

    @property
    def description(self) -> str:
        return "My custom processing step."

    def run(self, state: GraphState) -> GraphState:
        # 实现你的逻辑
        result = self._call_llm([...])
        return self._update_state(state, {"answer": result})
```

### 自定义检索器

```python
from astra_rag.agents.multi_retriever import BaseRetriever

class MyVectorRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        # 连接你的向量数据库
        ...
```

---

## 技术栈

- **LangGraph** — 多 Agent 工作流编排
- **LangChain + LangChain-OpenAI** — LLM 调用和工具链
- **OpenAI GPT-4o** — 核心大模型
- **Pydantic v2** — 数据校验和结构化输出
- **Tenacity** — LLM 调用重试
- **Redis** — 对话历史持久化
- **ChromaDB / FAISS** — 向量知识库

---

## License

MIT

