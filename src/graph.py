from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

# 1. 定义状态 (State)
# 这是在各个节点之间流转的数据结构
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    is_hallucination: bool

# 2. 定义节点 (Nodes)
def retrieve(state: GraphState):
    """从向量数据库检索相关文档"""
    print("---RETRIEVE---")
    question = state["question"]
    # 伪代码：调用你的检索器
    # documents = vector_store.similarity_search(question)
    documents = [Document(page_content="模拟检索到的狼疮肾炎指南片段...")] 
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """基于文档生成回答"""
    print("---GENERATE---")
    question = state["question"]
    docs = state["documents"]
    # 伪代码：调用 LLM
    # generation = rag_chain.invoke({"context": docs, "question": question})
    generation = "根据指南，建议使用..." 
    return {"documents": docs, "question": question, "generation": generation}

def grade_documents(state: GraphState):
    """(可选) 评分：检索到的文档真的和问题相关吗？"""
    print("---CHECK RELEVANCE---")
    # 这里可以用 LLM 快速判断相关性
    return state

# 3. 构建图 (Graph Construction)
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# 定义边 (Edges) - 这里的逻辑是线性的，高级版可以加判断条件做回环
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 编译
app = workflow.compile()

# 4. 运行
if __name__ == "__main__":
    result = app.invoke({"question": "狼疮肾炎的诱导治疗方案是什么？"})
    print(f"Final Answer: {result['generation']}")