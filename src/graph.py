import os
from typing import List, TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 导入你之前写的模块
from retrieval import MedicalVectorStore
from chains import MedicalRAGChain

# 1. 定义状态 (State)
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    # 可以在这里扩展，比如：
    # steps: List[str] 

# 2. 定义辅助组件：文档评分器 (Grader)
# 这一步是解决幻觉的关键：判断检索到的内容是否真的和问题相关
class GradeAnswer(BaseModel):
    """用于判断文档相关性的二元评分结构"""
    binary_score: str = Field(description="相关性评分，'yes' 或 'no'")

def get_grader():
    llm = ChatOpenAI(
        model="qwen-flash", 
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0
    )
    # 强制 LLM 输出结构化数据
    structured_llm = llm.with_structured_output(GradeAnswer)
    
    system_prompt = """你是一个医学文献质量评估员。你的任务是判断给定的参考资料是否与用户的问题直接相关。
    如果是医学上的相关信息（如病因、诊断、治疗建议等），请回答 'yes'，否则回答 'no'。"""
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "参考资料: \n\n {document} \n\n 用户问题: {question}")
    ])
    
    return grader_prompt | structured_llm

# 3. 定义节点 (Nodes)

# 初始化组件 (全局，避免重复加载)
project_root = Path(__file__).resolve().parent.parent
db_dir = str(project_root / "data" / "vector_store")
store_manager = MedicalVectorStore(db_dir)
rag_chain_manager = MedicalRAGChain(store_manager)
doc_grader = get_grader()

def retrieve(state: GraphState):
    print("--- 步骤1：执行向量搜索 ---")
    question = state["question"]
    # 搜索 Top-5 片段
    results = store_manager.search(question, k=5)
    # 注意：search 返回的是 (doc, score) 元组，我们只需要 doc
    docs = [res[0] for res in results]
    return {"documents": docs, "question": question}

def grade_documents(state: GraphState):
    print("--- 步骤2：评估文档相关性 ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        # 调用评分器
        score = doc_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print(f"  [✓] 相关: {d.metadata.get('source')[:20]}...")
            filtered_docs.append(d)
        else:
            print(f"  [✗] 无关: {d.metadata.get('source')[:20]}...")
            
    return {"documents": filtered_docs, "question": question}

def generate(state: GraphState):
    print("--- 步骤3：生成最终回复 ---")
    question = state["question"]
    docs = state["documents"]
    
    if not docs:
        return {"generation": "抱歉，我在提供的临床指南中没有找到与该问题相关的确切信息。", "documents": docs}
    
    # 复用之前 chains.py 里的逻辑
    rag_chain = rag_chain_manager.get_chain()
    # 注意：我们的 chain 内部已经包含了 retriever。
    # 这里为了演示 Graph，我们可以直接调用 LLM，或者修改 chain 接收 docs 
    # 这里采用直接调用封装好的 chain (它会二次搜索，但更稳定)
    generation = rag_chain.invoke(question)
    
    return {"generation": generation, "documents": docs}

# 4. 构建图 (Graph Construction)

workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_docs", grade_documents)
workflow.add_node("generate", generate)

# 建立连接
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_docs")
workflow.add_edge("grade_docs", "generate")
workflow.add_edge("generate", END)

# 编译应用
app = workflow.compile()

# 5. 运行测试
if __name__ == "__main__":
    test_question = "对于高钾血症患者，血液透析是如何清除钾离子的？"
    
    print(f"\n🚀 启动医疗 RAG 工作流 \n问题: {test_question}\n")
    
    # 设置一个递归上限（虽然后面只有直线逻辑，但在循环图中很重要）
    config = {"recursion_limit": 10}
    
    for output in app.stream({"question": test_question}, config):
        # 打印当前正在运行的节点名
        for key, value in output.items():
            pass # 节点内部已经有打印输出了
            
    final_result = app.invoke({"question": test_question})
    print("\n" + "★" * 30 + " 最终诊断建议 " + "★" * 30)
    print(final_result["generation"])