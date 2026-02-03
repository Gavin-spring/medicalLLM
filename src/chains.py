import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class MedicalRAGChain:
    def __init__(self, vector_store_manager):
        self.vector_store = vector_store_manager.get_vector_store()
        
        # 1. 定义 LLM
        self.llm = ChatOpenAI(
            model="qwen-flash-2025-07-28", 
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0
        )
        
        # 2. 定义 Prompt 模板 (RAG 的灵魂)
        self.template = """你是一个专业的肾内科助手。请根据以下提供的【参考资料】回答用户的问题。
            如果你在资料中找不到答案，请直接回答“根据现有临床共识，无法给出确切建议”，切勿编造。
            回答请保持专业、严谨，并列出参考的文档来源。

            【参考资料】
            {context}

            【用户问题】
            {question}

            【专家建议】"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _format_docs(self, docs):
        """格式化检索到的文档，带上来源，方便模型引用"""
        formatted = []
        for d in docs:
            formatted.append(f"内容: {d.page_content}\n来源: {d.metadata.get('source')} (第{d.metadata.get('page')}页)")
        return "\n\n---\n\n".join(formatted)

    def get_chain(self):
        """构建 RAG 链"""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

if __name__ == "__main__":
    from retrieval import MedicalVectorStore
    from pathlib import Path
    
    # 初始化环境
    project_root = Path(__file__).resolve().parent.parent
    db_dir = str(project_root / "data" / "vector_store")
    
    store = MedicalVectorStore(db_dir)
    rag_system = MedicalRAGChain(store)
    
    # 第一次完整 RAG 测试
    chain = rag_system.get_chain()
    # question = "CKD患者血钾管理中，环硅酸锆钠的用法是什么？"
    question = "如何修理呼吸机？"
    
    print("\n🩺 AI 正在思考中...")
    response = chain.invoke(question)
    print("\n" + "="*50)
    print(response)
    print("="*50)