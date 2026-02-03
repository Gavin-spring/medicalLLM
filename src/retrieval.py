import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class MedicalVectorStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        # 1. 定义 Embedding 模型 (使用 BGE-Small 适配 4G 显存)
        # 第一次运行会自动下载，约 100MB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cuda'}, # 必须用 GPU 加速
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def get_vector_store(self):
        """获取或创建向量数据库实例"""
        return Chroma(
            collection_name="nephrology_kb",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents):
        """分批存入文档，防止 OOM (内存溢出)"""
        vector_store = self.get_vector_store()
        # 4966 个 chunk 建议分 10 批存入
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vector_store.add_documents(batch)
            print(f"  📥 已存入 {i + len(batch)} / {len(documents)} 条数据...")
 
    def search(self, query: str, k: int = 4):
        """
        语义搜索测试
        k: 返回最相关的片段数量
        """
        vector_store = self.get_vector_store()
        # similarity_search_with_score 会返回 (Document, score)
        # score 越小表示越相似（基于 L2 距离）
        results = vector_store.similarity_search_with_score(query, k=k)
        return results

if __name__ == "__main__":
    # 测试脚本
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    vector_db_dir = str(project_root / "data" / "vector_store")
    
    db_manager = MedicalVectorStore(vector_db_dir)
    
    # 模拟一个临床问题
    test_query = "CKD患者高钾血症的常用降钾药物有哪些？"
    print(f"\n🔍 测试查询: {test_query}")
    
    results = db_manager.search(test_query, k=3)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n--- 匹配结果 {i+1} (相似度分值: {score:.4f}) ---")
        print(f"来源: {doc.metadata.get('source')} | 第 {doc.metadata.get('page')} 页")
        print(f"内容: {doc.page_content[:200]}...")