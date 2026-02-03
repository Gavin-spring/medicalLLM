import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class MedicalIngestor:
    def __init__(self, base_dir: Path):
        # 自动定位路径，不管你在哪运行脚本都能找到 data 文件夹
        self.base_dir = base_dir
        if not self.base_dir.exists():
            raise FileNotFoundError(f"路径不存在: {self.base_dir}")
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

    def _extract_metadata(self, file_path: Path) -> Dict:
        # 获取相对于 data/raw 的相对路径，从而准确提取类别
        relative_path = file_path.relative_to(self.base_dir)
        category = relative_path.parts[0] # 获取第一级子目录名
        
        return {
            "source": file_path.name,
            "category": category,
            "disease": self._guess_disease(file_path.name),
            "path": str(relative_path)
        }

    def _guess_disease(self, filename: str) -> str:
        disease_map = {"BK": "BKV", "Alport": "Alport", "狼疮": "LN", "CKD": "CKD"}
        for k, v in disease_map.items():
            if k.lower() in filename.lower(): return v
        return "General"

    def process_pdf(self, file_path: Path) -> List[Document]:
        print(f"  -> 正在解析: {file_path.name}")
        try:
            doc = fitz.open(file_path)
            base_meta = self._extract_metadata(file_path)
            chunks = []
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if len(text) < 50: continue
                
                # 创建原始文档对象
                page_doc = Document(
                    page_content=text, 
                    metadata={**base_meta, "page": i+1}
                )
                # 切分
                chunks.extend(self.text_splitter.split_documents([page_doc]))
            doc.close()
            return chunks
        except Exception as e:
            print(f"  ❌ 解析失败 {file_path.name}: {e}")
            return []

    def run(self, target_category: Optional[str] = None, limit: int = 999):
        """
        target_category: 指定处理哪个子文件夹，如 'Consensus'
        """
        # 路径搜索逻辑：如果是 Consensus，就只搜 raw/Consensus/*.pdf
        search_path = self.base_dir / target_category if target_category else self.base_dir
        print(f"🔍 搜索目录: {search_path.absolute()}")
        
        # 使用 rglob 进行递归搜索所有 PDF
        pdf_files = list(search_path.rglob("*.pdf"))[:limit]
        print(f"📂 找到 {len(pdf_files)} 个待处理文件")
        
        all_chunks = []
        for f in pdf_files:
            all_chunks.extend(self.process_pdf(f))
            
        print(f"✨ 处理完成: 生成 {len(all_chunks)} 个 Chunks")
        return all_chunks

if __name__ == "__main__":
    from retrieval import MedicalVectorStore
    
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    vector_db_dir = str(project_root / "data" / "vector_store") # 数据库路径
    
    ingestor = MedicalIngestor(raw_data_dir)
    
    # 1. 提取 Consensus
    print("🚀 开始解析 PDF...")
    consensus_chunks = ingestor.run(target_category="Consensus")
    
    # 2. 存入数据库
    if consensus_chunks:
        print(f"📦 正在初始化数据库并存入 {len(consensus_chunks)} 个 Chunks...")
        db_manager = MedicalVectorStore(vector_db_dir)
        db_manager.add_documents(consensus_chunks)
        print("✅ 数据库构建完成！")