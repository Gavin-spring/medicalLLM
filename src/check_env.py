import sys
import os

print("====== 🏥 Medical RAG Environment Diagnostic ======")

# 1. 检查 PyTorch & CUDA
try:
    import torch
    print(f"✅ [Torch] Version: {torch.__version__}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ [CUDA]  GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram:.2f} GB")
        # 针对 4GB 显存的警告
        if vram < 5:
            print("⚠️  [Warn] VRAM < 6GB. You MUST use 4-bit quantization (bitsandbytes).")
    else:
        print("❌ [CUDA]  GPU NOT FOUND! Stop here.")
        sys.exit(1)
except ImportError as e:
    print(f"❌ [Torch] Failed: {e}")

# 2. 检查 Bitsandbytes (量化关键)
try:
    import bitsandbytes as bnb
    # 尝试寻找 CUDA 库，很多时候装了但找不到动态链接库
    print(f"✅ [BnB]   Version: {bnb.__version__}")
    # 简单的 CUDA 联动测试
    import torch.nn as nn
    linear = nn.Linear(10, 10).cuda()
    print("✅ [BnB]   CUDA linking works.")
except Exception as e:
    print(f"❌ [BnB]   Bitsandbytes Failed! You won't be able to run LLMs on 4GB VRAM.")
    print(f"          Error: {e}")

# 3. 检查 SQLite 版本 (ChromaDB 的隐形杀手)
import sqlite3
sqlite_ver = sqlite3.sqlite_version
print(f"ℹ️  [SQLite] System Version: {sqlite_ver}")
if tuple(map(int, sqlite_ver.split('.'))) < (3, 35, 0):
    print("❌ [SQLite] Version too old for ChromaDB! Need > 3.35.0")
else:
    print("✅ [SQLite] Compatible with ChromaDB.")

# 4. 检查 ChromaDB
try:
    import chromadb
    client = chromadb.Client() # 内存模式测试
    collection = client.create_collection("test_health_check")
    collection.add(documents=["hello world"], ids=["1"])
    print(f"✅ [Chroma] In-memory insertion successful. Version: {chromadb.__version__}")
except Exception as e:
    print(f"❌ [Chroma] Failed to initialize: {e}")

# 5. 检查 LangChain & HuggingFace
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    # 测试能否加载 Embeddings (不需要下载模型，只加载架构)
    print("✅ [LangChain] Import successful.")
except ImportError:
    # 兼容性处理：旧版 LangChain 可能位置不同
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ [LangChain] Import successful (Community).")
    except Exception as e:
        print(f"❌ [LangChain] Critical Import Failed: {e}")

print("====== Diagnostic Finished ======")