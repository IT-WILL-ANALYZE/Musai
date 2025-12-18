import json
import os
from datetime import datetime
from loguru import logger

# ----------------------------------------------------
# Knowledgebase 저장
# ----------------------------------------------------
def store_knowledgebase(file_url, chunked_docs, vector_ids):
    logger.info(f"Start store_knowledgebase : {file_url, chunked_docs, vector_ids}")

    base_name = os.path.splitext(os.path.basename(file_url))[0]

    base_path = "rag_resources/knowledge-base"
    os.makedirs(base_path, exist_ok=True)

    # 중복된 이름 처리
    file_name = base_name
    counter = 1
    while os.path.exists(os.path.join(base_path, f"{file_name}.json")):
        file_name = f"{base_name}_{counter}"
        counter += 1

    data = {
        "meta": {
            "file_name": file_name,
            "total_chunks": len(chunked_docs),
            "created_at": datetime.now().isoformat()
        },
        "chunks": []
    }

    for i, doc in enumerate(chunked_docs):
        data["chunks"].append({
            "id": f"chunk_{i+1:03}",
            "content": doc.page_content,
            "metadata": doc.metadata,
            "vector_id": vector_ids[i] if vector_ids else None
        })

    save_path = os.path.join(base_path, f"{file_name}.json")

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.success(f"Done store_knowledgebase : {save_path}")
        return save_path

    except Exception as e:
        logger.exception(f"Failed store_knowledgebase : {e}")
        raise
