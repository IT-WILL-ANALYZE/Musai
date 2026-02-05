import streamlit as st
import json
import os
from datetime import datetime
from loguru import logger
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# ------------------------------
# 설정
# ------------------------------
KNOWLEDGE_BASE_PATH = "rag_resources/knowledge-base"
VECTORDB_DIR = "rag_resources/vectordb/chroma_store"
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

st.set_page_config(
    page_title="MusAi - 데이터 관리",
    page_icon="🔧",
    layout="wide"
)

# ------------------------------
# 세션 상태 초기화
# ------------------------------
if "admin_verified" not in st.session_state:
    st.session_state.admin_verified = False
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "knowledge_data" not in st.session_state:
    st.session_state.knowledge_data = None
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = {}


# ------------------------------
# JSON 파일 관련 함수
# ------------------------------
def get_json_files():
    """knowledge-base 폴더의 모든 JSON 파일 목록 반환"""
    try:
        files = [f for f in os.listdir(KNOWLEDGE_BASE_PATH) if f.endswith('.json')]
        return sorted(files)
    except Exception as e:
        logger.error(f"Failed to get JSON files: {e}")
        return []


def load_json_file(filename):
    """JSON 파일 로드"""
    try:
        file_path = os.path.join(KNOWLEDGE_BASE_PATH, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file {filename}: {e}")
        st.error(f"파일 로드 실패: {e}")
        return None


def save_json_file(filename, data):
    """JSON 파일 저장"""
    try:
        file_path = os.path.join(KNOWLEDGE_BASE_PATH, filename)
        # total_chunks 업데이트
        data["meta"]["total_chunks"] = len(data["chunks"])
        data["meta"]["updated_at"] = datetime.now().isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.success(f"Saved JSON file: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file {filename}: {e}")
        st.error(f"파일 저장 실패: {e}")
        return False


def delete_json_file(filename):
    """JSON 파일 및 VectorDB 데이터 삭제"""
    try:
        # 1. JSON 파일 로드하여 vector_ids 수집
        file_path = os.path.join(KNOWLEDGE_BASE_PATH, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vector_ids = [chunk.get("vector_id") for chunk in data.get("chunks", []) if chunk.get("vector_id")]
        
        # 2. VectorDB에서 벡터 삭제
        if vector_ids:
            try:
                vectordb = Chroma(
                    embedding_function=embedding,
                    persist_directory=VECTORDB_DIR
                )
                vectordb.delete(ids=vector_ids)
                logger.info(f"Deleted {len(vector_ids)} vectors from VectorDB")
            except Exception as e:
                logger.warning(f"Failed to delete vectors from VectorDB: {e}")
        
        # 3. JSON 파일 삭제
        os.remove(file_path)
        logger.success(f"Deleted JSON file: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {e}")
        st.error(f"파일 삭제 실패: {e}")
        return False


def sync_to_vectordb(data):
    """JSON 데이터를 VectorDB에 동기화"""
    try:
        logger.info("Start sync_to_vectordb")
        
        # 1. JSON chunks를 Document 객체로 변환
        documents = []
        old_vector_ids = []
        
        for chunk in data.get("chunks", []):
            doc = Document(
                page_content=chunk.get("content", ""),
                metadata=chunk.get("metadata", {})
            )
            documents.append(doc)
            
            # 기존 vector_id 수집 (삭제용)
            vector_id = chunk.get("vector_id")
            if vector_id:
                old_vector_ids.append(vector_id)
        
        if not documents:
            st.warning("동기화할 chunk가 없습니다.")
            return False, data
        
        # 2. VectorDB 연결
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=VECTORDB_DIR
        )
        
        # 3. 기존 vector 삭제 (vector_id가 있는 경우에만)
        if old_vector_ids:
            try:
                vectordb.delete(ids=old_vector_ids)
                logger.info(f"Deleted {len(old_vector_ids)} old vectors")
            except Exception as e:
                logger.warning(f"Failed to delete old vectors: {e}")
        
        # 4. 새 vector 추가
        documents = filter_complex_metadata(documents)
        new_vector_ids = vectordb.add_documents(documents)
        logger.info(f"Added {len(new_vector_ids)} new vectors")
        
        # 5. JSON 데이터에 새 vector_id 업데이트
        for i, chunk in enumerate(data["chunks"]):
            if i < len(new_vector_ids):
                chunk["vector_id"] = new_vector_ids[i]
        
        return True, data
        
    except Exception as e:
        logger.exception(f"Failed sync_to_vectordb: {e}")
        st.error(f"VectorDB 동기화 실패: {e}")
        return False, data


# ------------------------------
# UI 렌더링
# ------------------------------
st.title("🔧 Vector DB 데이터 관리")
st.caption("knowledge-base의 JSON 파일을 조회하고 수정할 수 있습니다.")

# ------------------------------
# 파일 선택 영역
# ------------------------------
json_files = get_json_files()

if not json_files:
    st.warning("📁 knowledge-base 폴더에 JSON 파일이 없습니다.")
    st.stop()

# 사이드바에 파일 목록 표시(관리자 모드)
if st.session_state.admin_verified:
    with st.sidebar:
        st.header("📂 파일 목록")
        
        for idx, file in enumerate(json_files):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if st.button(
                    file, 
                    key=f"select_file_{idx}_{file}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_file == file else "secondary"
                ):
                    st.session_state.selected_file = file
                    st.session_state.knowledge_data = load_json_file(file)
                    st.rerun()
            
            with col2:
                if st.button("🗑", key=f"delete_file_{idx}", help="파일 및 VectorDB 삭제"):
                    with st.spinner(f"🗑 {file} 삭제 중..."):
                        if delete_json_file(file):
                            if st.session_state.selected_file == file:
                                st.session_state.selected_file = None
                                st.session_state.knowledge_data = None
                            st.success(f"✅ {file} 삭제 완료! (JSON + VectorDB)")
                            st.rerun()
                        else:
                            st.error(f"❌ {file} 삭제 실패!")
        
        st.divider()
        st.caption(f"총 {len(json_files)}개의 파일")


# ------------------------------
# 메인 영역: 선택된 파일 표시
# ------------------------------
if not st.session_state.selected_file:
    if st.session_state.admin_verified:
        st.info("👈 사이드바에서 파일을 선택해주세요.")
    else:
        st.error("🚨 관리자 모드에서만 접근 가능합니다.")
    st.stop()
    

data = st.session_state.knowledge_data

if not data:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# 파일 메타정보 표시
st.subheader(f"📄 {st.session_state.selected_file}")

meta = data.get("meta", {})
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("총 Chunk 수", meta.get("total_chunks", 0))
with col2:
    st.metric("생성일", meta.get("created_at", "N/A")[:10])
with col3:
    if "updated_at" in meta:
        st.metric("수정일", meta.get("updated_at", "N/A")[:10])

st.divider()

# ------------------------------
# Chunk 관리 영역
# ------------------------------
st.subheader("🧩 Chunk 관리")

chunks = data.get("chunks", [])

# VectorDB 동기화 상태 확인
synced_count = sum(1 for c in chunks if c.get("vector_id"))
total_count = len(chunks)

if total_count > 0:
    sync_status = f"🟢 {synced_count}/{total_count} Chunks 동기화됨" if synced_count == total_count else f"🟡 {synced_count}/{total_count} Chunks 동기화됨"
    st.info(sync_status)

if not chunks:
    st.info("이 파일에는 chunk가 없습니다.")
else:
    # 삭제/추가할 chunk 추적
    delete_chunk_idx = None
    add_after_idx = None
    save_chunk_idx = None
    
    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("id", f"chunk_{idx}")
        vector_id = chunk.get('vector_id')
        
        # Vector ID 표시 (동기화 여부 확인)
        if vector_id:
            vector_status = f"✅ Vector ID: {vector_id[:8]}..."
        else:
            vector_status = "❌ 미동기화"
        
        with st.expander(f"📦 {chunk_id} - {vector_status}", expanded=False):
            # Metadata 표시
            metadata = chunk.get("metadata", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"📌 Source: {metadata.get('source', 'N/A')}")
            with col2:
                st.text(f"🏷 Category: {metadata.get('category', 'N/A')}")
            
            st.markdown("---")
            
            # Content 편집
            edited_content = st.text_area(
                "Content",
                value=chunk.get("content", ""),
                height=200,
                key=f"edit_{st.session_state.selected_file}_{idx}",
                label_visibility="collapsed"
            )
            
            # 버튼 영역
            st.markdown("---")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("💾 저장", key=f"save_{st.session_state.selected_file}_{idx}", type="primary", use_container_width=True):
                    # 내용 업데이트
                    chunk["content"] = edited_content
                    save_chunk_idx = idx
            
            with btn_col2:
                if st.button("➕ 추가", key=f"add_{st.session_state.selected_file}_{idx}", type="secondary", use_container_width=True):
                    add_after_idx = idx
            
            with btn_col3:
                if st.button("🗑 삭제", key=f"delete_{st.session_state.selected_file}_{idx}", type="secondary", use_container_width=True):
                    delete_chunk_idx = idx

    # 저장 처리 (해당 chunk만)
    if save_chunk_idx is not None:
        with st.spinner(f"💾 {chunks[save_chunk_idx]['id']} 저장 중..."):
            success, updated_data = sync_to_vectordb(data)
            if success and save_json_file(st.session_state.selected_file, updated_data):
                st.success(f"✅ {chunks[save_chunk_idx]['id']} 저장 완료!")
                st.session_state.knowledge_data = updated_data
                st.balloons()
                st.rerun()
            else:
                st.error("❌ 저장 실패!")

    # 삭제 처리
    if delete_chunk_idx is not None:
        deleted_id = chunks[delete_chunk_idx]['id']
        with st.spinner(f"🗑 {deleted_id} 삭제 중..."):
            data["chunks"].pop(delete_chunk_idx)
            success, updated_data = sync_to_vectordb(data)
            if success and save_json_file(st.session_state.selected_file, updated_data):
                st.success(f"✅ {deleted_id} 삭제 완료!")
                st.session_state.knowledge_data = updated_data
                st.rerun()
            else:
                st.error("❌ 삭제 실패!")

    # 추가 처리
    if add_after_idx is not None:
        st.session_state.add_after_idx = add_after_idx
        st.rerun()

# 새 Chunk 추가 폼 (특정 위치 다음에)
if "add_after_idx" in st.session_state:
    add_idx = st.session_state.add_after_idx
    
    st.divider()
    st.subheader(f"➕ Chunk {chunks[add_idx]['id']} 다음에 추가")
    
    with st.form("add_chunk_form", clear_on_submit=True):
        new_content = st.text_area("새 Chunk 내용", height=150, placeholder="여기에 새로운 chunk 내용을 입력하세요...")
        
        col1, col2 = st.columns(2)
        with col1:
            new_source = st.text_input("Source", value="manual_input")
        with col2:
            new_category = st.text_input("Category", value="Manual")
        
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            submitted = st.form_submit_button("➕ 추가", type="primary", use_container_width=True)
        with form_col2:
            cancelled = st.form_submit_button("❌ 취소", use_container_width=True)
        
        if submitted and new_content.strip():
            with st.spinner("➕ 새 Chunk 추가 중..."):
                # 새 chunk ID 생성
                new_id = f"chunk_{len(chunks) + 1:03}"
                
                new_chunk = {
                    "id": new_id,
                    "content": new_content.strip(),
                    "metadata": {
                        "source": new_source,
                        "category": new_category
                    },
                    "vector_id": None
                }
                
                # 지정된 위치 다음에 삽입
                data["chunks"].insert(add_idx + 1, new_chunk)
                
                # VectorDB 동기화 후 JSON 저장
                success, updated_data = sync_to_vectordb(data)
                if success and save_json_file(st.session_state.selected_file, updated_data):
                    st.success(f"✅ 새 Chunk ({new_id}) 추가 완료!")
                    st.session_state.knowledge_data = updated_data
                    del st.session_state.add_after_idx
                    st.rerun()
                else:
                    st.error("❌ 추가 실패!")
        
        if cancelled:
            del st.session_state.add_after_idx
            st.rerun()
