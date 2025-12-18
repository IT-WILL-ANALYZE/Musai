import json
from langchain_core.documents import Document
from loguru import logger
from langchain_core.messages import AIMessage

# ----------------------------------------------------
# LLM이 반환한 마크다운 형태 변환
# ----------------------------------------------------
def clean_llm_json(text) -> str:
    """
    Normalize LLM output into clean JSON string.
    Accepts str or AIMessage.
    """

    # AIMessage → content 추출
    if isinstance(text, AIMessage):
        text = text.content

    if not isinstance(text, str):
        raise TypeError(f"clean_llm_json expects str or AIMessage, got {type(text)}")

    text = text.strip()

    # ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = text.split("```")[1]

    # remove leading 'json\n'
    text = text.replace("json\n", "", 1).strip()

    return text


# ----------------------------------------------------
# LLM이 반환한 구조화 JSON을 LangChain Document list로 변환
# ----------------------------------------------------
def parse_structured_json(
    structured_result: str | dict,
    source: str | None = None,
    page: int | None = None,
    structure_type: str | None = None,
    structure_confidence: float | None = None,
):
    logger.info(f"Start parse_structured_json : {structured_result, source, page, structure_type, structure_confidence}")

    if not structured_result:
        return []

    # JSON 파싱
    try:
        if isinstance(structured_result, str):
            cleaned = clean_llm_json(structured_result)
            structured_data = json.loads(cleaned)
        else:
            structured_data = structured_result
    except Exception:
        logger.error("Failed to parse structured JSON")
        logger.debug(structured_result)
        return []

    items = structured_data.get("items")
    if not isinstance(items, list):
        logger.warning("Structured JSON has no 'items' list")
        return []

    docs: list[Document] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        # 구조 타입별 content 생성
        if structure_type == "qa_pairs":
            content = f"Q: {item.get('question')} | A: {item.get('answer')}"

        elif structure_type == "definition":
            content = f"{item.get('term')}: {item.get('definition')}"

        elif structure_type == "timeline":
            content = f"{item.get('time')}: {item.get('event')}"

        elif structure_type == "spec":
            content = f"{item.get('attribute')}: {item.get('value')}"

        elif structure_type == "list":
            content = item.get("item")

        elif structure_type == "table":
            headers = item.get("headers", [])
            row = item.get("row", [])
            content = " | ".join(
                f"{h}: {v}" for h, v in zip(headers, row)
            )

        else:
            content = " | ".join(
                f"{k}: {v}" for k, v in item.items() if v
            )

        if not content:
            continue

        metadata = {
            "type": "structured",
            "structure_type": structure_type,
            "row_index": idx,
        }

        if structure_confidence is not None:
            metadata["structure_confidence"] = structure_confidence
        if source:
            metadata["source"] = source
        if page is not None:
            metadata["page"] = page

        docs.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    logger.success(f"Done parse_structured_json : Parsed {len(docs)} structured items (type={structure_type})")

    return docs

