from loguru import logger
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.partition.xlsx import partition_xlsx


def load_by_unstructured (file_url: str, ext: str):
    logger.info(f"Start load_by_unstructured : {file_url, ext}")
    
    docs = []
    try : 
        if ext == ".pdf":
            docs = partition_pdf(file_url, strategy="hi_res")
        elif ext == ".docx":
            docs = partition_docx(file_url)
        elif ext == ".md":
            docs = partition_md(file_url)
        elif ext == ".html":
            docs = partition_html(file_url)
        elif ext in [".xlsx", ".xls"]:
            docs = partition_xlsx(file_url)
        else:
            raise ValueError("Unsupported format")
    
    except Exception as e:
        logger.exception(f"Failed load_by_unstructured : {e}")
        raise

    logger.success(f"Done load_by_unstructured : {len(docs)}")
    return docs