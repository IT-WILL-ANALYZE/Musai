import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def get_bin_path(tool_name: str) -> str:
    """
    1. config/bin 하위의 특정 도구 경로를 반환합니다.
    2. 해당 경로를 시스템 PATH에 자동으로 추가하여 라이브러리들이 바로 인식하게 합니다.
    """
    # 현재 파일 위치에 따라 parents[n] 값을 조정하세요. 
    # (이 파일이 utils/path.py 등에 있다면 parents[1]이 프로젝트 루트입니다)
    root = Path(__file__).resolve().parents[1] 
    
    bin_path = root / "config" / "bin" / tool_name
    abs_path = str(bin_path.absolute())
    
    # 1. 경로 존재 여부 확인
    if not bin_path.exists():
        logger.warning(f"Binary path not found at: {abs_path}")
        return abs_path
    
    # 2. 시스템 PATH에 추가 (이미 있으면 중복 추가 방지)
    # os.pathsep은 Windows에서는 ';', Linux에서는 ':'를 자동으로 선택해줍니다.
    current_path = os.environ.get("PATH", "")
    if abs_path not in current_path:
        os.environ["PATH"] = abs_path + os.pathsep + current_path
        logger.info(f"Successfully added to PATH: {abs_path}")
    
    return abs_path