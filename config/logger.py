from loguru import logger
import sys
import os

# 로그 폴더 생성
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger():
    logger.remove()  # 기본 로그 제거
    
    # 콘솔 출력용
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
        level="INFO",
        colorize=True,
    )

    # 파일 저장용 (DEBUG~)
    logger.add(
        f"{LOG_DIR}/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",    # 10MB 넘어가면 자동 회전
        retention="10 days",  # 10일 보관
        encoding="utf8"
    )

    logger.info("Logger initialized")
