"""
Client for communicating with the NN model service.
"""

import asyncio
import logging
import httpx
from config import settings

logger = logging.getLogger(__name__)

# Fallback system prompt — used when NN service is unavailable
FALLBACK_SYSTEM_PROMPT = (
    "Ты — опытный лайф-коуч и психолог. Твоя задача — помогать людям "
    "разобраться в себе, ставить цели и находить мотивацию. "
    "Отвечай ТОЛЬКО на русском языке. Будь эмпатичным, задавай уточняющие вопросы."
)


async def get_system_prompt() -> str:
    """Fetch system prompt from NN service, with fallback."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{settings.NN_SERVICE_URL}/system-prompt")
            resp.raise_for_status()
            return resp.json()["system_prompt"]
    except Exception as e:
        logger.warning(f"NN service unavailable for system prompt, using fallback: {e}")
        return FALLBACK_SYSTEM_PROMPT


async def generate_response(messages: list[dict]) -> str:
    """
    Send conversation messages to NN service and get response.
    Includes retry logic for transient failures.
    """
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
    last_error = None

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{settings.NN_SERVICE_URL}/generate",
                    json={"messages": messages},
                )
                resp.raise_for_status()
                return resp.json()["response"]
        except httpx.ConnectError as e:
            last_error = e
            logger.warning(f"NN service connect error (attempt {attempt + 1}/3): {e}")
            await asyncio.sleep(2)
        except httpx.ReadTimeout as e:
            # Don't retry timeouts — model is just slow
            raise Exception(f"Model generation timed out after 600s: {e}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"NN service error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            last_error = e
            logger.warning(f"NN service error (attempt {attempt + 1}/3): {e}")
            await asyncio.sleep(2)

    raise Exception(f"NN service unavailable after 3 attempts: {last_error}")


async def check_health() -> bool:
    """Check if NN service is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.NN_SERVICE_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False
