# bot.py
import os
import asyncio
from collections import deque
from typing import Deque, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

try:
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
except ValueError:
    TEMPERATURE = 0.7

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "30"))
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "3500"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

History = Deque[dict]
chat_histories: Dict[int, History] = {}


def get_history(chat_id: int) -> History:
    history = chat_histories.get(chat_id)
    if history is None:
        history = deque(maxlen=MAX_HISTORY)
        chat_histories[chat_id] = history
    return history


def build_payload(history: List[dict]) -> dict:
    payload = {
        "contents": history,
        "generationConfig": {
            "temperature": TEMPERATURE,
        },
    }
    if SYSTEM_PROMPT:
        payload["systemInstruction"] = {
            "parts": [{"text": SYSTEM_PROMPT}]
        }
    return payload


def extract_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""

    texts = []
    for part in parts:
        text = part.get("text")
        if text:
            texts.append(text)
    return "".join(texts).strip()


def trim_for_telegram(text: str) -> str:
    if len(text) <= MAX_REPLY_CHARS:
        return text
    suffix = "…"
    cut = MAX_REPLY_CHARS - len(suffix)
    return text[: max(0, cut)].rstrip() + suffix


async def call_gemini(history: List[dict]) -> str:
    payload = build_payload(history)
    headers = {
        "x-goog-api-key": GEMINI_API_KEY or "",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(GEMINI_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    return extract_text(data)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет! Я бот с интеграцией Google Gemini.\n"
        "Напишите сообщение — я отвечу.\n"
        "Команда /new очищает историю диалога."
    )
    await update.message.reply_text(text)


async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    chat_histories.pop(chat_id, None)
    await update.message.reply_text("История диалога очищена.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        await update.message.reply_text(
            "Ошибка конфигурации: проверьте TELEGRAM_BOT_TOKEN и GEMINI_API_KEY в .env"
        )
        return

    chat_id = update.effective_chat.id
    history = get_history(chat_id)

    user_text = update.message.text.strip()
    history.append({"role": "user", "parts": [{"text": user_text}]})

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        reply_text = await call_gemini(list(history))
        if not reply_text:
            reply_text = "Gemini вернул пустой ответ. Попробуйте переформулировать запрос."
    except httpx.HTTPStatusError as exc:
        reply_text = (
            "Ошибка Gemini API. Проверьте ключ и модель, затем повторите запрос. "
            f"(HTTP {exc.response.status_code})"
        )
    except httpx.TimeoutException:
        reply_text = "Таймаут запроса к Gemini. Попробуйте позже."
    except Exception:
        reply_text = "Непредвиденная ошибка при обращении к Gemini."

    history.append({"role": "model", "parts": [{"text": reply_text}]})

    await update.message.reply_text(trim_for_telegram(reply_text))


def validate_env() -> Optional[str]:
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        return "Отсутствуют переменные окружения: " + ", ".join(missing)
    return None


def main() -> None:
    error = validate_env()
    if error:
        print(error)
        print("Создайте .env и заполните его значениями.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("new", new_chat))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()


if __name__ == "__main__":
    main()
