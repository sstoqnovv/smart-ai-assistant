"""
Smart Multimodal AI Assistant
============================

Уеб приложение за чат с изкуствен интелект, изградено със Streamlit.

Поддържа три режима на работа:
  1. Cloud  – Google Gemini 3.8 Flash (мултимодален, текст + изображения)
  2. Local  – Ollama (Llama 3.2 / Gemma 2 и др.) на локален сървър
  3. Hybrid – верига, в която Gemini извлича данните от снимката,
              а Ollama генерира крайния отговор

Стартиране:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import ollama as ollama_lib
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from PIL import Image

# ---------------------------------------------------------------------------
# Константи и конфигурация
# ---------------------------------------------------------------------------

GEMINI_MODEL: str = "gemini-3.8-flash"
# Резервен модел при претоварване на основния (HTTP 503 / UNAVAILABLE).
# Бележка: `gemini-3.8-flash-lite` НЕ съществува в Google API (връща 404),
# затова се използва наличният lite модел, който е проверен в работещо състояние.
GEMINI_FALLBACK_MODEL: str = "gemini-3.5-flash-lite"
DEFAULT_OLLAMA_MODEL: str = "gemma2:2b"
DEFAULT_OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS: int = 120

# Режими (стойността се използва в session_state, label-ът се показва в UI)
MODE_CLOUD: str = "cloud"
MODE_LOCAL: str = "local"
MODE_HYBRID: str = "hybrid"

MODE_LABELS: dict[str, str] = {
    MODE_CLOUD: "☁️ Cloud: Gemini 3.8 Flash (Multimodal)",
    MODE_LOCAL: "🏠 Local: Ollama (Llama 3.2 / Gemma 2)",
    MODE_HYBRID: "🔗 Hybrid Chain (Auto-Routing)",
}

ALLOWED_IMAGE_TYPES: tuple[str, ...] = ("png", "jpg", "jpeg", "webp")

# Зареждаме .env веднъж при стартиране на приложението.
load_dotenv()


# ---------------------------------------------------------------------------
# Малки помощни функции
# ---------------------------------------------------------------------------


def get_gemini_api_key() -> str:
    """Връща GEMINI_API_KEY от средата / .env (може да е празен низ)."""
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def get_ollama_base_url() -> str:
    """Връща URL адреса на Ollama сървъра (по подразбиране localhost:11434)."""
    url = (os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    return url


def image_mime_type(filename: str | None) -> str:
    """Определя MIME типа на каченото изображение по разширението."""
    ext = (filename or "png").rsplit(".", 1)[-1].lower()
    ext = ext if ext in ALLOWED_IMAGE_TYPES else "png"
    return f"image/{'jpeg' if ext == 'jpg' else ext}"


def encode_image(image_bytes: bytes) -> str:
    """Кодира байтове на изображение в base64 низ (за Ollama API-то)."""
    return base64.b64encode(image_bytes).decode("ascii")


def clear_uploaded_image() -> None:
    """
    Callback, който занулява стойността на file_uploader widget-а.

    Изпълнява се преди ререндъра на скрипта, затова е безопасно да се
    пипа session_state-ът на widget с ключ.
    """
    st.session_state["image_uploader"] = None


# ---------------------------------------------------------------------------
# Cloud модел – Google Gemini
# ---------------------------------------------------------------------------


def build_gemini_client() -> genai.Client:
    """
    Създава клиент към Google Gemini.

    При липсващ ключ се изхвърля ValueError с четимо съобщение,
    което по-късно се показва като st.error в интерфейса.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError(
            "Липсва `GEMINI_API_KEY`.\n\n"
            "Как да оправите:\n"
            "1. Създайте файл `.env` в корена на проекта (копирайте `.env.example`).\n"
            "2. Добавете реда `GEMINI_API_KEY=вашият_ключ`.\n"
            "3. Вземете ключ от https://aistudio.google.com/apikey и рестартирайте приложението."
        )
    return genai.Client(api_key=api_key)


def _is_model_unavailable(exc: Exception) -> bool:
    """
    Разпознава ли основният модел е временно недостъпен (претоварване).

    Третира като "недостъпен" грешка с код 503 / `ServerError` на Google SDK
    или съобщение, съдържащо UNAVAILABLE, "high demand" / "overloaded".
    В такъв случай има смисъл автоматично да се пробва резервният модел.
    """
    # 1. Стандартен код за статус от APIError / ServerError
    code = getattr(exc, "code", None)
    if code == 503:
        return True
    if isinstance(exc, genai_errors.ServerError):
        return True

    # 2. Резервен текстов признак – за различни версии на SDK / прокси
    text = str(exc).upper()
    return any(
        token in text
        for token in ("UNAVAILABLE", "HIGH DEMAND", "OVERLOADED", "503")
    )


def _generate_text_with(client: genai.Client, model: str, contents: list[Any]) -> str:
    """
    Едно извикване на `generate_content` към конкретен модел.

    Returns:
        Изчистен текстов отговор.

    Raises:
        Суровото изключение от SDK-то (проверява се от повикващия).
        ValueError: Ако моделът върне празен отговор.
    """
    response = client.models.generate_content(model=model, contents=contents)
    text = (response.text or "").strip()
    if not text:
        raise ValueError(
            f"Моделът `{model}` върна празен отговор. Опитайте с по-ясен prompt."
        )
    return text


def call_gemini(
    prompt: str,
    image_bytes: bytes | None = None,
    mime_type: str = "image/png",
) -> str:
    """
    Изпраща prompt (и опционално изображение) към Gemini 3.8 Flash.

    Ако основният модел е претоварен (HTTP 503 / UNAVAILABLE), заявката се
    повтаря автоматично с резервния модел `gemini-3.5-flash-lite`, преди да
    се покаже грешка на потребителя.

    Args:
        prompt: Въпросът / инструкцията на потребителя.
        image_bytes: Сурови байтове на каченото изображение (ако има).
        mime_type: MIME тип на изображението.

    Returns:
        Текстовият отговор от модела.

    Raises:
        ValueError: Ако липсва API ключ или моделите върнат празен отговор.
        RuntimeError: При грешка от API-то на Google (и при двата модела).
    """
    client = build_gemini_client()

    contents: list[Any] = []
    if image_bytes:
        contents.append(
            genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        )
    contents.append(prompt)

    # --- Стъпка 1: основен модел ---------------------------------------
    primary_exc: Exception | None = None
    try:
        return _generate_text_with(client, GEMINI_MODEL, contents)
    except ValueError:
        # Празният отговор не се "лекува" с друг модел – това не е 503.
        raise
    except Exception as exc:  # Google SDK хвърля разнообразни изключения
        if not _is_model_unavailable(exc):
            raise _gemini_error(exc, GEMINI_MODEL) from exc
        primary_exc = exc

    # --- Стъпка 2: автоматичен fallback при 503 / UNAVAILABLE -----------
    try:
        return _generate_text_with(client, GEMINI_FALLBACK_MODEL, contents)
    except Exception as fallback_exc:
        # Тук се обвива и празният отговор (ValueError), за да получи
        # потребителят четимо съобщение, а не сурово изключение.
        raise _gemini_error(
            fallback_exc,
            GEMINI_FALLBACK_MODEL,
            primary_exc=primary_exc,
        ) from fallback_exc


def _gemini_error(
    exc: Exception,
    model: str,
    primary_exc: Exception | None = None,
) -> RuntimeError:
    """Сглобява четимо съобщение за грешка от Gemini API (за st.error)."""
    if primary_exc is not None:
        details = (
            f"Основният модел `{GEMINI_MODEL}` е недоступен: {primary_exc}\n"
            f"Резервният модел `{model}` също се провали: {exc}"
        )
    else:
        details = f"Детайли: {exc}"
    return RuntimeError(
        "Грешка при извикване на Gemini API.\n\n"
        f"{details}\n\n"
        "Проверете дали `GEMINI_API_KEY` е валиден и дали имате достъп до модела "
        f"`{model}`."
    )


# ---------------------------------------------------------------------------
# Локален модел – Ollama
# ---------------------------------------------------------------------------


def ollama_status() -> tuple[bool, str]:
    """
    Проверява дали Ollama сървърът е достъпен.

    Returns:
        (доступен ли е, съобщение за грешка или празен низ)
    """
    base_url = get_ollama_base_url()
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return True, ""
    except requests.exceptions.ConnectionError:
        return False, (
            "Неуспешна връзка с Ollama сървъра на "
            f"`{base_url}`.\n\n"
            "Как да оправите:\n"
            "1. Инсталирайте Ollama от https://ollama.com\n"
            "2. Стартирайте сървъра: `ollama serve` (обикновено стартира автоматично).\n"
            "3. Изтеглете модела: `ollama run gemma2:2b`\n"
            "4. Ако Ollama е на друг адрес, сложете го в `.env` като "
            "`OLLAMA_BASE_URL=http://localhost:11434`."
        )
    except requests.exceptions.HTTPError as exc:
        return False, f"Ollama сървърът отговори с грешка: {exc}"
    except Exception as exc:
        return False, f"Неочаквана грешка при свързване с Ollama: {exc}"


def ollama_model_supports_vision(model: str) -> bool:
    """
    Проверява чрез `/api/show` дали локалният модел поддържа vision.

    При неуспешна проверка приемаме, че НЕ поддържа (по-безопасно поведение –
    избягваме неясни грешки от самия модел).
    """
    base_url = get_ollama_base_url()
    try:
        resp = requests.post(
            f"{base_url}/api/show",
            json={"name": model},
            timeout=10,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        capabilities = data.get("capabilities") or []
        return "vision" in capabilities
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """
    Връща списъка с локално наличните Ollama модели (чрез официалната
    библиотека `ollama`). При недостъпен сървър връща празен списък,
    за да не се срива UI-ят.
    """
    try:
        # Библиотеката ползва OLLAMA_HOST от средата; гарантираме, че е зададен.
        os.environ.setdefault("OLLAMA_HOST", get_ollama_base_url())
        response = ollama_lib.list()
        models = getattr(response, "models", None) or []
        names: list[str] = []
        for m in models:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.append(str(name))
        return sorted(names)
    except Exception:
        return []


def describe_image(image_bytes: bytes, mime_type: str) -> str:
    """
    Валидира изображението с Pillow и връща кратко описание (размер + формат).
    Използва се в sidebar-а, за да потвърди потребителят какво е качил.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            fmt = (img.format or mime_type.split("/")[-1].upper()).upper()
        return f"{width}×{height} px · {fmt} · {len(image_bytes) // 1024} KB"
    except Exception:
        return "невалиден или неподдържан файл"


def call_ollama(
    prompt: str,
    model: str,
    image_bytes: bytes | None = None,
    mime_type: str = "image/png",
) -> str:
    """
    Изпраща prompt (и опционално изображение) към локален Ollama модел.

    Args:
        prompt: Въпросът / инструкцията на потребителя.
        model: Име на локалния модел (напр. `gemma2:2b`, `gemma2`).
        image_bytes: Сурови байтове на изображението, ако моделът е vision-capable.
        mime_type: MIME тип на изображението (използва се за съобщение при отказ).

    Returns:
        Текстовият отговор от локалния модел.

    Raises:
        RuntimeError: При недостъпен сървър или върната грешка от Ollama.
        ValueError: Ако моделът не поддържа изображения.
    """
    available, err = ollama_status()
    if not available:
        raise RuntimeError(err)

    # Проверка за vision поддръжка – локалните LLM често са само текстови.
    if image_bytes and not ollama_model_supports_vision(model):
        raise ValueError(
            f"Локалният модел `{model}` няма vision поддръжка и не може да анализира изображения.\n\n"
            "Опции:\n"
            "• Използвайте режим **Hybrid Chain** – там Gemini анализира снимката, "
            "а Ollama обработва резултата.\n"
            "• Изберете **Cloud: Gemini** за директен анализ на изображението.\n"
            "• Сменете модела на vision-capable вариант (напр. `llava`)."
        )

    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_bytes:
        message["images"] = [encode_image(image_bytes)]

    payload: dict[str, Any] = {
        "model": model,
        "messages": [message],
        "stream": False,
        "options": {"temperature": 0.7},
    }

    base_url = get_ollama_base_url()
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Изгубена връзка с Ollama при изпращане на заявката.\n"
            "Уверете се, че `ollama serve` работи."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        body = ""
        try:
            body = resp.json().get("error", "")
        except Exception:
            body = resp.text[:300]
        raise RuntimeError(
            f"Ollama върна грешка: {exc}\n\n{body}\n\n"
            f"Ако моделът `{model}` не е изтеглен, стартирайте: `ollama run {model}`"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Грешка при комуникация с Ollama: {exc}") from exc

    if data.get("error"):
        raise RuntimeError(
            f"Ollama: {data['error']}\n\n"
            f"Ако моделът `{model}` липсва, стартирайте: `ollama run {model}`"
        )

    text = ((data.get("message") or {}).get("content") or "").strip()
    if not text:
        raise ValueError("Ollama върна празен отговор.")
    return text


# ---------------------------------------------------------------------------
# Хибридна верига (Auto-Routing)
# ---------------------------------------------------------------------------


def detect_language_instruction(user_prompt: str) -> str:
    """
    Груба езикова експертиза: ако prompt-ът съдържа кирилица,
    отговорът се иска на български, иначе – на английски.
    """
    cyrillic = sum(1 for ch in user_prompt if "а" <= ch.lower() <= "я" or ch == "ё")
    return "на български език" if cyrillic > 0 else "in English"


def call_hybrid(
    prompt: str,
    image_bytes: bytes | None = None,
    mime_type: str = "image/png",
    local_model: str = DEFAULT_OLLAMA_MODEL,
) -> str:
    """
    Хибридна верига с auto-routing:

      изображение ──► Gemini (извличане на детайли)
                          │
                          ▼
      prompt + извлечени данни ──► Ollama (краен отговор)

    Без изображение веригата се свежда до директно извикване на Ollama.
    """
    language = detect_language_instruction(prompt)

    # Стъпка 1: ако има снимка – Cloud моделът я анализира.
    extracted: str = ""
    if image_bytes:
        extraction_prompt = (
            "Ти си аналитик на изображения. Опиши детайлно и структурирано "
            "съдържанието на това изображение: обекти, текстове, хора, цветове, "
            "сцени, графики или таблици (ако има). Изброи всички полезни данни, "
            "които биха послужили за по-нататъшен въпрос. Отговори "
            f"{language}."
        )
        extracted = call_gemini(
            extraction_prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

    # Стъпка 2: Local моделът обработва контекста и генерира финалния отговор.
    if image_bytes:
        final_prompt = (
            "Използвай данните по-долу, за да отговориш на въпроса на потребителя. "
            "Ако данните не са достатъчни – кажи го изрично.\n\n"
            f"=== ИЗВЛЕЧЕНИ ДАНИ ОТ ИЗОБРАЖЕНИЕТО ===\n{extracted}\n"
            f"=== ВЪПРОС НА ПОТРЕБИТЕЛЯ ===\n{prompt}\n\n"
            f"Отговори {language}."
        )
    else:
        final_prompt = prompt

    return call_ollama(final_prompt, model=local_model)


# ---------------------------------------------------------------------------
# Инициализация на session_state
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Създава необходимите ключове в st.session_state (идемпотентно)."""
    if "messages" not in st.session_state:
        # Всеки елемент: {"role": "user"|"assistant", "content": str, "meta": str}
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = MODE_HYBRID
    if "local_model" not in st.session_state:
        st.session_state.local_model = DEFAULT_OLLAMA_MODEL


def render_sidebar() -> dict[str, Any]:
    """
    Рисува страничната лента и връща текущите настройки.

    Returns:
        Речник с настройки: mode, local_model, image_bytes, mime_type, filename
    """
    st.sidebar.title("⚙️ Настройки")

    # --- Избор на модел/режим -------------------------------------------
    # Ясният `key` е задължителен: без него Streamlit не запомня избора
    # коректно при повторни reruns (смяна на режима се "откатва").
    # `index` се чете само при първото рендиране – после важи state-ът.
    mode_label = st.sidebar.selectbox(
        "Модел / режим на работа",
        options=list(MODE_LABELS.values()),
        index=list(MODE_LABELS.values()).index(MODE_LABELS[st.session_state.mode]),
        key="mode_select",
        help="Изберете облачен, локален или хибриден режим.",
    )
    mode = next(k for k, v in MODE_LABELS.items() if v == mode_label)
    st.session_state.mode = mode

    # --- Име на локалния модел ------------------------------------------
    local_model = st.session_state.local_model
    if mode in (MODE_LOCAL, MODE_HYBRID):
        local_model = st.sidebar.text_input(
            "Име на локалния Ollama модел",
            value=st.session_state.local_model,
            key="local_model_input",
            help="Напр. `gemma2:2b`, `gemma2`, `llava` (за vision).",
        ).strip() or DEFAULT_OLLAMA_MODEL
        st.session_state.local_model = local_model

        # Помощна информация: наличните локални модели (ако Ollama е жив)
        available = list_ollama_models()
        if available:
            st.sidebar.caption(
                "Налични локални модели: " + ", ".join(f"`{m}`" for m in available)
            )
        elif ollama_status()[0]:
            st.sidebar.caption("Няма свалени локални модели (напр. `ollama run gemma2:2b`).")

    # --- Качване на изображение -----------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖼️ Изображение")
    uploaded = st.sidebar.file_uploader(
        "Качи снимка за анализ",
        type=list(ALLOWED_IMAGE_TYPES),
        accept_multiple_files=False,
        help="Поддържани формати: .png, .jpg, .jpeg, .webp",
        key="image_uploader",
    )

    image_bytes: bytes | None = None
    mime_type = "image/png"
    filename: str | None = None

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        filename = uploaded.name
        mime_type = image_mime_type(filename)

        # Преглед (thumbnail) на каченото изображение
        st.sidebar.image(
            image_bytes,
            caption=filename,
            width="stretch",
        )

        # Предупреждение, ако избраният режим не може да гледа снимки
        if mode == MODE_LOCAL and not ollama_model_supports_vision(local_model):
            st.sidebar.warning(
                f"⚠️ Моделът `{local_model}` няма vision поддръжка.\n\n"
                "При изпращане приложението ще предложи прехвърляне към "
                "**Hybrid Chain** (Gemini анализира снимката → Ollama отговаря)."
            )
        st.sidebar.caption(
            f"Файл: `{filename}` · MIME: `{mime_type}`\n\n"
            f"📏 {describe_image(image_bytes, mime_type)}"
        )
        if st.sidebar.button(
            "🧹 Премахни изображението",
            use_container_width=True,
            on_click=clear_uploaded_image,
        ):
            pass  # обработката става в callback преди ререндъра

    # --- Информационен блок ---------------------------------------------
    st.sidebar.markdown("---")
    with st.sidebar.expander("📡 Състояние на услугите"):
        gemini_ok = bool(get_gemini_api_key())
        st.write(
            "✅" if gemini_ok else "❌",
            "Gemini API key",
            "готов" if gemini_ok else "липсва в .env",
        )
        ollama_ok, _ = ollama_status()
        st.write(
            "✅" if ollama_ok else "❌",
            f"Ollama ({get_ollama_base_url()})",
            "достъпен" if ollama_ok else "недостъпен",
        )
        if mode in (MODE_LOCAL, MODE_HYBRID) and ollama_ok:
            vision_ok = ollama_model_supports_vision(local_model)
            st.write(
                "✅" if vision_ok else "⚠️",
                f"Модел `{local_model}`",
                "vision" if vision_ok else "само текст",
            )

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Изчисти историята", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    return {
        "mode": mode,
        "local_model": local_model,
        "image_bytes": image_bytes,
        "mime_type": mime_type,
        "filename": filename,
    }


# ---------------------------------------------------------------------------
# Диспечер на режимите
# ---------------------------------------------------------------------------


def generate_response(
    prompt: str,
    mode: str,
    local_model: str,
    image_bytes: bytes | None,
    mime_type: str,
) -> str:
    """
    Изпраща заявката към правилния модел спрямо избрания режим.

    Raises:
        ValueError / RuntimeError с четими инструкции за отстраняване.
    """
    if mode == MODE_CLOUD:
        return call_gemini(prompt, image_bytes=image_bytes, mime_type=mime_type)

    if mode == MODE_LOCAL:
        return call_ollama(
            prompt,
            model=local_model,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

    if mode == MODE_HYBRID:
        # В хибридния режим изискваме поне локален модел; при наличие на снимка
        # автоматично минава през Gemini (Cloud).
        return call_hybrid(
            prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            local_model=local_model,
        )

    raise ValueError(f"Непознат режим: {mode}")


def handle_send(prompt: str, settings: dict[str, Any]) -> None:
    """
    Обработва изпращането на съобщение: записва го в историята,
    извиква модела и показва отговора (или четима грешка).
    """
    image_bytes: bytes | None = settings["image_bytes"]
    mime_type: str = settings["mime_type"]

    # 1. Съобщението на потребителя
    with st.chat_message("user"):
        st.markdown(prompt)
        if image_bytes:
            st.image(image_bytes, caption=settings["filename"], width=240)

    user_meta = f"{MODE_LABELS[settings['mode']]} · {settings['local_model']}"
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "meta": user_meta,
            "image": image_bytes,
            "filename": settings["filename"],
        }
    )

    # 2. Отговор от модела
    with st.chat_message("assistant"):
        with st.spinner("Моделят размишлява..."):
            try:
                answer = generate_response(
                    prompt=prompt,
                    mode=settings["mode"],
                    local_model=settings["local_model"],
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ {exc}",
                        "meta": "Грешка",
                        "image": None,
                        "filename": None,
                    }
                )
                return

        st.markdown(answer)
        st.caption(MODE_LABELS[settings["mode"]])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "meta": MODE_LABELS[settings["mode"]],
            "image": None,
            "filename": None,
        }
    )


# ---------------------------------------------------------------------------
# Точка на влизане
# ---------------------------------------------------------------------------


def main() -> None:
    """Конфигурира страницата, рисува UI-я и управлява чат цикъла."""
    st.set_page_config(
        page_title="Smart Multimodal AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # --- Заглавие ---------------------------------------------------------
    st.title("🤖 Smart Multimodal AI Assistant")
    st.caption(
        "Анализ на изображения, документи и текстови задачи чрез "
        "Cloud (Gemini), Local (Ollama) и Hybrid режим."
    )

    # --- Странична лента --------------------------------------------------
    settings = render_sidebar()

    # Подсказка, ако е избран Local режим със снимка без vision
    if (
        settings["image_bytes"] is not None
        and settings["mode"] == MODE_LOCAL
        and not ollama_model_supports_vision(settings["local_model"])
    ):
        st.info(
            "ℹ️ Избран е **Local** режим без vision поддръжка. При натискане на "
            "*Send* приложението ще пренасочи анализа на снимката автоматично "
            "към **Hybrid Chain**."
        )

    # --- История на съобщенията ------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("image"):
                st.image(
                    msg["image"],
                    caption=msg.get("filename") or "изображение",
                    width=240,
                )
            if msg.get("meta"):
                st.caption(msg["meta"])

    # Празно състояние – кратко ръководство
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "**Здравейте!** Аз съм вашият умен мултимодален асистент. "
                "Можете да:\n"
                "• Пишете въпроси на български или английски език.\n"
                "• Качите изображение в sidebar-а и да го анализирате.\n"
                "• Изберете между **Cloud (Gemini)**, **Local (Ollama)** и "
                "**Hybrid Chain** режим."
            )

    # --- Чат поле ---------------------------------------------------------
    prompt = st.chat_input(
        "Напишете въпрос... (или качете изображение вляво)"
    )
    if not prompt:
        return

    prompt = prompt.strip()
    if not prompt:
        return

    # Автоматично пренасочване: Local режим + снимка без vision → Hybrid
    if (
        settings["image_bytes"] is not None
        and settings["mode"] == MODE_LOCAL
        and not ollama_model_supports_vision(settings["local_model"])
    ):
        st.warning(
            f"↪️ Моделът `{settings['local_model']}` не поддържа изображения – "
            "заявката е пренасочена към **Hybrid Chain** (Gemini → Ollama)."
        )
        settings["mode"] = MODE_HYBRID

    handle_send(prompt, settings)


if __name__ == "__main__":
    main()
