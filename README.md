# 🤖 Smart Multimodal AI Assistant

**Умен мултимодален AI асистент** – уеб приложение на Python (Streamlit) за анализ на изображения, документи и текстови задачи, с избор между облачен, локален и хибриден модел.

*A smart multimodal AI assistant – a Python (Streamlit) web app for image, document and text analysis with cloud, local and hybrid model routing.*

---

## 📌 Съдържание / Table of contents

1. [Описание на проекта](#-описание-на-проекта)
2. [Ключови функционалности](#-ключови-функционалности)
3. [Архитектура](#-архитектура)
4. [Инсталация стъпка по стъпка](#-инсталация-стъпка-по-стъпка)
5. [Стартиране](#-стартиране)
6. [Структура на файловете](#-структура-на-файловете)
7. [Скриншоти](#-скриншоти)
8. [Отстраняване на често срещани грешки](#-отстраняване-на-често-срещани-грешки)
9. [License](#-license)

---

## 📝 Описание на проекта

**Smart Multimodal AI Assistant** е чат интерфейс, който позволява на потребителя да общува с изкуствен интелект на български или английски език и да анализира качени изображения (`.png`, `.jpg`, `.jpeg`, `.webp`).

Избираемите режими в страничната лента (sidebar) са три:

| Режим | Двигател | Поддържа изображения? | Изисквания |
|---|---|---|---|
| ☁️ **Cloud: Gemini 3.8 Flash (Multimodal)** | Google `google-genai` SDK | ✅ Да | `GEMINI_API_KEY` в `.env` |
| 🏠 **Local: Ollama (Llama 3.2 / Gemma 2)** | Локален Ollama сървър | ⚠️ Само ако моделът е vision (напр. `llava`) | Инсталиран Ollama + модел |
| 🔗 **Hybrid Chain (Auto-Routing)** | Gemini → Ollama верига | ✅ Да (Gemini анализира снимката) | И двете от горните |

### Ключови функционалности / Key features

- **Избор на повече от 1 модел** – dropdown в sidebar-а с три режима на работа.
- **Мултимодалност (Vision)** – `st.file_uploader` за снимки, преглед (thumbnail) в реално време и изпращане на изображението заедно с текстовия prompt.
- **Auto-Routing при липса на vision** – ако е избран локален модел без vision поддръжка, приложението показва предупреждение и автоматично пренасочва заявката през Hybrid режима.
- **История на чата** – всички съобщения се пазят в `st.session_state` и се рисуват с `st.chat_message`.
- **Смяна на локалния модел** – текстово поле в sidebar-а (по подразбиране `gemma2:2b`, може `gemma2`, `llava` и т.н.).
- **Устойчивост на грешки** – липсващ `GEMINI_API_KEY` или спрян Ollama не сриват приложението, а показват четимо `st.error` с инструкции за отстраняване.
- **Панел за състояние** – експандър в sidebar-а показва статуса на API ключа, Ollama сървъра и vision възможностите на модела.

---

## 🏗 Архитектура

```
                 ┌──────────────────────────────────────────┐
                 │              Streamlit UI                │
                 │  st.chat_input / st.chat_message         │
                 │  st.session_state["messages"]            │
                 │  Sidebar: mode dropdown, uploader, ...   │
                 └───────────────────┬──────────────────────┘
                                     │  generate_response()
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐
   │  ☁️ CLOUD MODE    │   │  🏠 LOCAL MODE    │   │  🔗 HYBRID CHAIN        │
   │  call_gemini()   │   │  call_ollama()   │   │  call_hybrid()          │
   │                  │   │                  │   │                         │
   │ google-genai SDK │   │ POST             │   │ Шаг 1 (ако има снимка): │
   │ gemini-3.8-flash │   │ /api/chat        │   │  Gemini извлича данни   │
   │ text + image     │   │ localhost:11434  │   │        │                │
   └──────────────────┘   └──────────────────┘   │        ▼                │
                                                 │ Шаг 2: Ollama обработва │
                                                 │ контекста → финален     │
                                                 │ отговор (BG / EN)       │
                                                 └─────────────────────────┘
```

### 1. Cloud режим (Gemini)

- Използва **официалния SDK `google-genai`**.
- Ключът се чете от променливата на средата `GEMINI_API_KEY` (зареждана от `.env` чрез `python-dotenv`).
- Изображението се кодира като `genai_types.Part.from_bytes(...)` и се изпраща заедно с текста към модела `gemini-3.8-flash`.

### 2. Local режим (Ollama)

- Прави `POST http://localhost:11434/api/chat` с `stream: false`.
- Преди заявката се проверява дали сървърът е жив чрез `GET /api/tags`.
- Името на модела е променливо от полето в sidebar-а (по подразбиране `gemma2:2b`).
- Чрез `POST /api/show` се проверява дали моделът има capability `vision`; ако няма, заявката с изображение се отказва с ясно съобщение (или се пренасочва).

### 3. Hybrid Chain (Auto-Routing)

1. **Ако има качено изображение** → облачният модел (Gemini) получава специализиран extraction prompt и връща структурирано описание: обекти, текстове, сцени, таблици, детайли.
2. Извлечената информация + оригиналният въпрос на потребителя се предават на **локалния модел (Ollama)**, който генерира крайния отговор.
3. Езикът на отговора се определя автоматично – ако prompt-ът съдържа кирилица, отговорът е на български, иначе на английски.
4. **Ако няма изображение** → веригата се свежда до директно извикване на локалния модел.

---

## 📦 Инсталация – стъпка по стъпка

### Стъпка 1: Клониране и виртуална среда

```bash
git clone https://github.com/<your-user>/smart-ai-assistant.git
cd smart-ai-assistant

# Създаване на виртуална среда
python3 -m venv .venv

# Активиране (Linux / macOS)
source .venv/bin/activate

# Активиране (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### Стъпка 2: Инсталиране на зависимостите

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Стъпка 3: Настройка на Ollama (Local режим)

1. Инсталирайте Ollama за вашата OS от **<https://ollama.com>**.
2. Стартирайте сървъра (обикновено стартира автоматично; ръчно: `ollama serve`).
3. Изтеглете моделите:

```bash
ollama run gemma2:2b      # текстов модел (по подразбиране в приложението)
ollama run gemma2        # алтернативен текстов модел
ollama run llava         # опционален vision модел за директен анализ на снимки
```

4. Проверка, че всичко работи:

```bash
curl http://localhost:11434/api/tags
```

### Стъпка 4: Настройка на `.env` (Cloud / Hybrid режим)

```bash
cp .env.example .env
```

Редактирайте `.env`:

```env
GEMINI_API_KEY=your_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

- Безплатен Gemini ключ: **<https://aistudio.google.com/apikey>**
- `OLLAMA_BASE_URL` е опционален – ако липсва, се използва `http://localhost:11434`.

> ⚠️ Файлът `.env` е в `.gitignore` и **никога** не се комитва.

### Стъпка 5: Стартиране на приложението

```bash
streamlit run app.py
```

Браузърът ще се отвори автоматично на **http://localhost:8501**.

---

## 🗂 Структура на файловете

```
smart-ai-assistant/
├── app.py              # Основен Streamlit код (Gemini / Ollama / Hybrid функции)
├── requirements.txt    # Зависимости: streamlit, google-genai, ollama, requests,
│                       #              pillow, python-dotenv
├── .env.example        # Примерен конфигурационен файл
├── .gitignore          # Стандартен Python .gitignore (.env, venv/, __pycache__/)
├── README.md           # Тази документация
└── .streamlit/         # (опционално) конфигурация на темата
```

---

## 🖼 Скриншоти

> Плейсхолдъри – да се попълнят по-късно.

### 1. Начален екран и избор на модел в sidebar-а

![Screenshot – Main UI and model dropdown](screenshots/01-main-ui.png)

*Описание: Главният чат интерфейс с dropdown за избор на режим и поле за локален модел.*

### 2. Качено изображение и preview в sidebar-а

![Screenshot – Image upload and preview](screenshots/02-image-upload.png)

*Описание: Заглавен преглед (thumbnail) на качената снимка и предупреждение при модел без vision.*

### 3. Анализ на снимка в Cloud режим (Gemini)

![Screenshot – Gemini image analysis](screenshots/03-gemini-vision.png)

*Описание: Отговор на Gemini 3.8 Flash с детайлен анализ на изображението.*

### 4. Hybrid Chain (Auto-Routing) в действие

![Screenshot – Hybrid chain](screenshots/04-hybrid-chain.png)

*Описание: Двуетапната верига – извличане на данни от снимката, после отговор от локалния модел.*

### 5. Локален режим с Ollama

![Screenshot – Local Ollama mode](screenshots/05-local-ollama.png)

*Описание: Чат с локалния модел `gemma2:2b`.*

### 6. Състояние на услугите и обработка на грешки

![Screenshot – Service status and error handling](screenshots/06-error-handling.png)

*Описание: Панелът за статус на API ключа/Ollama и четими `st.error` съобщения.*

---

## 🛠 Отстраняване на често срещани грешки

| Симптом | Причина | Решение |
|---|---|---|
| `st.error`: „Липсва GEMINI_API_KEY“ | Няма `.env` или ключът е празен | Копирайте `.env.example` → `.env` и сложете валиден ключ |
| `st.error`: „Неуспешна връзка с Ollama“ | Ollama не е стартиран | `ollama serve`, после `ollama run gemma2:2b` |
| `Ollama върна грешка: ... model ... not found` | Моделът не е изтеглен | `ollama run <име_на_модела>` |
| „Локалният модел няма vision поддръжка“ | Текстов модел (gemma2:2b/gemma2) | Използвайте Hybrid режим или `ollama run llava` |
| `ModuleNotFoundError: streamlit` | Липсваща виртуална среда | `source .venv/bin/activate && pip install -r requirements.txt` |
| „Cannot modify st.session_state ... after widget“ | State на widget се пипа след инстанциране | Използвайте `on_click` callback (вече е така в кода) |

---

## 🧪 Бърз тест без UI (опционално)

```bash
# Проверка на синтаксиса
python -m py_compile app.py

# Проверка, че Ollama отговаря
curl -s http://localhost:11434/api/tags | head
```

---

## 📄 License

MIT License. Свободно е за употреба и модифициране.
