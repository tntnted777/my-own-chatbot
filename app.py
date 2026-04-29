import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import time
import datetime
from PIL import Image

# ==========================================
# 1. 系統初始化與狀態管理
# ==========================================
st.set_page_config(page_title="QXQ Professional Agent v2", layout="wide")

GEMINI_API_KEY = "GEMINI API KEY"
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def init_system():
    # 長期記憶初始化
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    db_client = chromadb.PersistentClient(path="./gemini_pro_memory")
    db_collection = db_client.get_or_create_collection(name="history", embedding_function=emb_fn)
    # 模型獲取
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        flash = next((m for m in models if 'flash' in m), "models/gemini-1.5-flash")
    except:
        flash = "models/gemini-1.5-flash"
    return db_collection, flash

collection, FLASH_MODEL = init_system()

# --- Session State 初始化 ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"預設主題": []}
if "current_topic" not in st.session_state:
    st.session_state.current_topic = "預設主題"
if "reply_content" not in st.session_state:
    st.session_state.reply_content = None

# ==========================================
# 2. 側邊欄：進階主題管理 (Rename & Targeted Delete)
# ==========================================
with st.sidebar:
    st.header("📂 主題管理中心")
    
    # A. 新增主題
    with st.expander("➕ 新增對話主題"):
        new_topic_name = st.text_input("輸入名稱", key="new_topic_input")
        if st.button("確認新增"):
            if new_topic_name and new_topic_name not in st.session_state.chat_sessions:
                st.session_state.chat_sessions[new_topic_name] = []
                st.session_state.current_topic = new_topic_name
                st.rerun()

    st.divider()

    # B. 編輯與刪除 (針對當前選中主題)
    st.subheader(f"當前：{st.session_state.current_topic}")
    
    # 重命名功能
    with st.expander("📝 重命名此主題"):
        rename_val = st.text_input("輸入新名稱", value=st.session_state.current_topic)
        if st.button("更新名稱") and rename_val != st.session_state.current_topic:
            # 搬移數據到新的 Key
            st.session_state.chat_sessions[rename_val] = st.session_state.chat_sessions.pop(st.session_state.current_topic)
            # 同步更新向量資料庫中的標籤 (Metadata 遷移)
            # 注意：ChromaDB 本身不支持直接更新 metadata 條件，
            # 這裡的邏輯是之後的記憶會改用新標籤，刪除時會一併處理舊名稱。
            st.session_state.current_topic = rename_val
            st.rerun()

    # 刪除功能
    if st.button("🗑️ 刪除此主題與記憶", use_container_width=True):
        if len(st.session_state.chat_sessions) > 1:
            # 1. 執行標靶刪除：只刪除 Metadata 中 topic 符合的資料
            collection.delete(where={"topic": st.session_state.current_topic})
            # 2. 刪除 Session 紀錄
            del st.session_state.chat_sessions[st.session_state.current_topic]
            # 3. 強制切換到剩餘的第一個主題
            st.session_state.current_topic = list(st.session_state.chat_sessions.keys())[0]
            st.success("主題及其記憶已徹底消除")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("至少須保留一個主題。")

    st.divider()

    # C. 主題切換 (Radio)
    topic_list = list(st.session_state.chat_sessions.keys())
    selected_topic = st.radio("切換對話主題", topic_list, index=topic_list.index(st.session_state.current_topic))
    if selected_topic != st.session_state.current_topic:
        st.session_state.current_topic = selected_topic
        st.rerun()

    st.divider()
    uploaded_file = st.file_uploader("📸 多模態圖片輸入", type=["jpg", "png", "jpeg"])

# ==========================================
# 3. 主要對話區域與回覆功能
# ==========================================
st.title(f"♊ {st.session_state.current_topic}")

# 顯示引用回覆狀態
if st.session_state.reply_content:
    st.info(f"正在回覆：{st.session_state.reply_content[:50]}...")
    if st.button("取消引用"):
        st.session_state.reply_content = None
        st.rerun()

# 顯示歷史訊息
current_msgs = st.session_state.chat_sessions[st.session_state.current_topic]
for i, msg in enumerate(current_msgs):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if st.button(f"↩️ Reply", key=f"rep_{i}"):
            st.session_state.reply_content = msg["content"]
            st.rerun()

# ==========================================
# 4. 核心 Agent 邏輯 (隔離記憶 + 工具)
# ==========================================
if prompt := st.chat_input("請輸入指令..."):
    # 處理引用邏輯
    final_prompt = prompt
    if st.session_state.reply_content:
        final_prompt = f"> 【引用訊息】：{st.session_state.reply_content}\n\n{prompt}"
        st.session_state.reply_content = None

    # 更新 UI
    current_msgs.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    # --- 功能：記憶隔離檢索 (Where Metadata Filtering) ---
    results = collection.query(
        query_texts=[prompt], 
        n_results=1,
        where={"topic": st.session_state.current_topic} # 關鍵：只搜當前主題
    )
    relevant_memory = results['documents'][0][0] if results['documents'] and results['documents'][0] else "無此主題歷史"
    
    # 工具：系統時間
    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 執行生成
    with st.chat_message("assistant"):
        try:
            model = genai.GenerativeModel(model_name=FLASH_MODEL)
            
            # 組合上下文
            context = (
                f"你是有記憶的專業助手。當前主題：[{st.session_state.current_topic}]\n"
                f"此主題相關記憶：{relevant_memory}\n"
                f"目前系統時間：{now_time}\n"
            )
            
            payload = [f"{context}\nUser: {final_prompt}"]
            if uploaded_file:
                payload.append(Image.open(uploaded_file))
            
            response = model.generate_content(payload)
            full_response = response.text
            st.markdown(full_response)

            # --- 功能：帶標籤儲存記憶 ---
            collection.add(
                documents=[f"{prompt} -> {full_response}"],
                metadatas=[{"topic": st.session_state.current_topic}], # 標註當前主題
                ids=[f"id_{int(time.time())}"]
            )
            
            current_msgs.append({"role": "assistant", "content": full_response})
            st.rerun()

        except Exception as e:
            st.error(f"系統錯誤：{e}")
