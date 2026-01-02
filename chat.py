import os
import google.generativeai as genai
import pandas as pd
import streamlit as st
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv

# --- 1. การตั้งค่าหน้ากระดาษ (ต้องเป็นคำสั่งแรกของ Streamlit) ---
st.set_page_config(
    page_title="Computer Expert AI",
    page_icon="💻",
    layout="centered"
)

# --- 2. Custom CSS เพื่อความสวยงาม ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
        color: #1a1a1a; /* เปลี่ยนสีตัวอักษรหลักให้เข้มขึ้น */
    }

    .stApp {
        background-color: #f8f9fa;
    }

    /* ปรับแต่ง Header ให้เด่นชัด */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6); /* ไล่เฉดสีน้ำเงินเข้ม */
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }

    /* ปรับปรุงสีข้อความใน Chat Message ให้เข้มชัดเจน */
    .stChatMessage {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        margin-bottom: 10px;
    }

    /* เน้นสีข้อความที่ AI ตอบ (Model) */
    [data-testid="stChatMessage"] p {
        color: #1a1a1a !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* ปรับแต่ง Bullet points ให้เห็นชัด */
    [data-testid="stChatMessage"] li {
        color: #1a1a1a !important;
        font-weight: 500;
    }

    </style>
    <div class="main-header">💬 ผู้ช่วยอัจฉริยะด้านคอมพิวเตอร์</div>
    """, unsafe_allow_html=True)

# --- 3. โหลด Environment และตั้งค่า Gemini ---
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

PROMPT_INSTRUCTION = """
คุณคือผู้เชี่ยวชาญด้านคอมพิวเตอร์และสารสนเทศ หน้าที่ของคุณคือตอบคำถามโดยอ้างอิงจากฐานข้อมูลที่ให้มาเท่านั้น
กฎเหล็ก:
1. จับคู่ด้วยความหมาย (Semantic Match) เช่น "จอคอม" -> "จอมอนิเตอร์"
2. หากถามสั้นๆ ให้รวบรวมข้อมูลที่เกี่ยวข้องทั้งหมดมาตอบ
3. แก้ไขคำผิดอัตโนมัติ
4. ตอบด้วย "ค่ะ/คะ" เสมอ ห้ามใช้ Emoji ในเนื้อหาคำตอบ
5. หากไม่พบข้อมูลจริงๆ ให้ตอบว่า "ขออภัยค่ะ ฉันไม่พบข้อมูลที่คุณต้องการในขณะนี้"
"""

# หมายเหตุ: ปรับเป็น gemini-1.5-flash ซึ่งเป็นรุ่นที่เสถียรในปัจจุบัน
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    safety_settings=SAFETY_SETTINGS,
    generation_config=generation_config,
    system_instruction=PROMPT_INSTRUCTION
)

# --- 4. ฟังก์ชันโหลดข้อมูล ---
@st.cache_data
def load_data():
    try:
        file_name = "Prepare for chatbot.csv"
        df = pd.read_csv(file_name)
        df.columns = [col.strip() for col in df.columns]
        qa_df = df.dropna(subset=['User_query', 'Chatbot_response'])
        return qa_df.to_string(index=False)
    except Exception as e:
        return None

file_content = load_data()
if file_content is None:
    st.error("❌ ไม่พบไฟล์ 'Prepare for chatbot.csv' กรุณาตรวจสอบชื่อไฟล์นะคะ")
    st.stop()

# --- 5. Sidebar เมนูควบคุม ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.title("เมนูการใช้งาน")
    st.markdown("---")
    st.info("💡 **คำแนะนำ:** คุณสามารถสอบถามเรื่อง Windows, สเปกคอมพิวเตอร์ หรืออุปกรณ์ต่างๆ ได้ทันที")
    
    if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
        st.session_state["messages"] = [
            {"role": "model", "content": "สวัสดีค่ะ มีอะไรให้ช่วยสอบถามเพิ่มเติมไหมคะ"}
        ]
        st.rerun()
    
    st.markdown("---")
    st.caption("Version 1.0 | Powered by Gemini AI")

# --- 6. ส่วนการแสดง Chat ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีค่ะ ฉันคือผู้ช่วยอัจฉริยะ ยินดีให้คำปรึกษาเรื่องคอมพิวเตอร์ค่ะ"}
    ]

# แสดงข้อความทั้งหมดใน History
for msg in st.session_state["messages"]:
    avatar = "🤖" if msg["role"] == "model" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# --- 7. ส่วนการรับ Input และประมวลผล ---
if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    # แสดงข้อความฝั่ง User
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # ส่งข้อความให้ AI และแสดง Spinner
    with st.chat_message("model", avatar="🤖"):
        with st.spinner("กำลังค้นหาข้อมูล..."):
            history = [
                {"role": "user", "parts": [f"อ้างอิงข้อมูลจากตารางนี้:\n{file_content}"]},
                {"role": "model", "parts": ["รับทราบค่ะ ฉันจะวิเคราะห์และตอบคำถามจากข้อมูลชุดนี้เท่านั้นค่ะ"]}
            ]
            
            # ดึงประวัติล่าสุด 5 ข้อความ
            for msg in st.session_state["messages"][-5:]:
                history.append({"role": msg["role"], "parts": [msg["content"]]})

            try:
                chat_session = model.start_chat(history=history)
                response = chat_session.send_message(prompt)
                
                full_response = response.text
                st.write(full_response)
                st.session_state["messages"].append({"role": "model", "content": full_response})
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
