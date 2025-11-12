import streamlit as st # import thư viện Streamlit để xây dựng UI web
import re #import regex để parse URL YouTub
from urllib.parse import urlparse, parse_qs  # import hàm phân tích URL
from sklearn.metrics.pairwise import cosine_similarity # hàm tính cosine similarity
import numpy as np  #  numpy cho các thao tác mảng
 
from recommender import VideoRecommender, ChatbotRecommender # import 2 lớp recommender từ file recommender.py

st.set_page_config(page_title="Gợi ý video học CNTT + Chatbot", layout="wide") # cấu hình trang Streamlit (tiêu đề + layout)
st.title("🎓 Gợi ý video học tập ngành CNTT (Content-Based Filtering) + Chatbot")  # tiêu đề chính trên trang

# Load recommender
recommender = VideoRecommender("videos.csv")      # dùng cho chức năng recommend theo video đang xem
chat_rec = ChatbotRecommender(recommender.df)    # dùng cho chatbot (có thể truyền path string)
# Utility: extract youtube id and embed iframe
def extract_youtube_id(url: str) -> str | None:  # hàm lấy youtube video_id từ nhiều dạng URL
    if not url or not isinstance(url, str):  # kiểm tra url hợp lệ
        return None   # trả None nếu không hợp lệ
    parsed = urlparse(url)  # parse URL thành thành phần
    host = parsed.netloc.lower()  # lấy domain và chuyển thành chữ thường
    if "youtu.be" in host:  # trường hợp short link youtu.be/ID
        vid = parsed.path.lstrip("/")  # lấy phần path bỏ dấu '/'
        return vid or None  # trả ID hoặc None
    if "youtube" in host:  # trường hợp youtube.com
        m = re.search(r"/embed/([^/?&]+)", parsed.path)  # thử match embed URL
        if m:
            return m.group(1)  # trả ID nếu là embed
        qs = parse_qs(parsed.query)  # parse query string
        if "v" in qs and qs["v"]:  # trường hợp watch?v=ID
            return qs["v"][0]  # trả ID từ param v
    m2 = re.search(r"([A-Za-z0-9_-]{11})", url)  # fallback: tìm pattern 11 ký tự (thường là ID)
    if m2:
        return m2.group(1) # trả ID nếu tìm thấy
    return None # không tìm được -> None

def make_embed_html(video_id: str, width=320, height=180, autoplay=False):  # tạo iframe HTML nhúng YouTube
    if not video_id:  # nếu không có video_id
        return ""  # trả chuỗi rỗng
    auto = "1" if autoplay else "0"  # convert bool autoplay thành "1"/"0"
    src = f"https://www.youtube.com/embed/{video_id}?rel=0&showinfo=0&autoplay={auto}"  # URL embed kèm param
    return f'<iframe width="{width}" height="{height}" src="{src}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'  # trả iframe HTML

# Session state init
if "modal_url" not in st.session_state:  # nếu chưa có khóa modal_url trong session
    st.session_state["modal_url"] = None  # khởi tạo modal_url để lưu URL video muốn mở (nếu cần)
if "chat_history" not in st.session_state: # nếu chưa có chat_history
    st.session_state["chat_history"] = [] # khởi tạo danh sách lưu lịch sử chat
if "chat_counter" not in st.session_state:  # nếu chưa có chat_counter
    st.session_state["chat_counter"] = 0 # khởi tạo bộ đếm để tạo key duy nhất cho các button (nếu cần)
if "last_recs" not in st.session_state:  # nếu chưa có last_recs
    st.session_state["last_recs"] = [] # khởi tạo list lưu kết quả gợi ý gần nhất

# Recommender UI
st.header("Gợi ý video (Content-Based)")  # tiêu đề phần gợi ý
video_title = st.selectbox("Chọn video bạn đang xem:", recommender.df['title'].values)  # dropdown chọn video hiện tại từ dataframe
min_score = st.slider("Chọn ngưỡng độ tương đồng tối thiểu:", 0.0, 1.0, 0.0, 0.01) # slider chọn min_score (0.0..1.0), step 0.01
n = st.slider("Số lượng video gợi ý:", 1, 20, 10, 1) # slider chọn số lượng gợi ý, mặc định 10

if st.button("Gợi ý video tương tự"):  # khi người dùng nhấn nút Gợi ý
    recs = recommender.recommend(video_title, n, min_score=min_score)  # gọi hàm recommend theo video đã chọn
    if isinstance(recs, str): # nếu trả về string nghĩa là lỗi hoặc thông báo
        st.warning(recs) # hiển thị cảnh báo
        st.session_state["last_recs"] = [] # xóa last_recs (không có kết quả)
    else:
        st.session_state["last_recs"] = recs  # lưu kết quả vào session để hiển thị

# display last_recs (if any)
if st.session_state.get("last_recs"): # nếu có last_recs thì hiển thị
    st.subheader("🔎 Kết quả gợi ý:") # tiêu đề nhỏ
    for idx, r in enumerate(st.session_state["last_recs"]): # duyệt từng recommendation
        st.markdown(f"**🎥 {r['title']}** — Độ tương đồng: `{r['score']}`") # hiển thị tiêu đề + score
        url = r.get("url", "") # lấy url từ record
        vid = extract_youtube_id(url) # lấy youtube id nếu có
        if vid:
            embed = make_embed_html(vid, width=320, height=180, autoplay=False)  # tạo iframe
            st.markdown(embed, unsafe_allow_html=True) # render iframe (unsafe HTML)
            st.caption(url)  # hiển thị link dưới player như caption
        else:
            st.markdown(f"[Xem ngay]({url})") # nếu không parse được, hiện link thuần
        st.divider() # dòng ngăn cách giữa các mục

# Chatbot UI 
st.header("💬 Chatbot gợi ý video")  # tiêu đề phần chatbot
st.markdown("Hỏi bot: ví dụ: _'Cho tôi video về Python cơ bản'_, _'tài liệu học machine learning'_, hoặc _'gợi ý video OOP'_") # gợi ý cách hỏi

# Input + buttons
user_input = st.text_input("Nhập câu hỏi / yêu cầu tìm video ...", key="user_input")  # ô input cho user nhập câu hỏi
col_send, col_clear = st.columns([1, 1]) # tạo 2 cột nhỏ cho nút gửi và clear

with col_send: # cột nút gửi
    if st.button("Gửi", key="send_btn"):  # khi nhấn Gửi
        if user_input and user_input.strip(): # nếu input không rỗng
            # Ghi user message
            st.session_state["chat_history"].append({"role": "user", "text": user_input}) # lưu message user vào history
            st.session_state["chat_history"] = [] # Xóa lịch sử chat để chỉ giữ lại kết quả mới
            results = chat_rec.query_videos(user_input, n=n, min_score=min_score) # gọi chat_rec để tìm video theo query
            # Xuất kết quả bot
            if not results:
                bot_text = f"Xin lỗi — không có video nào có độ tương đồng ≥ {min_score}. Thử giảm ngưỡng hoặc thay từ khoá."
                st.session_state["chat_history"].append({"role": "bot", "text": bot_text})  # lưu bot text vào history
            else:
                st.session_state["chat_history"].append({"role": "bot", "content": results}) # lưu kết quả (list) vào history
            st.rerun()  # rerun app để cập nhật UI (moden Streamlit)
with col_clear: # cột nút clear
    if st.button("Clear chat", key="clear_chat"): # khi nhấn Clear chat
        st.session_state["chat_history"] = []  # xóa lịch sử chat
        st.rerun()  # rerun app để cập nhật UI
# Hiển thị lịch sử chat
for i, msg in enumerate(st.session_state["chat_history"]): # duyệt lịch sử chat để hiển thị
    role = msg.get("role")  # role là "user" hoặc "bot"
    if role == "user": # nếu tin nhắn của user
        st.markdown(f"**Bạn:** {msg.get('text')}")  # hiển thị theo format "Bạn: ..."
    else:
        content = msg.get("content") # bot có thể trả text hoặc content list
        if content and isinstance(content, list): # nếu bot trả list kết quả
            st.markdown("**Bot (gợi ý):**")  # tiêu đề cho phần bot gợi ý
            for item in content:  # duyệt từng item trong content
                st.markdown(f"- **{item['title']}** — Độ tương đồng: `{item['score']}`")  # hiển thị title + score
                vid = extract_youtube_id(item.get("url", "")) # lấy id youtube nếu có
                if vid:
                    emb = make_embed_html(vid, width=320, height=180, autoplay=False) # tạo iframe
                    st.markdown(emb, unsafe_allow_html=True) # render iframe
                    st.caption(item.get("url", "")) # hiển thị link dưới player như caption
                else:
                    st.markdown(f"[Xem ngay]({item.get('url', '')})") # nếu không parse được, hiện link thuần
            st.divider() # dòng ngăn cách
        else:
            st.markdown(f"**Bot:** {msg.get('text')}") # hiển thị text bot thông thường
# Hiển thị dataset gốc
with st.expander("Xem dữ liệu video (raw)"): # phần mở rộng để xem dữ liệu gốc
    st.dataframe(recommender.df.reset_index(drop=True)) # show dataframe để debug / kiểm tra
