# recommender.py
import pandas as pd  # pandas để đọc/ xử lý dataframe CSV
import re  # regex để tiền xử lý query và normalizing
from typing import List, Dict, Union  # kiểu dữ liệu cho type hints

from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vectorizer
from sklearn.metrics.pairwise import cosine_similarity  # tính cosine similarity giữa vectors
from sklearn.pipeline import FeatureUnion  # kết hợp nhiều vectorizer (word + char)

# optional fuzzy fallback
try:
    from rapidfuzz import process, fuzz  # thử import rapidfuzz để có fallback fuzzy matching (tùy cài)
    _HAS_RAPIDFUZZ = True  # đánh dấu có rapidfuzz
except Exception:
    _HAS_RAPIDFUZZ = False  # không có rapidfuzz -> fallback fuzzy sẽ không dùng

class VideoRecommender:
    """
    Giữ nguyên behavior ban đầu: recommend dựa trên một video đang có.
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)  # đọc CSV vào DataFrame
        # ensure columns exist
        for c in ['title', 'description', 'tags', 'url']:
            if c not in self.df.columns:
                self.df[c] = ""  # nếu thiếu cột nào thì tạo cột rỗng để tránh lỗi sau này
        self.df['content'] = (  # tạo cột content bằng cách ghép title + description + tags
            self.df['title'].fillna('') + ' ' +
            self.df['description'].fillna('') + ' ' +
            self.df['tags'].fillna('')
        ).astype(str)  # đảm bảo kiểu string

        # original TF-IDF (word-level) and precomputed cosine similarity matrix
        self.vectorizer = TfidfVectorizer(stop_words='english')  # TF-IDF mặc định (word-level), loại stopwords tiếng Anh
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['content'])  # fit & transform tài liệu
        self.similarity = cosine_similarity(self.tfidf_matrix)  # ma trận tương đồng cosine giữa tất cả các doc

    def recommend(self, video_title: str, n: int, min_score: float = 0.0) -> Union[str, List[Dict]]:
        """
        Gợi ý dựa trên một video đã chọn (không thay đổi logic cũ).
        Trả về danh sách dict hoặc thông báo lỗi (string).
        """
        if video_title not in self.df['title'].values:
            return f"Video '{video_title}' không có trong cơ sở dữ liệu."  # trả thông báo nếu title không tồn tại

        idx = self.df[self.df['title'] == video_title].index[0]  # lấy index của video được chọn
        sim_scores = list(enumerate(self.similarity[idx]))  # lấy scores của tất cả video đối với index đó
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]  # sắp xếp giảm dần, bỏ chính nó ở vị trí đầu

        recommendations = []  # list lưu kết quả
        for i, score in sim_scores:
            # keep tiny epsilon to avoid floating noise
            if score > 1e-8 and score >= min_score:  # lọc theo ngưỡng nhỏ để tránh số rất nhỏ do tính toán
                video = self.df.iloc[i]  # lấy record video
                recommendations.append({
                    "title": video["title"],  # tiêu đề video
                    "score": round(float(score), 3),  # điểm tương đồng làm tròn 3 chữ số
                    "url": video.get("url", "")  # url (nếu có)
                })

        if not recommendations:
            return f"⚠️ Không có video nào có độ tương đồng ≥ {min_score}"  # thông báo nếu rỗng

        return recommendations  # trả về list các recommendations


class ChatbotRecommender:
    """
    Lớp riêng cho Chatbot: improved search for short/typo queries.
    - Kết hợp word-level TF-IDF (1..2 grams) + char_wb TF-IDF (2..4)
    - Tiền xử lý alias (c++, c#)
    - Method query_videos(query, n, min_score)
    - Nếu rapidfuzz có sẵn, có fallback fuzzy matching
    """
    def __init__(self, csv_path_or_df: Union[str, pd.DataFrame]):
        if isinstance(csv_path_or_df, str):
            df = pd.read_csv(csv_path_or_df)  # nếu truyền path thì đọc CSV
        else:
            df = csv_path_or_df.copy()  # nếu truyền DataFrame thì copy để an toàn

        # ensure columns
        for c in ['title', 'description', 'tags', 'url']:
            if c not in df.columns:
                df[c] = ""  # tạo cột rỗng nếu thiếu
        df['title'] = df['title'].fillna('').astype(str)  # đảm bảo không có NaN
        df['description'] = df['description'].fillna('').astype(str)  # đảm bảo không có NaN
        df['tags'] = df['tags'].fillna('').astype(str)  # đảm bảo không có NaN

        # build a normalized content field (lowercase, with some token mapping)
        content = (df['title'] + ' ' + df['description'] + ' ' + df['tags']).str.lower().astype(str)  # ghép và lowercase toàn bộ nội dung
        # normalize common language tokens (c++, c# -> cpp, csharp)
        content = content.str.replace(r'c\+\+', 'cpp', regex=True)  # chuyển c++ -> cpp để dễ match
        content = content.str.replace(r'c#', 'csharp', regex=True)  # chuyển c# -> csharp để tránh ký tự #
        df['content_norm'] = content  # lưu cột chuẩn hóa

        self.df = df.reset_index(drop=True)  # reset index để mapping dễ hơn

        # Word-level vectorizer (captures semantic/phrases)
        self.word_vect = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),  # unigram + bigram
            token_pattern=r'(?u)\b\w+\b',  # token pattern chuẩn
            stop_words='english',
            min_df=1
        )
        # Char-level vectorizer (helps matching short tokens / typos)
        self.char_vect = TfidfVectorizer(
            analyzer='char_wb',  # char n-grams within word boundaries
            ngram_range=(2, 4),  # 2..4-grams character
            min_df=1
        )

        self.union = FeatureUnion([
            ('word', self.word_vect),  # feature "word"
            ('char', self.char_vect),  # feature "char"
        ])  # kết hợp hai vectorizer thành một feature vector tổng hợp

        # fit on normalized content
        self.tfidf_matrix = self.union.fit_transform(self.df['content_norm'])  # fit_transform trên content chuẩn

        # prepare fuzzy choices if available
        if _HAS_RAPIDFUZZ:
            # combine title + tags as choices for fuzzy fallback
            self._fuzzy_choices = (self.df['title'] + " | " + self.df['tags']).tolist()  # list string dùng cho rapidfuzz

    @staticmethod
    def _normalize_query(q: str) -> str:
        q2 = q.lower().strip()  # lowercase và strip
        q2 = re.sub(r'c\+\+', 'cpp', q2)  # map c++ -> cpp
        q2 = re.sub(r'c#', 'csharp', q2)  # map c# -> csharp
        return q2  # trả về query đã chuẩn hóa

    def query_videos(self, query: str, n: int = 5, min_score: float = 0.0) -> List[Dict]:
        """
        Tìm video phù hợp với một câu truy vấn (dành cho chatbot).
        Trả về list các dict: {"title","score","url"}.
        """
        if not query or not isinstance(query, str):
            return []  # trả rỗng nếu query không hợp lệ

        q = self._normalize_query(query)  # chuẩn hóa query (lowercase, map aliases)

        # If query is extremely short (length 1) we can optionally require user to expand,
        # but we'll still try char n-grams.
        q_vec = self.union.transform([q])  # transform query thành vector theo union đã fit
        sims = cosine_similarity(q_vec, self.tfidf_matrix).ravel()  # cosine similarity giữa query và tất cả docs
        idxs = sims.argsort()[::-1]  # sắp xếp index giảm dần theo score

        results = []  # danh sách kết quả
        for idx in idxs[:50]:  # scan top candidates (tối đa 50)
            score = float(sims[idx])  # convert sang float
            if score >= min_score and score > 0:  # chỉ lấy các item có score >= min_score và >0
                row = self.df.iloc[idx]  # lấy row tương ứng
                results.append({
                    "title": row.get("title", ""),  # lấy title
                    "score": round(score, 3),  # làm tròn score
                    "url": row.get("url", "")  # lấy url nếu có
                })
                if len(results) >= n:  # dừng khi đủ n kết quả
                    break

        # Fallback: if no results and rapidfuzz available, use fuzzy match on titles/tags
        if not results and _HAS_RAPIDFUZZ:
            fuzzy_matches = process.extract(query, self._fuzzy_choices, scorer=fuzz.WRatio, limit=n)  # lấy các match fuzzy tốt nhất
            fallback = []  # kết quả fallback
            for match in fuzzy_matches:
                matched_text, score100, choice_idx = match  # choice_idx là index trong _fuzzy_choices
                # convert score 0..100 to 0..1
                fallback.append({
                    "title": self.df.iloc[choice_idx]['title'],  # title tương ứng
                    "score": round(score100 / 100.0, 3),  # chuẩn hóa score về 0..1
                    "url": self.df.iloc[choice_idx].get("url", "")  # lấy url
                })
            return fallback  # trả fallback nếu có

        return results  # trả results (có thể rỗng)
