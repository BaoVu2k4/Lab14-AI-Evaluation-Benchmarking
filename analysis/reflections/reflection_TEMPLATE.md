<!--
HƯỚNG DẪN: Copy file này thành reflection_[TênCủaBạn].md.
Phần "2. Technical Depth" đã viết sẵn (kiến thức chung của hệ thống) — bạn ĐỌC HIỂU
và chỉnh theo lời mình. Phần 1, 3, 4 mỗi người TỰ ĐIỀN theo đóng góp thật của mình
(thay [TÊN], [MODULE], [...]). Nhớ để commit Git mang tên bạn để chứng minh đóng góp.
-->

# Reflection cá nhân — [TÊN SINH VIÊN] ([MSSV])
**Nhóm mission phụ trách:** [Data / AI-Backend / DevOps-Analyst]

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)
> Liệt kê module/PR cụ thể bạn làm, kèm bằng chứng commit Git.

- **Module phụ trách:** [vd: `engine/llm_judge.py` — Multi-Judge consensus engine]
- **Việc đã làm cụ thể:**
    - [vd: triển khai `evaluate_multi_judge` gọi song song 2 judge khác họ model;]
    - [vd: viết `cohen_kappa` đo độ đồng thuận; `check_position_bias` đảo vị trí A/B;]
    - [vd: cơ chế cascading tiebreaker chỉ gọi Gemini khi 2 judge lệch > 1 điểm.]
- **Bằng chứng Git:** [commit hash / PR link mang tên bạn]

## 2. Technical Depth (kiến thức cốt lõi — đã viết sẵn, hãy đọc & diễn đạt lại)

### 2.1. MRR (Mean Reciprocal Rank)
MRR đo **chất lượng thứ hạng** của Retriever. Với mỗi truy vấn, tìm vị trí (1-indexed)
của tài liệu đúng đầu tiên trong danh sách trả về; Reciprocal Rank = 1/vị_trí (không
thấy → 0). MRR là trung bình các Reciprocal Rank trên toàn bộ truy vấn.
- Tài liệu đúng ở hạng 1 → 1.0; hạng 2 → 0.5; hạng 3 → 0.33.
- Khác Hit Rate (chỉ quan tâm "có trúng trong top-k hay không", nhị phân), MRR
  **thưởng cho việc xếp tài liệu đúng lên cao** → phản ánh chất lượng ranking.
- Trong lab: `engine/retrieval_eval.py::calculate_mrr`. Kết quả MRR = 1.0 nghĩa là
  chunk đúng luôn nằm ở hạng 1.

### 2.2. Cohen's Kappa (độ tin cậy giữa 2 giám khảo)
Kappa đo mức đồng thuận giữa 2 judge **đã loại trừ phần đồng thuận do may rủi**:
`κ = (Po − Pe) / (1 − Pe)`, với Po = tỉ lệ chấm trùng thực tế, Pe = tỉ lệ trùng kỳ
vọng nếu chấm ngẫu nhiên.
- κ = 1: đồng thuận hoàn hảo; κ = 0: như ngẫu nhiên; κ < 0: tệ hơn ngẫu nhiên.
- Thang quy ước: 0–0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate, 0.6–0.8 substantial.
- Trong lab: `engine/llm_judge.py::cohen_kappa`. Kết quả κ ≈ 0.13–0.28 ("fair") cho
  thấy 2 judge **chưa thật sự đồng thuận** → cần calibrate rubric (xem failure analysis).

### 2.3. Position Bias (thiên vị vị trí của LLM-as-judge)
LLM khi so sánh 2 câu trả lời thường thiên vị câu **đặt trước** (hoặc đặt sau) bất kể
nội dung. Cách phát hiện: hỏi judge cặp (A,B), rồi đảo thành (B,A); nếu judge vẫn chọn
**cùng một vị trí** (vd luôn chọn "first") dù nội dung đã đổi chỗ → có position bias.
- Khắc phục: chấm 2 lượt đảo vị trí rồi lấy trung bình, hoặc chấm theo rubric tuyệt đối
  (như lab này) thay vì so sánh tương đối.
- Trong lab: `engine/llm_judge.py::check_position_bias`.

### 2.4. Trade-off Chi phí ↔ Chất lượng
- Model lớn (70B) chính xác hơn nhưng **chậm (~2.2x latency)** và đắt hơn model nhỏ (8B).
- Giải pháp tối ưu: **cascading / routing** — dùng model rẻ cho case dễ, chỉ "leo thang"
  lên model đắt/judge thứ 3 khi 2 judge bất đồng. Trong lab, tiebreaker (Gemini) chỉ
  được gọi ở các case xung đột (8/20) → tiết kiệm ~một nửa chi phí judge.
- Đo bằng `cost_per_eval_usd` trong `reports/summary.json` (~$0.00035/eval).

## 3. Problem Solving (vấn đề gặp & cách giải quyết)
> Chọn vấn đề bạn trực tiếp xử lý. Dưới đây là các vấn đề THẬT của dự án để tham khảo:

- **Rate-limit free tier (TPM) làm benchmark treo:** ban đầu OpenRouter tiebreaker mất
  ~164s rồi 429 → đổi sang model nhanh; thêm **rate-limiter theo từng model** + **retry
  backoff** để không bị 429 dồn.
- **Giám khảo "thông đồng":** ban đầu dùng 2 model cùng họ Llama → sửa thành **3 họ khác
  nhau** (Meta + OpenAI + Google) để consensus khách quan.
- **Cohere hết quota giữa buổi:** retrieval phụ thuộc embedding bị chặn → thêm **fallback
  lexical (TF-IDF)** + **circuit breaker** để pipeline luôn chạy xong, không lỗi.
- **Judge gọi tuần tự:** phát hiện `judge_a`/`judge_b` await nối tiếp → đổi sang
  `asyncio.gather` chạy song song.

## 4. Bài học rút ra
- [vd: hệ thống eval thực tế phải coi rate-limit/độ tin cậy nhà cung cấp là first-class concern;]
- [vd: một con số (avg_score) chưa đủ — phải nhìn agreement/kappa để biết điểm có đáng tin không.]
