# Reflection cá nhân — Vũ Quang Bảo (2A202600610)
**Mission phụ trách:** Toàn bộ hệ thống Evaluation Factory (Data + AI-Backend + DevOps/Analyst)

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

Tôi xây dựng end-to-end toàn bộ pipeline đánh giá, gồm các module chính:

- **`engine/providers.py` — tầng truy cập model dùng chung:** client Groq/Mistral/Gemini/Cohere/Qdrant, **theo dõi token & chi phí** (`CostTracker`), **retry + backoff**, **rate-limiter theo từng model** (TPM) và **circuit breaker** tự ngắt provider hết quota.
- **`agent/retriever.py` + `data/corpus.py` — tầng Retrieval:** RAG qua Cohere embedding + Qdrant vector search + Cohere rerank, có **fallback lexical (TF-IDF)** khi mất mạng/hết quota.
- **`agent/main_agent.py` — Agent V1/V2:** V1 (8B, prompt lỏng) vs V2 (70B, rerank, prompt grounded), trả `retrieved_ids` để tính Hit Rate/MRR.
- **`data/synthetic_gen.py` — SDG:** sinh 69 golden case (grounded theo từng chunk + 6 red-team: out-of-context, prompt injection, goal hijacking, ambiguous, false premise).
- **`engine/retrieval_eval.py`:** Hit Rate, MRR và faithfulness/relevancy (embedding, có fallback token-overlap).
- **`engine/llm_judge.py` — Multi-Judge consensus:** 2 judge khác họ chạy song song + tiebreaker khác họ, agreement rate, **Cohen's Kappa**, **position-bias check**.
- **`engine/runner.py` + `main.py`:** async batch runner + **regression V1↔V2 + Release Gate tự động** (quality/cost/latency) + báo cáo chi phí + đề xuất giảm 30%.

**Bằng chứng Git:** commit `8b8b121` ("feat: Implement full AI Evaluation Factory…") trên nhánh `main`, tài khoản `BaoVu2k4`.

## 2. Technical Depth

### 2.1. MRR (Mean Reciprocal Rank)
MRR đo **chất lượng thứ hạng** của Retriever: với mỗi truy vấn lấy 1/(vị trí 1-indexed của
tài liệu đúng đầu tiên), không thấy thì 0, rồi lấy trung bình. Hạng 1 → 1.0, hạng 2 → 0.5,
hạng 3 → 0.33. Khác Hit Rate (nhị phân, chỉ xét "có trúng trong top-k không"), MRR **thưởng
cho việc xếp tài liệu đúng lên cao**. Trong lab MRR = 1.0 nghĩa là chunk đúng luôn ở hạng 1
(`engine/retrieval_eval.py::calculate_mrr`).

### 2.2. Cohen's Kappa
Đo đồng thuận giữa 2 judge **sau khi loại phần trùng do may rủi**: `κ = (Po − Pe)/(1 − Pe)`,
Po = tỉ lệ chấm trùng thực tế, Pe = tỉ lệ trùng kỳ vọng nếu chấm ngẫu nhiên. κ=1 hoàn hảo,
κ=0 như ngẫu nhiên. Thang: 0.2–0.4 = "fair", 0.6–0.8 = "substantial". Lab đạt κ ≈ 0.13–0.28
("fair") → 2 judge chưa thực sự đồng thuận, cần calibrate rubric (`cohen_kappa`).

### 2.3. Position Bias
LLM-as-judge thường thiên vị câu trả lời đặt ở một vị trí cố định. Tôi phát hiện bằng cách
chấm cặp (A,B) rồi đảo (B,A): nếu judge vẫn chọn **cùng vị trí** dù nội dung đã đổi chỗ →
có position bias. Lab này né bias bằng cách **chấm theo rubric tuyệt đối 1–5** thay vì so
sánh tương đối (`check_position_bias`).

### 2.4. Trade-off Chi phí ↔ Chất lượng
Model 70B chính xác hơn nhưng chậm ~2.2x và đắt hơn 8B. Tôi tối ưu bằng **cascading**: panel
2 judge nhanh, chỉ "leo thang" gọi tiebreaker (Gemini) khi 2 judge lệch > 1 điểm (8/20 case)
→ tiết kiệm ~một nửa chi phí judge mà gần như không giảm độ chính xác. Đo bằng
`cost_per_eval_usd` ≈ $0.00035/eval trong `reports/summary.json`.

## 3. Problem Solving (vấn đề thật đã xử lý)

- **Benchmark treo 34 phút:** truy ra tiebreaker OpenRouter free mất ~164s rồi 429. Khắc phục:
  đổi sang model nhanh + thêm **rate-limiter theo model** + retry/backoff để 429 không dồn.
- **Giám khảo "thông đồng":** ban đầu 2 judge cùng họ Llama → thiên vị tương quan. Sửa thành
  **3 họ model khác nhau** (Meta + OpenAI + Google) để consensus khách quan.
- **Cohere hết quota giữa buổi:** mọi truy vấn RAG phụ thuộc embedding Cohere → toàn bộ case
  lỗi. Khắc phục: **fallback lexical** ở tầng search + **circuit breaker** (lỗi 1 lần thì tắt
  Cohere cả phiên, fallback tức thì, không phí backoff) → pipeline luôn hoàn tất.
- **Judge chạy tuần tự:** `judge_a`/`judge_b` đang await nối tiếp → đổi sang `asyncio.gather`
  cho song song, giảm latency mỗi case.

## 4. Bài học rút ra
- Hệ thống đánh giá thực tế phải coi **rate-limit và độ tin cậy nhà cung cấp** là vấn đề
  cốt lõi, không phải chuyện phụ — fallback/circuit-breaker quan trọng ngang thuật toán.
- **Một con số trung bình (avg_score) chưa đủ tin:** phải nhìn agreement rate và Cohen's
  Kappa để biết điểm số có đáng tin không — κ "fair" cho thấy cần calibrate giám khảo.
- Đa dạng **họ model** khi làm LLM-as-judge quan trọng để tránh thiên vị hệ thống.
