# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> Số liệu lấy từ `reports/summary.json` & `reports/benchmark_results.json`
> (lần chạy 2026-06-16, mẫu 20/69 case — `BENCH_LIMIT=20`).
> Lưu ý môi trường: key **Cohere (trial) bị rate-limit giữa buổi**, nên Retriever
> tự động fallback sang **lexical (TF-IDF)** và faithfulness/relevancy được tính
> bằng **proxy token-overlap** thay vì embedding. Điều này làm 2 chỉ số RAGAS
> này **bị hạ thấp giả tạo** (xem Cluster #3).

## 1. Tổng quan Benchmark
- **Tổng số cases (mẫu):** 20 (14 grounded + 6 red-team), trên golden set 69 case.
- **Tỉ lệ Pass/Fail:** V1 = 20/0 (pass_rate 1.00) · V2 = 19/1 (pass_rate 0.95).
- **Điểm RAGAS trung bình (proxy token-overlap, do Cohere bị giới hạn):**
    - Faithfulness: V1 0.326 · V2 0.365
    - Relevancy:    V1 0.155 · V2 0.298
- **Retrieval (lexical):** Hit Rate = 1.00 · MRR = 1.00 (15 case có ground-truth chunk).
- **Điểm LLM-Judge trung bình:** V1 **3.775** / 5.0 · V2 **3.475** / 5.0.
- **Độ tin cậy Judge:** Agreement V1 0.738 / V2 0.788 · **Cohen's Kappa** V1 0.127 / V2 0.28 (mức "fair").
- **Hiệu năng:** avg latency V1 3.43s · V2 7.48s (V2 dùng model 70B nên chậm hơn 2.18x).
- **Chi phí:** ~$0.00035/eval · tổng ~$0.007 cho 20 case.

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng (V2) | Nguyên nhân dự kiến |
|----------|:---:|---------------------|
| Judge Conflict (2 giám khảo lệch > 1 điểm) | 8/20 | Sai khác độ "khắt khe" giữa Llama-70B và gpt-oss-120B; kéo nhiều câu trả lời ĐÚNG xuống điểm 3 |
| Over-refusal (V2 từ chối quá đà) | 2/20 (1 thành fail) | Prompt V2 ép "chỉ trả lời theo ngữ cảnh / nói không biết" kích hoạt sai ở câu hỏi cài bẫy |
| RAGAS faithfulness/relevancy thấp | toàn bộ | Cohere rate-limit → dùng proxy token-overlap (không phản ánh đúng độ trung thực ngữ nghĩa) |
| Retrieval noise (top-3 lẫn chunk lạc) | ~6/20 | Lexical lấy đúng chunk #1 nhưng kèm chunk nhiễu ở vị trí 2-3 (thiếu rerank do Cohere tắt) |

> Retrieval **không phải** nguồn lỗi chính ở lần chạy này: Hit Rate/MRR = 1.0 (chunk đúng luôn ở hạng 1).
> Lỗi tập trung ở **tầng đánh giá (judge variance)** và **prompting (over-refusal của V2)**.

## 3. Phân tích 5 Whys (3 case đáng chú ý nhất)

### Case #1 — GS-029 (V2 FAIL, score 2.5): Over-refusal ở câu hỏi cài bẫy
*Câu hỏi:* "Tại sao hàng khuyến mãi cuối mùa **có thể** được đổi trả?" — tiền đề sai (REF-1 nói hàng final-sale **không** được đổi trả trừ lỗi nặng).
1. **Symptom:** V2 trả lời "tôi không tìm thấy thông tin... liên hệ tổng đài" thay vì đính chính tiền đề sai.
2. **Why 1:** Prompt V2 yêu cầu "nếu ngữ cảnh không chứa thông tin thì nói không biết".
3. **Why 2:** Ngữ cảnh (REF-1) chứa thông tin *phủ định* tiền đề, nhưng prompt không hướng dẫn Agent xử lý câu hỏi có **tiền đề sai (false premise)**.
4. **Why 3:** Agent diễn giải "không có thông tin khẳng định" = "không biết", nên từ chối.
5. **Root Cause:** **Prompting** — prompt grounded quá cứng, thiếu nhánh "nếu ngữ cảnh mâu thuẫn với câu hỏi thì đính chính". (V1 prompt lỏng hơn nên không fail case này.)

### Case #2 — GS-001 (score 3, conflict): Judge bất đồng trên câu trả lời ĐÚNG
*Câu hỏi:* "Làm thế nào để đổi mật khẩu?" — Agent trả lời **chính xác** theo ACC-0.
1. **Symptom:** Câu trả lời đúng nhưng điểm cuối chỉ 3 (Judge A và B lệch nhau, agreement 0.5, phải gọi tiebreaker Gemini).
2. **Why 1:** Llama-70B chấm cao, gpt-oss-120B chấm thấp hơn ≥2 điểm.
3. **Why 2:** Hai model có "thước đo" completeness/tone khác nhau cho cùng một câu trả lời.
4. **Why 3:** Rubric chấm điểm còn để khoảng diễn giải rộng (chưa neo ví dụ điểm 3 vs 5).
5. **Root Cause:** **Calibration tầng đánh giá** — cần neo rubric bằng few-shot/anchor examples để giảm variance giữa các họ model. Cohen's Kappa 0.13-0.28 ("fair") xác nhận điều này.

### Case #3 — Sụt faithfulness/relevancy toàn cục: phụ thuộc hạ tầng
1. **Symptom:** Faithfulness ~0.36, Relevancy ~0.30 — thấp bất thường dù câu trả lời bám sát ngữ cảnh.
2. **Why 1:** Hai chỉ số này được tính bằng cosine trên embedding Cohere.
3. **Why 2:** Key Cohere (trial) bị `TooManyRequests` giữa buổi → circuit breaker tắt Cohere.
4. **Why 3:** Evaluator fallback sang **token-overlap** (đếm từ chung) — không đo được tương đồng ngữ nghĩa.
5. **Root Cause:** **Ingestion/Embedding pipeline** phụ thuộc một nhà cung cấp có hạn mức thấp. Cần key trả phí hoặc embedding cục bộ để có chỉ số RAGAS đáng tin.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] **Embedding ổn định:** dùng Cohere key trả phí (hoặc embedding cục bộ) để bật lại vector search + rerank → đo faithfulness/relevancy thật và kiểm chứng lại lợi thế V2.
- [ ] **Sửa over-refusal V2:** thêm nhánh prompt "nếu ngữ cảnh mâu thuẫn với câu hỏi, hãy đính chính thay vì từ chối".
- [ ] **Calibrate judge:** neo rubric bằng anchor examples (mẫu điểm 1/3/5) để kéo Cohen's Kappa lên mức "substantial" (>0.6).
- [ ] **Giảm latency V2:** cân nhắc model 70B chỉ cho case khó; dùng 8B cho case dễ (router theo độ khó).
- [ ] **Chạy đầy đủ 50+ case** khi hạn mức API cho phép (hiện chạy mẫu 20 do free-tier rate limit).
