# 🚀 Lab Day 14: AI Evaluation Factory (Team Edition)

## 🎯 Tổng quan
"Nếu bạn không thể đo lường nó, bạn không thể cải thiện nó." — Nhiệm vụ của nhóm bạn là xây dựng một **Hệ thống đánh giá tự động** chuyên nghiệp để benchmark AI Agent. Hệ thống này phải chứng minh được bằng con số cụ thể: Agent đang tốt ở đâu và tệ ở đâu.

---

## 🕒 Lịch trình thực hiện (4 Tiếng)
- **Giai đoạn 1 (45'):** Thiết kế Golden Dataset & Script SDG. Tạo ra ít nhất 50 test cases chất lượng.
- **Giai đoạn 2 (90'):** Phát triển Eval Engine (RAGAS, Custom Judge) & Async Runner.
- **Giai đoạn 3 (60'):** Chạy Benchmark, Phân cụm lỗi (Failure Clustering) & Phân tích "5 Whys".
- **Giai đoạn 4 (45'):** Tối ưu Agent dựa trên kết quả & Hoàn thiện báo cáo nộp bài.

---

## 🛠️ Các nhiệm vụ chính (Expert Mission)

### 1. Retrieval & SDG (Nhóm Data)
- **Retrieval Eval:** Tính toán Hit Rate và MRR cho Vector DB. Bạn phải chứng minh được Retrieval stage hoạt động tốt trước khi đánh giá Generation.
- **SDG:** Tạo 50+ cases, bao gồm cả Ground Truth IDs của tài liệu để tính Hit Rate.

### 2. Multi-Judge Consensus Engine (Nhóm AI/Backend)
- **Consensus logic:** Sử dụng ít nhất 2 model Judge khác nhau. 
- **Calibration:** Tính toán hệ số đồng thuận (Agreement Rate) và xử lý xung đột điểm số tự động.

### 3. Regression Release Gate (Nhóm DevOps/Analyst)
- **Delta Analysis:** So sánh kết quả của Agent phiên bản mới với phiên bản cũ.
- **Auto-Gate:** Viết logic tự động quyết định "Release" hoặc "Rollback" dựa trên các chỉ số Chất lượng/Chi phí/Hiệu năng.

---

## 📤 Danh mục nộp bài (Submission Checklist)
Nhóm nộp 1 đường dẫn Repository (GitHub/GitLab) chứa:
1. [ ] **Source Code**: Toàn bộ mã nguồn hoàn chỉnh.
2. [ ] **Reports**: File `reports/summary.json` và `reports/benchmark_results.json` (được tạo ra sau khi chạy `main.py`).
3. [ ] **Group Report**: File `analysis/failure_analysis.md` (đã điền đầy đủ).
4. [ ] **Individual Reports**: Các file `analysis/reflections/reflection_[Tên_SV].md`.

---

## 🏆 Bí kíp đạt điểm tuyệt đối (Expert Tips)

### ✅ Đánh giá Retrieval (15%)
Nhóm nào chỉ đánh giá câu trả lời mà bỏ qua bước Retrieval sẽ không thể đạt điểm tối đa. Bạn cần biết chính xác chunk nào đang gây ra lỗi Hallucination.

### ✅ Multi-Judge Reliability (20%)
Việc chỉ tin vào một Judge (ví dụ GPT-4o) là một sai lầm trong sản phẩm thực tế. Hãy chứng minh hệ thống của bạn khách quan bằng cách so sánh nhiều Judge model và tính toán độ tin cậy của chúng.

### ✅ Tối ưu hiệu năng & Chi phí (15%)
Hệ thống Expert phải chạy cực nhanh (Async) và phải có báo cáo chi tiết về "Giá tiền cho mỗi lần Eval". Hãy đề xuất cách giảm 30% chi phí eval mà không giảm độ chính xác.

### ✅ Phân tích nguyên nhân gốc rễ (Root Cause) (20%)
Báo cáo 5 Whys phải chỉ ra được lỗi nằm ở đâu: Ingestion pipeline, Chunking strategy, Retrieval, hay Prompting.

---

## 🔧 Hướng dẫn chạy

### Cấu hình `.env` (bắt buộc, KHÔNG commit)
```env
GROQ_API_KEY=...        # Judge A (Llama-70B) + Agent
GROQ_API_KEY_2=...      # Judge B (gpt-oss-120B) + Agent V1
MISTRAL_API_KEY=...     # (dự phòng)
GEMINI_API_KEY=...      # Tiebreaker (Gemini) khi 2 judge bất đồng
COHERE_API_KEY=...      # Embedding + Rerank (RAG)
QDRANT_URL=...          # Vector DB
QDRANT_API_KEY=...
```
> Hệ thống có **fallback**: nếu Cohere/Qdrant bị rate-limit, Retriever tự chuyển
> sang tìm kiếm **lexical (TF-IDF)** và faithfulness/relevancy dùng proxy
> token-overlap — pipeline vẫn chạy xong, không lỗi.

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Tạo Golden Dataset (chạy trước khi benchmark)
python data/synthetic_gen.py

# 3. Chạy Benchmark & tạo reports (V1 vs V2 + Release Gate)
python main.py
#    Chạy nhanh trên mẫu N case (free-tier rate-limit chậm):
#    BENCH_LIMIT=20 python main.py

# 4. Kiểm tra định dạng trước khi nộp
python check_lab.py
```

### 🧩 Kiến trúc giám khảo (chống "thông đồng")
Panel dùng **3 họ model khác nhau** để tránh thiên vị tương quan:
- **Judge A:** Groq Llama-3.3-70B (Meta) · **Judge B:** Groq gpt-oss-120B (OpenAI)
- **Tiebreaker:** Google Gemini-2.5-flash — chỉ gọi khi 2 judge lệch > 1 điểm (cascading, tiết kiệm chi phí).

---

## ⚠️ Lưu ý quan trọng
- **Bắt buộc** chạy `python data/synthetic_gen.py` trước để tạo file `data/golden_set.jsonl`. File này không được commit sẵn trong repo.
- Trước khi nộp bài, hãy chạy `python check_lab.py` để đảm bảo định dạng dữ liệu đã chuẩn. Bất kỳ lỗi định dạng nào dẫn đến việc script chấm điểm tự động không chạy được sẽ bị trừ 5 điểm thủ tục.
- File `.env` chứa API Key **KHÔNG** được push lên GitHub.

---
*Chúc nhóm bạn xây dựng được một Evaluation Factory thực sự mạnh mẽ!*
