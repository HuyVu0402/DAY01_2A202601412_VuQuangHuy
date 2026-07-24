# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi temperature = 0.0, câu trả lời ổn định, chính xác và ít sáng tạo. Khi tăng lên 0.7 và 1.2, phản hồi trở nên đa dạng, tự nhiên và sáng tạo hơn nhưng vẫn giữ được sự mạch lạc. Ở khoảng 1.8, câu trả lời bắt đầu kém mạch lạc hơn, dễ lan man hoặc đưa ra những chi tiết không cần thiết.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Đối với trợ lý soạn thảo hợp đồng pháp lý, tôi sẽ đặt temperature khoảng 0.0–0.2 để đảm bảo câu trả lời chính xác, nhất quán và hạn chế sáng tạo không cần thiết. Đối với trợ lý viết slogan quảng cáo, tôi sẽ đặt temperature khoảng 0.8–1.2 để tạo ra nhiều ý tưởng đa dạng và sáng tạo. Hai ứng dụng có mục tiêu khác nhau nên cần mức độ ngẫu nhiên khác nhau.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Mỗi ngày có 20.000 × 2 = 40.000 lượt gọi API, mỗi lượt khoảng 500 output token, tương đương 20.000.000 output token/ngày. Với GPT-4o, chi phí khoảng 200 USD/ngày (20.000 × 0.01 USD/1K token). Với GPT-4o-mini, chi phí khoảng 12 USD/ngày (20.000 × 0.0006 USD/1K token). Model lớn phù hợp với các tác vụ yêu cầu chất lượng cao như phân tích tài liệu pháp lý hoặc y tế, còn model nhỏ phù hợp với chatbot hỗ trợ khách hàng hoặc trả lời các câu hỏi đơn giản để tiết kiệm chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona là nhà thơ, câu trả lời giàu hình ảnh, sử dụng nhiều phép so sánh và hạn chế thuật ngữ kỹ thuật. Với persona là kỹ sư phần mềm senior, câu trả lời chi tiết hơn, sử dụng thuật ngữ chính xác và có thể kèm ví dụ hoặc đoạn mã minh họa. Điều này cho thấy system prompt có thể điều khiển giọng văn, mức độ chuyên môn, phong cách trình bày và độ dài của phản hồi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Kết quả đếm bằng tiktoken thường khác so với cách ước lượng theo số từ, với sai lệch khoảng 10–20% tùy nội dung. Đối với tiếng Việt, nếu chỉ dùng công thức ước lượng theo số từ thì dự toán chi phí có thể không chính xác vì token không tương ứng trực tiếp với số từ và còn phụ thuộc vào cách tokenizer chia văn bản. Vì vậy, sử dụng tiktoken sẽ cho kết quả đáng tin cậy hơn khi tính chi phí API.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Streaming mang lại lợi ích lớn nhất cho chatbot văn bản và đặc biệt là trợ lý giọng nói, vì người dùng có thể đọc hoặc nghe phản hồi ngay khi model bắt đầu sinh nội dung, giúp giảm cảm giác chờ đợi. Đối với pipeline dịch tài liệu chạy ngầm ban đêm, streaming hầu như không cần thiết vì người dùng chỉ quan tâm đến kết quả cuối cùng chứ không tương tác trực tiếp trong quá trình xử lý.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff giúp các client không gửi lại yêu cầu quá dồn dập khi API đang quá tải, từ đó giảm áp lực lên máy chủ và tăng khả năng yêu cầu thành công ở lần thử sau. Nếu tất cả client đều retry theo cùng một khoảng thời gian cố định thì chúng có thể tiếp tục gửi yêu cầu cùng lúc. Kỹ thuật jitter thêm một khoảng trễ ngẫu nhiên vào mỗi lần retry để tránh hiện tượng nhiều client đồng thời gửi lại request (thundering herd).

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: "Bạn là trợ giảng AI thân thiện, trả lời ngắn gọn bằng tiếng Việt, giải thích rõ ràng và đưa ví dụ khi cần thiết." Nếu bỏ cụm "trả lời ngắn gọn", phản hồi sẽ dài và chi tiết hơn. Nếu bỏ cụm "đưa ví dụ khi cần thiết", câu trả lời sẽ ít ví dụ minh họa, khiến người học khó hiểu hơn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Nếu cuộc hội thoại kéo dài và thông tin quan trọng xuất hiện ở những lượt đầu, việc chỉ giữ lại 4 lượt hội thoại gần nhất có thể khiến trợ lý quên ngữ cảnh và trả lời không chính xác. Một cách khắc phục là tóm tắt các lượt hội thoại cũ rồi lưu bản tóm tắt vào history, hoặc chỉ giữ lại những thông tin quan trọng thay vì cắt theo số lượng message cố định.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
