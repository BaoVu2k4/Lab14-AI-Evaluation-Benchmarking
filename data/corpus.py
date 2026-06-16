"""
Knowledge Base (Golden Corpus) cho Evaluation Factory.

Đây là "nguồn sự thật" mà RAG Agent truy xuất và là tài liệu gốc để SDG sinh
câu hỏi. Mỗi chunk có ID ổn định (vd: ACC-0) -> chính là Ground Truth ID dùng
để tính Hit Rate / MRR.

Nội dung mô phỏng cẩm nang hỗ trợ khách hàng của một sàn TMĐT giả định ("ShopVN").
"""
from __future__ import annotations

from typing import Dict, List

# Mỗi tài liệu: prefix ID + tiêu đề + danh sách chunk (mỗi chunk 1 đoạn).
_DOCUMENTS: List[Dict] = [
    {
        "prefix": "ACC",
        "title": "Tài khoản & Đăng nhập",
        "chunks": [
            "Để đổi mật khẩu, vào Cài đặt > Bảo mật > Đổi mật khẩu, nhập mật khẩu hiện tại "
            "rồi nhập mật khẩu mới hai lần. Mật khẩu mới phải dài tối thiểu 8 ký tự, gồm cả "
            "chữ hoa, chữ thường và số.",
            "Nếu quên mật khẩu, chọn 'Quên mật khẩu' tại màn hình đăng nhập. Hệ thống gửi "
            "đường dẫn đặt lại mật khẩu tới email đã đăng ký, đường dẫn này có hiệu lực trong "
            "30 phút và chỉ dùng được một lần.",
            "Tài khoản sẽ bị khoá tạm thời trong 15 phút sau 5 lần nhập sai mật khẩu liên tiếp. "
            "Đây là cơ chế chống dò mật khẩu tự động (brute-force).",
            "Bạn có thể bật Xác thực hai lớp (2FA) qua ứng dụng Authenticator hoặc qua mã OTP "
            "gửi về SMS. Khi bật 2FA, mỗi lần đăng nhập trên thiết bị mới sẽ cần thêm mã 6 số.",
        ],
    },
    {
        "prefix": "ORD",
        "title": "Đơn hàng & Vận chuyển",
        "chunks": [
            "Sau khi đặt hàng, đơn sẽ ở trạng thái 'Chờ xác nhận' trong tối đa 2 giờ. Bạn có thể "
            "huỷ đơn miễn phí khi đơn vẫn đang ở trạng thái 'Chờ xác nhận' hoặc 'Chờ lấy hàng'.",
            "Thời gian giao hàng tiêu chuẩn là 3-5 ngày làm việc với nội thành và 5-7 ngày với "
            "khu vực tỉnh xa. Đơn giao nhanh (Hoả tốc) chỉ áp dụng nội thành và giao trong 4 giờ.",
            "Phí vận chuyển được tính theo trọng lượng và khoảng cách. Đơn hàng có giá trị từ "
            "300.000đ trở lên được miễn phí vận chuyển tiêu chuẩn trên toàn quốc.",
            "Bạn có thể theo dõi hành trình đơn hàng tại mục 'Đơn mua' > chọn đơn > 'Theo dõi'. "
            "Mã vận đơn được cung cấp ngay khi đơn chuyển sang trạng thái 'Đang giao'.",
        ],
    },
    {
        "prefix": "REF",
        "title": "Đổi trả & Hoàn tiền",
        "chunks": [
            "Chính sách đổi trả cho phép trả hàng trong vòng 7 ngày kể từ ngày nhận đối với sản "
            "phẩm lỗi do nhà sản xuất, giao sai mẫu hoặc thiếu phụ kiện. Sản phẩm phải còn nguyên "
            "tem mác và hộp đựng.",
            "Hàng thời trang đã qua sử dụng, đồ lót, và sản phẩm khuyến mãi cuối mùa (final sale) "
            "không được áp dụng đổi trả, trừ trường hợp lỗi nghiêm trọng từ nhà sản xuất.",
            "Sau khi yêu cầu hoàn tiền được duyệt, tiền sẽ được hoàn về phương thức thanh toán gốc "
            "trong 5-7 ngày làm việc với thẻ ngân hàng, hoặc hoàn ngay vào Ví ShopVN trong 24 giờ.",
            "Để tạo yêu cầu trả hàng, vào 'Đơn mua' > chọn đơn > 'Trả hàng/Hoàn tiền', chọn lý do "
            "và đính kèm ảnh hoặc video sản phẩm lỗi để bộ phận kiểm duyệt xử lý nhanh hơn.",
        ],
    },
    {
        "prefix": "PAY",
        "title": "Thanh toán",
        "chunks": [
            "ShopVN hỗ trợ thanh toán khi nhận hàng (COD), thẻ ATM nội địa, thẻ tín dụng "
            "Visa/Mastercard, ví điện tử MoMo/ZaloPay và Ví ShopVN. Thanh toán COD có giới hạn "
            "tối đa 20.000.000đ cho mỗi đơn hàng.",
            "Khi thanh toán bằng thẻ tín dụng, giao dịch được mã hoá theo chuẩn PCI-DSS và ShopVN "
            "không lưu trữ số thẻ đầy đủ của khách hàng trên hệ thống.",
            "Trả góp 0% lãi suất áp dụng cho đơn hàng từ 3.000.000đ qua thẻ tín dụng của các ngân "
            "hàng liên kết, với kỳ hạn 3, 6 hoặc 12 tháng tuỳ chính sách từng ngân hàng.",
        ],
    },
    {
        "prefix": "SEC",
        "title": "Bảo mật & Quyền riêng tư",
        "chunks": [
            "ShopVN không bao giờ yêu cầu khách hàng cung cấp mật khẩu hoặc mã OTP qua điện thoại, "
            "email hay tin nhắn. Mọi yêu cầu như vậy đều là dấu hiệu lừa đảo, hãy bỏ qua và báo cáo.",
            "Dữ liệu cá nhân của khách hàng được lưu trữ và xử lý theo Nghị định 13/2023/NĐ-CP về "
            "bảo vệ dữ liệu cá nhân. Khách hàng có quyền yêu cầu truy cập, chỉnh sửa hoặc xoá dữ liệu.",
            "Nếu phát hiện hoạt động bất thường trên tài khoản, hệ thống sẽ gửi cảnh báo qua email "
            "và tạm thời yêu cầu xác minh lại danh tính trước khi cho phép giao dịch giá trị lớn.",
        ],
    },
    {
        "prefix": "TEC",
        "title": "Hỗ trợ kỹ thuật & Sự cố",
        "chunks": [
            "Nếu ứng dụng bị treo hoặc tải chậm, hãy thử xoá bộ nhớ đệm (cache) trong phần Cài đặt "
            "ứng dụng, cập nhật lên phiên bản mới nhất, hoặc gỡ và cài đặt lại ứng dụng.",
            "Lỗi 'Không thể kết nối máy chủ' thường do mạng không ổn định. Hãy chuyển đổi giữa Wi-Fi "
            "và 4G, kiểm tra lại kết nối, sau đó thử lại sau vài phút.",
            "Tổng đài hỗ trợ khách hàng hoạt động từ 8h đến 22h hằng ngày qua số 1900-1234. Kênh "
            "chat trực tuyến trong ứng dụng hoạt động 24/7 với trợ lý ảo và nhân viên trực ca.",
        ],
    },
]


def load_chunks() -> List[Dict]:
    """Trả về danh sách chunk phẳng: {id, doc_title, text}."""
    chunks: List[Dict] = []
    for doc in _DOCUMENTS:
        for i, text in enumerate(doc["chunks"]):
            chunks.append(
                {
                    "id": f"{doc['prefix']}-{i}",
                    "doc_title": doc["title"],
                    "text": " ".join(text.split()),
                }
            )
    return chunks


def chunks_by_id() -> Dict[str, Dict]:
    return {c["id"]: c for c in load_chunks()}


if __name__ == "__main__":
    cs = load_chunks()
    print(f"Tổng số chunk: {len(cs)}")
    for c in cs[:3]:
        print(c["id"], "-", c["text"][:60], "...")
