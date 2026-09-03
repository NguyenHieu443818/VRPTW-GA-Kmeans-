from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


SOURCE = Path(r"C:\Users\admin\OneDrive\Tài liệu\nguyễn minh hiếu\Mô_hình_bài_toán_đã_sửa.docx")
OUTPUT = Path(r"D:\Code\Python\hieunm\output\Mô_hình_bài_toán_đã_cập_nhật.docx")


def set_font(run, name="Times New Roman", size=13, bold=False, italic=False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:cs"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_text(doc, text, *, bold_prefix=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.3
    if bold_prefix and text.startswith(bold_prefix):
        set_font(p.add_run(bold_prefix), bold=True)
        set_font(p.add_run(text[len(bold_prefix):]))
    else:
        set_font(p.add_run(text))
    return p


def add_equation(doc, equation):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    set_font(p.add_run(equation), name="Cambria Math", size=12)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.style = f"Heading {level}"
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_font(run, size=15 if level == 1 else 13, bold=True)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.2
    set_font(p.add_run(text))
    return p


def build():
    # Open the source once to make the revision intentionally based on it.
    Document(SOURCE)
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(13)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(14)
    set_font(title.add_run("MÔ HÌNH BÀI TOÁN VÀ THUẬT TOÁN ĐỀ XUẤT"), size=16, bold=True)

    heading(doc, "1. Mô hình đánh giá lời giải VRPTW", 1)
    add_text(doc, "Bài toán gồm depot 0 và tập khách hàng C = {1, 2, ..., n}. Tập đỉnh là N = {0} ∪ C; tập xe là V = {1, 2, ..., K}. Mỗi khách hàng i có nhu cầu qᵢ, cửa sổ thời gian [eᵢ, lᵢ] và thời gian phục vụ sᵢ. Mỗi xe có tải trọng tối đa Q. Chi phí và thời gian di chuyển từ i đến j lần lượt là cᵢⱼ và tᵢⱼ.")
    add_text(doc, "Biến xᵢⱼᵏ nhận giá trị 1 nếu xe k đi trực tiếp từ đỉnh i đến j; ngược lại nhận giá trị 0. Với mỗi cung được sử dụng, aⱼᵏ là thời điểm xe đến khách hàng j, wⱼᵏ là thời gian chờ, uⱼᵏ là thời điểm bắt đầu phục vụ và pⱼᵏ là thời gian phục vụ trễ.")

    heading(doc, "1.1. Quy tắc tính thời gian trên một tuyến", 2)
    add_text(doc, "Với xe k, nếu xᵢⱼᵏ = 1 thì i là đỉnh đi ngay trước khách hàng j. Các đại lượng thời gian được tính tuần tự như sau:")
    add_equation(doc, "aⱼᵏ = { t₀ⱼ, nếu i = 0;   uᵢᵏ + sᵢ + tᵢⱼ, nếu i ≠ 0 }")
    add_equation(doc, "wⱼᵏ = max{eⱼ − aⱼᵏ, 0};     uⱼᵏ = aⱼᵏ + wⱼᵏ")
    add_equation(doc, "pⱼᵏ = max{uⱼᵏ − lⱼ, 0}")
    add_text(doc, "Cửa sổ thời gian của khách hàng là mềm: khách hàng vẫn được phục vụ khi pⱼᵏ > 0, nhưng độ trễ này làm tăng fitness. Depot và sức chứa là ràng buộc cứng.")

    heading(doc, "1.2. Ràng buộc cứng và quy tắc tách tuyến", 2)
    add_text(doc, "Trong khi giải mã một hoán vị khách hàng thành tuyến, khách tiếp theo j chỉ được thêm vào tuyến hiện tại nếu đồng thời thỏa mãn:")
    add_equation(doc, "Lᵢᵏ + qⱼ ≤ Q")
    add_equation(doc, "uⱼᵏ + sⱼ + tⱼ₀ ≤ l₀")
    add_text(doc, "Nếu một trong hai điều kiện trên không thỏa mãn, tuyến hiện tại được đóng bằng cách quay về depot và xe mới bắt đầu phục vụ j từ depot. Vì depot có thời hạn cứng, mỗi khách hàng phải thỏa điều kiện tiền xử lý max{eᵢ, t₀ᵢ} + sᵢ + tᵢ₀ ≤ l₀; nếu không, bản thân khách hàng i không có tuyến khả thi.")

    heading(doc, "1.3. Hàm fitness", 2)
    add_text(doc, "Fitness thống nhất với mô hình ban đầu: tổng chi phí di chuyển, tổng thời gian chờ và tổng thời gian trễ. Không sử dụng hệ số phạt riêng theo từng khách hàng αᵢ hoặc βᵢ.")
    add_equation(doc, "D(S) = Σ₍ᵢ,ⱼ₎∈S cᵢⱼ;     W(S) = Σ wⱼᵏ;     P(S) = Σ pⱼᵏ")
    add_equation(doc, "Fitness(S) = D(S) + W(S) + P(S)")
    add_text(doc, "Mục tiêu là tối thiểu hóa Fitness(S). Nếu bộ dữ liệu dùng cùng đơn vị cho chi phí và thời gian di chuyển thì có thể đặt cᵢⱼ = tᵢⱼ; nếu không, cᵢⱼ phải được dùng riêng trong D(S).")

    heading(doc, "2. Phân rã bằng Fuzzy STD C-medoids", 1)
    add_text(doc, "Mỗi khách hàng i được biểu diễn bởi vector đặc trưng τᵢ = (xᵢ, yᵢ, θᵢ, eᵢ, lᵢ, sᵢ, qᵢ), trong đó θᵢ = arctan2(yᵢ − y₀, xᵢ − x₀).")
    add_equation(doc, "Sˢᵢⱼ = √[(xⱼ−xᵢ)² + (yⱼ−yᵢ)² + λ(θⱼ−θᵢ)²]")
    add_equation(doc, "fᵢⱼ = lⱼ − (eᵢ + sᵢ + tᵢⱼ);     hᵢⱼ = max{eⱼ − (lᵢ + sᵢ + tᵢⱼ), 0}")
    add_equation(doc, "Ṡˢᵗᵈᵢⱼ = Sˢᵢⱼ[2 − (fᵢⱼ − hᵢⱼ)/(l₀−e₀) + (qᵢ+qⱼ)/Q]")
    add_equation(doc, "Sˢᵗᵈᵢⱼ = min{Ṡˢᵗᵈᵢⱼ, Ṡˢᵗᵈⱼᵢ}")
    add_text(doc, "Fuzzy c-medoids tạo ma trận membership U = (μᵢ,ₚ), với Σₚ μᵢ,ₚ = 1. Sau hội tụ, cụm chính của khách hàng i là main(i) = argmaxₚ μᵢ,ₚ.")

    heading(doc, "2.1. Khách hàng biên và subproblem chồng lấn", 2)
    add_text(doc, "Không sử dụng OverlapScore hay ngưỡng theo phân vị. Người dùng truyền trực tiếp ngưỡng cố định ρ ∈ [0, 1]. Khách hàng i được xem là khách hàng biên nếu mức độ thuộc về cụm chính của nó không vượt quá ngưỡng này:")
    add_equation(doc, "B = { i ∈ C | μᵢ,ₘₐᵢₙ₍ᵢ₎ ≤ ρ }")
    add_text(doc, "Khách hàng không thuộc B chỉ nằm trong cụm chính. Mỗi khách hàng thuộc B được thêm vào đúng một cụm phụ có membership lớn thứ hai, vì vậy một khách hàng xuất hiện trong nhiều nhất hai subproblem:")
    add_equation(doc, "alt(i) = argmax₍ₚ ≠ main(i)₎ μᵢ,ₚ")
    add_equation(doc, "Cₚ = {i | main(i)=p} ∪ {i ∈ B | alt(i)=p}")
    add_text(doc, "Nếu một subproblem vượt quá size_limit, chỉ giữ các khách hàng biên có μᵢ,ₚ cao nhất trong phần cụm phụ; toàn bộ khách hàng lõi của cụm phải luôn được giữ lại.")

    heading(doc, "3. Sinh route ứng viên", 1)
    add_text(doc, "GA giant-tour và PSO swap-sequence được chạy trên từng subproblem Cₚ để sinh các hoán vị khách hàng. Decoder dùng các quy tắc tại Mục 1.2: chỉ cắt tuyến khi vượt tải trọng hoặc khi sau khi phục vụ khách kế tiếp xe không thể quay về depot trước l₀. Vi phạm cửa sổ thời gian của khách hàng không làm cắt tuyến mà chỉ được cộng vào P(S).")
    add_text(doc, "Mỗi route ứng viên r lưu danh sách khách hàng, tổng chi phí di chuyển D(r), tổng thời gian chờ W(r), tổng trễ P(r), và chi phí route:")
    add_equation(doc, "cᵣ = D(r) + W(r) + P(r)")
    add_text(doc, "Route chỉ được đưa vào route pool Ω khi thỏa tải trọng và quay về depot trước l₀. Route trùng thứ tự khách hàng chỉ lưu một bản có chi phí thấp nhất.")

    heading(doc, "4. Chọn nghiệm toàn cục bằng Set Partitioning", 1)
    add_text(doc, "Route pool Ω tổng hợp route từ mọi subproblem, bao gồm các route được tạo quanh khách hàng biên. Gọi aᵢᵣ = 1 nếu route r phục vụ khách hàng i, và bằng 0 nếu ngược lại. Bài toán Set Partitioning là:")
    add_equation(doc, "min Σᵣ∈Ω cᵣxᵣ")
    add_equation(doc, "Σᵣ∈Ω aᵢᵣxᵣ = 1, ∀i ∈ C;     xᵣ ∈ {0,1}")
    add_text(doc, "Nếu số xe khả dụng bị giới hạn bởi K, bổ sung ràng buộc Σᵣ∈Ω xᵣ ≤ K. Để bảo đảm pool phủ được mọi khách hàng, thêm route singleton [i] cho từng khách hàng thỏa điều kiện quay về depot ở Mục 1.2.")
    add_text(doc, "Với phương pháp greedy thay cho ILP, chỉ được chọn route r nếu toàn bộ khách hàng trong r chưa được phủ, tức r ⊆ uncovered. Quy tắc này bảo đảm không có khách hàng nào bị phục vụ lặp lại.")

    heading(doc, "5. Cải thiện cục bộ dựa trên dữ liệu mờ", 1)
    add_text(doc, "Sau khi có tập route, local search ưu tiên các cụm có chi phí trung bình trên tuyến cao và các tuyến có mức sử dụng tải trọng thấp. Di chuyển liên tuyến chỉ xét các cụm lân cận theo ma trận Sˢᵗᵈ và khách hàng lân cận theo Sˢᵗᵈ. Khi dùng Fuzzy c-medoids, khách hàng biên thỏa μᵢ,ₘₐᵢₙ₍ᵢ₎ ≤ ρ được ưu tiên trong các phép relocate, swap và cross-over. Mọi move chỉ được chấp nhận nếu các tuyến sau cập nhật vẫn thỏa tải trọng và hạn về depot, đồng thời làm giảm Fitness(S).")

    heading(doc, "Tài liệu tham khảo", 1)
    add_text(doc, "[1] Kerscher, C., & Minner, S. (2024). Spatial-temporal-demand clustering for solving large-scale vehicle routing problems with time windows. arXiv:2402.00041.")
    add_text(doc, "[2] Emambocus, B. A. S., Jasser, M. B., Hamzah, M., Mustapha, A., & Amphawan, A. (2021). An Enhanced Swap Sequence-Based Particle Swarm Optimization Algorithm to Solve TSP. IEEE Access, 9, 164820-164836.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print("Document created.")


if __name__ == "__main__":
    build()
