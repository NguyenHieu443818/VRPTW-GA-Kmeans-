# Mô Hình Đề Xuất Cho Bài Toán VRPTW Soft Customer Time Window

## 1. Ý Tưởng Tổng Quan

Mô hình đề xuất kết hợp phân rã bằng Fuzzy STD C-medoids, sinh route ứng viên bằng các thuật toán tiến hóa và chọn nghiệm toàn cục bằng Set Partitioning (SP). Cửa sổ thời gian tại khách hàng là mềm: xe vẫn được phục vụ khách đến muộn, nhưng độ trễ được cộng vào fitness. Tải trọng xe và thời hạn quay về depot là các ràng buộc cứng.

```text
Dữ liệu VRPTW
    ↓
Fuzzy STD C-medoids → U, S_std
    ↓
Tạo subproblem chồng lấn bằng ngưỡng rho cố định
    ↓
GA giant-tour và/hoặc PSO trên từng subproblem
    ↓
Decode: cắt route khi vượt tải hoặc không thể về depot đúng hạn
    ↓
Route Pool Ω
    ↓
Set Partitioning → nghiệm cuối cùng
```

## 2. Dữ Liệu Và Mô Hình Đánh Giá Lời Giải

```text
C = {1, 2, ..., n}: tập khách hàng
0: depot
V = {1, 2, ..., K}: tập xe
q_i: nhu cầu khách hàng i
Q: sức chứa tối đa của xe
[e_i, l_i]: cửa sổ thời gian mềm của khách hàng i
s_i: thời gian phục vụ khách hàng i
t_ij: thời gian di chuyển từ i đến j
c_ij: chi phí di chuyển từ i đến j
[e_0, l_0]: thời gian hoạt động của depot
```

Giả sử `x_ij^k = 1` khi xe `k` đi trực tiếp từ đỉnh `i` đến khách hàng `j`. Khi đó, `a_j^k` là thời điểm đến, `w_j^k` là thời gian chờ, `u_j^k` là thời điểm bắt đầu phục vụ và `p_j^k` là thời gian trễ tại khách hàng `j`.

### 2.1. Propagation thời gian

Với `i` là đỉnh ngay trước `j` trên tuyến xe `k`:

```text
a_j^k = t_0j                         nếu i = 0
a_j^k = u_i^k + s_i + t_ij           nếu i ≠ 0

w_j^k = max(e_j - a_j^k, 0)
u_j^k = a_j^k + w_j^k
p_j^k = max(u_j^k - l_j, 0)
```

Khách hàng có `p_j^k > 0` vẫn được phục vụ. Không dùng các hệ số phạt riêng theo khách hàng `alpha_i` hoặc `beta_i`; mức phạt chính là thời gian chờ và thời gian trễ được tính trực tiếp trong fitness.

### 2.2. Ràng buộc cứng và decoder

Khi decoder xét thêm khách hàng `j` vào route hiện tại, khách hàng chỉ được giữ lại trên route đó nếu cả hai điều kiện sau đúng:

```text
load_current + q_j ≤ Q
u_j^k + s_j + t_j0 ≤ l_0
```

Nếu một điều kiện bị vi phạm, decoder đóng route hiện tại bằng cách thêm depot, sau đó mở route mới từ depot để phục vụ `j`. Vì thời hạn depot là cứng, cần kiểm tra trước với mọi khách hàng:

```text
max(e_i, t_0i) + s_i + t_i0 ≤ l_0
q_i ≤ Q
```

Nếu một khách hàng không thỏa một trong hai điều kiện này thì không tồn tại route khả thi để phục vụ khách hàng đó.

### 2.3. Fitness

Với lời giải `S` gồm các route, đặt:

```text
D(S) = tổng chi phí di chuyển của tất cả các cung được sử dụng
W(S) = tổng thời gian chờ tại khách hàng
P(S) = tổng thời gian trễ tại khách hàng

Fitness(S) = D(S) + W(S) + P(S)
```

Nếu benchmark dùng cùng đơn vị cho chi phí và thời gian di chuyển, có thể đặt `c_ij = t_ij`. Nếu không, `c_ij` chỉ dùng trong `D(S)` và `t_ij` chỉ dùng khi lan truyền thời gian.

## 3. Phase 1: Fuzzy STD C-medoids

Mỗi khách hàng được biểu diễn bởi vector:

```text
tau_i = (x_i, y_i, theta_i, e_i, l_i, s_i, q_i)
theta_i = arctan2(y_i - y_0, x_i - x_0)
```

Khoảng cách không gian mở rộng:

```text
S_s_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2
              + lambda * (theta_i - theta_j)^2)
```

Các thành phần thời gian:

```text
f_ij = l_j - (e_i + s_i + t_ij)
h_ij = max(e_j - (l_i + s_i + t_ij), 0)
```

Khoảng cách STD có hướng và khoảng cách đối xứng dùng cho clustering:

```text
S_tilde_std_ij = S_s_ij * (2 - (f_ij - h_ij)/(l_0 - e_0)
                            + (q_i + q_j)/Q)

S_std_ij = min(S_tilde_std_ij, S_tilde_std_ji)
```

Fuzzy c-medoids tạo ma trận membership `U`, với:

```text
sum_p U[i,p] = 1, với mọi khách hàng i
main(i) = argmax_p U[i,p]
```

## 4. Phase 2: Tạo Subproblem Chồng Lấn Bằng Rho

Người dùng truyền ngưỡng cố định `rho ∈ [0, 1]`. Khách hàng biên là khách hàng có mức độ thuộc về cụm chính không vượt quá ngưỡng này:

```text
B = { i ∈ C | U[i, main(i)] ≤ rho }
```

Mỗi khách hàng không thuộc `B` chỉ nằm trong cụm chính. Với khách hàng biên, xác định cụm phụ có membership lớn thứ hai:

```text
alt(i) = argmax_{p ≠ main(i)} U[i,p]

C_p = {i | main(i) = p} ∪ {i ∈ B | alt(i) = p}
```

Do đó, một khách hàng thuộc tối đa hai subproblem: cụm chính và một cụm phụ. Nếu `|C_p| > size_limit`, chỉ cắt bớt phần khách hàng biên của cụm phụ; không được loại khách hàng lõi.

## 5. Phase 3: Sinh Route Ứng Viên

GA giant-tour và/hoặc PSO swap-sequence được chạy trên mỗi subproblem `C_p`. Một cá thể là hoán vị không lặp của các khách hàng trong `C_p`.

```text
pi_p = [v_1, v_2, ..., v_m]
```

Decoder thực hiện theo quy tắc sau:

```text
route = [0]
current_load = 0
current_time = e_0

FOR mỗi khách v trong pi_p:
    tính a_v, w_v, u_v nếu v được nối sau khách cuối của route
    IF current_load + q_v > Q
       OR u_v + s_v + t_v0 > l_0:
        đóng route hiện tại bằng depot
        mở route mới từ depot
        tính lại a_v, w_v, u_v

    thêm v vào route hiện tại
    cập nhật current_load và current_time = u_v + s_v

đóng route cuối bằng depot
```

Vi phạm time window của khách hàng không làm cắt route; nó chỉ được phản ánh qua `p_v`. Mọi route đưa vào pool phải thỏa tải trọng và phải quay về depot không muộn hơn `l_0`.

Với mỗi route `r`:

```text
D(r) = tổng c_ij trên route r, bao gồm cung đi và về depot
W(r) = tổng w_i trên route r
P(r) = tổng p_i trên route r
c_r  = D(r) + W(r) + P(r)
```

Định kỳ, thu thập các route từ nghiệm tốt của GA/PSO vào route pool `Omega`. Hai route có cùng thứ tự khách hàng chỉ giữ một bản có `c_r` nhỏ nhất.

## 6. Phase 4: Route Pool Và Set Partitioning

Với mỗi route `r ∈ Omega`, đặt:

```text
a_ir = 1 nếu route r phục vụ khách hàng i, ngược lại bằng 0
x_r  = 1 nếu route r được chọn trong nghiệm cuối, ngược lại bằng 0
```

Set Partitioning master problem:

```text
minimize    sum_{r ∈ Omega} c_r * x_r

subject to  sum_{r ∈ Omega} a_ir * x_r = 1,  với mọi i ∈ C
            sum_{r ∈ Omega} x_r ≤ K,          nếu số xe K là giới hạn cứng
            x_r ∈ {0, 1},                     với mọi r ∈ Omega
```

Route singleton `[0, i, 0]` được thêm vào pool cho mỗi khách hàng thỏa điều kiện tiền xử lý ở Mục 2.2. Điều này bảo đảm SP có ít nhất một phương án phủ mỗi khách hàng.

Nếu dùng greedy thay ILP solver, chỉ chọn route `r` khi tất cả khách hàng của route đều chưa được phủ:

```text
r ⊆ uncovered
```

Quy tắc này tránh việc một khách hàng bị phủ nhiều hơn một lần.

## 7. Phase 5: Cải Thiện Cục Bộ

Local search ưu tiên các cụm có chi phí trung bình trên route cao và các route có mức sử dụng tải trọng thấp. Relocate, swap và cross-over chỉ được xét giữa các subproblem/khách hàng lân cận theo `S_std`.

Khi dùng Fuzzy c-medoids, khách hàng biên thỏa `U[i, main(i)] ≤ rho` được ưu tiên trong các move liên tuyến. Một move chỉ được chấp nhận nếu các route sau cập nhật vẫn thỏa tải trọng, quay về depot trước `l_0`, và làm giảm fitness.

## 8. Các Khác Biệt Đã Loại Bỏ So Với Bản Trước

```text
- Không dùng alpha_i, beta_i hoặc trọng số penalty alpha riêng.
- Không dùng fixed vehicle cost F trong cost của route.
- Không dùng OverlapScore, entropy hoặc percentile threshold.
- Không coi depot deadline là soft: route bị cắt nếu không thể quay về depot đúng hạn.
- Không dùng khẳng định “mọi route đều feasible”; chỉ customer time window là mềm.
```

## 9. Tên Mô Hình

```text
Fuzzy STD Rho-overlapping Decomposition with Route-pool Set Partitioning
Tên rút gọn: FSRD-SP
```
