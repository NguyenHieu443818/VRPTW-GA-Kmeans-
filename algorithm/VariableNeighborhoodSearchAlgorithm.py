import random
import math


class VNS():

    def __init__(self, list_n_l: list, beta_0: int, beta_1: int, generation: int):
        # Danh sách [n1-n6] là số lần chọn điểm hoặc đoạn tối đa cho từng thuật toán
        self.list_n_l = list_n_l
        self.beta_0 = beta_0  # Tham số thứ 1 tính Max_n
        self.beta_1 = beta_1  # Tham số thứ 2 tính Max_n
        self.generation = generation  # Số đời quần thể (số vòng lặp)

        # Danh sách [rn1-rn6] là số lần chọn điểm hoặc đoạn cho từng thuật toán
        self.r_n_l = [random.randint(1, i) for i in self.list_n_l]
        # Tổng số vòng lặp tối đa
        self.max_n = 6*self.beta_0 + 6 * \
            math.floor(self.beta_1 * math.sqrt(self.generation))
        # Số vòng lặp tối đa cho mỗi thuật toán con
        self.S_l = math.floor(self.max_n/6)

    def insertion_operator(self, num_ran_point: int, gene: list):
        if len(gene) < num_ran_point or num_ran_point <= 0:
            return gene

        ran_val = random.sample(gene, num_ran_point)
        gene = [x for x in gene if x not in ran_val]

        ran_pos = random.sample(range(len(gene) + 1), num_ran_point)
        for index, pos in enumerate(ran_pos):
            gene.insert(pos, ran_val[index])

        return gene

    def pairwise_exchange_operator(self, num_ran_pair_point: int, gene: list):
        if len(gene) < 2 * num_ran_pair_point or num_ran_pair_point <= 0:
            return gene

        indices_to_swap = random.sample(
            range(len(gene)), 2 * num_ran_pair_point)

        for i in range(0, 2 * num_ran_pair_point, 2):
            idx1 = indices_to_swap[i]
            idx2 = indices_to_swap[i + 1]
            gene[idx1], gene[idx2] = gene[idx2], gene[idx1]

        return gene

    def fragment_part_reverse_operator(self, len_gen: int, gene: list):
        if len(gene) < len_gen or len_gen <= 1:
            return gene

        start_pos = random.randint(0, len(gene) - len_gen)
        end_pos = start_pos + len_gen

        gene[start_pos:end_pos] = gene[start_pos:end_pos][::-1]
        return gene

    def fragment_two_part_inversion_operator(self, len_gen: int, gene: list):
        # Cần ít nhất 2 * len_gen phần tử để tìm được 2 đoạn không chồng lấn
        if len(gene) < 2 * len_gen or len_gen <= 0:
            return gene

        max_start = len(gene) - 2 * len_gen
        start_pos_1 = random.randint(0, max_start)
        start_pos_2 = random.randint(start_pos_1 + len_gen, len(gene) - len_gen)

        end_pos_1, end_pos_2 = start_pos_1 + len_gen, start_pos_2 + len_gen
        gene[start_pos_1:end_pos_1], gene[start_pos_2:end_pos_2] = (
            gene[start_pos_2:end_pos_2],
            gene[start_pos_1:end_pos_1],
        )

        return gene

    def random_exchange_operator(self, num_ran_point: int, gene: list):
        if len(gene) < num_ran_point or num_ran_point <= 1:
            return gene

        ran_indices = random.sample(range(len(gene)), num_ran_point)
        values = [gene[indice] for indice in ran_indices]

        random.shuffle(ran_indices)

        for i, index in enumerate(ran_indices):
            gene[index] = values[i]

        return gene

    def fragment_translation_operator(self, len_gen: int, gene: list):
        if len(gene) <= len_gen or len_gen <= 0:
            return gene

        start_pos = random.randint(0, len(gene) - len_gen)
        end_pos = start_pos + len_gen

        part_to_move = gene[start_pos:end_pos]

        # Xóa phần đoạn cần dịch chuyển
        gene = gene[:start_pos] + gene[end_pos:]

        ran_insert_point = random.randint(0, len(gene))
        gene = gene[:ran_insert_point] + part_to_move + gene[ran_insert_point:]

        return gene

    def choose_operator(self, index: int, r_n: int, gene: list):
        match index:
            case 0:
                return self.insertion_operator(num_ran_point=r_n, gene=gene)
            case 1:
                return self.pairwise_exchange_operator(num_ran_pair_point=r_n, gene=gene)
            case 2:
                return self.fragment_part_reverse_operator(len_gen=r_n, gene=gene)
            case 3:
                return self.fragment_two_part_inversion_operator(len_gen=r_n, gene=gene)
            case 4:
                return self.random_exchange_operator(num_ran_point=r_n, gene=gene)
            case 5:
                return self.fragment_translation_operator(len_gen=r_n, gene=gene)

    def is_better(self, new_gene: list, gene: list):
        # check xem gen mới tốt hơn gen cũ không
        pass

    def fit(self, gene: list):

        # Thứ tự các thuật toán
        order_operator = random.sample(range(6), 6)
        # Duyệt từng thuật toán 1
        for i in range(6):
            # Xét từng vòng lặp
            for _ in range(self.S_l):
                new_gene = self.choose_operator(index=order_operator[i], r_n=self.r_n_l[i], gene=gene.copy())
                    
                if (self.is_better(new_gene, gene)):
                    gene = new_gene.copy()
                    break
   
        return gene
