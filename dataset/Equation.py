import torch

op_one = ["g_half", "g_double", "g_equal", "g_sin", "g_cos", "g_tan", "cal_circle_area", "cal_circle_perimeter", "g_acos", "g_asin"]
op_two = ["g_minus", "g_add", "gougu_add", "cal_cone", "gougu_minus", "g_mul", "g_divide"]
op_three = ["g_bili"]

oe_constant = ["C_0", "C_1", "C_2", "C_3", "C_4", "C_5", "C_6"]
oe_number = ["N_0", "N_1", "N_2", "N_3", "N_4", "N_5", "N_6", "N_7", "N_8", "N_9", "N_10", "N_11"]
oe_previous_result = ["V_0", "V_1", "V_2"]

max_equation_length = 4
max_operand_length = 3

op_list = op_one + op_two + op_three
oe_list = oe_constant + oe_number + oe_previous_result


class Equation:
    def __init__(self, text = None):
        self.text = text
        self.equation = None
        self.operator = None
        self.operator_idx = None
        self.operand = None
        self.operand_idx = None

        if text is not None:
            self.text_to_equation(text)

    def solve_equation(self, equation):
        return eval(equation)

    def text_to_equation(self, text):
        target_text = text.split()
        equation = []
        operator = []
        operator_idx = []
        operand = []
        operand_idx = []
        cur_operand_idx = []
        for i, cur in enumerate(target_text):
            if cur.startswith('g') or cur.startswith('cal_'):
                if cur_operand_idx:
                    operand_idx.append(cur_operand_idx)

                cur_operand_idx = []

                operator.append(cur)
                operator_idx.append(op_list.index(cur))
            elif cur.startswith('C') or cur.startswith('N') or cur.startswith('V'):
                operand.append(cur)
                cur_operand_idx.append(oe_list.index(cur))
            else:
                print(cur)
                assert False
            equation.append(cur)
        if cur_operand_idx:
            operand_idx.append(cur_operand_idx)

        self.equation = equation
        self.operator = operator
        self.operator_idx = operator_idx
        self.operand = operand
        self.operand_idx = operand_idx

    def get_operator_ids(self):
        operator_idx = self.operator_idx + [len(op_list)] * (max_equation_length - len(self.operator_idx))

        return torch.LongTensor(operator_idx)

    def get_operand_ids(self):
        operand_idx = []
        for cur in self.operand_idx:
            operand_idx.append(cur + [len(oe_list)] * (max_operand_length - len(cur)))
        operand_idx = operand_idx + [[len(oe_list)] * max_operand_length ] * (max_equation_length - len(self.operand_idx))

        return torch.LongTensor(operand_idx)

