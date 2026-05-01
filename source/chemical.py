"""化学分子式处理模块 - MatPreML"""

import pandas as pd
import numpy as np
from .config import PERIODIC_TABLE, ATOMIC_NUMBERS


class ChemicalFormulaProcessor:
    """处理化学分子式的类"""

    def __init__(self):
        self.periodic_table = PERIODIC_TABLE
        self.atomic_numbers = ATOMIC_NUMBERS

    def is_chemical_formula(self, text):
        """判断文本是否为化学分子式"""
        if not isinstance(text, str):
            return False
        text = text.replace(' ', '')
        if not text:
            return False
        if not all(c.isalnum() or c == '.' for c in text):
            return False
        i = 0
        while i < len(text):
            if text[i].isupper():
                if i + 1 < len(text) and text[i + 1].islower():
                    element = text[i:i + 2]
                    i += 2
                else:
                    element = text[i]
                    i += 1
                if element not in self.periodic_table:
                    return False
                while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                    i += 1
            else:
                return False
        return True

    def parse_chemical_formula(self, formula):
        """解析化学分子式，返回元素和数量的列表"""
        if not isinstance(formula, str):
            return []
        formula = formula.replace(' ', '')
        elements = []
        i = 0
        while i < len(formula):
            if formula[i].isupper():
                if i + 1 < len(formula) and formula[i + 1].islower():
                    element = formula[i:i + 2]
                    i += 2
                else:
                    element = formula[i]
                    i += 1
                count = ''
                while i < len(formula) and (formula[i].isdigit() or formula[i] == '.'):
                    count += formula[i]
                    i += 1
                if count:
                    count = float(count) if '.' in count else int(count)
                else:
                    count = 1
                atomic_number = self.periodic_table.get(element, 0)
                if atomic_number > 0:
                    elements.append((atomic_number, count))
        return elements

    def process_chemical_data(self, data, reference_elements=None):
        """处理包含化学分子式的数据

        Args:
            data: 要处理的数据
            reference_elements: 参考元素列表（用于预测数据），如果为None则使用数据中所有元素
        Returns:
            tuple: (处理后的数据, 元素列表) 或 (处理后的数据, None)
        """
        if data is None or data.empty:
            return data, None

        first_column = data.iloc[:, 0]
        chemical_count = 0
        total_count = len(first_column)
        for value in first_column:
            if pd.isna(value):
                continue
            if self.is_chemical_formula(str(value)):
                chemical_count += 1

        if chemical_count / total_count > 0.5 and total_count > 0:
            all_elements = set()
            formulas_elements = []
            for formula in first_column:
                if pd.isna(formula):
                    formulas_elements.append([])
                    continue
                if self.is_chemical_formula(str(formula)):
                    elements = self.parse_chemical_formula(str(formula))
                    formulas_elements.append(elements)
                    if reference_elements is None:
                        for atomic_number, _ in elements:
                            all_elements.add(atomic_number)
                else:
                    formulas_elements.append([])

            if reference_elements is not None:
                sorted_elements = sorted(list(reference_elements))
            else:
                sorted_elements = sorted(list(all_elements))

            element_symbols = [self.atomic_numbers.get(z, '') for z in sorted_elements]
            new_data = pd.DataFrame(index=data.index, columns=element_symbols)
            for idx, elements in enumerate(formulas_elements):
                element_counts = {z: 0 for z in sorted_elements}
                for atomic_number, count in elements:
                    if atomic_number in sorted_elements:
                        element_counts[atomic_number] += count
                for atomic_number, count in element_counts.items():
                    element_symbol = self.atomic_numbers.get(atomic_number, '')
                    new_data.loc[idx, element_symbol] = count

            new_data = new_data.fillna(0)
            result_data = new_data
            if len(data.columns) > 1:
                result_data = pd.concat([result_data, data.iloc[:, 1:]], axis=1)

            if reference_elements is None:
                return result_data, sorted_elements
            else:
                return result_data, None
        else:
            return data, None
