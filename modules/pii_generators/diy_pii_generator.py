import random


class DIYPIIGenerator:
    LIST = 1
    INT = 2
    PROMPT = 3
    FLOAT = 4

    def __init__(self, generator_type):
        self.generator_type = generator_type

    def generate(self, *params):
        """
        Generate a value based on the generator type and parameters.

        - LIST: Select a random item from a list of options.
        - INT: Generate a random integer between a given range.
        - PROMPT: Return a placeholder value for manual input.
        """
        if self.generator_type == DIYPIIGenerator.LIST:
              # First parameter is the list of options
            params = params[0]
            if isinstance(params, list):
                return random.choice(params)
            else:
                raise ValueError("LIST generator requires a list as the first parameter.")

        elif self.generator_type == DIYPIIGenerator.INT:
            if len(params) >= 2:
                min_val, max_val = params[0], params[1]
                return random.randint(min_val, max_val)
            else:
                raise ValueError("INT generator requires two parameters: min and max.")

        elif self.generator_type == DIYPIIGenerator.FLOAT:
            if len(params) >= 2:
                min_val, max_val = params[0], params[1]

                # 计算 min_val 和 max_val 中的小数位数
                min_decimal_places = len(str(min_val).split('.')[-1]) if '.' in str(min_val) else 0
                max_decimal_places = len(str(max_val).split('.')[-1]) if '.' in str(max_val) else 0
                decimal_places = max(min_decimal_places, max_decimal_places)  # 保留更高的小数位数

                # 生成随机浮动值
                result = random.uniform(min_val, max_val)

                # 保留小数位数
                return round(result, decimal_places)
            else:
                raise ValueError("INT generator requires two parameters: min and max.")
        elif self.generator_type == DIYPIIGenerator.PROMPT:
            return f"[${params[0]}$]"

        else:
            raise ValueError("Invalid generator type.")