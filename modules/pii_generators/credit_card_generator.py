import random
from randomtimestamp import randomtimestamp
from datetime import datetime


class CC():
    '''Individual card info and methods.
    '''
    # `len_num` - Number of digits in the CC number.
    # `len_cvv` - Number of digits in the CVV number.
    #  `pre` - List of the digits a card can start with.
    # `remaining` - Number of digits after `pre`.

    CCDATA = {
        'amex': {
            'len_num': 15,
            'len_cvv': 4,
            'pre': [34, 37],
            'remaining': 13
        },
        'discover': {
            'len_num': 16,
            'len_cvv': 3,
            'pre': [6001],
            'remaining': 12
        },
        'mastercard': {
            'len_num': 16,
            'len_cvv': 3,
            'pre': [51, 55],
            'remaining': 14
        },
        'visa13': {
            'len_num': 13,
            'len_cvv': 3,
            'pre': [4],
            'remaining': 12
        },
        'visa': {
            'len_num': 16,
            'len_cvv': 3,
            'pre': [4],
            'remaining': 15
        },
    }

    def __init__(self):
        self.cc_type = None
        self.cc_len = None
        self.cc_num = None
        self.cc_cvv = None
        self.cc_exp = None
        self.cc_prefill = []

    def generate_cc_exp(self):
        '''Generates a card expiration date that is
        between 1 and 3 years from today. Sets `cc_exp`.
        '''
        # pattern = "%d-%m-%Y %H:%M:%S"
        self.cc_exp = randomtimestamp(
            start_year=datetime.now().year + 1,
            text=True,
            end_year=datetime.now().year + 3,
            start=None,
            end=None,
            pattern="%m-%Y")

    def generate_cc_cvv(self):
        '''Generates a type-specific CVV number.
        Sets `cc_cvv`.
        '''
        this = []
        length = self.CCDATA[self.cc_type]['len_cvv']

        for x_ in range(length):
            this.append(random.randint(0, 9))

        self.cc_cvv = ''.join(map(str, this))

    def generate_cc_prefill(self):
        '''Generates the card's starting numbers
        and sets `cc_prefill`.
        '''
        this = self.CCDATA[self.cc_type]['pre']
        self.cc_prefill = random.choices(this)

    def generate_cc_num(self):
        '''Uses Luhn algorithm to generate a theoretically
        valid credit card number. Sets `cc_num`.
        '''
        # Generate all but the last digit
        remaining = self.CCDATA[self.cc_type]['remaining']
        working = self.cc_prefill + [random.randint(1, 9) for x in range(remaining - 1)]

        # `check_offset` determines if final list length is
        # even or odd, which affects the check_sum calculation.
        # Also avoids reversing the list, which is specified in Luhn.
        check_offset = (len(working) + 1) % 2
        check_sum = 0

        for i, n in enumerate(working):
            if (i + check_offset) % 2 == 0:
                n_ = n * 2
                check_sum += n_ - 9 if n_ > 9 else n_
            else:
                check_sum += n

        temp = working + [10 - (check_sum % 10)]
        self.cc_num = "".join(map(str, temp))

    def generate_card(self):
        card_type = random.choice(['amex', 'discover', 'mastercard', 'visa'])
        self.cc_type = card_type
        self.generate_cc_cvv()
        self.generate_cc_exp()
        self.generate_cc_prefill()
        self.generate_cc_num()
        return {'cc_brand': self.cc_type,
                'cc_num': self.cc_num,
                'cc_cvv': self.cc_cvv,
                'cc_exp': self.cc_exp}

if __name__ == "__main__":
    ins = CC()
    print(ins.generate_card())