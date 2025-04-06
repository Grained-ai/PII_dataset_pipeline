import json
from pathlib import Path

from faker import Faker
import usaddress
import random_address
from modules.pii_generators.phone_cellphone_number_generator import generate_cell_phone_number
from modules.pii_generators.credit_card_generator import CC
import random
import string


class PersonGenerator:
    def __init__(self, sex=None, age_min=18, age_max=70, if_middle=None, state_abbr=None):
        # Set locale to 'en_US' for consistent U.S.-based data
        self.fake = Faker('en_US')
        # self.fake.add_provider(person)
        # self.fake.seed_instance(random.randint(1, 1000))

        # Sex - Randomly assign male or female
        ADDRESSES_PATH = Path(__file__).parent / 'addresses.json'
        with open(ADDRESSES_PATH, 'r') as f:
            addresses = json.load(f)
        self.Sex = random.choice(['Male', 'Female']) if not sex else sex  # Randomly select Male or Female
        self.if_middle_name = random.choice([1, 0, 0, 0, 0]) if if_middle else 0

        # Generate names based on Sex
        if self.Sex == 'Male':
            # self.FirstName = random.choice([self.fake.first_name_male(), self.fake.first_name_male_est()])
            self.FirstName = self.fake.first_name_male()

        else:
            # self.FirstName = random.choice([self.fake.first_name_female(), self.fake.first_name_female_est()])  # Generate female first name
            self.FirstName = self.fake.first_name_female()  # Generate female first name

        self.MiddleName = self.generate_middle_name(self.Sex) if self.if_middle_name else ""

        self.LastName = self.fake.last_name()
        # self.FullName = random.choice([f"{self.FirstName} {self.LastName}", f"{self.FirstName} {self.LastName}".upper(),
        #                                f"{self.FirstName}{self.LastName}",
        #                                f"{self.FirstName} {self.MiddleName} {self.LastName}"])
        self.FullName = random.choice([f"{self.FirstName} {self.LastName}", f"{self.FirstName} {self.LastName}".upper(),
                                       f"{self.FirstName} {self.LastName}"]) if not self.if_middle_name else f"{self.FirstName} {self.MiddleName[0].upper()}. {self.LastName}"

        self.UserName = self.fake.user_name()
        self.Initials = ".".join(
            [name[0].upper() for name in [self.FirstName, self.LastName]]) if not self.if_middle_name else ".".join(
            [name[0].upper() for name in [self.FirstName, self.MiddleName, self.LastName]])
        self.EmailAddress = self.generate_email()

        # Address details
        self.Country = "United States"  # Fixed to the U.S.
        if state_abbr:
            self.StateAbbr = state_abbr
        else:
            while 1:
                self.StateAbbr = self.fake.state_abbr()
                if addresses.get(self.StateAbbr):
                    break
        abbrev_to_us_state = {
            "AL": "Alabama",
            "AK": "Alaska",
            "AZ": "Arizona",
            "AR": "Arkansas",
            "CA": "California",
            "CO": "Colorado",
            "CT": "Connecticut",
            "DE": "Delaware",
            "FL": "Florida",
            "GA": "Georgia",
            "HI": "Hawaii",
            "ID": "Idaho",
            "IL": "Illinois",
            "IN": "Indiana",
            "IA": "Iowa",
            "KS": "Kansas",
            "KY": "Kentucky",
            "LA": "Louisiana",
            "ME": "Maine",
            "MD": "Maryland",
            "MA": "Massachusetts",
            "MI": "Michigan",
            "MN": "Minnesota",
            "MS": "Mississippi",
            "MO": "Missouri",
            "MT": "Montana",
            "NE": "Nebraska",
            "NV": "Nevada",
            "NH": "New Hampshire",
            "NJ": "New Jersey",
            "NM": "New Mexico",
            "NY": "New York",
            "NC": "North Carolina",
            "ND": "North Dakota",
            "OH": "Ohio",
            "OK": "Oklahoma",
            "OR": "Oregon",
            "PA": "Pennsylvania",
            "RI": "Rhode Island",
            "SC": "South Carolina",
            "SD": "South Dakota",
            "TN": "Tennessee",
            "TX": "Texas",
            "UT": "Utah",
            "VT": "Vermont",
            "VA": "Virginia",
            "WA": "Washington",
            "WV": "West Virginia",
            "WI": "Wisconsin",
            "WY": "Wyoming"
        }
        self.State = abbrev_to_us_state.get(self.StateAbbr)
        # self.State = self.fake.state()

        # self.ZipCode = self.fake.zipcode_in_state(self.StateAbbr)

        # self.address_raw = random_address.real_random_address_by_state(self.StateAbbr)

        # self.Address = " ".join([self.address_raw.get('address1', ""), self.address_raw.get('city', self.City),
        #                          self.address_raw.get('postalCode', self.ZipCode),
        #                          self.address_raw.get('state', self.StateAbbr)])
        while 1:
            try:
                self.Address = random.choice(addresses.get(self.StateAbbr))
                tagged_address = usaddress.tag(self.Address)
                if tagged_address[1] == 'Street Address':
                    tagged_address = tagged_address[0]
                    break
            except:
                pass

        self.ZipCode = tagged_address.get('ZipCode').split('-')[0]
        self.StreetNumber = tagged_address.get('AddressNumber')
        street_name_parts = [tagged_address[k] for k in tagged_address.keys() if
                             k not in ['ZipCode', 'CountryName', 'AddressNumber', 'StateName', 'PlaceName']]
        self.City = tagged_address.get('PlaceName')
        self.City = self.City[0].upper() + self.City[1:].lower()
        self.StreetName = " ".join([i for i in street_name_parts if i])
        # self.StreetNumber = self.fake.building_number()
        # self.StreetName = self.fake.street_name()
        # self.Address = f"{self.StreetNumber} {self.StreetName}, {self.City}, {self.State}, {self.ZipCode}, {self.Country}"
        # self.PhoneNumber = self.fake.phone_number()
        self.PhoneNumber = generate_cell_phone_number(self.StateAbbr)
        self.FaxNumber = generate_cell_phone_number(self.StateAbbr, True)
        self.OfficePhoneNumber = generate_cell_phone_number(self.StateAbbr)
        # self.HomeNumber = self.fake.cellphone_number()
        # Identity details
        self.PassportNumber = self.generate_passport_number()
        self.DriverLicense = self.generate_driver_license()
        self.SocialSecurityNumber = self.fake.ssn()
        self.GeneralIDs = self.generate_general_id()

        # Financial-samples details
        credit_card = CC().generate_card()
        self.CreditCardNumber = credit_card['cc_num']
        self.CreditCardDetails = credit_card
        self.BankAccountNumber = self.generate_bank_account_number()

        # Date details
        self.Date = self.fake.date_between(start_date='-3y', end_date='today')
        self.Timestamps = self.Date

        self.DateOfBirth = self.fake.date_of_birth(minimum_age=age_min, maximum_age=age_max)

        # Network details

        self.IPAddress = self.fake.ipv4_private()
        self.MACAddress = self.fake.mac_address()

        self.MedID = self.generate_random_medical_id()
        # State Work related:
        self.JobSeriesNum = random.randint(1, 10)
        self.pay_plan = f"GS_{self.JobSeriesNum}"
        self.JobSeriesNum_full = str(self.JobSeriesNum) if self.JobSeriesNum>=10 else f"0{self.JobSeriesNum}"
        self.fpl = f"GS_{self.JobSeriesNum+random.randint(2, 4)}"
        self.position_description_number = f"GS-{random.randint(20, 24)}0{random.randint(1,9)}-{self.JobSeriesNum_full}-{random.randint(10000, 99999)}"

    def generate_middle_name(self, sex):
        if sex == 'Male':
            return random.choice([
                "James", "Henry", "Jude", "Finn", "William", "Jameson", "Miles", "Oscar", "Jude", "Alexander",
                "Kai", "River", "Elias", "Atlas", "Finnian", "Jude", "Apollo", "Nash", "Leo", "Grey", "Jude",
                "William", "Elijah", "Rhys", "Bennett", "Beau", "Oliver", "Arthur", "Cole", "Orion", "Maximus",
                "Jude", "Gray", "Patrick", "Oliver", "Brooks", "Bennett", "Wells", "Reeve", "Rex", "Silas",
                "Ezra", "Mason", "Atticus", "Elijah", "Beau", "Zane", "Isaac", "Sawyer", "Gideon", "Quinn",
                "Theo", "Emmett", "Quinn", "Roscoe", "Asher", "Xavier", "Boone", "Gray", "August", "Maxwell",
                "Luke", "Rowan", "Griffin", "Kit", "Greyson", "Theodore", "Zeke", "Rhett", "Solomon", "Knox",
                "Beau", "Archer", "Wilder", "Leo", "Jett", "Phoenix", "Cash", "Cruz", "August", "Beckett",
                "George", "Elias", "Chase", "Montgomery", "Sullivan", "Weston", "Josiah", "Jasper", "Solomon",
                "Paxton", "Colton", "Justice", "Beck", "Finnick", "Hunter", "Wyatt", "Kai", "Kingston", "Thatcher"
            ])
        else:
            return random.choice([
                "Rose", "Grace", "Mae", "Jane", "Elizabeth", "Claire", "Lynn", "Anne", "Hope", "May",
                "Faith", "Louise", "Kate", "Victoria", "Marie", "Belle", "Rae", "Dawn", "Faye", "Lila",
                "Ella", "Paige", "Charlotte", "Lou", "Beth", "Mia", "Olivia", "Sophia", "Maeve", "Alice",
                "Eve", "Sage", "Violet", "Willow", "Ava", "Emma", "Kate", "Ella", "Zoe", "Nora", "Lily",
                "Grace", "Hazel", "Ruth", "Skye", "Blair", "Amelia", "Piper", "Harper", "Ruby", "Camille",
                "Luna", "Maggie", "Ivy", "Isla", "Madeline", "Charlotte", "Adeline", "Genevieve", "Juliet",
                "Isabel", "Lillian", "Marley", "Ariana", "Wren", "Penelope", "Leona", "Seraphina", "Cora",
                "Camilla", "Vivian", "Madeline", "Evelyn", "Sophie", "Lacey", "Estelle", "Sienna", "Aurora",
                "Matilda", "Rosemary", "Juliana", "Lydia", "Adele", "Willow", "Athena", "Vera", "Tess",
                "Harriet", "Mabel", "Blythe", "Josephine", "Charlotte", "Sloane", "Indigo", "Hazel", "Zara",
                "Clementine", "Coraline", "Amelia", "Blossom", "Marlene", "Camden", "Cleo", "Katherine",
                "Olive", "Carmen", "Gwendolyn", "Sophie", "Aspen", "Adelaide", "Veda", "Eden", "Marina",
                "Felicity", "Leila", "Iris", "Vivienne", "Eleanor", "Josephine", "Veronica", "Cora", "Greer",
                "Avery", "Tatum", "Emery", "Quinn", "Bliss", "Adalyn", "Larkin", "Autumn", "Leah", "Haven"
            ])

    def generate_email_username(self):
        # 基础部分：名字和姓氏的组合方式
        templates = [
            '{fname}',  # 只有名字
            '{lname}',  # 只有姓氏
            '{fname}.{lname}',  # 名字.姓氏
            '{fname}_{lname}',  # 名字_姓氏
            '{fname}{year}',  # 名字+年份
            '{lname}{year}',  # 姓氏+年份
            '{fname[0]}{lname}',  # 名字首字母+姓氏
            '{fname}{lname[0]}',  # 名字+姓氏首字母
            '{fname[0:3]}{lname[0:3]}',  # 名字前三个字母+姓氏前三个字母
        ]

        # 随机选择一个模板
        template = random.choice(templates)

        # 生成随机年份或数字（有时人们会用生日或幸运数字）
        year_or_number = str(random.randint(1950, 2025)) if 'year' in template else ''.join(
            random.choices(string.digits, k=4))

        # 格式化模板
        email_username = template.format(fname=self.FirstName.lower(), lname=self.LastName.lower(), year=year_or_number)

        return email_username

    def generate_email(self):
        """Generate an email address based on first and last name."""
        domain = random.choice([random.choice([
            "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
            "protonmail.com", "icloud.com", "mail.com", "yandex.com", "zoho.com",
            "gmx.com", "fastmail.com", "tutanota.com", "hushmail.com", "posteo.net",
            "live.com", "rediffmail.com", "lycos.com", "rocketmail.com"]),
            self.fake.free_email_domain()])
        while 1:
            try:
                email_name = self.generate_email_username()
                break
            except:
                pass

        return f"{email_name}@{domain}"

    def generate_passport_number(self):
        """Generate a U.S. passport number (9 digits)."""
        return ''.join(random.choices(string.digits, k=9))

    def generate_driver_license(self):
        """Generate a U.S. driver license (15 alphanumeric characters)."""
        license_formats = {
            'AL': ['{:08d}'.format(random.randint(0, 99999999))],  # 1-8 Numeric
            'AK': ['{:07d}'.format(random.randint(0, 9999999))],  # 1-7 Numeric
            'AZ': ['{}{:08d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999)),
                   '{:09d}'.format(random.randint(0, 999999999))],  # 1 Alpha + 8 Numeric or 9 Numeric
            'AR': ['{:09d}'.format(random.randint(0, 999999999))],  # 4-9 Numeric (for simplicity, using 9 digits)
            'CA': ['{}{:07d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 9999999))],
            # 1 Alpha + 7 Numeric
            'CO': ['{:09d}'.format(random.randint(0, 999999999)),
                   '{}{:06d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999)),
                   '{}{}{:05d}'.format(random.choice(string.ascii_uppercase), random.choice(string.ascii_uppercase),
                                       random.randint(0, 99999))],
            # 9 Numeric or 1 Alpha + 3-6 Numeric or 2 Alpha + 2-5 Numeric
            'CT': ['{:09d}'.format(random.randint(0, 999999999))],  # 9 Numeric
            'DE': ['{:07d}'.format(random.randint(0, 9999999))],  # 1-7 Numeric
            'DC': ['{:07d}'.format(random.randint(0, 9999999)), '{:09d}'.format(random.randint(0, 999999999))],
            # 7 Numeric or 9 Numeric
            'FL': ['{}{:012d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999999))],
            # 1 Alpha + 12 Numeric
            'GA': ['{:09d}'.format(random.randint(0, 999999999))],  # 7-9 Numeric (for simplicity, using 9 digits)
            'HI': ['{}{:08d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999)),
                   '{:09d}'.format(random.randint(0, 999999999))],  # 1 Alpha + 8 Numeric or 9 Numeric
            'ID': ['{}{:06d}{}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999),
                                       random.choice(string.ascii_uppercase)),
                   '{:09d}'.format(random.randint(0, 999999999))],  # 2 Alpha + 6 Numeric + 1 Alpha or 9 Numeric
            'IL': ['{}{:011d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999999)),
                   '{}{:012d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999999))],
            # 1 Alpha + 11-12 Numeric
            'IN': ['{}{:09d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999)),
                   '{:09d}'.format(random.randint(0, 999999999)), '{:010d}'.format(random.randint(0, 9999999999))],
            # 1 Alpha + 9 Numeric or 9-10 Numeric
            'IA': ['{:09d}'.format(random.randint(0, 999999999)),
                   '{:03d}{}{:04d}'.format(random.randint(0, 999), random.choice(string.ascii_uppercase),
                                           random.randint(0, 9999))],  # 9 Numeric or 3 Numeric + 2 Alpha + 4 Numeric
            'KS': ['{}{}{}{}{}'.format(random.choice(string.ascii_uppercase), random.randint(0, 9),
                                       random.choice(string.ascii_uppercase), random.randint(0, 9),
                                       random.choice(string.ascii_uppercase)),
                   '{}{:08d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999)),
                   '{:09d}'.format(random.randint(0, 999999999))],
            # 1 Alpha + 1 Numeric + 1 Alpha + 1 Numeric + 1 Alpha or 1 Alpha + 8 Numeric or 9 Numeric
            'KY': ['{}{:08d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999)),
                   '{}{:09d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999)),
                   '{:09d}'.format(random.randint(0, 999999999))],
            # 1 Alpha + 8 Numeric or 1 Alpha + 9 Numeric or 9 Numeric
            'LA': ['{:09d}'.format(random.randint(0, 999999999))],  # 1-9 Numeric (for simplicity, using 9 digits)
            'ME': ['{:07d}'.format(random.randint(0, 9999999)),
                   '{:07d}{}'.format(random.randint(0, 9999999), random.choice(string.ascii_uppercase)),
                   '{:08d}'.format(random.randint(0, 99999999))],  # 7 Numeric or 7 Numeric + 1 Alpha or 8 Numeric
            'MD': ['{}{:012d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999999))],
            # 1 Alpha + 12 Numeric
            'MA': ['{}{:08d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999)),
                   '{:09d}'.format(random.randint(0, 999999999))],  # 1 Alpha + 8 Numeric or 9 Numeric
            'MI': ['{}{:010d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 9999999999)),
                   '{}{:012d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999999))],
            # 1 Alpha + 10 Numeric or 1 Alpha + 12 Numeric
            'MN': ['{}{:012d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999999))],
            # 1 Alpha + 12 Numeric
            'MS': ['{:09d}'.format(random.randint(0, 999999999))],  # 9 Numeric
            'MO': ['{:03d}{}{:06d}'.format(random.randint(0, 999), random.choice(string.ascii_uppercase),
                                           random.randint(0, 999999)),
                   '{}{:05d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999)),
                   '{}{:06d}R'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999)),
                   '{:08d}{}'.format(random.randint(0, 99999999), ''.join(random.choices(string.ascii_uppercase, k=2))),
                   '{:09d}{}'.format(random.randint(0, 999999999), random.choice(string.ascii_uppercase)),
                   '{:09d}'.format(random.randint(0, 999999999))],
            # 3 Numeric + 1 Alpha + 6 Numeric or 1 Alpha + 5-9 Numeric or 1 Alpha + 6 Numeric + R or 8 Numeric + 2 Alpha or 9 Numeric + 1 Alpha or 9 Numeric
            'MT': ['{}{:08d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999)),
                   '{:09d}'.format(random.randint(0, 999999999)), '{:013d}'.format(random.randint(0, 9999999999999)),
                   '{:014d}'.format(random.randint(0, 99999999999999))],
            # 1 Alpha + 8 Numeric or 9 Numeric or 13-14 Numeric
            'NE': ['{}{:06d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999)),
                   '{}{:08d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999))],
            # 1 Alpha + 6-8 Numeric
            'NV': ['{:09d}'.format(random.randint(0, 999999999)), '{:010d}'.format(random.randint(0, 9999999999)),
                   '{:012d}'.format(random.randint(0, 999999999999)), 'X{:08d}'.format(random.randint(0, 99999999))],
            # 9-10 Numeric or 12 Numeric or X + 8 Numeric
            'NH': ['{:02d}{}{:05d}'.format(random.randint(0, 99), random.choice(string.ascii_uppercase),
                                           random.randint(0, 99999))],  # 2 Numeric + 3 Alpha + 5 Numeric
            'NJ': ['{}{:014d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999999999))],
            # 1 Alpha + 14 Numeric
            'NM': ['{:08d}'.format(random.randint(0, 99999999)), '{:09d}'.format(random.randint(0, 999999999))],
            # 8-9 Numeric
            'NY': ['{}{:07d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 9999999)),
                   '{}{:018d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999999999999)),
                   '{:08d}'.format(random.randint(0, 99999999)), '{:09d}'.format(random.randint(0, 999999999)),
                   '{:016d}'.format(random.randint(0, 9999999999999999)),
                   ''.join(random.choices(string.ascii_uppercase, k=8))],
            # 1 Alpha + 7 Numeric or 1 Alpha + 18 Numeric or 8-9 Numeric or 16 Numeric or 8 Alpha
            'NC': ['{:012d}'.format(random.randint(0, 999999999999))],  # 1-12 Numeric (for simplicity, using 12 digits)
            'ND': ['{}{:06d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999)),
                   '{:09d}'.format(random.randint(0, 999999999))],  # 3 Alpha + 6 Numeric or 9 Numeric
            'OH': ['{}{:08d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 99999999)),
                   '{}{}{:07d}'.format(random.choice(string.ascii_uppercase), random.choice(string.ascii_uppercase),
                                       random.randint(0, 9999999)), '{:08d}'.format(random.randint(0, 99999999))],
            # 1 Alpha + 4-8 Numeric or 2 Alpha + 3-7 Numeric or 8 Numeric
            'OK': ['{}{:09d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999999)),
                   '{:09d}'.format(random.randint(0, 999999999))],  # 1 Alpha + 9 Numeric or 9 Numeric
            'OR': ['{:09d}'.format(random.randint(0, 999999999))],  # 1-9 Numeric (for simplicity, using 9 digits)
            'PA': ['{:08d}'.format(random.randint(0, 99999999))],  # 8 Numeric
            'RI': ['{:07d}'.format(random.randint(0, 9999999)),
                   '{}{:06d}'.format(random.choice(string.ascii_uppercase), random.randint(0, 999999))],
            # 7 Numeric or 1 Alpha + 6 Numeric
            'SC': ['{:011d}'.format(random.randint(0, 99999999999))],  # 5-11 Numeric (for simplicity, using 11 digits)
            'SD': ['{:010d}'.format(random.randint(0, 9999999999))],  # 6-10 Numeric (for simplicity, using 10 digits)
            'TN': ['{:09d}'.format(random.randint(0, 999999999))],  # 7-9 Numeric (for simplicity, using 9 digits)
            'TX': ['{:08d}'.format(random.randint(0, 99999999)), '{:07d}'.format(random.randint(0, 9999999))],
            # 7-8 Numeric (for simplicity, using 8 digits)
            'UT': ['{:010d}'.format(random.randint(0, 9999999999))],  # 4-10 Numeric (for simplicity, using 10 digits)
            'VT': ['{:08d}'.format(random.randint(0, 99999999)), '{:07d}A'.format(random.randint(0, 9999999))]
            # 8 Numeric or 7 Numeric + A
        }

        if self.StateAbbr in license_formats:
            return random.choice(license_formats[self.StateAbbr])
        else:
            return None

    def generate_general_id(self):
        """Generate a general ID (12 alphanumeric characters)."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

    def generate_random_medical_id(self):
        """
        随机生成一个符合以下三种模式之一的美国医疗ID：
        1. Medicare ID
        2. 医院或医疗保险ID
        3. 自定义医疗ID（如American Medical ID标签）
        :return: 随机生成的医疗ID字符串
        """
        type_choice = random.choice(['medicare', 'hospital', 'custom'])

        if type_choice == 'medicare':
            # Medicare ID
            return ''.join(random.choices(string.ascii_uppercase + string.digits, k=2)) + '-' + \
                ''.join(random.choices(string.ascii_uppercase + string.digits, k=2)) + '-' + \
                ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        elif type_choice == 'hospital':
            # Hospital or Insurance ID
            prefix = random.choice(['MC', 'HP', 'HI'])  # 示例前缀，可根据实际情况扩展
            return prefix + '-' + ''.join(random.choices(string.digits, k=8))
        else:  # 'custom'
            # Custom Medical ID
            birth_year = self.DateOfBirth.year
            birth_month = self.DateOfBirth.month
            birth_day = self.DateOfBirth.day  # 简化处理，不考虑不同月份天数差异
            birthday_part = f"{birth_year:04d}{birth_month:02d}{birth_day:02d}"
            random_part = ''.join(random.choices(string.digits, k=4))
            return f"{birthday_part}-{random_part}"

    def generate_case_number(self):
        prefix = "AFRA"
        number = random.randint(100000000, 999999999)  # 生成9位随机数
        return f"{prefix}{number}"

    def generate_bank_account_number(self):
        """Generate a bank account number (12 digits)."""
        return ''.join(random.choices(string.digits, k=12))

    def summary(self):
        """Provide a summary of all attributes."""
        return {
            "Sex": self.Sex,
            "FirstName": self.FirstName,
            "LastName": self.LastName,
            "MiddleName": self.MiddleName,
            "FullName": self.FullName,
            "Initials": self.Initials,
            "EmailAddress": self.EmailAddress,
            "PhoneNumber": self.PhoneNumber,
            "FaxNumber": self.FaxNumber,
            "OfficePhoneNumber": self.OfficePhoneNumber,
            "Address": self.Address,
            "StreetNumber": self.StreetNumber,
            "StreetName": self.StreetName,
            "StreetAddress": f"{self.StreetNumber} {self.StreetName}",
            "ZipCode": self.ZipCode,
            "City": self.City,
            "City&State": f"{self.City}, {self.State}",
            "Country": self.Country,
            "State": self.State,
            "StateAbbr": self.StateAbbr,
            "PassportNumber": self.PassportNumber,
            "DriverLicense": self.DriverLicense,
            "SocialSecurityNumber": self.SocialSecurityNumber,
            # "GeneralIDs": self.GeneralIDs,
            "CreditCardNumber": self.CreditCardNumber,
            "CreditCardDetails": self.CreditCardDetails,
            "BankAccountNumber": self.BankAccountNumber,
            "Date": self.Date,
            "DateOfBirth": self.DateOfBirth,
            "IPAddress": self.IPAddress,
            "MACAddress": self.MACAddress,
            "MedID": self.MedID,
            "CaseNumber": self.generate_case_number(),
            "StateEmail": f"{self.FirstName}.{self.LastName}@{''.join(self.State.split(' ')).lower()}.gov",
            "GovID": f"{self.FirstName[0]}{self.LastName[0]}{str(random.randint(100000, 999999999))}",
            "PositionDesID": self.position_description_number,
            "PayPlan": self.pay_plan,
            "fpl": self.fpl,
            "FullName&Address": f"{self.FullName}, {self.Address}"
        }

    def summary_serializable(self):
        out = {}
        for key, value in self.summary().items():
            out[key] = str(value) if value else ""
        return out

    def summary_text(self):
        out = []
        for key, value in self.summary().items():
            out.append(f"{key}: {value}")
        return '\n'.join(out)


# Example Usage
if __name__ == "__main__":
    person = PersonGenerator()  # Generate a U.S.-based person
    print("Person Summary:")
    for key, value in person.summary().items():
        print(f"{key}: {value}")
