from faker import Faker

import random
import string


class PersonGenerator:
    def __init__(self):
        # Set locale to 'en_US' for consistent U.S.-based data
        self.fake = Faker('en_US')
        # self.fake.add_provider(person)
        # self.fake.seed_instance(random.randint(1, 1000))

        # Sex - Randomly assign male or female
        self.Sex = random.choice(['Male', 'Female'])  # Randomly select Male or Female

        self.if_middle_name = random.choice([1, 1, 1, 0, 0])

        # Generate names based on Sex
        if self.Sex == 'Male':
            # self.FirstName = random.choice([self.fake.first_name_male(), self.fake.first_name_male_est()])
            self.FirstName = self.fake.first_name_male()

        else:
            # self.FirstName = random.choice([self.fake.first_name_female(), self.fake.first_name_female_est()])  # Generate female first name
            self.FirstName = self.fake.first_name_female()  # Generate female first name

        self.MiddleName = self.generate_middle_name(self.Sex) if self.if_middle_name else None

        self.LastName = self.fake.last_name()
        # self.FullName = random.choice([f"{self.FirstName} {self.LastName}", f"{self.FirstName} {self.LastName}".upper(),
        #                                f"{self.FirstName}{self.LastName}",
        #                                f"{self.FirstName} {self.MiddleName} {self.LastName}"])
        self.FullName = random.choice([f"{self.FirstName} {self.LastName}", f"{self.FirstName} {self.LastName}".upper(),
                                       f"{self.FirstName}{self.LastName}"]) if not self.if_middle_name else f"{self.FirstName} {self.MiddleName} {self.LastName}"

        self.UserName = self.fake.user_name()
        self.Initials = ".".join(
            [name[0].upper() for name in [self.FirstName, self.LastName]]) if not self.if_middle_name else ".".join(
            [name[0].upper() for name in [self.FirstName, self.MiddleName, self.LastName]])
        self.EmailAddress = self.generate_email()
        self.PhoneNumber = self.fake.phone_number()

        # Address details
        self.Country = "United States"  # Fixed to the U.S.
        self.State = self.fake.state()
        self.City = self.fake.city()
        self.ZipCode = self.fake.zipcode()
        self.StreetNumber = self.fake.building_number()
        self.StreetName = self.fake.street_name()
        self.Address = f"{self.StreetNumber} {self.StreetName}, {self.City}, {self.State}, {self.ZipCode}, {self.Country}"

        # Identity details
        self.PassportNumber = self.generate_passport_number()
        self.DriverLicense = self.generate_driver_license()
        self.SocialSecurityNumber = self.fake.ssn()
        self.GeneralIDs = self.generate_general_id()

        # Financial details
        self.CreditCardNumber = self.generate_credit_card_number()
        self.BankAccountNumber = self.generate_bank_account_number()

        # Date details
        self.Date = self.fake.date_between(start_date='-3y', end_date='today')
        self.DateOfBirth = self.fake.date_of_birth(minimum_age=18, maximum_age=70)

        # Network details

        self.IPAddress = self.fake.ipv4_private()
        self.MACAddress = self.fake.mac_address()

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

    def generate_email(self):
        """Generate an email address based on first and last name."""
        domain = random.choice([random.choice([
            "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
            "protonmail.com", "icloud.com", "mail.com", "yandex.com", "zoho.com",
            "gmx.com", "fastmail.com", "tutanota.com", "hushmail.com", "posteo.net",
            "live.com", "qq.com", "163.com", "126.com", "sina.com",
            "rediffmail.com", "lycos.com", "rocketmail.com", "bellsouth.net", "comcast.net",
            "shaw.ca", "sympatico.ca", "btinternet.com", "ntlworld.com", "sky.com",
            "verizon.net", "earthlink.net", "cox.net", "me.com", "mac.com",
            "att.net", "optonline.net", "charter.net", "frontiernet.net", "roadrunner.com",
            "windstream.net", "wowway.com", "embarqmail.com", "kabelmail.de", "t-online.de",
            "web.de", "freenet.de", "gmx.de", "bluewin.ch", "wanadoo.fr",
            "orange.fr", "laposte.net", "free.fr", "sfr.fr", "neuf.fr",
            "alice.it", "libero.it", "virgilio.it", "tin.it", "tiscali.it",
            "live.co.uk", "hotmail.co.uk", "yahoo.co.uk", "btinternet.co.uk", "virginmedia.com",
            "talktalk.net", "sky.com", "blueyonder.co.uk", "plus.net", "zoho.eu",
            "yandex.ru", "rambler.ru", "mail.ru", "bk.ru", "inbox.ru",
            "list.ru", "ya.ru", "seznam.cz", "centrum.cz", "volny.cz",
            "atlas.cz", "post.cz", "azet.sk", "zoznam.sk", "centrum.sk",
            "nate.com", "daum.net", "hanmail.net", "naver.com", "kakao.com",
            "mail.ee", "online.ee", "suomi24.fi", "elisa.fi", "kolumbus.fi",
            "tele2.se", "comhem.se", "bredband.net", "home.se", "outlook.jp",
            "icloud.cn", "21cn.com"]),
            self.fake.free_email_domain()])

        return f"{self.FirstName.lower()}.{self.LastName.lower()}@{domain}"

    def generate_passport_number(self):
        """Generate a U.S. passport number (9 digits)."""
        return ''.join(random.choices(string.digits, k=9))

    def generate_driver_license(self):
        """Generate a U.S. driver license (15 alphanumeric characters)."""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))

    def generate_general_id(self):
        """Generate a general ID (12 alphanumeric characters)."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

    def generate_credit_card_number(self):
        """Generate a credit card number (16 digits)."""
        return ''.join(random.choices(string.digits, k=16))

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
            "Address": self.Address,
            "StreetNumber": self.StreetNumber,
            "StreetName": self.StreetName,
            "ZipCode": self.ZipCode,
            "City": self.City,
            "Country": self.Country,
            "State": self.State,
            "PassportNumber": self.PassportNumber,
            "DriverLicense": self.DriverLicense,
            "SocialSecurityNumber": self.SocialSecurityNumber,
            # "GeneralIDs": self.GeneralIDs,
            "CreditCardNumber": self.CreditCardNumber,
            "BankAccountNumber": self.BankAccountNumber,
            "Date": self.Date,
            "DateOfBirth": self.DateOfBirth,
            "IPAddress": self.IPAddress,
            "MACAddress": self.MACAddress,
        }

    def summary_serializable(self):
        out = {}
        for key, value in self.summary().items():
            out[key] = str(value)
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
