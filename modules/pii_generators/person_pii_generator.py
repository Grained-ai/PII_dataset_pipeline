from faker import Faker
import random
import string


class PersonGenerator:
    def __init__(self):
        # Set locale to 'en_US' for consistent U.S.-based data
        self.fake = Faker('en_US')
        self.fake.seed_instance(random.randint(1, 1000))

        # Sex - Randomly assign male or female
        self.Sex = random.choice(['Male', 'Female'])  # Randomly select Male or Female

        # Generate names based on Sex
        if self.Sex == 'Male':
            self.FirstName = self.fake.first_name_male()
        else:
            self.FirstName = self.fake.first_name_female()  # Generate female first name

        self.LastName = self.fake.last_name()
        self.FullName = f"{self.FirstName} {self.LastName}"
        self.Initials = "".join([name[0].upper() for name in self.FullName.split()])
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
        self.Date = self.fake.date_between(start_date='-30y', end_date='today')
        self.DateOfBirth = self.fake.date_of_birth(minimum_age=18, maximum_age=80)

        # Network details
        self.IPAddress = self.fake.ipv4()
        self.MACAddress = self.fake.mac_address()

    def generate_email(self):
        """Generate an email address based on first and last name."""
        domain = self.fake.free_email_domain()
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
            "GeneralIDs": self.GeneralIDs,
            "CreditCardNumber": self.CreditCardNumber,
            "BankAccountNumber": self.BankAccountNumber,
            "Date": self.Date,
            "DateOfBirth": self.DateOfBirth,
            "IPaddress": self.IPAddress,
            "MACAddress": self.MACAddress,
        }


# Example Usage
if __name__ == "__main__":
    person = PersonGenerator()  # Generate a U.S.-based person
    print("Person Summary:")
    for key, value in person.summary().items():
        print(f"{key}: {value}")
