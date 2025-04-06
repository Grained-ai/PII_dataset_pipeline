from pathlib import Path

from pandas.tseries.holiday import after_nearest_workday

from modules.llm_factory import LLMFactory
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from loguru import logger

class ScoreResult(BaseModel):
    final_score: int = Field(
        description="Final score according to the metrics"
    )
    reason: str = Field(
        description="Reason for the given score"
    )

class SampleScoring:
    def __init__(self):
        self.__llm_factory = LLMFactory()

    def unit_score_synthetic_sample(self, seed_content: str, synthetic_content: str):
        prompt_path = Path(__file__).parent.parent / 'prompts' / 'score_synthetic_sample.prompt'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        parser = PydanticOutputParser(pydantic_object=ScoreResult)
        model_instance = self.__llm_factory.create_llm_instance()

        prompt = template.format(format_instruction=parser.get_format_instructions(),
                                 seed_document_content=seed_content,
                                 synthesized_document_content=synthetic_content)
        res_content = model_instance.invoke(prompt)
        answer = parser.parse(res_content.content)
        logger.success(answer)
        return answer.model_dump()

if __name__ == "__main__":
    seed = """Duplicate Invoice
Apple Distribution
Hollyhill Industrial Estate
HollyhillCork
Ireland
VAT No:GB117223643
InvoiceNumber:6927022343
Bill to
Apple Order Number:2818048457
James Fox
Invoice Date:04.11.2018
2
Due Date:05.11.2018
Perthcelyn Cottages
Customer Number:200225
Mountain Ash
CF453YJ
GREAT BRITAIN
Ship To
James Fox
Purchase Order Number:W576106559
2
Payment Terms:Credit Card Visa/MCard
Perthcelyn Cottages
Date Ordered:02.11.2018
Mountain Ash
Delivery Note/Date:8570070728/02.11.2018
CF453Y
GREAT BRITAIN
Item
Quantit
Value Per
Extended
Number
Unit
Value
Tax
Material
ProductDescription
y
Rate
Number
Shipped
Excl.VAT
(Excl.VAT
%
Terms of Delivery/Incoterms: CP Carriage InsurancePaid To
000010
MT562B/A
IPHONE XSMAXSPACE GREY 256GB-GBR
1
1,040.85
1,040.85
20.00
Country Of OriginCN
EEE PRODUCER REGISTRATION NUMBER-WEE/CEOOSSTS
This invoice amount has been paid by Credit Card Visa/MCard
Thank you for shopping at the Apple Store.
If you have any questions regarding this invoice or the goods delivered
Please contact our Customer Service Department on
Phone:0800 048 0408 Monday-Friday 08:00-21:00
Web:www.apple.com/ukstore
MEI Numbers for Item 000010
357287090672090357287090767056
Serial Numbers far Item 000010
F2LXC26EKPH7
VAT Basis
VAT Amount
VAT Rate
For Apple Store terms and conditions and details of your after sale
1,040.85
208.15
20.00%
Warranty and support,please refer to
Total Price (Incl.VAT
G8P1,249.00
http:/store.apple.com/Catalog/uk/Images/salespolicies_consumers.html"""
    synthetic = """Duplicate Invoice
Apple Distribution
VincentRoadsEstate
SarahmouthMichigan
UnitedStates
VAT No:US366261234
InvoiceNumber:8453217690
Bill to
Apple Order Number:4321876590
Jessica Smith
Invoice Date:04.09.2023
2
Due Date:05.09.2023
StevenRouteCottages
Customer Number:225200
NewRalphAlabama
CF69518
GREAT BRITAIN
Ship To
Jessica Smith
Purchase Order Number:W659076321
2
Payment Terms:Credit Card Visa/MCard
StevenRouteCottages
Date Ordered:25.07.2024
NewRalphAlabama
Delivery Note/Date:7280078520/25.07.2024
CF69518
GREAT BRITAIN
Item
Quantit
Value Per
Extended
Number
Unit
Value
Tax
Material
ProductDescription
y
Rate
Number
Shipped
Excl.VAT
(Excl.VAT
%
Terms of Delivery/Incoterms: CP Carriage InsurancePaid To
000010
MT562B/A
IPHONE XSMAXSPACE GREY 256GB-GBR
1
1,040.85
1,040.85
20.00
Country Of OriginCN
EEE PRODUCER REGISTRATION NUMBER-WEE/CEOOSSTS
This invoice amount has been paid by Credit Card Visa/MCard
Thank you for shopping at the Apple Store.
If you have any questions regarding this invoice or the goods delivered
Please contact our Customer Service Department on
Phone:+1-312-924-6016 Monday-Friday 08:00-21:00
Web:jessica.smith@aol.com
MEI Numbers for Item 000010
357287090672090357287090767056
Serial Numbers far Item 000010
F2LXC26EKPH7
VAT Basis
VAT Amount
VAT Rate
For Apple Store terms and conditions and details of your after sale
1,040.85
208.15
20.00%
Warranty and support,please refer to
Total Price (Incl.VAT
1,249.00
http://store.apple.com/Catalog/uk/Images/salespolicies_consumers.html"""
    ins = SampleScoring()
    ins.unit_score_synthetic_sample(seed_content=seed,
                                    synthetic_content=synthetic)

