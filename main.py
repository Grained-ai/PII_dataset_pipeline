from modules.PII_extraction import PIIExtraction
from modules.sample_generation import SampleGeneration

class PIISampleGenPipeline:
    def __init__(self):
        self.pii_extractor = PIIExtraction()
        self.sample_generator = SampleGeneration()

    def sample_to_sample(self, input, pii_category='general'):
        # Extract PII
        piis = self.pii_extractor.main(pii_category, input, 5)
        # Replace PII

        # Refine
        pass

    def template_to_sample(self):
        pass




