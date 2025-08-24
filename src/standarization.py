import contractions
from flashtext import KeywordProcessor
import requests
from nltk.tokenize import TweetTokenizer


# URLs for slang and acronym dictionaries
slang_abbr_url = "https://raw.githubusercontent.com/rishabhverma17/sms_slang_translator/master/slang.txt"
acronym_url = "https://raw.githubusercontent.com/prajwalkhairnar/abbreviations_py/main/abbreviations_py/textes/abbreviation_mappings.json"


class TextStandardizer:
    def __init__(self):
        self.contraction_map = contractions.fix # Contraction to full form mapping

        # Initialize keyword processor for slang replacement
        self.slang_processor = KeywordProcessor()

        # Initialize tokenizer for tweet text processing
        self.tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

        # Initialize slang and acronym dictionaries
        self.slang_dict = None
        self.abbrev_dict = None

        # Initialize dictionaries on creation
        self.load_slang_abbr()
        self.load_acronyms()

    # Load slang dictionary from GitHub url
    def load_slang_abbr(self):
        if self.slang_dict is None:
            # Get slang dictionary from GitHub url
            response = requests.get(slang_abbr_url, timeout=5)
            lines = response.text.splitlines() # Split lines into a list
            self.slang_dict = {}
            for line in lines:
                if '=' in line: # Check if line contains '='
                    k, v = line.lower().split('=', 1) # Split line into key and value
                    k = k.strip() # Strip whitespace from key
                    v = v.strip() # Strip whitespace from value
                    # Store value as a list to satisfy flashtext requirement
                    self.slang_dict[v] = [k]
            self.slang_processor.add_keywords_from_dict(self.slang_dict) # Add slang dictionary to keyword processor

    # Load acronym dictionary from GitHub url
    def load_acronyms(self):
        if self.abbrev_dict is None: # Check if acronym dictionary is not loaded
            self.abbrev_dict = requests.get(acronym_url).json() # Get acronym dictionary from GitHub url

    # Replace contractions in text
    def replace_contractions(self, text):
        return contractions.fix(text)

    # Replace slang in text
    def replace_slang(self, text):
        return self.slang_processor.replace_keywords(text)

    # Replace acronyms in text
    def replace_acronyms(self, text):
        tokens = self.tokenizer.tokenize(text) # split text into tokens
        return ' '.join([self.abbrev_dict.get(token.lower(), token) for token in tokens]) # replace acronyms with their full forms

    # Standardize text
    def standardize_text(self, text):
        # Process all replacements in a single pass
        text = self.replace_contractions(text)
        text = self.replace_slang(text)
        text = self.replace_acronyms(text)
        return text