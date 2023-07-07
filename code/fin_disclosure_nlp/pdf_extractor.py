### This is adapted from https://github.com/dafrie/fin-disclosures-nlp and to be run with the dependencies referenced there

from tika import parser
from bs4 import BeautifulSoup
import os
from io import StringIO
import yaml
import logging
from pathlib import Path

import tika
tika.initVM()

class PdfExtractor:
    def __init__(self, input_file, output_folder=None, parser="tika", ocr_strategy="no_ocr", **kwargs):
        self.logger = logging.getLogger('pdf_extractor')
        self.input_file = input_file
        self.output_folder = output_folder
        self.parser = parser
        self.ocr_strategy = ocr_strategy
        self.process_document()

    def get_pages_text(self):
        """ Returns the extracted text of the document (if found)"""
        return self.pages_text

    def extract_with_tika(self):
        """
        Note that pytika can be additionally configured via environment variables in the docker-compose file!
        """
        pages_text = []
        # Read PDF file and export to XML to keep page information
        data = parser.from_file(
            str(self.input_file),
            xmlContent=True,
            requestOptions={'timeout': 30},
            # 'X-Tika-PDFextractInlineImages' : true # Unfortunately does not really work
            # Options: 'no_ocr', 'ocr_only' and 'ocr_and_text'
            headers={'X-Tika-PDFOcrStrategy': self.ocr_strategy}
        )
        xhtml_data = BeautifulSoup(data['content'], features="lxml")

        pages = xhtml_data.find_all('div', attrs={'class': 'page'})
        for i, content in enumerate(pages):
            _buffer = StringIO()
            _buffer.write(str(content))
            parsed_content = parser.from_buffer(_buffer.getvalue())
            text = ''
            if parsed_content['content']:
                text = parsed_content['content'].strip()
            pages_text.append({"page_no": i+1, "text": text})
        return pages_text

    def write_output(self):
        output = {
            "pages": self.pages_text
        }
        filename = Path(self.input_file).stem
        out_file_path = os.path.join(self.output_folder, filename)
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        with open(out_file_path + '.yml', 'w') as fp:
            yaml.dump(output, fp)

    def process_document(self):
        self.pages_text = self.extract_with_tika()
        self.write_output()