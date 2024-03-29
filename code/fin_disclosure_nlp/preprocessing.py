### This is adapted from https://github.com/dafrie/fin-disclosures-nlp and to be run with the dependencies referenced there

import re

class DocumentPreprocessor():
    """A custom document preprocessor to clean a document from unnecessary content

    TODO: Long description
    Check out also the Test Suite (in test.py) to see what each method should achieve

    Usage:
        The default methods can be run with:
            doc = DocumentPreprocessor(doc).process()

        To chain individual methods:
            doc = DocumentPreprocessor(doc).\
                fix_co2().\
                strip_numbers()
    """

    def __init__(self, doc):
        self.doc = doc

    def fix_co2(self):
        """Find lines that have a "2" in a single line, following a "CO" and concatenate it all together"""
        self.doc = re.sub(r'(\s[cC][oO0]\n*\n[2]\n)', ' CO2 ', self.doc)
        return self.doc

    def replace_tabs(self):
        self.doc = re.sub(r'\t', ' ', self.doc)
        self.doc = re.sub(r'\u2003', ' ', self.doc)
        return self.doc

    def strip_urls(self):
        """Remove URLs from the text"""
        self.doc = re.sub(
            r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)+[^\s]+', ' ', self.doc)
        return self.doc

    def strip_numbers(self):
        """Strip all lines that only contain numbers"""
        # self.doc = re.sub(r'(\n+[^a-zA-Z\n]+)', '', self.doc)
        # (\n)[€£$_%\-$()]*(\d+[,.\-']\d+|\d+)*[\d€£$_ %\-$()]*(?=$|\n)
        # self.doc = re.sub(r'\n+[(\d)]*[\d]+(?=(\n|$))', '', self.doc)
        # if the numbers are very long the regex gets hung up
        l = self.doc.split("\n")
        output = ""
        for x in l:
            if len(x) > 10:
                if re.findall(r"[^\d]", x):
                    output += ("\n"+x)
            else:
                output += ("\n" + re.sub(r"(^|\n)[€£$_%\-$()]*((\d+[,.\-']\d+)|\d+)+[\d€£$_%\-$()]*(?=$|\n)", '', x))
        self.doc = output
        return self.doc

    def strip_whitespaces(self):
        # Match any sequence of whitespaces > 1 and replace
        self.doc = re.sub(r'(?<= ) +', '', self.doc)
        # Match whitespaces on a new line.
        self.doc = re.sub(r'(?<=\n) +', '', self.doc)
        # Match whitespaces at end of a line --> Not necessary!
        # self.doc = re.sub(r' (?=\n)', '', self.doc)
        self.doc = re.sub(r' \n(?!(\n))', ' ', self.doc)
        return self.doc

    def strip_empty_lines(self):
        """Strip all excessive empty lines (also containing whitespaces), i.e. no more than 2 in sequence"""
        self.doc = re.sub(r'(\n+[ +\n]*\n+)', '\n\n', self.doc)
        return self.doc

    def strip_tables(self):
        """Removes lines which are most likely from tables, e.g. have a high percentage of digits to numbers"
            Goes line by line, counts words, numbers in each and the processes
        """
        result = []
        for line in self.doc.splitlines():
            #   [\s]+[^a-zA-Z\s]+")
            # numbers_reg = re.compile(r"(^|[\s]+)[^a-zA-Z\s]+(?=$|\s)")
            # Match any number that optionally starts/ends with special characters, i.e. $10,000, (30000)
            # TODO: How to handle placeholders? "-"
            numbers_reg = re.compile(
                r"(^|[\s]+)[€£$_%\-$()]*(\d+[,.\-']\d+|\d+)*[\d€£$_%\-$()]*(?=$|\s|[,.!])")
            numbers = len(numbers_reg.findall(line))
            words = len(re.findall(r"[a-zA-Z]{2,}", line))

            if words + numbers == 0:
                pass

            elif words < 5 and max(words, 1) / (max(words, 1) + numbers) < 0.5:
                result.append("")
                continue
            result.append(line)

        if len(self.doc) > 0:
            self.doc = "\n".join(result) + ("\n" if "\n" in self.doc[-1] else "")
        #else:
        #<    print(self.doc)
        return self.doc

    def strip_titles(self):
        """If the first paragraph doesn't consist of a "normal" sentence, remove it"""
        paragraphs = self.doc.split("\n\n")
        if len(paragraphs) >= 1:
            first_p = paragraphs[0]
            if len(first_p.split("\n")) < 3 and re.match(r"^[^.?!:,]+$ *", first_p):
                self.doc = "\n\n".join(paragraphs[1:])
        return self.doc

    def fix_paragraph_breaks(self):
        """Replaces paragraph breaks that are most likely wrong, i.e. not ending with a punctuation and continuuing with a lower-case letter"""
        # TODO: For now simple regex implementation. Does not work with nouns in the new line! Should be switching to a Part Of Speech approach!

        # paragraphs = self.doc.split("\n\n")
        # result = []
        # for idx, p in enumerate(paragraphs, 1):
        #     # Check if previous paragrap does end with a "suspect" ending
        #     if idx < len(paragraphs) and re.match(r"(?<=[a-z\-,])+ *", p):
        #       pass
        #  self.doc = "\n\n".join(result)

        self.doc = re.sub(
            r'(?<=[0-9a-z\-,&+`])+ *\n{2,}(?=[a-z$£€0-9\>\-*])', '\n', self.doc)
        return self.doc

    def fix_linebreaks(self):
        """Replace word that is split by linebreaks"""
        # Note the capturing group!
        self.doc = re.sub(r'-\n *(\w+[;:,.?!]* *)', r'\1\n', self.doc)
        return self.doc

    def fix_bullet_points(self):
        """Bullet points often have a pattern of introduction and then a list"""
        self.doc = re.sub(r"(?<=[:;]) *\n{2,} *", '\n', self.doc)
        return self.doc

    def keep_semantic_paragraphs(self):
        """Only keep paragraphs that actually could make sense."""
        paragraphs = self.doc.split("\n\n")
        result = []
        for p in paragraphs:
            # TODO: For now just simple cutoff
            if len(p) > 20:
                result.append(p)
            else:
                # TODO: For now simple regex matcher
                match = re.match(r"^[A-Z]+.+[?.!:]$", p, flags=re.DOTALL)
                if match:
                    result.append(p)
                # TODO: Add if keyword list is hit!
                # TODO: Use spacy to check for a simple rule, i.e. has subject, has verb
                # doc = nlp(p)
                # if len(p) > 4:
                #     result.append(p)
        self.doc = "\n\n".join(result)

    def fix_missing_paragraph_breaks(self):
        """If a paragraph is very long, check if there are missed paragraph breaks by inserting a paragraph break if a sentence ends on a line before the average line length."""
        paragraphs = self.doc.split("\n\n")
        result = []
        for p in paragraphs:
            if p.count('\n') >= 4:
                # Get the average line length of this paragraph
                lines = p.splitlines()
                avg_line_length = (sum([len(line.strip('\n'))
                                        for line in lines]) / len(lines))

                # Loop through all ended sentences and add a paragraph in between if below a threshold
                start_idx = 0
                copy_p = p
                ended_sentences = re.finditer(r"(\n)(.+\. *\n)", p)
                for match in ended_sentences:
                    line_length = match.end() - match.start()
                    if line_length < avg_line_length * 0.9:
                        result.append(copy_p[start_idx:match.end()])
                        p = copy_p[match.end():]
                        start_idx = match.end()
            result.append(p)
        self.doc = "\n\n".join(result)

    def process(self):
        if self.doc and len(self.doc) > 0:
            # self.fix_co2()
            self.replace_tabs()
            self.strip_urls()
            self.strip_whitespaces()
            self.strip_empty_lines()
            self.strip_numbers()
            self.strip_tables()
            self.strip_titles()
            self.fix_paragraph_breaks()

            # Note: Run again to cleanup
            self.strip_whitespaces()
            self.strip_empty_lines()

            self.fix_linebreaks()
            self.fix_bullet_points()
            self.fix_missing_paragraph_breaks()
            self.keep_semantic_paragraphs()
        return self.doc
