# -*- coding: utf-8 -*-
'''
Original code from:
https://codereview.stackexchange.com/questions/149068/parse-run-length-encoded-file-for-cellular-automaton-data
'''
from pprint import pformat

class RunLengthEncodedParser:
    """
    Parser for Run Length Encode (RLE) strings / files.
    More information: http://www.conwaylife.com/w/index.php?title=Run_Length_Encoded
    """
    def __init__(self, rle_string):
        self.rle_string = rle_string
        self.name = ""
        self._comments = []    # Note underscore
        self.author = ""
        self.size_x = 0
        self.size_y = 0
        self.rule_birth = []
        self.rule_survival = []
        self.pattern_raw = ""

        # Fill in instance attributes by parsing the raw strings
        self.populate_attributes(self.rle_string.strip().splitlines())
        self.pattern_2d_array = self.populate_pattern(self.pattern_raw, self.size_x, self.size_y)

    def populate_attributes(self, lines):
        """
        This method performs all the string parsing required to parse the various 
        fields of data into their respective data members.
        """
        for line in lines:
            # Name of the pattern
            if line.startswith("#N"):
                self.name = line.lstrip("#N ")
            # Comments accompanying the pattern
            elif line.startswith("#C") or line.startswith("#c"):
                self._comments.append(line.lstrip("#Cc "))
            # Authorship of the pattern
            elif line.startswith("#O"):
                self.author = line.lstrip("#O ")
            # Grid sizes and rules
            elif line.startswith("x"):
                data = line.split(",")
                for d in data:
                    # Grid sizes
                    if d.strip().startswith("x"):
                        _, x = d.split("=")
                        self.size_x = int(x.strip())
                    elif d.strip().startswith("y"):
                        _, y = d.split("=")
                        self.size_y = int(y.strip())
                    # Rules
                    elif d.strip().startswith("rule"):
                        _, rule = d.split("=")
                        for r in rule.strip().split("/"):
                            if r.startswith("B"):
                                for digit in list(r.lstrip("B")):
                                    self.rule_birth.append(int(digit))
                            if r.startswith("S"):
                                for digit in list(r.lstrip("S")):
                                    self.rule_survival.append(int(digit))
            # Other lines should contain the actual pattern
            else:
                self.pattern_raw += line.strip(" \n\r\t")

    def populate_pattern(self, pattern_raw, size_x, size_y, default_cell='b'):
        pattern = []
        pattern_rows = pattern_raw.rstrip("!").split("$")
        #~ assert len(pattern_rows) == size_y, \
        #~ "Number of data rows {0} does not match size y = {1}".format(len(pattern_rows), size_y)
        for y in range(len(pattern_rows)):
            pattern.append([])
            tmp_num_str = ""
            for c in pattern_rows[y]:
                if self.isdigit(c):
                    tmp_num_str += c
                else:
                    if tmp_num_str == "":
                        num_cells = 1
                    else:
                        num_cells = int(tmp_num_str)
                    for n in range(num_cells):
                        pattern[y].append(c)
                    #reset count until another number is encountered
                    tmp_num_str = ""
            #fill in empty spaces at end of each row
            for _ in range(len(pattern[y]), size_x):
                pattern[y].append(default_cell)
        return pattern

    def isdigit(self, c):
        """Returns True is the character is a digit"""
        return '0' <= c <= '9'

    def __str__(self):
        return self.rle_string

    def __format__(self, fmt):
        return 'name: {self.name}\n' \
               'comments: {self._comments}\n' \
               'author: {self.author}\n' \
               'size_x: {self.size_x}\n' \
               'size_y: {self.size_y}\n' \
               'rule_birth: {self.rule_birth}\n' \
               'rule_survival: {self.rule_survival}\n' \
               'pattern_raw: {self.pattern_raw}\n' \
               'human_friendly_pattern: {self.human_friendly_pattern}\n'.format(self=self)

    @property
    def human_friendly_pattern(self):
        pattern_str = ""
        for row in self.pattern_2d_array:
            row_str = ""
            for c in row:
                if c == 'b':
                    row_str += '.'
                else:
                    row_str += c
            pattern_str += row_str + '\n'
        return pattern_str

    @property
    def comments(self):
        return pformat(self._comments)