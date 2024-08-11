from string import Template
from collections import defaultdict
import pandas as pd


class NoteTemplate(Template):
    delimiter = '$'
    idpattern = r'[^{}]+'

    def __init__(self, template, prefixes=None, suffixes=None, defaults=None, preprocess=None, postprocess=None,
                 fns=None):
        super().__init__(template)

        if preprocess is None:
            preprocess = {}
        if postprocess is None:
            postprocess = {}
        self.preprocess = preprocess
        self.postprocess = postprocess

        def initialize_default_dict(d, default):
            if d is None:
                d = {}
            return defaultdict(lambda: default, d)

        self.prefixes = initialize_default_dict(prefixes, '')
        self.suffixes = initialize_default_dict(suffixes, '')
        self.defaults = initialize_default_dict(defaults, '')

        # Custom functions
        self.fns = [] if fns is None else fns

        # Formatters
        self.format_timestamp = lambda dt: (dt.to_pydatetime()).strftime('%B %-d, %Y')
        self.format_timedelta = lambda td: str((td.to_pytimedelta()).days)

    def generate_note(self, mapping, data_type, tune_type):
        if isinstance(mapping, pd.Series):
            mapping = mapping.to_dict()
        assert type(mapping) is dict

        # Pre-formatting
        mapping = {k: self.preprocess[k](mapping[k]) if k in self.preprocess.keys() else mapping[k] for k in
                   mapping.keys()}

        # Remove empty string or None
        mapping = {k: mapping[k] for k in mapping.keys() if (mapping[k] != "" and not pd.isna(mapping[k]))}

        # Format special datatypes
        for k in mapping.keys():
            if isinstance(mapping[k], pd.Timestamp):
                mapping[k] = self.format_timestamp(mapping[k])
            if isinstance(mapping[k], pd.Timedelta):
                mapping[k] = self.format_timedelta(mapping[k])

        # For existing groups add prefixes and suffixes
        for k in mapping.keys():
            mapping[k] = self.prefixes[k] + str(mapping[k]) + self.suffixes[k]

        # Post-formatting
        mapping = {k: self.postprocess[k](mapping[k]) if k in self.postprocess.keys() else mapping[k] for k in
                   mapping.keys()}

        # For non existing keys add defaults
        for _, _, k, _ in self.pattern.findall(self.template):
            if k not in mapping.keys():
                mapping[k] = self.defaults[k]
        print(mapping)
        text = super().substitute(mapping)

        for fn in self.fns:
            text = fn(text)

        return text
