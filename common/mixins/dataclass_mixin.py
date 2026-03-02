from dataclasses import asdict


class DataClassMixin:
    def to_dict(self): return asdict(self)