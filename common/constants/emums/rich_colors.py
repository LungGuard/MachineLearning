from enum import StrEnum


class Color(StrEnum):
    CYAN = 'cyan'
    YELLOW = 'yellow'
    RED = 'red'
    GREEN = 'green'
    MAGENTA = 'magenta'
    BLUE = 'blue'
    WHITE = 'white'

    def decorate_text(self, msg: str, decoration: str = "") -> str:
        tag = f"{decoration} {self.value}".strip()
        return f'[{tag}]{msg}[/{tag}]'

    def __call__(self, msg: str, decoration: str = "") -> str:
        return self.decorate_text(msg, decoration)


class Decoration(StrEnum):
    BOLD = "bold"
    DIM = "dim"
    BOLD_RED = "bold red"
    BOLD_GREEN = "bold green"
    BOLD_YELLOW = "bold yellow"
    BOLD_CYAN = "bold cyan"