from enum import Enum, auto, IntEnum


class RawNoteType(Enum):
    NONE = auto()
    CLICK = auto()
    FLICK = auto()
    SLIDE_START = auto()
    SLIDE_END = auto()
    SLIDE_INTERMEDIATE = auto()
    SLIDE_CHECKPOINT = auto()

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class NoteModifier(Enum):
    NONE = auto()
    SLIDE_EASEIN = auto()
    SLIDE_EASEOUT = auto()
    CHECKPOINT_LABEL_ONLY = auto()
    FLICK_LEFT = auto()
    FLICK_UP = auto()  # specially added for flick after slide support
    FLICK_RIGHT = auto()


class RawNoteData:
    def __init__(self, type: RawNoteType, t_pos: float, group_id: int, x_pos: int, width: int,
                 *modifiers: NoteModifier):
        self.type = type
        self.action_ms = t_pos
        self.group = group_id
        self.left_pos = x_pos
        self.width = width
        self.modifiers = modifiers

    def __str__(self):
        return f'G{self.group} {self.type} Note Event at {self.action_ms} ms, {self.left_pos} with width {self.width}, mod {self.modifiers}'

    def __repr__(self) -> str:
        return f'RawNoteData({self.type}, {self.action_ms} ms, {self.group}, {self.left_pos}, {self.width}, {self.modifiers})'

    @property
    def right_pos(self):
        return self.left_pos + self.width - 1

    def is_slide(self):
        return self.is_slide_type(self.type)

    @staticmethod
    def is_slide_type(type):
        return type in [RawNoteType.SLIDE_START, RawNoteType.SLIDE_END, RawNoteType.SLIDE_INTERMEDIATE,
                        RawNoteType.SLIDE_CHECKPOINT]

    @staticmethod
    def is_accepted_type_match(first: RawNoteType, second: RawNoteType) -> bool:
        if first == RawNoteType.CLICK:
            return second == RawNoteType.CLICK
        if first == RawNoteType.FLICK:
            return second == RawNoteType.FLICK
        return second != RawNoteType.CLICK and second != RawNoteType.FLICK
