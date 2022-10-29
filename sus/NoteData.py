from sus.RawNoteData import *
from enum import Enum, auto
from typing import List


class NoteType(Enum):
    CLICK = auto()
    FLICK = auto()
    SLIDE = auto()


class NoteData:
    def __init__(self, *data: RawNoteData):
        if len(data) == 0:
            raise ValueError('empty note')
        first = data[0]
        if not all(map(lambda d: first.group == d.group and RawNoteData.is_accepted_type_match(d.type, first.type),
                       data[1:])):
            raise ValueError('mismatch group or type in note data')
        if first.type == RawNoteType.CLICK:
            self.type = NoteType.CLICK
        elif first.type == RawNoteType.FLICK:
            self.type = NoteType.FLICK
        else:
            self.type = NoteType.SLIDE
        self.raw_notes = sorted(data, key=lambda d: d.action_ms)
        if self.type == NoteType.SLIDE:
            if self.raw_notes[0].type != RawNoteType.SLIDE_START:
                raise ValueError('first slide note is not starting note')
            if self.raw_notes[-1].type != RawNoteType.SLIDE_END:
                raise ValueError('last slide note is not ending note')
            if any(map(lambda d: d.type in [RawNoteType.SLIDE_START, RawNoteType.SLIDE_END], self.raw_notes[1:-1])):
                raise ValueError('inner slide note containing start or end note')
        elif len(self.raw_notes) > 1:
            raise ValueError('too many notes for click/flick notes')

    def __str__(self):
        if self.type == NoteType.SLIDE:
            return f'{self.type} Note from {self.start_time} ms to {self.end_time} ms, {len(self.raw_notes)} note events'
        else:
            return f'{self.type} Note at {self.start_time} ms'

    def __repr__(self):
        if self.type == NoteType.SLIDE:
            return f'Note({self.type},{self.start_time},{self.end_time},count={len(self.raw_notes)})'
        else:
            return f'Note({self.type},{self.start_time})'

    @property
    def sorted_raw_notes(self) -> List[RawNoteData]:
        return sorted(self.raw_notes, key=lambda d: d.action_ms)

    @property
    def start_time(self) -> float:
        return self.sorted_raw_notes[0].action_ms

    @property
    def end_time(self) -> float:
        return self.sorted_raw_notes[-1].action_ms


def bake_note_data(raw_notes: List[RawNoteData]) -> List[NoteData]:
    baked = []
    slide_dict = {}
    time_info = {
        'curr': None,
        'groups': []
    }
    datum: RawNoteData
    for datum in sorted(sorted(raw_notes, key=lambda d: d.type, reverse=True), key=lambda d: d.action_ms):
        if datum.action_ms != time_info['curr']:
            time_info['groups'].clear()
            time_info['curr'] = datum.action_ms
        if not datum.is_slide():
            while datum.group in slide_dict:
                # shift group
                datum.group += 1
            while datum.group in time_info['groups']:
                datum.group += 1
            baked.append(NoteData(datum))
            time_info['groups'].append(datum.group)
        else:
            # good chart has already set up groups for slide notes, so we don't need to reallocate for them
            if datum.type == RawNoteType.SLIDE_START:
                if datum.group in slide_dict:
                    print(f'warn: duplicate slide start note for group {datum.group} at {datum.action_ms} ms')
                    continue
                slide_dict[datum.group] = [datum]
            else:
                if datum.group not in slide_dict:
                    print(f'warn: stray non-start slide note for group {datum.group} at {datum.action_ms} ms')
                    continue
                slide_dict[datum.group].append(datum)
                if datum.type == RawNoteType.SLIDE_END:
                    r1 = slide_dict[datum.group]
                    baked.append(NoteData(*r1))
                    slide_dict.pop(datum.group)
                    time_info['groups'].append(datum.group)  # since we popped up the group in slide_dict, add to time_info
    baked = sorted(baked, key=lambda d: d.start_time)
    return baked
