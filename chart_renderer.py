from typing import List

import numpy as np
from collections import deque
from sus import *
import utils


def _has_flick_mod(modifiers: List[NoteModifier]):
    for m in [NoteModifier.FLICK_UP, NoteModifier.FLICK_LEFT, NoteModifier.FLICK_RIGHT]:
        if m in modifiers:
            return True
    return False


def _get_type_color(note: RawNoteData):
    if note.type == RawNoteType.CLICK or note.type == RawNoteType.SLIDE_INTERMEDIATE or note.type == RawNoteType.SLIDE_START:
        return 1
    elif note.type == RawNoteType.SLIDE_END and not _has_flick_mod(note.modifiers):
        return 1
    elif note.type == RawNoteType.SLIDE_CHECKPOINT:
        return 2
    elif note.type == RawNoteType.NONE:
        return 0
    elif NoteModifier.FLICK_UP in note.modifiers:
        return 3
    elif NoteModifier.FLICK_LEFT in note.modifiers:
        return 4
    elif NoteModifier.FLICK_RIGHT in note.modifiers:
        return 5
    print(f'warn: cannot determine color of note {note}')
    return 0


class SlideNoteRenderer:
    def __init__(self, note: NoteData, frame_length: int = 10):
        if note.type != NoteType.SLIDE:
            raise ValueError('not a slide note')
        self.note = note
        self.t0 = None
        self.should_remove = False
        self.frame_length = frame_length

    def draw(self, t: float, alpha: np.ndarray, color: np.ndarray, channel: np.ndarray):
        """
        t: in frame
        """
        from math import ceil
        if self.t0 is None:
            self.t0 = t - 1
        if self.t0 > t:
            print(f'warn: t0 ({self.t0}) > t ({t})')
            return
        t_ms = t * self.frame_length
        prev_ctrl: RawNoteData = None
        prev: RawNoteData = None
        next: RawNoteData = None
        next_ctrl: RawNoteData = None
        for x in self.note.sorted_raw_notes:
            if x.action_ms <= t_ms:
                prev = x
                if not NoteModifier.CHECKPOINT_LABEL_ONLY in x.modifiers:
                    prev_ctrl = x
            elif x.action_ms > t_ms:
                if next is None:
                    next = x
                if not NoteModifier.CHECKPOINT_LABEL_ONLY in x.modifiers:
                    next_ctrl = x
                    break
        if next is None:
            alpha[prev.left_pos:prev.right_pos + 1] = 1
            color[prev.left_pos:prev.right_pos + 1] = _get_type_color(prev)
            channel[prev.left_pos:prev.right_pos + 1] |= prev.group
            self.should_remove = True
        else:
            interpole_t = utils.inverse_lerp(prev_ctrl.action_ms, next_ctrl.action_ms, t_ms)
            prev_l_coord = np.array([prev_ctrl.left_pos, 0])
            next_l_coord = np.array([next_ctrl.left_pos, 1])
            prev_r_coord = np.array([prev_ctrl.right_pos, 0])
            next_r_coord = np.array([next_ctrl.right_pos, 1])
            if NoteModifier.SLIDE_EASEIN in prev_ctrl.modifiers:
                left = utils.cubic_ease(prev_l_coord, next_l_coord, np.array([prev_ctrl.left_pos, 0.5]), next_l_coord,
                                        interpole_t)[0]
                right = utils.cubic_ease(prev_r_coord, next_r_coord, np.array([prev_ctrl.right_pos, 0.5]), next_r_coord,
                                         interpole_t)[0]
            elif NoteModifier.SLIDE_EASEOUT in prev_ctrl.modifiers:
                left = utils.cubic_ease(prev_l_coord, next_l_coord, prev_l_coord, np.array([next_ctrl.left_pos, 0.5]),
                                        interpole_t)[0]
                right = utils.cubic_ease(prev_r_coord, next_r_coord, prev_r_coord, np.array([next_ctrl.right_pos, 0.5]),
                                         interpole_t)[0]
            else:
                left = utils.linear_ease(prev_l_coord, next_l_coord, interpole_t)[0]
                right = utils.linear_ease(prev_r_coord, next_r_coord, interpole_t)[0]
            if abs(int(left) - left) > 1e-2 and ceil(left - 1) >= 0:
                alpha[ceil(left) - 1] = 1 - (left - int(left))
            if abs(int(right) - right) > 1e-2 and int(right) + 1 < 12:
                alpha[ceil(left):ceil(right)] = 1
                alpha[int(right) + 1] = right - int(right)
            else:
                alpha[ceil(left):int(right) + 1] = 1
            channel[int(left):ceil(right) + 1] |= prev.group
            # coloring
            if abs(prev.action_ms - self.t0 * self.frame_length) < self.frame_length and abs(
                    prev.action_ms - self.t0 * self.frame_length) > abs(prev.action_ms - t_ms):
                # previous note is closer to this frame than last frame
                color[int(left):ceil(right) + 1] = _get_type_color(prev)
            elif abs(prev.action_ms - t_ms) < self.frame_length and abs(next.action_ms - t_ms) < self.frame_length:
                if abs(prev.action_ms - t_ms) < abs(next.action_ms - t_ms):
                    color[int(left):ceil(right) + 1] = _get_type_color(prev)
                else:
                    color[int(left):ceil(right) + 1] = _get_type_color(next)
            elif abs(next.action_ms - t_ms) < self.frame_length and abs(
                    next.action_ms - (t_ms + self.frame_length)) > abs(next.action_ms - t_ms):
                color[int(left):ceil(right) + 1] = _get_type_color(next)
            else:
                color[int(left):ceil(right) + 1] = 1
        self.t0 = t


def draw(notes: List[NoteData], front_pad=0, frame_length=10, threshold_alpha=True):
    """
    draw a chart into ndarray.
    ndarray returned should have shape (frames, width).
    width is always fixed to be 12.
    note that a good chart should only have 2 groups at the same time, thus we use or operator in group/channel to merge some overlap notes
    variable:
    notes: a list of note
    front_pad: how many seconds shall be padded before chart's zero. We take 10 ms as 1 frame in default so front_pad=1 means padding 100 frames
    frame_length: how long a frame is (in ms). default: 10
    """
    alpha_list = [np.zeros(12)] * int(front_pad * 1000 / frame_length)
    color_list = [np.zeros(12)] * int(front_pad * 1000 / frame_length)
    channel_list = [np.zeros(12)] * int(front_pad * 1000 / frame_length)
    unread_notes = deque(notes)
    progressing_notes: List[SlideNoteRenderer] = []
    frame = 0
    while len(unread_notes) > 0 or len(progressing_notes) > 0:
        alpha = np.zeros(12)
        color = np.zeros(12)
        channel = np.zeros(12, dtype=np.int32)
        t = (frame + 0.5) * frame_length  # add half frame such that we can take care of notes that are little ahead
        # from frame time
        for renderer in progressing_notes:
            renderer.draw(frame, alpha, color, channel)
        while unread_notes:
            if unread_notes[0].start_time < t:
                note = unread_notes.popleft()
                first_raw = note.sorted_raw_notes[0]
                alpha[first_raw.left_pos: first_raw.right_pos + 1] += 1
                color[first_raw.left_pos: first_raw.right_pos + 1] = _get_type_color(first_raw)
                channel[first_raw.left_pos: first_raw.right_pos + 1] |= first_raw.group
                if note.type == NoteType.SLIDE:
                    progressing_notes.append(SlideNoteRenderer(note, frame_length))
            else:
                break
        if threshold_alpha:
            alpha = (alpha >= 0.6).astype(int)  # threshold at 0.6
        else:
            alpha = np.clip(alpha, None, 1)
        alpha_list.append(alpha)
        color_list.append(color)
        channel_list.append(channel)
        progressing_notes = list(filter(lambda x: not x.should_remove, progressing_notes))
        frame += 1
    return np.stack(alpha_list), np.stack(channel_list), np.stack(color_list)
