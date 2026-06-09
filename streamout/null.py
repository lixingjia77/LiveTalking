###############################################################################
#  Output — Null 输出
###############################################################################

from typing import TYPE_CHECKING, Optional

from streamout.base_output import BaseOutput
from registry import register

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


@register("streamout", "null")
class NullOutput(BaseOutput):
    """丢弃音视频帧的输出模式，供自动化 benchmark / CI 使用。"""

    def __init__(self, opt=None, parent: Optional["BaseAvatar"] = None, **kwargs):
        super().__init__(opt, parent)
        self.video_frames = 0
        self.audio_frames = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def push_video_frame(self, frame) -> None:
        self.video_frames += 1

    def push_audio_frame(self, frame, eventpoint=None) -> None:
        self.audio_frames += 1

    def stop(self) -> None:
        self.started = False
