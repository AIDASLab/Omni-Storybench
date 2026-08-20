from pydantic import BaseModel


class SpeakerMetadata(BaseModel):
    """Metadata for a single speaker's line."""
    name: str
    line: str
    emotion: str
    speed: str
    pitch: str
    gender: str


class SpeechMetadata(BaseModel):
    """Speech metadata containing speaker information."""
    speaker: SpeakerMetadata


class NextPageOutput(BaseModel):
    """Output format for next page generation."""
    text: str
    t2i: str
    speech_metadata: SpeechMetadata
