from pydantic import BaseModel, Field, ConfigDict

class NovelRequest(BaseModel):
    prompt: str = Field(min_length=25, max_length=10000)
    genre: str = Field(default="fantasy", min_length=2, max_length=80)
    tone: str = Field(default="immersive", min_length=2, max_length=80)
    chapters: int = Field(default=10, ge=1, le=40)
    words_per_chapter: int = Field(default=1200, ge=200, le=2500)

class ChapterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    number: int
    title: str
    content: str
    summary: str

class NovelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    genre: str
    tone: str
    chapters_target: int
    words_target: int
    status: str
    progress: int
    error: str | None

class NovelDetail(NovelOut):
    state: dict
    chapters: list[ChapterOut]

class RenameInstruction(BaseModel):
    character_id: str = Field(min_length=1, max_length=80)
    new_name: str = Field(min_length=2, max_length=80)

class ContinueRequest(BaseModel):
    plot: str = Field(min_length=25, max_length=4000)
    additional_chapters: int = Field(default=1, ge=1, le=10)
    renames: list[RenameInstruction] = Field(default_factory=list, max_length=30)
