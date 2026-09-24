"""The transport contract is deliberately narrower than a chat/generation API."""
from __future__ import annotations

from typing import Annotated, Literal
import string

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from . import MODEL_ID


def supported_text(value):
    if any(character not in string.ascii_letters + string.punctuation + " \t\r\n" for character in value):
        raise ValueError("Model text supports ASCII letters, whitespace and punctuation only; numeric and non-ASCII content would be lost by this tokenizer")
    return value


Name = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128)]
QuestionText = Annotated[str, StringConstraints(min_length=1, max_length=512), AfterValidator(supported_text)]
CandidateText = Annotated[str, StringConstraints(min_length=1, max_length=256), AfterValidator(supported_text)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Base64Source(StrictModel):
    type: Literal["base64"]
    media_type: Literal["image/png", "image/jpeg", "audio/wav"]
    data: str = Field(min_length=1, max_length=2_800_000)


class DataURLSource(StrictModel):
    type: Literal["data_url"]
    url: str = Field(min_length=1, max_length=2_800_100)


MediaSource = Annotated[Base64Source | DataURLSource, Field(discriminator="type")]


class ImageBlock(StrictModel):
    type: Literal["image"]
    source: MediaSource


class AudioBlock(StrictModel):
    type: Literal["audio"]
    source: MediaSource


class TextBlock(StrictModel):
    type: Literal["text"]
    id: Name
    text: QuestionText


ContentBlock = Annotated[ImageBlock | AudioBlock | TextBlock, Field(discriminator="type")]


class Candidate(StrictModel):
    id: Identifier
    text: CandidateText


class Level(Candidate):
    value: float = Field(ge=-1_000_000_000, le=1_000_000_000)


class Question(StrictModel):
    question: QuestionText | None = None
    question_ref: Name | None = None

    @model_validator(mode="after")
    def exactly_one_question(self):
        if (self.question is None) == (self.question_ref is None):
            raise ValueError("Supply exactly one of question or question_ref")
        return self


class ChoiceQuestion(Question):
    type: Literal["choice"]
    choices: list[Candidate] = Field(min_length=2, max_length=32)


class RankingQuestion(Question):
    type: Literal["ranking"]
    choices: list[Candidate] = Field(min_length=2, max_length=32)


class NoulQuestion(Question):
    type: Literal["noul"]


class ScoreQuestion(Question):
    type: Literal["score"]
    levels: list[Level] = Field(min_length=2, max_length=32)

    @model_validator(mode="after")
    def distinct_values(self):
        if len({level.value for level in self.levels}) != len(self.levels):
            raise ValueError("Score rubric values must be distinct")
        return self


TypedQuestion = Annotated[ChoiceQuestion | RankingQuestion | NoulQuestion | ScoreQuestion,
                          Field(discriminator="type")]


class DecisionRequest(StrictModel):
    model: str = Field(default=MODEL_ID, min_length=1, max_length=128)
    input: list[ContentBlock] = Field(min_length=2, max_length=34)
    questions: dict[Name, TypedQuestion] = Field(min_length=1, max_length=32)
    abstain_threshold: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_content(self):
        for kind in ("image", "audio"):
            if sum(block.type == kind for block in self.input) != 1:
                raise ValueError(f"This model requires exactly one {kind} block")
        text_blocks = [block for block in self.input if isinstance(block, TextBlock)]
        ids = [block.id for block in text_blocks]
        if len(set(ids)) != len(ids):
            raise ValueError("Text block IDs must be unique")
        references = {q.question_ref for q in self.questions.values() if q.question_ref is not None}
        if references != set(ids):
            raise ValueError("Every question_ref must resolve and every text block must be referenced; free context is unsupported")
        return self

    def neural_requests(self):
        texts = {block.id: block.text for block in self.input if isinstance(block, TextBlock)}
        requests = []
        for q in self.questions.values():
            if isinstance(q, NoulQuestion):
                candidates = [{"id": "true", "text": "yes"}, {"id": "false", "text": "no"}]
            else:
                source = q.levels if isinstance(q, ScoreQuestion) else q.choices
                candidates = [{"id": candidate.id, "text": candidate.text} for candidate in source]
            requests.append({"question": q.question if q.question is not None else texts[q.question_ref],
                             "candidates": candidates})
        return requests


class RankedCandidate(StrictModel):
    id: str
    rank: int
    probability: float


class CategoricalResult(StrictModel):
    probabilities: dict[str, float]
    ranking: list[RankedCandidate]
    prediction: str | None
    decision: str | None
    abstained: bool
    ties: list[str]
    confidence: float


class ChoiceResult(CategoricalResult):
    type: Literal["choice"] = "choice"


class RankingResult(CategoricalResult):
    type: Literal["ranking"] = "ranking"


class NoulResult(CategoricalResult):
    type: Literal["noul"] = "noul"
    probability_true: float
    value: bool | None


class ScoreResult(CategoricalResult):
    type: Literal["score"] = "score"
    expected_value: float
    value: float | None
    range: list[float]


DecisionResult = Annotated[ChoiceResult | RankingResult | NoulResult | ScoreResult,
                           Field(discriminator="type")]


class DecisionResponse(StrictModel):
    id: str
    object: Literal["decision.response"] = "decision.response"
    created: int
    model: str
    checkpoint_sha256: str
    results: dict[str, DecisionResult]
    input_summary: dict
    semantics: dict
