from pydantic import BaseModel

from ..tsumego.schemas import TsumegoStone


class Move(BaseModel):
    col: int
    row: int


class Review(BaseModel):
    verdict: str           # "correct" | "incorrect"
    reviewed_at: str


class Attempt(BaseModel):
    """Student-facing attempt with the full per-teacher reviews map."""
    id: str
    problem_id: str
    moves: list[Move]
    submitted_at: str
    sent_to: list[str] = []
    sent_at: str | None = None
    reviews: dict[str, Review] = {}
    acked_at: str | None = None


class TeacherAttempt(BaseModel):
    """Teacher-facing attempt — only this teacher's review is exposed."""
    id: str
    problem_id: str
    moves: list[Move]
    submitted_at: str
    sent_at: str | None = None
    review: Review | None = None


class SubmitAttemptRequest(BaseModel):
    problem_id: str
    moves: list[Move]


class AttemptsResponse(BaseModel):
    attempts: list[Attempt]


class ProblemSummary(BaseModel):
    """Just enough to render the position alongside an attempt."""
    id: str
    source: str
    source_board_idx: int
    black_to_play: bool
    stones: list[TsumegoStone]
    has_image: bool


class AttemptWithProblem(BaseModel):
    attempt: Attempt
    problem: ProblemSummary


class TeacherAttemptWithProblem(BaseModel):
    attempt: TeacherAttempt
    problem: ProblemSummary


class AttemptsBundleResponse(BaseModel):
    items: list[AttemptWithProblem]


class ProblemStatus(BaseModel):
    last_verdict: str | None = None  # "correct" | "incorrect" | None


class ProblemStatusesResponse(BaseModel):
    statuses: dict[str, ProblemStatus]


class TeacherBundleResponse(BaseModel):
    items: list[TeacherAttemptWithProblem]


# --- teachers ---


class Teacher(BaseModel):
    id: str
    label: str
    created_at: str


class TeacherWithUrl(Teacher):
    token: str
    url: str               # frontend route, e.g. "/teacher/<token>"


class TeachersResponse(BaseModel):
    teachers: list[TeacherWithUrl]


class CreateTeacherRequest(BaseModel):
    label: str


class UpdateTeacherRequest(BaseModel):
    label: str


# --- batch ---


class SendBatchRequest(BaseModel):
    teacher_id: str


class SendBatchResponse(BaseModel):
    sent_count: int
    teacher_id: str
    sent_at: str


class BatchResponse(BaseModel):
    items: list[AttemptWithProblem]


class Submission(BaseModel):
    sent_at: str
    teacher_id: str
    state: str             # "pending" | "returned" | "acked"
    items: list[AttemptWithProblem]


class SubmissionsResponse(BaseModel):
    submissions: list[Submission]


class AckSubmissionResponse(BaseModel):
    sent_at: str
    acked_count: int


# --- teacher-side identity (so the UI knows who it's grading as) ---


class TeacherMe(BaseModel):
    id: str
    label: str
    student: str           # who's asking the teacher to review (user_id)


class ReviewRequest(BaseModel):
    verdict: str           # "correct" | "incorrect"
