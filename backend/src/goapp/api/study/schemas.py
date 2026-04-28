from pydantic import BaseModel

from ..tsumego.schemas import TsumegoStone


class Move(BaseModel):
    col: int
    row: int


class Review(BaseModel):
    verdict: str           # "correct" | "incorrect"
    reviewed_at: str


class Attempt(BaseModel):
    """Student-facing attempt with the full per-reviewer reviews map."""
    id: str
    problem_id: str
    moves: list[Move]
    submitted_at: str
    sent_to: list[str] = []   # reviewer user_ids
    sent_at: str | None = None
    reviews: dict[str, Review] = {}  # keyed by reviewer user_id
    acked_at: str | None = None


class TeacherAttempt(BaseModel):
    """Reviewer-facing attempt — only this reviewer's review is exposed."""
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


# --- linked users (teachers as seen from the student, students as seen
# from the teacher) ---


class LinkedUser(BaseModel):
    user_id: str
    display_name: str       # falls back to user_id
    email: str | None = None


class LinkedTeachersResponse(BaseModel):
    teachers: list[LinkedUser]


class LinkedStudentsResponse(BaseModel):
    students: list[LinkedUser]


# --- batch ---


class SendBatchRequest(BaseModel):
    teacher_user_id: str    # reviewer's user_id


class SendBatchResponse(BaseModel):
    sent_count: int
    teacher_user_id: str
    sent_at: str


class BatchResponse(BaseModel):
    items: list[AttemptWithProblem]


class Submission(BaseModel):
    sent_at: str
    reviewer_id: str        # reviewer user_id
    reviewer_name: str      # display name of the reviewer
    state: str              # "pending" | "returned" | "acked"
    items: list[AttemptWithProblem]


class SubmissionsResponse(BaseModel):
    submissions: list[Submission]


class AckSubmissionResponse(BaseModel):
    sent_at: str
    acked_count: int


# --- profile (per-user settings) ---


class Profile(BaseModel):
    display_name: str | None = None
    email: str | None = None
    default_role: str | None = None  # "student" | "teacher"


class UpdateProfileRequest(BaseModel):
    display_name: str | None = None
    default_role: str | None = None


class ReviewRequest(BaseModel):
    verdict: str           # "correct" | "incorrect"
