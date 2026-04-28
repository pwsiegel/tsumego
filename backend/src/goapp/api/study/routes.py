"""Solve attempts, batches, and reviewer-side review endpoints.

Two surfaces, both gated by the same IAP-derived user_id:

* `/api/study/...`         — student endpoints (the caller's own data).
* `/api/study/teacher/...` — reviewer endpoints (data of students who
                              have linked the caller as their teacher).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from ...auth import user_id_from_request
from ...links import is_teacher_of, list_teachers, students_of
from ...paths import tsumego_dir
from ...profile import display_name, email_for, load_profile, save_profile
from ...study import (
    ack_submission,
    attempts_for_problem,
    list_submissions,
    list_unsent,
    load_attempt,
    pending_for_reviewer,
    problem_statuses,
    reviewed_attempts,
    reviewed_by_reviewer,
    save_attempt,
    send_to_reviewer,
    set_review,
)
from ...tsumego import load_problem
from ..tsumego.schemas import TsumegoStone
from .schemas import (
    AckSubmissionResponse,
    Attempt,
    AttemptsBundleResponse,
    AttemptsResponse,
    AttemptWithProblem,
    BatchResponse,
    LinkedStudentsResponse,
    LinkedTeachersResponse,
    LinkedUser,
    Move,
    Profile,
    ProblemStatus,
    ProblemStatusesResponse,
    ProblemSummary,
    Review,
    ReviewRequest,
    SendBatchRequest,
    SendBatchResponse,
    Submission,
    SubmissionsResponse,
    SubmitAttemptRequest,
    TeacherAttempt,
    TeacherAttemptWithProblem,
    TeacherBundleResponse,
    UpdateProfileRequest,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["study"])

UserId = Depends(user_id_from_request)


def _attempt_schema(a: dict) -> Attempt:
    return Attempt(
        id=a["id"],
        problem_id=a["problem_id"],
        moves=[Move(col=int(m["col"]), row=int(m["row"])) for m in a.get("moves", [])],
        submitted_at=a["submitted_at"],
        sent_to=list(a.get("sent_to", [])),
        sent_at=a.get("sent_at"),
        reviews={k: Review(**v) for k, v in a.get("reviews", {}).items()},
        acked_at=a.get("acked_at"),
    )


def _teacher_attempt_schema(a: dict, reviewer_uid: str) -> TeacherAttempt:
    review = a.get("reviews", {}).get(reviewer_uid)
    return TeacherAttempt(
        id=a["id"],
        problem_id=a["problem_id"],
        moves=[Move(col=int(m["col"]), row=int(m["row"])) for m in a.get("moves", [])],
        submitted_at=a["submitted_at"],
        sent_at=a.get("sent_at"),
        review=Review(**review) if review else None,
    )


def _problem_summary(p: dict) -> ProblemSummary:
    return ProblemSummary(
        id=p["id"],
        source=p["source"],
        source_board_idx=p["source_board_idx"],
        black_to_play=p.get("black_to_play", True),
        stones=[TsumegoStone(**s) for s in p.get("stones", [])],
        has_image=bool(p.get("image")),
    )


def _bundle(user_id: str, attempts: list[dict]) -> list[AttemptWithProblem]:
    out: list[AttemptWithProblem] = []
    for a in attempts:
        p = load_problem(user_id, a["problem_id"])
        if p is None:
            continue
        out.append(AttemptWithProblem(
            attempt=_attempt_schema(a),
            problem=_problem_summary(p),
        ))
    return out


def _teacher_bundle(
    student_uid: str, reviewer_uid: str, attempts: list[dict],
) -> list[TeacherAttemptWithProblem]:
    out: list[TeacherAttemptWithProblem] = []
    for a in attempts:
        p = load_problem(student_uid, a["problem_id"])
        if p is None:
            continue
        out.append(TeacherAttemptWithProblem(
            attempt=_teacher_attempt_schema(a, reviewer_uid),
            problem=_problem_summary(p),
        ))
    return out


def _linked(user_id: str) -> LinkedUser:
    return LinkedUser(
        user_id=user_id,
        display_name=display_name(user_id),
        email=email_for(user_id),
    )


def _require_teacher_of(reviewer_uid: str, student_uid: str) -> None:
    if not is_teacher_of(reviewer_uid, student_uid):
        raise HTTPException(status_code=403, detail="not a linked teacher of this student")


# --- student endpoints: attempts ---


@router.post("/api/study/attempts", response_model=Attempt)
def submit_attempt_endpoint(
    req: SubmitAttemptRequest, user_id: str = UserId,
) -> Attempt:
    if load_problem(user_id, req.problem_id) is None:
        raise HTTPException(status_code=404, detail=req.problem_id)
    rec = save_attempt(
        user_id, req.problem_id,
        [{"col": m.col, "row": m.row} for m in req.moves],
    )
    return _attempt_schema(rec)


@router.get("/api/study/problems/{problem_id}/attempts", response_model=AttemptsResponse)
def list_problem_attempts_endpoint(
    problem_id: str, user_id: str = UserId,
) -> AttemptsResponse:
    items = attempts_for_problem(user_id, problem_id)
    return AttemptsResponse(attempts=[_attempt_schema(a) for a in items])


@router.get("/api/study/reviewed", response_model=AttemptsBundleResponse)
def list_reviewed_endpoint(user_id: str = UserId) -> AttemptsBundleResponse:
    return AttemptsBundleResponse(items=_bundle(user_id, reviewed_attempts(user_id)))


@router.get("/api/study/problem-status", response_model=ProblemStatusesResponse)
def problem_status_endpoint(user_id: str = UserId) -> ProblemStatusesResponse:
    return ProblemStatusesResponse(statuses={
        pid: ProblemStatus(**s) for pid, s in problem_statuses(user_id).items()
    })


# --- student endpoints: batch ---


@router.get("/api/study/batch", response_model=BatchResponse)
def get_batch_endpoint(user_id: str = UserId) -> BatchResponse:
    return BatchResponse(items=_bundle(user_id, list_unsent(user_id)))


@router.post("/api/study/batch/send", response_model=SendBatchResponse)
def send_batch_endpoint(
    req: SendBatchRequest, user_id: str = UserId,
) -> SendBatchResponse:
    if not req.teacher_user_id:
        raise HTTPException(status_code=400, detail="teacher_user_id required")
    try:
        sent = send_to_reviewer(user_id, req.teacher_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    sent_at = sent[0]["sent_at"] if sent else ""
    return SendBatchResponse(
        sent_count=len(sent),
        teacher_user_id=req.teacher_user_id,
        sent_at=sent_at,
    )


# --- student endpoints: submissions ---


def _submission_schema(user_id: str, sub: dict) -> Submission:
    return Submission(
        sent_at=sub["sent_at"],
        reviewer_id=sub["reviewer_id"],
        reviewer_name=display_name(sub["reviewer_id"]),
        state=sub["state"],
        items=_bundle(user_id, sub["attempts"]),
    )


@router.get("/api/study/submissions", response_model=SubmissionsResponse)
def list_submissions_endpoint(user_id: str = UserId) -> SubmissionsResponse:
    """In-flight submissions only (pending or returned). Acked
    submissions move into /reviewed (graded history)."""
    subs = [s for s in list_submissions(user_id) if s["state"] != "acked"]
    return SubmissionsResponse(
        submissions=[_submission_schema(user_id, s) for s in subs],
    )


@router.get("/api/study/submissions/{sent_at}", response_model=Submission)
def get_submission_endpoint(sent_at: str, user_id: str = UserId) -> Submission:
    for s in list_submissions(user_id):
        if s["sent_at"] == sent_at:
            return _submission_schema(user_id, s)
    raise HTTPException(status_code=404, detail=sent_at)


@router.post(
    "/api/study/submissions/{sent_at}/ack",
    response_model=AckSubmissionResponse,
)
def ack_submission_endpoint(
    sent_at: str, user_id: str = UserId,
) -> AckSubmissionResponse:
    n = ack_submission(user_id, sent_at)
    return AckSubmissionResponse(sent_at=sent_at, acked_count=n)


# --- student endpoints: linked teachers ---


@router.get("/api/study/teachers", response_model=LinkedTeachersResponse)
def list_linked_teachers_endpoint(user_id: str = UserId) -> LinkedTeachersResponse:
    return LinkedTeachersResponse(
        teachers=[_linked(uid) for uid in list_teachers(user_id)],
    )


# --- student endpoints: profile ---


@router.get("/api/study/profile", response_model=Profile)
def get_profile_endpoint(user_id: str = UserId) -> Profile:
    return Profile(**load_profile(user_id))


@router.patch("/api/study/profile", response_model=Profile)
def update_profile_endpoint(
    req: UpdateProfileRequest, user_id: str = UserId,
) -> Profile:
    fields = req.model_dump(exclude_unset=True)
    saved = save_profile(user_id, **fields)
    return Profile(**saved)


# --- reviewer endpoints (authenticated; gated by links.is_teacher_of) ---


@router.get("/api/study/teacher/students", response_model=LinkedStudentsResponse)
def list_linked_students_endpoint(user_id: str = UserId) -> LinkedStudentsResponse:
    return LinkedStudentsResponse(
        students=[_linked(uid) for uid in students_of(user_id)],
    )


@router.get(
    "/api/study/teacher/students/{student_uid}/queue",
    response_model=TeacherBundleResponse,
)
def teacher_queue_endpoint(
    student_uid: str, user_id: str = UserId,
) -> TeacherBundleResponse:
    _require_teacher_of(user_id, student_uid)
    items = pending_for_reviewer(student_uid, user_id)
    return TeacherBundleResponse(items=_teacher_bundle(student_uid, user_id, items))


@router.get(
    "/api/study/teacher/students/{student_uid}/reviewed",
    response_model=TeacherBundleResponse,
)
def teacher_reviewed_endpoint(
    student_uid: str, user_id: str = UserId,
) -> TeacherBundleResponse:
    _require_teacher_of(user_id, student_uid)
    items = reviewed_by_reviewer(student_uid, user_id)
    return TeacherBundleResponse(items=_teacher_bundle(student_uid, user_id, items))


@router.get(
    "/api/study/teacher/students/{student_uid}/attempts/{attempt_id}",
    response_model=TeacherAttemptWithProblem,
)
def teacher_attempt_endpoint(
    student_uid: str, attempt_id: str, user_id: str = UserId,
) -> TeacherAttemptWithProblem:
    _require_teacher_of(user_id, student_uid)
    a = load_attempt(student_uid, attempt_id)
    if a is None or user_id not in a.get("sent_to", []):
        raise HTTPException(status_code=404, detail=attempt_id)
    p = load_problem(student_uid, a["problem_id"])
    if p is None:
        raise HTTPException(status_code=404, detail="problem missing")
    return TeacherAttemptWithProblem(
        attempt=_teacher_attempt_schema(a, user_id),
        problem=_problem_summary(p),
    )


@router.post(
    "/api/study/teacher/students/{student_uid}/attempts/{attempt_id}/review",
    response_model=TeacherAttempt,
)
def teacher_review_endpoint(
    student_uid: str, attempt_id: str, req: ReviewRequest,
    user_id: str = UserId,
) -> TeacherAttempt:
    _require_teacher_of(user_id, student_uid)
    try:
        out = set_review(student_uid, user_id, attempt_id, req.verdict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    if out is None:
        raise HTTPException(status_code=404, detail=attempt_id)
    return _teacher_attempt_schema(out, user_id)


@router.get(
    "/api/study/teacher/students/{student_uid}/problems/{problem_id}/image.png"
)
def teacher_problem_image_endpoint(
    student_uid: str, problem_id: str, user_id: str = UserId,
) -> Response:
    _require_teacher_of(user_id, student_uid)
    meta = load_problem(student_uid, problem_id)
    if meta is None or not meta.get("image"):
        raise HTTPException(status_code=404, detail="image not found")
    path = tsumego_dir(student_uid) / meta["image"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return Response(content=path.read_bytes(), media_type="image/png")
