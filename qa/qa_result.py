from pydantic import BaseModel
from collections.abc import Sequence
from collections.abc import Callable

# TODO: add Elicit commit, ICE commit, other metadata
class QAResult(BaseModel):
    question_short_name: str
    document_id: str
    answer: str | None
    experiment: str
    excerpts: Sequence[str]
    recipe: str
    time: str

class ElicitQAResult(QAResult):
    elicit_commit: str