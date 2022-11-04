from pydantic import BaseModel
from collections.abc import Sequence
from collections.abc import Callable

# TODO: add ICE commit
class QAResult(BaseModel):
    question_short_name: str
    document_id: str
    answer: str | None
    experiment: str
    excerpts: Sequence[str]
    recipe: str
    time: str
    title: str

class ElicitQAResult(QAResult):
    elicit_commit: str
    recipe: str = "Elicit QA"