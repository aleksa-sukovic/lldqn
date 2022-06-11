from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from models.task import Task


class KnowledgeBase:
    def __init__(self, tasks: List["Task"] = []) -> None:
        self.tasks = tasks

    def add_task(self, task: "Task") -> None:
        self.tasks.append(task)
