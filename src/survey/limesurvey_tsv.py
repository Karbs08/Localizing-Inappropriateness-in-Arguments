from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_COLUMNS = [
    "id",
    "related_id",
    "class",
    "type/scale",
    "name",
    "relevance",
    "text",
    "help",
    "language",
    "validation",
    "mandatory",
    "other",
    "default",
    "same_default",
]

# Advanced question attributes can be included as additional columns.
EXTRA_COLUMNS = [
    "always_hide",
    "hide_tip",
    "answer_width",
    "random_group",
    "em_validation_q",
    "em_validation_q_tip",
]

ALL_COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS


@dataclass
class LimeSurveyTSVBuilder:
    language: str = "en"
    rows: list[dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1
    _current_group_id: int | None = None

    def _new_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def _row(self, **kwargs: Any) -> dict[str, Any]:
        row = {column: "" for column in ALL_COLUMNS}
        row.update({key: value for key, value in kwargs.items() if value is not None})
        self.rows.append(row)
        return row

    def survey_parameter(self, name: str, value: Any) -> None:
        self._row(**{"class": "S", "name": name, "text": value})

    def language_parameter(self, name: str, value: Any) -> None:
        self._row(
            **{
                "class": "SL",
                "name": name,
                "text": value,
                "language": self.language,
            }
        )

    def group(self, name: str, description: str = "", relevance: str = "1") -> int:
        group_id = self._new_id()
        self._current_group_id = group_id
        self._row(
            **{
                "id": group_id,
                "class": "G",
                "name": name,
                "relevance": relevance,
                "text": description,
                "language": self.language,
            }
        )
        return group_id

    def question(
        self,
        qtype: str,
        name: str,
        text: str,
        *,
        relevance: str = "1",
        help_text: str = "",
        mandatory: bool = False,
        default: str = "",
        validation: str = "",
        always_hide: bool = False,
        hide_tip: bool = True,
        em_validation_q: str = "",
        em_validation_q_tip: str = "",
    ) -> int:
        if self._current_group_id is None:
            raise RuntimeError("Add a group before adding questions.")
        question_id = self._new_id()
        self._row(
            **{
                "id": question_id,
                "class": "Q",
                "type/scale": qtype,
                "name": name,
                "relevance": relevance,
                "text": text,
                "help": help_text,
                "language": self.language,
                "validation": validation,
                "mandatory": "Y" if mandatory else "N",
                "default": default,
                "always_hide": "1" if always_hide else "0",
                "hide_tip": "1" if hide_tip else "0",
                "em_validation_q": em_validation_q,
                "em_validation_q_tip": em_validation_q_tip,
            }
        )
        return question_id

    def subquestion(
        self,
        parent_question_id: int,
        code: str,
        text: str,
        *,
        scale: int = 0,
    ) -> int:
        subquestion_id = self._new_id()
        self._row(
            **{
                "id": subquestion_id,
                "class": "SQ",
                "type/scale": scale,
                "name": code,
                "relevance": "1",
                "text": text,
                "language": self.language,
            }
        )
        return subquestion_id

    def answer(
        self,
        parent_question_id: int,
        code: str,
        text: str,
        *,
        scale: int = 0,
        assessment_value: str | int = "",
    ) -> None:
        self._row(
            **{
                "id": parent_question_id,
                "class": "A",
                "type/scale": scale,
                "name": code,
                "relevance": assessment_value,
                "text": text,
                "language": self.language,
            }
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # LimeSurvey's own exports use UTF-8 with BOM.
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=ALL_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
