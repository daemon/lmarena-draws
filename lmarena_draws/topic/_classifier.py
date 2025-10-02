import json
from typing import Unpack, override

from ._labeler import Labeler, LabelerKwargs, OpenAIChatCompletions


class TopicClassifier(Labeler[tuple[str, list[str]], dict[str, float]]):
    def __init__(
        self,
        chat_completions: OpenAIChatCompletions,
        **kwargs: Unpack[LabelerKwargs],
    ):
        super().__init__(**kwargs)
        self.chat_completions = chat_completions
    
    def _make_prompt(self, item: tuple[str, list[str]]) -> str:
        prompt = f"""Given a list of topics and a conversation, output a probability distribution over the topics indicating the relevance of each topic to the prompt.

        Topics: {json.dumps(item[1])}

        Prompt: {item[0]}

        Do not explain; just output the probability distribution over the topics in the format:

        {{
            "topic1": 0.1,
            "topic2": 0.2,
            "topic3": 0.3,
            ...
        }}"""

        return prompt

    @override
    @property
    def error_value(self) -> dict[str, float]:
        return {}

    @override
    def _label(self, item: tuple[str, list[str]]) -> dict[str, float]:
        prompt = self._make_prompt(item)
        return json.loads(self.chat_completions.label([dict(role="user", content=prompt)])["content"])

    @override
    async def _alabel(self, item: tuple[str, list[str]]) -> dict[str, float]:
        prompt = self._make_prompt(item)
        return json.loads((await self.chat_completions.alabel([dict(role="user", content=prompt)]))["content"])


class TopicExtractor(Labeler[tuple[str, list[str]], list[str]]):
    def __init__(
        self,
        chat_completions: OpenAIChatCompletions,
        **kwargs: Unpack[LabelerKwargs],
    ):
        super().__init__(**kwargs)
        self.chat_completions = chat_completions
    
    def _make_prompt(self, item: tuple[str, list[str]]) -> str:
        prompt = f"""Given a list of known topics and a conversation, output a list of very high-level topics that are relevant to the conversation, adding new topics only if needed.

        Known topics: {json.dumps(item[1])}

        Conversation:
        {item[0]}

        Do not explain; just output the list of topics in the format:

        ["topic1", "topic2", "topic3", ...]
        """

        return prompt
    
    @override
    @property
    def error_value(self) -> list[str]:
        return []

    @override
    def _label(self, item: tuple[str, list[str]]) -> list[str]:
        prompt = self._make_prompt(item)
        return json.loads(self.chat_completions.label([dict(role="user", content=prompt)])["content"])

    @override
    async def _alabel(self, item: tuple[str, list[str]]) -> list[str]:
        prompt = self._make_prompt(item)
        return json.loads((await self.chat_completions.alabel([dict(role="user", content=prompt)]))["content"])


class DifficultyExtractor(Labeler[tuple[str, str, str], float]):
    def __init__(
        self,
        chat_completions: OpenAIChatCompletions,
        **kwargs: Unpack[LabelerKwargs],
    ):
        super().__init__(**kwargs)
        self.chat_completions = chat_completions
    
    def _make_prompt(self, item: tuple[str, str, str]) -> str:
        prompt = f"""Given a prompt and two responses from language models, tell me the difficulty of the prompt.

        Prompt: {item[0]}
        Response A: {item[1]}
        Response B: {item[2]}

        Output your reasoning in fewer than 20 words, then return a new line and a number between 0 and 5, where 0 is the easiest and 5 is the hardest"""

        return prompt
    
    @override
    @property
    def error_value(self) -> float:
        return -1

    @override
    def _label(self, item: tuple[str, str, str]) -> float:
        prompt = self._make_prompt(item)
        out = self.chat_completions.label([dict(role="user", content=prompt)])["content"].split()[-1]
        return float(out)

    @override
    async def _alabel(self, item: tuple[str, str, str]) -> float:
        prompt = self._make_prompt(item)
        out = (await self.chat_completions.alabel([dict(role="user", content=prompt)]))["content"].split()[-1]
        return float(out)
    

class SubjectivityExtractor(Labeler[tuple[str, str, str], float]):
    def __init__(
        self,
        chat_completions: OpenAIChatCompletions,
        **kwargs: Unpack[LabelerKwargs],
    ):
        super().__init__(**kwargs)
        self.chat_completions = chat_completions
    
    def _make_prompt(self, item: tuple[str, str, str]) -> str:
        prompt = f"""Given a prompt and two responses from language models, tell me the subjectivity of the prompt.

        Prompt: {item[0]}
        Response A: {item[1]}
        Response B: {item[2]}

        Output your reasoning in fewer than 20 words, then return a new line and a number between 0 and 5, where 0 is the most objective and 5 is the most subjective"""

        return prompt
    
    @override
    @property
    def error_value(self) -> float:
        return -1

    @override
    def _label(self, item: tuple[str, str, str]) -> float:
        prompt = self._make_prompt(item)
        out = self.chat_completions.label([dict(role="user", content=prompt)])["content"].split()[-1]
        return float(out)

    @override
    async def _alabel(self, item: tuple[str, str, str]) -> float:
        prompt = self._make_prompt(item)
        out = (await self.chat_completions.alabel([dict(role="user", content=prompt)]))["content"].split()[-1]
        return float(out)


class WinReasonExtractor(Labeler[tuple[str, str, str, str], str]):
    def __init__(
        self,
        chat_completions: OpenAIChatCompletions,
        **kwargs: Unpack[LabelerKwargs],
    ):
        super().__init__(**kwargs)
        self.chat_completions = chat_completions
    
    def _make_prompt(self, item: tuple[str, str, str, str]) -> str:
        prompt = f"""Given a prompt for a language model, two responses from two language models, and the winner (or draw), tell me why the user chose that response.

        Prompt: {item[0]}
        Response A: {item[1]}
        Response B: {item[2]}
        Outcome: {item[3]}

        Return no more than 10 words.
        """

        return prompt
    
    @override
    @property
    def error_value(self) -> str:
        return ""

    @override
    def _label(self, item: tuple[str, str, str, str]) -> str:
        prompt = self._make_prompt(item)
        return self.chat_completions.label([dict(role="user", content=prompt)])["content"]

    @override
    async def _alabel(self, item: tuple[str, str, str, str]) -> str:
        prompt = self._make_prompt(item)
        return (await self.chat_completions.alabel([dict(role="user", content=prompt)]))["content"]