import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AsyncExitStack
import logging
import random
import time
from typing import Literal, TypedDict, Unpack, override

import openai
from tenacity import retry, wait_incrementing, stop_after_attempt
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm


class LabelerKwargs(TypedDict, total=False):
    max_simultaneous: int
    max_retries: int
    timeout: int
    delay_secs: float
    global_max_simultaneous: int
    executor: Literal["thread", "process"]
    on_error: Literal["raise", "ignore"]
    delay_range: float


class Labeler[T, U]:
    def __init__(
        self,
        **kwargs: Unpack[LabelerKwargs],        
    ):
        max_simultaneous = kwargs.get("max_simultaneous", 50)
        max_retries = kwargs.get("max_retries", 1)
        timeout = kwargs.get("timeout", 20)
        global_max_simultaneous = kwargs.get("global_max_simultaneous", float("inf"))
        executor = kwargs.get("executor", "thread")
        on_error = kwargs.get("on_error", "ignore")
        self.delay_secs = kwargs.get("delay_secs", 0)
        self.delay_range = kwargs.get("delay_range", 0)

        if max_retries > 1:
            # TODO: fix this not being contained in the try-except
            self.label = retry(
                wait=wait_incrementing(start=0, increment=min(1, timeout // max_retries), max=timeout),
                stop=stop_after_attempt(max_retries),
            )(self.label)

            self.alabel = retry(
                wait=wait_incrementing(start=0, increment=min(1, timeout // max_retries), max=timeout),
                stop=stop_after_attempt(max_retries),
            )(self.alabel)

        self.max_simultaneous = max_simultaneous
        self.global_max_simultaneous = global_max_simultaneous
        self._create_global_async_semaphore(global_max_simultaneous)
        self.timeout = timeout
        self.on_error = on_error

        match executor:
            case "thread":
                self.executor = ThreadPoolExecutor(max_workers=max_simultaneous)
            case "process":
                self.executor = ProcessPoolExecutor(max_workers=max_simultaneous)
            case _:
                raise ValueError(f"Invalid executor: {executor}")
    
    @classmethod
    def _create_global_async_semaphore(cls, global_max_simultaneous: int):
        cls._global_sem = None if global_max_simultaneous == float("inf") else asyncio.Semaphore(global_max_simultaneous)

    @property
    def error_value(self) -> U:
        raise NotImplementedError
    
    @property
    def exception_info(self) -> str:
        return ""
    
    def roll_delay_secs(self) -> float:
        return random.uniform(self.delay_secs - self.delay_range, self.delay_secs + self.delay_range)

    def label(self, item: T) -> U:
        try:
            if self.delay_secs > 0:
                time.sleep(self.roll_delay_secs())  # intentionally blocking

            return self._label(item)
        except Exception as e:
            logging.exception(e)
            
            if self.on_error == "raise":
                raise e

            return self.error_value

    def _label(self, item: T) -> U:
        raise NotImplementedError
    
    async def alabel(self, item: T) -> U:
        async with AsyncExitStack() as es:
            try:
                if self._global_sem is not None:
                    await es.enter_async_context(self._global_sem)

                if self.delay_secs > 0:
                    time.sleep(self.roll_delay_secs())  # intentionally blocking

                async with asyncio.timeout(self.timeout):
                    return await self._alabel(item)
            except Exception as e:
                logging.exception(e)
                
                if self.on_error == "raise":
                    raise e
                
                return self.error_value

    async def _alabel(self, item: T) -> U:
        raise NotImplementedError

    def batch_label(self, items: list[T]) -> list[U]:
        with self.executor as executor:
            return list(tqdm(executor.map(self.label, items), total=len(items)))

    async def abatch_label(self, items: list[T]) -> list[U]:
        async def _label(item: T) -> U:
            async with sem:
                return await self.alabel(item)

        sem = asyncio.Semaphore(self.max_simultaneous)
        return await tqdm_asyncio.gather(*(_label(item) for item in items), total=len(items))


class MultipleLabeler[T, U](Labeler[list[T], list[U]]):
    def __init__(self, labelers: list[Labeler[T, U]], **kwargs: Unpack[LabelerKwargs]):
        kwargs.setdefault("max_simultaneous", 7)
        kwargs.setdefault("timeout", labelers[0].timeout)
        kwargs.setdefault("on_error", labelers[0].on_error)
        super().__init__(**kwargs)
        self.labelers = labelers

    def _label(self, items: list[T]) -> list[U]:
        return [labeler.label(item) for labeler, item in zip(self.labelers, items)]

    async def _alabel(self, items: list[T]) -> list[U]:
        return await asyncio.gather(*(labeler.alabel(item) for labeler, item in zip(self.labelers, items)))

    @property
    def error_value(self) -> list[U]:
        return [labeler.error_value for labeler in self.labelers]


type Conversation = list[dict[str, str]]
type Message = dict[str, str]


class OpenAIChatCompletions(Labeler[Conversation, Message]):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        timeout: int = 20,
        async_client: openai.AsyncOpenAI | None = None,
        sync_client: openai.OpenAI | None = None,
        **kwargs,
    ):
        kwargs.setdefault("on_error", "raise")
        super().__init__(**kwargs)
        self.model = model
        self.temperature = temperature
        self.async_client = async_client or openai.AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.sync_client = sync_client or openai.OpenAI(api_key=api_key, timeout=timeout)

    @override
    async def _alabel(self, conversation: Conversation) -> Message:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        return response.choices[0].message.to_dict()

    @override
    def _label(self, conversation: Conversation) -> Message:
        return self.sync_client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            timeout=self.timeout,
        ).choices[0].message.to_dict()

    @override
    @property
    def error_value(self) -> Message:
        return dict(role="assistant", content="")


class AzureOpenAIChatCompletions(Labeler[Conversation, Message]):
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str = "2025-01-01-preview",
        timeout: int = 20,
        async_client: openai.AsyncAzureOpenAI | None = None,
        sync_client: openai.AzureOpenAI | None = None,
        **kwargs,
    ):
        kwargs.setdefault("on_error", "raise")
        super().__init__(**kwargs)
        self.model = model
        self.temperature = temperature
        self.async_client = async_client or openai.AsyncAzureOpenAI(
            api_key=api_key,
            timeout=timeout,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
        )
        self.sync_client = sync_client or openai.AzureOpenAI(
            api_key=api_key,
            timeout=timeout,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
        )

    @override
    async def _alabel(self, conversation: Conversation) -> Message:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        return response.choices[0].message.to_dict()

    @override
    def _label(self, conversation: Conversation) -> Message:
        return self.sync_client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            timeout=self.timeout,
        ).choices[0].message.to_dict()

    @override
    @property
    def error_value(self) -> Message:
        return dict(role="assistant", content="")