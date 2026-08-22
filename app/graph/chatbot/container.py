"""Dependency wiring for the hierarchical chatbot graph."""

from __future__ import annotations

from app.models.chat_model import ChatModel
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_scraper_repository import (
    JOB_SCRAPER_REPOSITORY,
    JobScraperRepository,
)
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.repositories.chatbot.subgraph_repository import (
    SUBGRAPH_REPOSITORY,
    SubgraphRepository,
)
from app.services.chatbot.cv_workflow_service import CvWorkflowService
from app.services.chatbot.event_state_service import EventStateService
from app.services.chatbot.hierarchical_planning_service import (
    HierarchicalPlanningService,
)
from app.services.chatbot.job_workflow_service import JobWorkflowService
from app.services.chatbot.message_reader import MessageReader
from app.services.chatbot.response_service import ResponseService
from app.services.chatbot.result_projection_service import ResultProjectionService
from app.services.chatbot.scrape_parser import ScrapeResponseParser
from app.services.chatbot.upload_parser import UploadParser


class ChatbotContainer:
    """Construct and expose active chatbot repositories and services."""

    def __init__(
        self,
        *,
        chat_model: ChatModel | None = None,
        subgraphs: SubgraphRepository | None = None,
        scraper: JobScraperRepository | None = None,
    ) -> None:
        self.chat_model: ChatModel = chat_model or ChatModel.from_env()
        self.subgraphs: SubgraphRepository = subgraphs or SUBGRAPH_REPOSITORY
        self.scraper: JobScraperRepository = scraper or JOB_SCRAPER_REPOSITORY

        # [Stateless helpers]
        self.messages: MessageReader = MessageReader()
        self.uploads: UploadParser = UploadParser()
        self.parser: ScrapeResponseParser = ScrapeResponseParser()

        # [State repositories]
        self.state: ConversationStateRepository = ConversationStateRepository(
            messages=self.messages,
        )
        self.cvs: CvStateRepository = CvStateRepository(state=self.state)
        self.jobs: JobStateRepository = JobStateRepository(
            state=self.state,
            cvs=self.cvs,
        )
        self.events: EventStateService = EventStateService(state=self.state)
        self.hierarchical: HierarchicalPlanningService = HierarchicalPlanningService(
            state=self.state,
            messages=self.messages,
            chat_model=self.chat_model,
        )

        # [Services]
        self.projection: ResultProjectionService = ResultProjectionService(
            state=self.state,
            jobs=self.jobs,
            subgraphs=self.subgraphs,
        )
        self.cv_workflow: CvWorkflowService = CvWorkflowService(
            state=self.state,
            cvs=self.cvs,
            subgraphs=self.subgraphs,
            projection=self.projection,
            chat_model=self.chat_model,
        )
        self.job_workflow: JobWorkflowService = JobWorkflowService(
            state=self.state,
            cvs=self.cvs,
            jobs=self.jobs,
            scraper=self.scraper,
            parser=self.parser,
            subgraphs=self.subgraphs,
            projection=self.projection,
        )
        self.response: ResponseService = ResponseService(
            messages=self.messages,
            chat_model=self.chat_model,
        )
