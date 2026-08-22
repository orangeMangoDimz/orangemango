"""Dependency wiring for the chatbot graph.

Repositories are built first, then services, then routers. The subgraph and MCP
repositories default to module-level singletons so repeated ``build_graph``
calls reuse the loaded child graphs and the resolved scrape tool.
"""

from __future__ import annotations

from app.graph.chatbot.routers import (
    ChatbotRouter,
    CvSubagentRouter,
    JobSubagentRouter,
)
from app.models.chat_model import ChatModel
from app.repositories.chatbot.catalog_repository import RoutingCatalogRepository
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
from app.services.chatbot.action_reuse_service import ActionReuseService
from app.services.chatbot.conversation_service import ConversationService
from app.services.chatbot.cv_workflow_service import CvWorkflowService
from app.services.chatbot.execution_service import ExecutionService
from app.services.chatbot.ingest_service import IngestService
from app.services.chatbot.job_workflow_service import JobWorkflowService
from app.services.chatbot.match_presentation_service import MatchPresentationService
from app.services.chatbot.memory_service import ConversationMemoryService
from app.services.chatbot.message_reader import MessageReader
from app.services.chatbot.plan_validation_service import PlanValidationService
from app.services.chatbot.presentation_service import PresentationService
from app.services.chatbot.response_service import ResponseService
from app.services.chatbot.result_projection_service import ResultProjectionService
from app.services.chatbot.routing_service import RequestRoutingService
from app.services.chatbot.scrape_parser import ScrapeResponseParser
from app.services.chatbot.upload_parser import UploadParser


class ChatbotContainer:
    """Construct and expose every chatbot repository, service, and router."""

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
        self.matches: MatchPresentationService = MatchPresentationService()

        # [State repositories]
        self.state: ConversationStateRepository = ConversationStateRepository(
            messages=self.messages,
        )
        self.cvs: CvStateRepository = CvStateRepository(state=self.state)
        self.jobs: JobStateRepository = JobStateRepository(
            state=self.state,
            cvs=self.cvs,
        )
        self.catalogs: RoutingCatalogRepository = RoutingCatalogRepository(
            state=self.state,
            cvs=self.cvs,
            jobs=self.jobs,
        )

        # [Services]
        self.projection: ResultProjectionService = ResultProjectionService(
            state=self.state,
            jobs=self.jobs,
            subgraphs=self.subgraphs,
        )
        self.reuse: ActionReuseService = ActionReuseService(
            state=self.state,
            cvs=self.cvs,
            jobs=self.jobs,
        )
        self.execution: ExecutionService = ExecutionService(
            state=self.state,
            cvs=self.cvs,
            jobs=self.jobs,
            reuse=self.reuse,
            projection=self.projection,
        )
        self.conversation: ConversationService = ConversationService(
            state=self.state,
            messages=self.messages,
        )
        self.memory: ConversationMemoryService = ConversationMemoryService(
            state=self.state,
            conversation=self.conversation,
            chat_model=self.chat_model,
        )
        self.routing: RequestRoutingService = RequestRoutingService(
            state=self.state,
            cvs=self.cvs,
            jobs=self.jobs,
            catalogs=self.catalogs,
            reuse=self.reuse,
            conversation=self.conversation,
            chat_model=self.chat_model,
        )
        self.plan: PlanValidationService = PlanValidationService(
            state=self.state,
            cvs=self.cvs,
            jobs=self.jobs,
            catalogs=self.catalogs,
            reuse=self.reuse,
            execution=self.execution,
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
        self.presentation: PresentationService = PresentationService(
            state=self.state,
            cvs=self.cvs,
            jobs=self.jobs,
            projection=self.projection,
            matches=self.matches,
        )
        self.response: ResponseService = ResponseService(
            state=self.state,
            presentation=self.presentation,
            conversation=self.conversation,
            messages=self.messages,
            chat_model=self.chat_model,
        )
        self.ingest: IngestService = IngestService(
            state=self.state,
            cvs=self.cvs,
            uploads=self.uploads,
        )

        # [Routers]
        self.router: ChatbotRouter = ChatbotRouter(state=self.state, cvs=self.cvs)
        self.cv_router: CvSubagentRouter = CvSubagentRouter(
            state=self.state,
            cvs=self.cvs,
        )
        self.job_router: JobSubagentRouter = JobSubagentRouter(
            state=self.state,
            jobs=self.jobs,
            reuse=self.reuse,
        )
