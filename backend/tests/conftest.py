"""
Shared test fixtures and configuration for RAG system tests
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import Config
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Handle Windows file locking issues with ChromaDB
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        # On Windows, ChromaDB files might be locked
        # Try to remove them with a delay
        import time

        time.sleep(0.1)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # If still locked, just ignore the cleanup error
            # The temp directory will be cleaned by the OS eventually
            pass


@pytest.fixture
def test_config(temp_directory):
    """Create a test configuration with temporary paths"""
    config = Config()
    config.CHROMA_PATH = str(Path(temp_directory) / "test_chroma")
    config.MAX_RESULTS = 5  # Override the problematic 0 value for testing
    config.ANTHROPIC_API_KEY = "test-key-123"
    return config


@pytest.fixture
def sample_course():
    """Create a sample course object for testing"""
    lessons = [
        Lesson(
            lesson_number=0,
            title="Introduction",
            lesson_link="http://example.com/lesson0",
        ),
        Lesson(
            lesson_number=1,
            title="Getting Started",
            lesson_link="http://example.com/lesson1",
        ),
        Lesson(lesson_number=2, title="Advanced Topics", lesson_link=None),
    ]
    return Course(
        title="Test Course: Introduction to Testing",
        instructor="Dr. Test",
        course_link="http://example.com/course",
        lessons=lessons,
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = []
    for i, lesson in enumerate(sample_course.lessons):
        for j in range(2):  # 2 chunks per lesson
            chunk = CourseChunk(
                course_title=sample_course.title,
                lesson_number=lesson.lesson_number,
                chunk_index=i * 2 + j,
                content=f"This is content for lesson {lesson.lesson_number}, chunk {j}. "
                f"It contains information about {lesson.title}.",
            )
            chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)

    # Configure default return values
    mock_store.search.return_value = SearchResults(
        documents=["Sample document content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_store._resolve_course_name.return_value = (
        "Test Course: Introduction to Testing"
    )
    mock_store.get_lesson_link.return_value = "http://example.com/lesson1"
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": "Test Course: Introduction to Testing",
            "instructor": "Dr. Test",
            "course_link": "http://example.com/course",
            "lessons": [
                {
                    "lesson_number": 0,
                    "lesson_title": "Introduction",
                    "lesson_link": "http://example.com/lesson0",
                },
                {
                    "lesson_number": 1,
                    "lesson_title": "Getting Started",
                    "lesson_link": "http://example.com/lesson1",
                },
            ],
        }
    ]

    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()

    # Create a mock response object
    mock_response = Mock()
    mock_response.stop_reason = "stop"
    mock_response.content = [Mock(text="Test response from AI")]

    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Create a mock tool use block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": "test query"}
    mock_tool_block.id = "tool_123"

    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Create search results with error for testing"""
    return SearchResults.empty("Test error message")


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "This is a test document about machine learning",
            "Another document about data science",
        ],
        metadata=[
            {"course_title": "ML Course", "lesson_number": 1},
            {"course_title": "DS Course", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    from unittest.mock import Mock
    
    # Create test app
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    mock_rag_system.query.return_value = ("Test answer", ["Source 1", "Source 2"])
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    mock_rag_system.session_manager.create_session.return_value = "test-session-123"
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for testing"""
    mock_rag = Mock()
    mock_rag.query.return_value = ("Test answer", ["Source 1", "Source 2"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    return mock_rag


@pytest.fixture
def sample_query_request():
    """Sample query request data for testing"""
    return {
        "query": "What is machine learning?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response data for testing"""
    return {
        "answer": "Machine learning is a subset of artificial intelligence...",
        "sources": ["Course: ML Basics, Lesson 1", "Course: AI Fundamentals, Lesson 2"],
        "session_id": "test-session-123"
    }
