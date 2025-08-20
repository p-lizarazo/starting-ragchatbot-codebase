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
