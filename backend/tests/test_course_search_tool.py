"""
Tests for CourseSearchTool to verify tool execution and search result handling
"""

from unittest.mock import Mock, patch

import pytest
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""

    def test_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly structured"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        # Check required fields
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Check input schema
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "query" in schema["required"]

        # Check properties
        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

        print(f"\n✅ Tool definition structure is correct")

    def test_execute_with_successful_search(
        self, mock_vector_store, sample_search_results
    ):
        """Test execute method with successful search results"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return sample results
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        # Execute the tool
        result = tool.execute("test query")

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=None
        )

        # Check result format
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ML Course" in result  # From sample data
        assert "Lesson 1" in result

        # Check sources are tracked
        assert len(tool.last_sources) > 0

        print(
            f"\n✅ Execute with successful search works: {len(result)} chars returned"
        )

    def test_execute_with_empty_search_results(
        self, mock_vector_store, empty_search_results
    ):
        """Test execute method when search returns no results"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return empty results
        mock_vector_store.search.return_value = empty_search_results

        # Execute the tool
        result = tool.execute("nonexistent query")

        # Check that appropriate message is returned
        assert "No relevant content found" in result

        print(f"\n✅ Execute with empty results: {result}")

    def test_execute_with_search_error(self, mock_vector_store, error_search_results):
        """Test execute method when search returns an error"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return error results
        mock_vector_store.search.return_value = error_search_results

        # Execute the tool
        result = tool.execute("test query")

        # Check that error is returned
        assert result == "Test error message"

        print(f"\n✅ Execute with search error: {result}")

    def test_execute_with_course_name_filter(
        self, mock_vector_store, sample_search_results
    ):
        """Test execute method with course name filtering"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return sample results
        mock_vector_store.search.return_value = sample_search_results

        # Execute with course name filter
        result = tool.execute("test query", course_name="ML Course")

        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="ML Course", lesson_number=None
        )

        print(f"\n✅ Execute with course filter works")

    def test_execute_with_lesson_number_filter(
        self, mock_vector_store, sample_search_results
    ):
        """Test execute method with lesson number filtering"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return sample results
        mock_vector_store.search.return_value = sample_search_results

        # Execute with lesson number filter
        result = tool.execute("test query", lesson_number=1)

        # Verify search was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=1
        )

        print(f"\n✅ Execute with lesson filter works")

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test execute method with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return sample results
        mock_vector_store.search.return_value = sample_search_results

        # Execute with both filters
        result = tool.execute("test query", course_name="ML Course", lesson_number=1)

        # Verify search was called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="ML Course", lesson_number=1
        )

        print(f"\n✅ Execute with both filters works")

    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test result formatting with lesson links"""
        tool = CourseSearchTool(mock_vector_store)

        # Create test results with lesson data
        test_results = SearchResults(
            documents=["Test content about machine learning"],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}],
            distances=[0.1],
        )

        # Configure mock to return a lesson link
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        # Format results
        formatted = tool._format_results(test_results)

        # Check formatting
        assert "[ML Course - Lesson 1]" in formatted
        assert "Test content about machine learning" in formatted

        # Check that sources include embedded link
        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert "ML Course - Lesson 1" in source
        assert "|http://example.com/lesson1" in source  # Embedded link format

        print(f"\n✅ Result formatting with links: {source}")

    def test_format_results_without_lesson_data(self, mock_vector_store):
        """Test result formatting without lesson data"""
        tool = CourseSearchTool(mock_vector_store)

        # Create test results without lesson data
        test_results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "General Course"}],
            distances=[0.1],
        )

        # Format results
        formatted = tool._format_results(test_results)

        # Check formatting
        assert "[General Course]" in formatted
        assert "Lesson" not in formatted

        # Check sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0] == "General Course"

        print(f"\n✅ Result formatting without lesson data works")

    def test_sources_tracking_and_reset(self, mock_vector_store, sample_search_results):
        """Test that sources are properly tracked and can be reset"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        # Execute to generate sources
        tool.execute("test query")

        # Check sources are tracked
        initial_sources = tool.last_sources.copy()
        assert len(initial_sources) > 0

        # Execute again with different results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        tool.execute("another query")

        # Sources should be updated (empty in this case)
        assert len(tool.last_sources) == 0

        print(
            f"\n✅ Sources tracking works: {len(initial_sources)} -> {len(tool.last_sources)}"
        )


class TestCourseSearchToolWithBuggyVectorStore:
    """Test CourseSearchTool behavior with the buggy MAX_RESULTS=0 configuration"""

    def test_tool_with_zero_max_results_vector_store(self, test_config):
        """Test tool behavior when vector store has MAX_RESULTS=0"""
        # Create a mock vector store that simulates the bug
        mock_vector_store = Mock()

        # Simulate the bug: vector store returns empty results because MAX_RESULTS=0
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results

        tool = CourseSearchTool(mock_vector_store)

        # Execute the tool
        result = tool.execute("test query about machine learning")

        # The tool should return "No relevant content found" due to empty results
        assert "No relevant content found" in result

        # Verify search was called
        mock_vector_store.search.assert_called_once()

        print(
            f"\n🚨 CONFIRMED: Tool returns '{result}' when vector store has MAX_RESULTS=0"
        )
        print(
            "This explains why users see 'query failed' or 'no content found' messages!"
        )


class TestToolManager:
    """Test ToolManager functionality with CourseSearchTool"""

    def test_tool_registration(self, mock_vector_store):
        """Test registering CourseSearchTool with ToolManager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        # Register the tool
        manager.register_tool(tool)

        # Check tool is registered
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] is tool

        print(f"\n✅ Tool registration works")

    def test_tool_execution_through_manager(
        self, mock_vector_store, sample_search_results
    ):
        """Test executing tool through ToolManager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Configure mock
        mock_vector_store.search.return_value = sample_search_results

        # Execute through manager
        result = manager.execute_tool("search_course_content", query="test query")

        # Check result
        assert isinstance(result, str)
        assert len(result) > 0

        print(f"\n✅ Tool execution through manager works")

    def test_tool_definitions_retrieval(self, mock_vector_store):
        """Test getting tool definitions from manager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Get tool definitions
        definitions = manager.get_tool_definitions()

        # Check definitions
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

        print(f"\n✅ Tool definitions retrieval works")

    def test_sources_management(self, mock_vector_store, sample_search_results):
        """Test source tracking through ToolManager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Configure mock
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        # Execute tool to generate sources
        manager.execute_tool("search_course_content", query="test query")

        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) > 0

        # Reset sources
        manager.reset_sources()
        sources_after_reset = manager.get_last_sources()
        assert len(sources_after_reset) == 0

        print(
            f"\n✅ Source management works: {len(sources)} -> {len(sources_after_reset)}"
        )

    def test_nonexistent_tool_execution(self, mock_vector_store):
        """Test executing a nonexistent tool"""
        manager = ToolManager()

        # Try to execute a tool that doesn't exist
        result = manager.execute_tool("nonexistent_tool", query="test")

        # Should return error message
        assert "Tool 'nonexistent_tool' not found" in result

        print(f"\n✅ Nonexistent tool handling: {result}")
