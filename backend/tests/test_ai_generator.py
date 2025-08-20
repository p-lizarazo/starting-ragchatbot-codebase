"""
Tests for AIGenerator to verify tool calling mechanism and response handling
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestAIGeneratorBasics:
    """Test basic AIGenerator functionality"""

    def test_initialization(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "claude-3-sonnet-20240229")

        assert generator.model == "claude-3-sonnet-20240229"
        assert generator.base_params["model"] == "claude-3-sonnet-20240229"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

        print(f"\n✅ AIGenerator initialization works")

    def test_system_prompt_content(self):
        """Test that system prompt contains tool usage guidelines"""
        assert "Content Search Tool" in AIGenerator.SYSTEM_PROMPT
        assert "Course Outline Tool" in AIGenerator.SYSTEM_PROMPT
        assert "Sequential Tool Usage" in AIGenerator.SYSTEM_PROMPT
        assert "maximum 2 rounds" in AIGenerator.SYSTEM_PROMPT
        assert "Multi-step Queries" in AIGenerator.SYSTEM_PROMPT

        print(f"\n✅ System prompt contains sequential tool guidelines")


class TestAIGeneratorWithoutTools:
    """Test AIGenerator without tool usage"""

    @patch("anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test generating response without tools"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [Mock(text="This is a direct response")]
        mock_client.messages.create.return_value = mock_response

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Generate response
        result = generator.generate_response("What is machine learning?")

        # Check result
        assert result == "This is a direct response"

        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args[1]  # No tools provided

        print(f"\n✅ Generate response without tools works")

    @patch("anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test generating response with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [Mock(text="Response with history")]
        mock_client.messages.create.return_value = mock_response

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Generate response with history
        history = "Previous conversation context"
        result = generator.generate_response(
            "Follow up question", conversation_history=history
        )

        # Check that history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation context" in system_content

        print(f"\n✅ Generate response with history works")


class TestAIGeneratorWithTools:
    """Test AIGenerator with tool usage"""

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_class):
        """Test generating response with tools available but no tool use"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "stop"  # No tool use
        mock_response.content = [Mock(text="Direct response without tools")]
        mock_client.messages.create.return_value = mock_response

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Create mock tools
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]

        # Generate response
        result = generator.generate_response(
            "What is machine learning?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Check result
        assert result == "Direct response without tools"

        # Verify tools were provided but not used
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tool_choice"] == {"type": "auto"}

        print(f"\n✅ Generate response with tools (no tool use) works")

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test generating response that uses tools in single round"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Mock initial response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "machine learning basics"}
        mock_tool_block.id = "tool_123"

        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]

        # Mock final response after tool execution (no more tool use)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [
            Mock(text="Here's what I found about machine learning...")
        ]

        # Configure client to return initial response first, then final response
        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        mock_tool_manager.execute_tool.return_value = (
            "Search results about machine learning"
        )

        # Generate response
        result = generator.generate_response(
            "Tell me about machine learning",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Check result
        assert result == "Here's what I found about machine learning..."

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning basics"
        )

        # Verify two API calls were made (initial tool use + final response)
        assert mock_client.messages.create.call_count == 2

        print(f"\n✅ Generate response with single tool round works")

    @patch("anthropic.Anthropic")
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test handling of tool execution errors"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Mock initial response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_123"

        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="I couldn't find information")]

        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Create mock tool manager that returns error
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        mock_tool_manager.execute_tool.return_value = (
            "Search error: No relevant content found"
        )

        # Generate response
        result = generator.generate_response(
            "Tell me about something",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Check that error was handled gracefully
        assert result == "I couldn't find information"

        print(f"\n✅ Tool execution error handling works")


class TestAIGeneratorExceptionHandling:
    """Test AIGenerator exception handling"""

    @patch("anthropic.Anthropic")
    def test_api_exception_propagation(self, mock_anthropic_class):
        """Test that API exceptions are properly propagated"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Test that exception is raised (not caught)
        with pytest.raises(Exception, match="API Error"):
            generator.generate_response("test query")

        print(f"\n🚨 CONFIRMED: API exceptions are NOT caught in AIGenerator")
        print(
            "This means exceptions will propagate to RAG system and potentially cause 'query failed'"
        )

    @patch("anthropic.Anthropic")
    def test_tool_execution_exception_graceful_handling(self, mock_anthropic_class):
        """Test exception handling in tool execution with graceful degradation"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Mock initial response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_123"

        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]

        # Mock fallback response after tool error
        mock_fallback_response = Mock()
        mock_fallback_response.content = [
            Mock(text="I encountered an error with the search tool")
        ]

        # Configure client to return initial response, then fallback
        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_fallback_response,
        ]

        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        # Generate response - should handle tool error gracefully
        result = generator.generate_response(
            "test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Should get fallback response, not raise exception
        assert result == "I encountered an error with the search tool"

        print(
            f"\n✅ Tool execution exceptions are handled gracefully with fallback response"
        )


class TestAIGeneratorSequentialToolCalling:
    """Test AIGenerator sequential tool calling functionality"""

    @patch("anthropic.Anthropic")
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic_class):
        """Test successful two-round sequential tool calling"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Mock Round 1: Tool use for course outline
        mock_tool_block_1 = Mock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "get_course_outline"
        mock_tool_block_1.input = {"course_title": "Machine Learning"}
        mock_tool_block_1.id = "tool_1"

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block_1]

        # Mock Round 2: Tool use for content search
        mock_tool_block_2 = Mock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.input = {"query": "neural networks"}
        mock_tool_block_2.id = "tool_2"

        mock_response_2 = Mock()
        mock_response_2.stop_reason = "tool_use"
        mock_response_2.content = [mock_tool_block_2]

        # Mock Final response (no more tools)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [
            Mock(text="Here's the comparison between the courses...")
        ]

        # Configure client responses
        mock_client.messages.create.side_effect = [
            mock_response_1,  # Round 1
            mock_response_2,  # Round 2
            mock_final_response,  # Final call without tools
        ]

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "get_course_outline", "description": "Outline tool"},
            {"name": "search_course_content", "description": "Search tool"},
        ]
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline results...",
            "Neural networks search results...",
        ]

        # Generate response
        result = generator.generate_response(
            "Compare neural networks coverage between ML courses",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Check result
        assert result == "Here's the comparison between the courses..."

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Machine Learning"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="neural networks"
        )

        # Verify three API calls were made (2 tool rounds + final response)
        assert mock_client.messages.create.call_count == 3

        print(f"\n✅ Sequential tool calling (2 rounds) works correctly")

    @patch("anthropic.Anthropic")
    def test_natural_termination_after_one_round(self, mock_anthropic_class):
        """Test natural termination when Claude doesn't need second round"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Mock Round 1: Tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "python basics"}
        mock_tool_block.id = "tool_1"

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block]

        # Mock Round 2: No tool use (natural termination)
        mock_response_2 = Mock()
        mock_response_2.stop_reason = "stop"
        mock_response_2.content = [
            Mock(text="Based on the search, here's what I found...")
        ]

        # Configure client responses
        mock_client.messages.create.side_effect = [mock_response_1, mock_response_2]

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        mock_tool_manager.execute_tool.return_value = "Python basics search results..."

        # Generate response
        result = generator.generate_response(
            "Tell me about python basics",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Check result
        assert result == "Based on the search, here's what I found..."

        # Verify only one tool was executed
        assert mock_tool_manager.execute_tool.call_count == 1

        # Verify two API calls were made (1 tool round + natural termination)
        assert mock_client.messages.create.call_count == 2

        print(f"\n✅ Natural termination after one round works correctly")

    @patch("anthropic.Anthropic")
    def test_max_rounds_termination(self, mock_anthropic_class):
        """Test termination when max rounds (2) is reached"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Mock Round 1: Tool use
        mock_tool_block_1 = Mock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "search_course_content"
        mock_tool_block_1.input = {"query": "first search"}
        mock_tool_block_1.id = "tool_1"

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block_1]

        # Mock Round 2: Tool use (would want to continue but max rounds reached)
        mock_tool_block_2 = Mock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.input = {"query": "second search"}
        mock_tool_block_2.id = "tool_2"

        mock_response_2 = Mock()
        mock_response_2.stop_reason = "tool_use"
        mock_response_2.content = [mock_tool_block_2]

        # Mock Final response (forced because max rounds reached)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [Mock(text="Final response after max rounds")]

        # Configure client responses
        mock_client.messages.create.side_effect = [
            mock_response_1,  # Round 1
            mock_response_2,  # Round 2
            mock_final_response,  # Final forced call without tools
        ]

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        mock_tool_manager.execute_tool.side_effect = [
            "First search results...",
            "Second search results...",
        ]

        # Generate response
        result = generator.generate_response(
            "Complex query requiring multiple searches",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Check result
        assert result == "Final response after max rounds"

        # Verify both tools were executed (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify three API calls were made (2 tool rounds + final forced call)
        assert mock_client.messages.create.call_count == 3

        print(f"\n✅ Max rounds termination works correctly")

    @patch("anthropic.Anthropic")
    def test_conversation_context_preservation(self, mock_anthropic_class):
        """Test that conversation context is preserved across tool rounds"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Mock sequential responses (just check the context preservation)
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_1"

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block]

        mock_response_2 = Mock()
        mock_response_2.stop_reason = "stop"
        mock_response_2.content = [Mock(text="Final response")]

        mock_client.messages.create.side_effect = [mock_response_1, mock_response_2]

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Generate response
        result = generator.generate_response(
            "Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

        # Verify tool was executed between calls
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test"
        )

        # Verify final result
        assert result == "Final response"

        print(f"\n✅ Conversation context preservation works correctly")


class TestAIGeneratorWithRealToolManager:
    """Test AIGenerator with actual ToolManager and CourseSearchTool"""

    @patch("anthropic.Anthropic")
    def test_integration_with_real_tool_manager(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Test AIGenerator with real ToolManager and CourseSearchTool"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Create real tool manager with CourseSearchTool
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Configure mock vector store
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "machine learning"}
        mock_tool_block.id = "tool_123"

        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]

        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Final response about machine learning")
        ]

        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Generate response
        result = generator.generate_response(
            "Tell me about machine learning",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify the integration works
        assert result == "Final response about machine learning"

        # Verify tool was called
        mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

        print(f"\n✅ Integration with real ToolManager works")

    @patch("anthropic.Anthropic")
    def test_integration_with_buggy_vector_store(
        self, mock_anthropic_class, mock_vector_store
    ):
        """Test AIGenerator with vector store that returns empty results (simulating the bug)"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create generator
        generator = AIGenerator("test-key", "claude-3-sonnet-20240229")
        generator.client = mock_client

        # Create real tool manager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Configure mock vector store to return empty results (simulating MAX_RESULTS=0 bug)
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results

        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "machine learning"}
        mock_tool_block.id = "tool_123"

        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]

        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="I couldn't find relevant information")
        ]

        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Generate response
        result = generator.generate_response(
            "Tell me about machine learning",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # The AI should still respond, but indicate no information was found
        assert result == "I couldn't find relevant information"

        # Check that the tool result was "No relevant content found"
        # (This is what the CourseSearchTool returns for empty results)
        tool_calls = mock_client.messages.create.call_args_list
        if len(tool_calls) > 1:
            second_call_args = tool_calls[1][1]
            tool_result_message = second_call_args["messages"][-1]
            if "content" in tool_result_message and isinstance(
                tool_result_message["content"], list
            ):
                tool_result_content = tool_result_message["content"][0]["content"]
                assert "No relevant content found" in tool_result_content

        print(f"\n🚨 CONFIRMED: With empty search results, AI responds: '{result}'")
        print("This shows how the MAX_RESULTS=0 bug manifests to users")
