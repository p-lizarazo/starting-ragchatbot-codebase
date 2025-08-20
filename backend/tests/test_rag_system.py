"""
Tests for RAG system integration and end-to-end query processing
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson
from vector_store import SearchResults

class TestRAGSystemInitialization:
    """Test RAG system initialization"""
    
    def test_rag_system_initialization_with_buggy_config(self, temp_directory):
        """Test RAG system initialization with buggy MAX_RESULTS=0 config"""
        # Create config with the bug
        config = Config()
        config.CHROMA_PATH = str(Path(temp_directory) / "test_chroma")
        config.MAX_RESULTS = 0  # The bug!
        config.ANTHROPIC_API_KEY = "test-key"
        
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class:
            
            # Initialize RAG system
            rag_system = RAGSystem(config)
            
            # Verify VectorStore was initialized with buggy config
            mock_vector_store_class.assert_called_once_with(
                config.CHROMA_PATH, 
                config.EMBEDDING_MODEL, 
                0  # The buggy MAX_RESULTS value!
            )
            
            print(f"\n🚨 CONFIRMED: RAG system initializes VectorStore with MAX_RESULTS=0")
    
    def test_rag_system_tool_registration(self, test_config):
        """Test that tools are properly registered during initialization"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class:
            
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store
            
            # Initialize RAG system
            rag_system = RAGSystem(test_config)
            
            # Check that tools are registered
            assert "search_course_content" in rag_system.tool_manager.tools
            assert "get_course_outline" in rag_system.tool_manager.tools
            
            # Check that tools have access to the vector store
            search_tool = rag_system.tool_manager.tools["search_course_content"]
            outline_tool = rag_system.tool_manager.tools["get_course_outline"]
            
            assert search_tool.store is mock_vector_store
            assert outline_tool.store is mock_vector_store
            
            print(f"\n✅ Tools properly registered with vector store")

class TestRAGSystemQueryProcessing:
    """Test RAG system query processing"""
    
    def test_query_processing_without_tools_usage(self, test_config):
        """Test query processing when AI doesn't use tools"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class, \
             patch('rag_system.SessionManager') as mock_session_manager_class:
            
            # Setup mocks
            mock_ai_generator = Mock()
            mock_ai_generator_class.return_value = mock_ai_generator
            mock_ai_generator.generate_response.return_value = "General response about ML"
            
            mock_session_manager = Mock()
            mock_session_manager_class.return_value = mock_session_manager
            mock_session_manager.get_conversation_history.return_value = None
            
            # Initialize RAG system
            rag_system = RAGSystem(test_config)
            
            # Process query
            response, sources = rag_system.query("What is machine learning in general?")
            
            # Check response
            assert response == "General response about ML"
            assert sources == []  # No tools used, so no sources
            
            # Verify AI generator was called with tools
            mock_ai_generator.generate_response.assert_called_once()
            call_args = mock_ai_generator.generate_response.call_args
            assert "tools" in call_args[1]
            assert "tool_manager" in call_args[1]
            
            print(f"\n✅ Query processing without tool usage works")
    
    def test_query_processing_with_successful_tool_usage(self, test_config, sample_search_results):
        """Test query processing when AI uses tools successfully"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class, \
             patch('rag_system.SessionManager') as mock_session_manager_class:
            
            # Setup mocks
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store
            mock_vector_store.search.return_value = sample_search_results
            mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
            
            mock_ai_generator = Mock()
            mock_ai_generator_class.return_value = mock_ai_generator
            mock_ai_generator.generate_response.return_value = "Here's what I found about ML..."
            
            mock_session_manager = Mock()
            mock_session_manager_class.return_value = mock_session_manager
            mock_session_manager.get_conversation_history.return_value = None
            
            # Initialize RAG system
            rag_system = RAGSystem(test_config)
            
            # Simulate that the AI used the search tool
            # We need to manually trigger the tool execution to simulate this
            search_tool = rag_system.tool_manager.tools["search_course_content"]
            search_tool.execute("machine learning basics")  # This sets last_sources
            
            # Process query
            response, sources = rag_system.query("Tell me about machine learning")
            
            # Check response and sources
            assert response == "Here's what I found about ML..."
            assert len(sources) > 0  # Should have sources from tool usage
            
            print(f"\n✅ Query processing with successful tool usage works")
    
    def test_query_processing_with_empty_search_results(self, test_config):
        """Test query processing when search returns empty results (the bug scenario)"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class, \
             patch('rag_system.SessionManager') as mock_session_manager_class:
            
            # Setup mocks
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store
            
            # Configure vector store to return empty results (simulating MAX_RESULTS=0 bug)
            empty_results = SearchResults(documents=[], metadata=[], distances=[])
            mock_vector_store.search.return_value = empty_results
            
            mock_ai_generator = Mock()
            mock_ai_generator_class.return_value = mock_ai_generator
            mock_ai_generator.generate_response.return_value = "I couldn't find specific information"
            
            mock_session_manager = Mock()
            mock_session_manager_class.return_value = mock_session_manager
            mock_session_manager.get_conversation_history.return_value = None
            
            # Initialize RAG system
            rag_system = RAGSystem(test_config)
            
            # Simulate the search tool being used and returning empty results
            search_tool = rag_system.tool_manager.tools["search_course_content"]
            result = search_tool.execute("machine learning")
            assert "No relevant content found" in result  # Tool should return this message
            
            # Process query
            response, sources = rag_system.query("Tell me about machine learning")
            
            # Check response
            assert response == "I couldn't find specific information"
            assert sources == []  # No sources because search was empty
            
            print(f"\n🚨 CONFIRMED: Empty search results lead to '{response}'")
            print("This explains the user's 'query failed' experience")
    
    def test_query_exception_propagation(self, test_config):
        """Test that exceptions in query processing are NOT caught"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class:
            
            # Setup AI generator to raise exception
            mock_ai_generator = Mock()
            mock_ai_generator_class.return_value = mock_ai_generator
            mock_ai_generator.generate_response.side_effect = Exception("API connection failed")
            
            # Initialize RAG system
            rag_system = RAGSystem(test_config)
            
            # Test that exception is NOT caught
            with pytest.raises(Exception, match="API connection failed"):
                rag_system.query("test query")
            
            print(f"\n🚨 CONFIRMED: RAG system does NOT catch exceptions in query method")
            print("Exceptions propagate to the API layer, causing 'query failed' responses")

class TestRAGSystemWithRealComponents:
    """Test RAG system with more realistic component integration"""
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_end_to_end_query_processing_success(self, mock_vector_store_class, mock_ai_generator_class, test_config, sample_search_results):
        """Test end-to-end query processing with successful results"""
        # Setup vector store mock
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
        
        # Setup AI generator mock to simulate tool usage
        mock_ai_generator = Mock()
        mock_ai_generator_class.return_value = mock_ai_generator
        
        def mock_generate_response(query, conversation_history=None, tools=None, tool_manager=None):
            # Simulate the AI using the search tool
            if tool_manager:
                tool_result = tool_manager.execute_tool("search_course_content", query="machine learning")
                return f"Based on the course content: {tool_result[:50]}..."
            return "General response without tools"
        
        mock_ai_generator.generate_response.side_effect = mock_generate_response
        
        # Initialize RAG system
        rag_system = RAGSystem(test_config)
        
        # Process query
        response, sources = rag_system.query("Tell me about machine learning")
        
        # Check that the response includes tool results
        assert "Based on the course content" in response
        assert "ML Course - Lesson 1" in response  # From the formatted tool result
        
        # Check sources
        assert len(sources) > 0
        assert "ML Course - Lesson 1" in sources[0]
        
        print(f"\n✅ End-to-end query processing success works")
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_end_to_end_query_processing_with_buggy_vector_store(self, mock_vector_store_class, mock_ai_generator_class, test_config):
        """Test end-to-end query processing with buggy vector store (MAX_RESULTS=0)"""
        # Setup vector store mock with the bug
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        # Simulate the bug: empty results due to MAX_RESULTS=0
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        # Setup AI generator mock
        mock_ai_generator = Mock()
        mock_ai_generator_class.return_value = mock_ai_generator
        
        def mock_generate_response(query, conversation_history=None, tools=None, tool_manager=None):
            # Simulate the AI using the search tool and getting empty results
            if tool_manager:
                tool_result = tool_manager.execute_tool("search_course_content", query="machine learning")
                # Tool result would be "No relevant content found."
                return "I couldn't find any relevant course content about that topic."
            return "General response without tools"
        
        mock_ai_generator.generate_response.side_effect = mock_generate_response
        
        # Initialize RAG system
        rag_system = RAGSystem(test_config)
        
        # Process query
        response, sources = rag_system.query("Tell me about machine learning")
        
        # Check that the response indicates no content found
        assert "couldn't find" in response.lower()
        assert sources == []  # No sources due to empty results
        
        print(f"\n🚨 CONFIRMED: Buggy vector store leads to '{response}'")
        print("This demonstrates the user's experience with MAX_RESULTS=0")

class TestRAGSystemSessionManagement:
    """Test RAG system session management"""
    
    def test_session_creation_and_history(self, test_config):
        """Test session creation and conversation history"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_generator_class:
            
            mock_ai_generator = Mock()
            mock_ai_generator_class.return_value = mock_ai_generator
            mock_ai_generator.generate_response.return_value = "Test response"
            
            # Initialize RAG system
            rag_system = RAGSystem(test_config)
            
            # Create a session and process multiple queries
            session_id = "test-session-123"
            
            # First query
            response1, _ = rag_system.query("First question", session_id=session_id)
            
            # Second query with history
            response2, _ = rag_system.query("Follow up question", session_id=session_id)
            
            # Verify session manager was called correctly
            assert mock_ai_generator.generate_response.call_count == 2
            
            # Check that the second call included conversation history
            second_call_args = mock_ai_generator.generate_response.call_args_list[1]
            assert second_call_args[1]["conversation_history"] is not None
            
            print(f"\n✅ Session management works")

class TestRAGSystemCourseAnalytics:
    """Test RAG system course analytics"""
    
    def test_get_course_analytics(self, test_config):
        """Test getting course analytics"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator'):
            
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store
            mock_vector_store.get_course_count.return_value = 3
            mock_vector_store.get_existing_course_titles.return_value = [
                "Course 1", "Course 2", "Course 3"
            ]
            
            # Initialize RAG system
            rag_system = RAGSystem(test_config)
            
            # Get analytics
            analytics = rag_system.get_course_analytics()
            
            # Check analytics
            assert analytics["total_courses"] == 3
            assert len(analytics["course_titles"]) == 3
            assert "Course 1" in analytics["course_titles"]
            
            print(f"\n✅ Course analytics works")