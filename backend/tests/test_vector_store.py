"""
Tests for vector store functionality, specifically testing search behavior
and the MAX_RESULTS configuration issue
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk

class TestVectorStoreSearch:
    """Test vector store search functionality"""
    
    def test_search_results_creation(self):
        """Test SearchResults object creation and methods"""
        # Test normal results
        results = SearchResults(
            documents=["doc1", "doc2"],
            metadata=[{"course": "A"}, {"course": "B"}],
            distances=[0.1, 0.2]
        )
        
        assert not results.is_empty()
        assert results.error is None
        assert len(results.documents) == 2
        
        # Test empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        assert empty_results.is_empty()
        
        # Test error results
        error_results = SearchResults.empty("Test error")
        assert error_results.is_empty()
        assert error_results.error == "Test error"
    
    def test_vector_store_initialization_with_buggy_config(self, test_config):
        """Test vector store initialization with the buggy MAX_RESULTS=0 config"""
        # Set the buggy config value
        test_config.MAX_RESULTS = 0
        
        vector_store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        
        # Confirm the buggy value is set
        assert vector_store.max_results == 0
        print(f"\n🚨 Vector store initialized with MAX_RESULTS = {vector_store.max_results}")
    
    def test_vector_store_initialization_with_fixed_config(self, test_config):
        """Test vector store initialization with corrected MAX_RESULTS config"""
        # Set a fixed config value
        test_config.MAX_RESULTS = 5
        
        vector_store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        
        assert vector_store.max_results == 5
        print(f"\n✅ Vector store initialized with fixed MAX_RESULTS = {vector_store.max_results}")
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_zero_max_results(self, mock_embedding_fn, mock_client, test_config):
        """Test search behavior when MAX_RESULTS is 0 (the bug)"""
        # Setup mocks
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Create vector store with buggy config
        test_config.MAX_RESULTS = 0
        vector_store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        
        # Mock the course content collection query to return results
        mock_collection.query.return_value = {
            'documents': [["Document 1", "Document 2"]],
            'metadatas': [[{"course_title": "Test"}, {"course_title": "Test"}]],
            'distances': [[0.1, 0.2]]
        }
        
        # Perform search - this should call query with n_results=0
        result = vector_store.search("test query")
        
        # Check that query was called with n_results=0 (the bug!)
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=0,  # This is the bug!
            where=None
        )
        
        print(f"\n🚨 CONFIRMED BUG: search called with n_results={vector_store.max_results}")
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_fixed_max_results(self, mock_embedding_fn, mock_client, test_config):
        """Test search behavior when MAX_RESULTS is fixed"""
        # Setup mocks
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Create vector store with fixed config
        test_config.MAX_RESULTS = 5
        vector_store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        
        # Mock the course content collection query to return results
        mock_collection.query.return_value = {
            'documents': [["Document 1", "Document 2"]],
            'metadatas': [[{"course_title": "Test"}, {"course_title": "Test"}]],
            'distances': [[0.1, 0.2]]
        }
        
        # Perform search - this should call query with n_results=5
        result = vector_store.search("test query")
        
        # Check that query was called with n_results=5 (fixed!)
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,  # This is the fix!
            where=None
        )
        
        print(f"\n✅ FIXED: search called with n_results={vector_store.max_results}")
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_custom_limit_overrides_config(self, mock_embedding_fn, mock_client, test_config):
        """Test that providing a custom limit overrides the MAX_RESULTS config"""
        # Setup mocks
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Create vector store with buggy config
        test_config.MAX_RESULTS = 0
        vector_store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        
        # Mock the course content collection query
        mock_collection.query.return_value = {
            'documents': [["Document 1"]],
            'metadatas': [[{"course_title": "Test"}]],
            'distances': [[0.1]]
        }
        
        # Perform search with custom limit - this should override the buggy config
        result = vector_store.search("test query", limit=3)
        
        # Check that query was called with the custom limit, not the buggy MAX_RESULTS
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,  # Custom limit overrides the buggy config
            where=None
        )
        
        print(f"\n✅ Custom limit (3) overrides buggy MAX_RESULTS ({vector_store.max_results})")
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_error_handling(self, mock_embedding_fn, mock_client, test_config):
        """Test search error handling"""
        # Setup mocks
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Create vector store
        vector_store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=5
        )
        
        # Make the query method raise an exception
        mock_collection.query.side_effect = Exception("ChromaDB connection error")
        
        # Perform search - should handle the exception gracefully
        result = vector_store.search("test query")
        
        # Check that error is properly handled
        assert result.error is not None
        assert "Search error: ChromaDB connection error" in result.error
        assert result.is_empty()
        
        print(f"\n✅ Error handling works: {result.error}")
    
    def test_build_filter_logic(self, test_config):
        """Test the filter building logic"""
        vector_store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=5
        )
        
        # Test no filters
        filter_dict = vector_store._build_filter(None, None)
        assert filter_dict is None
        
        # Test course filter only
        filter_dict = vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}
        
        # Test lesson filter only
        filter_dict = vector_store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}
        
        # Test both filters
        filter_dict = vector_store._build_filter("Test Course", 1)
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 1}
        ]}
        assert filter_dict == expected

class TestVectorStoreDataOperations:
    """Test vector store data operations"""
    
    def test_course_metadata_addition(self, sample_course, test_config):
        """Test adding course metadata"""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            vector_store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=5
            )
            
            # Add course metadata
            vector_store.add_course_metadata(sample_course)
            
            # Verify the add method was called correctly
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args
            
            assert call_args[1]['documents'] == [sample_course.title]
            assert call_args[1]['ids'] == [sample_course.title]
            
            # Check metadata contains course information
            metadata = call_args[1]['metadatas'][0]
            assert metadata['title'] == sample_course.title
            assert metadata['instructor'] == sample_course.instructor
            assert metadata['course_link'] == sample_course.course_link
    
    def test_course_content_addition(self, sample_course_chunks, test_config):
        """Test adding course content chunks"""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            vector_store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=5
            )
            
            # Add course content
            vector_store.add_course_content(sample_course_chunks)
            
            # Verify the add method was called correctly
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args
            
            assert len(call_args[1]['documents']) == len(sample_course_chunks)
            assert len(call_args[1]['metadatas']) == len(sample_course_chunks)
            assert len(call_args[1]['ids']) == len(sample_course_chunks)
            
            # Check first chunk
            first_metadata = call_args[1]['metadatas'][0]
            first_chunk = sample_course_chunks[0]
            assert first_metadata['course_title'] == first_chunk.course_title
            assert first_metadata['lesson_number'] == first_chunk.lesson_number
            assert first_metadata['chunk_index'] == first_chunk.chunk_index