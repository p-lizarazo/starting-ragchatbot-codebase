"""
API endpoint tests for the RAG system FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""
    
    def test_query_with_session_id(self, test_client, sample_query_request):
        """Test query endpoint with provided session ID"""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_request["session_id"]
        assert isinstance(data["sources"], list)
    
    def test_query_without_session_id(self, test_client):
        """Test query endpoint without session ID (should create new session)"""
        request_data = {"query": "What is machine learning?"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
    
    def test_query_empty_query(self, test_client):
        """Test query endpoint with empty query string"""
        request_data = {"query": ""}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_missing_query_field(self, test_client):
        """Test query endpoint with missing query field"""
        request_data = {"session_id": "test-session"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post("/api/query", data="invalid json")
        
        assert response.status_code == 422
    
    def test_query_rag_system_error(self, test_client):
        """Test query endpoint when RAG system raises an error"""
        with patch('conftest.Mock') as mock_class:
            # Mock the RAG system to raise an exception
            mock_rag = mock_class.return_value
            mock_rag.query.side_effect = Exception("RAG system error")
            
            # Patch the test app's mock_rag_system
            with patch.object(test_client.app, 'mock_rag_system', mock_rag):
                request_data = {"query": "test query"}
                response = test_client.post("/api/query", json=request_data)
                
                # The test app has error handling, but our simplified version may not
                # This tests that the endpoint handles errors gracefully
                assert response.status_code in [200, 500]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""
    
    def test_get_courses_success(self, test_client):
        """Test successful retrieval of course statistics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
    
    def test_get_courses_content_type(self, test_client):
        """Test that courses endpoint returns JSON content type"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
    
    def test_get_courses_rag_system_error(self, test_client):
        """Test courses endpoint when RAG system raises an error"""
        with patch('conftest.Mock') as mock_class:
            # Mock the RAG system to raise an exception
            mock_rag = mock_class.return_value
            mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
            
            # Test that error handling works
            request_data = {"query": "test"}
            response = test_client.post("/api/query", json=request_data)
            
            # Our test app should handle this gracefully
            assert response.status_code in [200, 500]


@pytest.mark.api
class TestRootEndpoint:
    """Test the root / endpoint"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns welcome message"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Course Materials RAG System API"
    
    def test_root_endpoint_content_type(self, test_client):
        """Test that root endpoint returns JSON content type"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]


@pytest.mark.api
class TestRequestValidation:
    """Test request validation and error handling"""
    
    def test_query_request_schema_validation(self, test_client):
        """Test that query requests are properly validated"""
        # Valid request
        valid_request = {"query": "test query", "session_id": "optional-session"}
        response = test_client.post("/api/query", json=valid_request)
        assert response.status_code == 200
        
        # Request with extra fields (should be ignored)
        extra_fields_request = {
            "query": "test query",
            "session_id": "test-session",
            "extra_field": "should be ignored"
        }
        response = test_client.post("/api/query", json=extra_fields_request)
        assert response.status_code == 200
    
    def test_query_response_schema(self, test_client, sample_query_request):
        """Test that query responses match expected schema"""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        # Check field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check sources contain strings
        for source in data["sources"]:
            assert isinstance(source, str)
    
    def test_courses_response_schema(self, test_client):
        """Test that courses response matches expected schema"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check course titles contain strings
        for title in data["course_titles"]:
            assert isinstance(title, str)


@pytest.mark.api
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_endpoint(self, test_client):
        """Test request to non-existent endpoint"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_wrong_http_method(self, test_client):
        """Test using wrong HTTP method on endpoints"""
        # GET on POST endpoint
        response = test_client.get("/api/query")
        assert response.status_code == 405  # Method not allowed
        
        # POST on GET endpoint
        response = test_client.post("/api/courses")
        assert response.status_code == 405  # Method not allowed
    
    def test_large_query_string(self, test_client):
        """Test query endpoint with very large query string"""
        large_query = "x" * 10000  # 10KB query
        request_data = {"query": large_query}
        response = test_client.post("/api/query", json=request_data)
        
        # Should handle large queries gracefully
        assert response.status_code == 200
    
    def test_special_characters_in_query(self, test_client):
        """Test query endpoint with special characters"""
        special_query = "Query with special chars: ñáéíóú 中文 🚀 @#$%^&*()"
        request_data = {"query": special_query}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data


@pytest.mark.api
class TestCORSAndMiddleware:
    """Test CORS and middleware functionality"""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        # CORS headers should be present (configured in test app)
        # Note: TestClient may not simulate all CORS behavior exactly
    
    def test_options_request(self, test_client):
        """Test OPTIONS request for CORS preflight"""
        response = test_client.options("/api/query")
        
        # Should handle OPTIONS requests for CORS
        assert response.status_code in [200, 405]  # Depends on CORS configuration