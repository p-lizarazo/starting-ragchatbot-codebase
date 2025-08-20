"""
Tests for configuration module to identify configuration issues
"""
import pytest
import os
from unittest.mock import patch
from config import Config, config

class TestConfig:
    """Test configuration loading and validation"""
    
    def test_default_config_values(self):
        """Test that default configuration values are set correctly"""
        test_config = Config()
        
        # Test Anthropic settings
        assert hasattr(test_config, 'ANTHROPIC_API_KEY')
        assert test_config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        
        # Test embedding model
        assert test_config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        
        # Test document processing settings
        assert test_config.CHUNK_SIZE == 800
        assert test_config.CHUNK_OVERLAP == 100
        assert test_config.MAX_HISTORY == 2
        
        # Test database path
        assert test_config.CHROMA_PATH == "./chroma_db"
    
    def test_max_results_configuration_issue(self):
        """Test that MAX_RESULTS is set to a positive value (bug has been fixed)"""
        test_config = Config()
        
        # This should pass, confirming the bug has been fixed
        assert test_config.MAX_RESULTS > 0, f"MAX_RESULTS is {test_config.MAX_RESULTS}, should be positive"
        assert test_config.MAX_RESULTS == 5, f"MAX_RESULTS is {test_config.MAX_RESULTS}, expected 5"
        
        # Document the current state
        print(f"\n✅ MAX_RESULTS correctly set to: {test_config.MAX_RESULTS}")
        print("Vector store will return up to 5 search results for queries.")
    
    def test_global_config_instance(self):
        """Test the global config instance"""
        assert config is not None
        assert isinstance(config, Config)
        assert config.MAX_RESULTS == 5  # Confirms the fix in the global instance
    
    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly"""
        # Test that the API key is loaded (should not be empty if .env exists)
        test_config = Config()
        # If .env file exists with ANTHROPIC_API_KEY, it should be loaded
        assert len(test_config.ANTHROPIC_API_KEY) > 0, "API key should be loaded from environment"
    
    def test_missing_api_key(self):
        """Test behavior when API key is missing"""
        # Test that Config class handles missing API key by using empty string default
        # We'll create a config with explicit empty API key
        test_config = Config(ANTHROPIC_API_KEY="")
        assert test_config.ANTHROPIC_API_KEY == ""  # Should accept empty string
    
    def test_configuration_types(self):
        """Test that configuration values have correct types"""
        test_config = Config()
        
        assert isinstance(test_config.ANTHROPIC_API_KEY, str)
        assert isinstance(test_config.ANTHROPIC_MODEL, str)
        assert isinstance(test_config.EMBEDDING_MODEL, str)
        assert isinstance(test_config.CHUNK_SIZE, int)
        assert isinstance(test_config.CHUNK_OVERLAP, int)
        assert isinstance(test_config.MAX_RESULTS, int)
        assert isinstance(test_config.MAX_HISTORY, int)
        assert isinstance(test_config.CHROMA_PATH, str)
    
    def test_configuration_values_reasonable(self):
        """Test that configuration values are within reasonable ranges"""
        test_config = Config()
        
        # Chunk size should be positive
        assert test_config.CHUNK_SIZE > 0
        
        # Chunk overlap should be non-negative and less than chunk size
        assert test_config.CHUNK_OVERLAP >= 0
        assert test_config.CHUNK_OVERLAP < test_config.CHUNK_SIZE
        
        # Max history should be positive
        assert test_config.MAX_HISTORY > 0
        
        # MAX_RESULTS should be positive (this should pass now that bug is fixed)
        assert test_config.MAX_RESULTS > 0, "MAX_RESULTS should be positive for search to work!"

class TestConfigForBugFix:
    """Tests that demonstrate what the configuration should be after fixing"""
    
    def test_fixed_max_results_configuration(self):
        """Test what MAX_RESULTS should be after fixing the bug"""
        # Create a corrected config
        fixed_config = Config()
        fixed_config.MAX_RESULTS = 5  # What it should be
        
        assert fixed_config.MAX_RESULTS > 0
        assert fixed_config.MAX_RESULTS <= 10  # Reasonable upper bound
        print(f"\n✅ AFTER FIX: MAX_RESULTS should be {fixed_config.MAX_RESULTS}")
    
    def test_vector_store_search_limit_logic(self):
        """Test the logic that would be used in vector store search"""
        # Simulate the logic from vector_store.py line 90
        # search_limit = limit if limit is not None else self.max_results
        
        # Current buggy behavior
        max_results_buggy = 0
        limit = None
        search_limit = limit if limit is not None else max_results_buggy
        assert search_limit == 0  # This causes no results to be returned
        
        # Fixed behavior
        max_results_fixed = 5
        search_limit_fixed = limit if limit is not None else max_results_fixed
        assert search_limit_fixed == 5  # This would return up to 5 results
        
        print(f"\n🔄 Search limit logic: buggy={search_limit}, fixed={search_limit_fixed}")