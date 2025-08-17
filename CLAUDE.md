# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Install dependencies
uv sync

# Environment variables required in .env:
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Access Points
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials using semantic search and AI responses.

### Core Architecture Pattern
The system follows a **tool-based RAG architecture** where Claude dynamically decides when to search course content rather than always performing retrieval. This creates more natural conversations for general questions while providing accurate course-specific information when needed.

### Component Interaction Flow
```
Frontend (HTML/JS) → FastAPI → RAG System → AI Generator → Claude API
                                    ↓           ↑
                             Tool Manager → Course Search Tool → Vector Store
```

### Key Architectural Components

**RAG System (`rag_system.py`)**: Central orchestrator that coordinates all components. Manages the query processing pipeline and decides between direct AI responses vs tool-augmented responses.

**Tool-Based Search (`search_tools.py`)**: Implements Claude's tool calling interface. The `CourseSearchTool` provides semantic search capabilities that Claude can invoke when course-specific information is needed.

**AI Generator (`ai_generator.py`)**: Manages Claude API interactions with sophisticated tool execution handling. Processes tool use requests and synthesizes final responses.

**Vector Store (`vector_store.py`)**: ChromaDB-based semantic search with sentence-transformer embeddings. Supports course name fuzzy matching and lesson number filtering.

**Document Processor (`document_processor.py`)**: Handles structured course document parsing with specific format expectations (Course Title, Course Link, Course Instructor, followed by Lesson markers).

**Session Manager (`session_manager.py`)**: Maintains conversation context across queries for natural multi-turn conversations.

### Data Models (`models.py`)
- **Course**: Container for course metadata and lesson structure
- **Lesson**: Individual lesson with title and optional link
- **CourseChunk**: Text chunks with course/lesson attribution for vector storage

### Document Processing Pipeline
1. **Parse Structure**: Extracts course metadata from first 3 lines using regex patterns
2. **Lesson Segmentation**: Identifies lesson boundaries using `Lesson N: [title]` markers
3. **Content Chunking**: Sentence-based chunking with configurable overlap for context preservation
4. **Context Enhancement**: Prepends course/lesson context to chunks for better retrieval
5. **Vector Storage**: Converts to embeddings and stores in ChromaDB with metadata

### Configuration (`config.py`)
Key settings that affect system behavior:
- `CHUNK_SIZE`: 800 chars (affects retrieval granularity)
- `CHUNK_OVERLAP`: 100 chars (maintains context between chunks)
- `MAX_RESULTS`: 5 (search result limit)
- `MAX_HISTORY`: 2 (conversation memory depth)

### Frontend Integration
The frontend (`frontend/`) provides a chat interface that displays AI responses with collapsible source attribution. Sources are tracked through the tool execution pipeline and returned with each response.

## Course Document Format

Documents in `docs/` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional lesson url]
[lesson content...]

Lesson 1: [title]
[lesson content...]
```

## Key Implementation Details

**Tool Decision Logic**: Claude analyzes each query to determine if course content search is needed. General knowledge questions receive direct responses without tool use.

**Source Attribution**: The system tracks which course chunks contributed to each response through the tool execution pipeline, enabling transparent source citation.

**Semantic Course Matching**: Course name filters support partial matching (e.g., "MCP" matches "MCP: Build Rich-Context AI Apps with Anthropic").

**Session Persistence**: Conversation context is maintained in memory per session for natural multi-turn interactions.

**Error Handling**: Graceful degradation at each layer - if vector search fails, Claude can still provide general responses.
- always use uv to run the server never use pip
- make sure to use uv to manage all dependencies