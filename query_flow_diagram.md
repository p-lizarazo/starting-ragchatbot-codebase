# RAG System Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend (HTML/JS)
    participant API as FastAPI Endpoint
    participant RAG as RAG System
    participant Session as Session Manager
    participant AI as AI Generator
    participant Tools as Tool Manager
    participant Search as Course Search Tool
    participant Vector as Vector Store
    participant Claude as Anthropic Claude

    Note over User, Claude: User Query Processing Flow

    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: Disable input, show loading
    Frontend->>API: POST /api/query<br/>{query, session_id}

    API->>RAG: rag_system.query(query, session_id)
    
    RAG->>Session: get_conversation_history(session_id)
    Session-->>RAG: Previous conversation context
    
    RAG->>AI: generate_response(query, history, tools, tool_manager)
    
    AI->>Claude: API call with query + tool definitions
    Note over Claude: Claude analyzes query<br/>Decides if search needed
    
    alt Query needs course content search
        Claude-->>AI: Tool use request: search_course_content
        AI->>Tools: execute_tool("search_course_content", params)
        Tools->>Search: execute(query, course_name, lesson_number)
        Search->>Vector: search(query, filters)
        Vector->>Vector: Semantic search in ChromaDB
        Vector-->>Search: Relevant course chunks + metadata
        Search->>Search: Format results with context
        Search->>Search: Store sources for UI
        Search-->>Tools: Formatted search results
        Tools-->>AI: Search results string
        AI->>Claude: Follow-up API call with tool results
        Claude-->>AI: Final synthesized response
    else General knowledge query
        Claude-->>AI: Direct response (no tools)
    end
    
    AI-->>RAG: Generated response text
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: Source list from last search
    RAG->>Tools: reset_sources()
    RAG->>Session: add_exchange(session_id, query, response)
    RAG-->>API: (response, sources)
    
    API-->>Frontend: JSON response<br/>{answer, sources, session_id}
    
    Frontend->>Frontend: Remove loading animation
    Frontend->>Frontend: Display answer + sources
    Frontend->>Frontend: Re-enable input
    Frontend-->>User: Show response with sources
```

## Component Breakdown

### Frontend Layer
- **HTML/CSS**: User interface with chat input and display
- **JavaScript**: Handles user interactions, API calls, and DOM updates
- **Features**: Loading states, session management, source display

### API Layer  
- **FastAPI**: RESTful endpoints with CORS middleware
- **Pydantic Models**: Request/response validation
- **Error Handling**: HTTP status codes and error messages

### RAG System Core
- **Orchestration**: Coordinates all components
- **Session Management**: Maintains conversation context
- **Tool Integration**: Manages available AI tools

### AI Processing
- **Anthropic Claude**: Large language model for response generation
- **Tool Calling**: Dynamic decision to search course content
- **Context Aware**: Uses conversation history for better responses

### Search & Retrieval
- **Vector Store**: ChromaDB with sentence-transformer embeddings
- **Smart Search**: Course name and lesson number filtering
- **Source Tracking**: Maintains attribution for responses

## Data Flow Summary

1. **Input**: User query → Frontend validation
2. **Transport**: HTTP POST → API endpoint  
3. **Context**: Session history → AI context
4. **Intelligence**: Claude analysis → Tool decision
5. **Retrieval**: Semantic search → Relevant content
6. **Synthesis**: AI generation → Coherent response
7. **Attribution**: Source tracking → UI display
8. **Output**: Formatted response → User interface

## Key Features

- **Adaptive Search**: AI decides when course content search is needed
- **Context Preservation**: Session-based conversation memory
- **Source Attribution**: Tracks and displays content sources
- **Error Resilience**: Graceful handling at each stage
- **Performance**: Efficient vector search and caching