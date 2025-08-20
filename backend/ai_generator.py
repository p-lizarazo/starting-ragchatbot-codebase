from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- **Content Search Tool**: Use for questions about specific course content, lessons, or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, outlines, lesson lists, or course overview
- **Sequential Tool Usage**: You can make multiple tool calls across separate reasoning steps (maximum 2 rounds)
- **Tool Strategy**: Use tools sequentially to gather information, then reason about results
- **Multi-step Queries**: For complex questions, break into logical steps:
  1. First round: Gather initial information (e.g., get course outline, search specific content)
  2. Second round: Search for additional context or comparisons if needed
- **Reasoning Between Calls**: After each tool result, assess if additional information is needed
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Examples of multi-step usage:
- "Compare topic X between courses A and B" → outline course A → search course B for topic X
- "Find courses covering same topic as lesson N of course Y" → outline course Y → search for topic

Course Outline Responses:
When using the course outline tool, include in your response:
- Course title and link
- Course instructor
- Complete lesson list with lesson numbers and titles
- Lesson links when available

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use appropriate tools strategically
- **Complex queries**: Use multiple tool rounds if beneficial
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def _build_system_content(self, conversation_history: Optional[str] = None) -> str:
        """Build system content with optional conversation history"""
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

    def _make_api_call(
        self, messages: List[Dict], system_content: str, tools: Optional[List] = None
    ):
        """Make a single API call with consistent parameters"""
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)

    def _process_tool_round(self, response, messages: List[Dict], tool_manager):
        """
        Process tool execution and update conversation state.
        Returns updated messages list for next round.
        """
        # Add assistant's tool use response to conversation
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Graceful error handling - continue with error message
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {str(e)}",
                        }
                    )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with sequential tool calling support.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Initialize conversation state
        messages = [{"role": "user", "content": query}]
        system_content = self._build_system_content(conversation_history)

        # Sequential tool calling loop
        for round_num in range(max_rounds):
            # Make API call with tools available
            response = self._make_api_call(messages, system_content, tools)

            # Check termination conditions
            if response.stop_reason != "tool_use" or not tool_manager:
                # Natural termination - Claude doesn't want to use tools or no tool manager
                return response.content[0].text

            # Process tool execution and update conversation state
            try:
                messages = self._process_tool_round(response, messages, tool_manager)
            except Exception as e:
                # If tool processing fails completely, make final call without tools
                final_response = self._make_api_call(
                    messages, system_content, tools=None
                )
                return final_response.content[0].text

        # Max rounds reached - make final call without tools
        final_response = self._make_api_call(messages, system_content, tools=None)
        return final_response.content[0].text
