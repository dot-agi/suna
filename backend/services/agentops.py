"""
AgentOps integration service for Suna platform.

This service provides a clean, optional integration with AgentOps tracing
that coexists with the existing Langfuse tracing without conflicts.
"""

import os
import threading
from functools import wraps
from typing import Optional, Dict, Any, List, Union
from utils.logger import logger
from contextvars import ContextVar

try:
    import agentops
    from agentops.sdk.core import TraceContext
    from agentops.sdk.decorators import trace, agent, task, tool
    from agentops.semconv import (
        AgentAttributes,
        ToolAttributes, 
        CoreAttributes,
        SpanAttributes
    )
    from agentops.semconv.span_kinds import AgentOpsSpanKindValues, SpanKind
    from agentops.enums import TraceState
    AGENTOPS_AVAILABLE = True
    logger.info("AgentOps SDK successfully imported")
except ImportError:
    logger.info("AgentOps not available - integration disabled")
    AGENTOPS_AVAILABLE = False
    agentops = None
    TraceContext = None
    trace = None
    agent = None
    task = None
    tool = None
    AgentAttributes = None
    ToolAttributes = None
    CoreAttributes = None
    SpanAttributes = None
    AgentOpsSpanKindValues = None
    SpanKind = None
    TraceState = None


class AgentOpsService:
    """
    AgentOps integration service that provides optional tracing capabilities.
    
    This service is designed to be:
    - Optional: Only enabled when AGENTOPS_API_KEY is set
    - Non-invasive: Uses decorators to keep core logic clean
    - Compatible: Coexists with existing Langfuse tracing
    - Thread-safe: Handles concurrent access properly
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AgentOpsService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.api_key = os.getenv("AGENTOPS_API_KEY")
        self.log_level = os.getenv("AGENTOPS_LOG_LEVEL", "CRITICAL")
        self.enabled = bool(self.api_key and AGENTOPS_AVAILABLE)
        self._initialized = False
        
        # Use context variables for thread-safe context management
        self._context_var: ContextVar[Optional[TraceContext]] = ContextVar('agentops_context', default=None)
        
        # Mark as initialized to prevent re-initialization
        self._initialized = True
        
        if not AGENTOPS_AVAILABLE and self.api_key:
            logger.warning("AGENTOPS_API_KEY is set but AgentOps is not installed. Run: pip install agentops")
        
        logger.info(f"AgentOps service initialized - enabled: {self.enabled}")
    
    def init(self) -> bool:
        """Initialize AgentOps SDK."""
        if not self.enabled:
            logger.debug("AgentOps service disabled - skipping initialization")
            return False
            
        try:
            agentops.init(
                api_key=self.api_key,
                log_level=self.log_level,
                auto_start_session=False,  # We'll manage sessions manually
                instrument_llm_calls=False,  # We have our own LLM instrumentation
                fail_safe=True,  # Don't break the app if AgentOps fails
            )
            logger.info("AgentOps SDK initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AgentOps SDK: {e}")
            self.enabled = False
            return False
    
    def start_session_trace(
        self, 
        thread_id: str, 
        project_id: str, 
        user_id: str, 
        tags: Optional[List[str]] = None
    ) -> Optional[TraceContext]:
        """Start a new AgentOps session trace."""
        if not self.enabled:
            return None
            
        try:
            # Prepare tags for the session
            session_tags = [thread_id, project_id, user_id]

            if tags:
                session_tags.extend(tags)
                
            # Start trace using AgentOps
            trace_context = agentops.start_trace(
                trace_name="suna_session",
                tags=session_tags
            )
            
            if trace_context:
                logger.info(f"Started AgentOps session trace for thread {thread_id}")
                self.set_context(trace_context)
                return trace_context
            else:
                logger.warning("Failed to start AgentOps session trace")
                return None
                
        except Exception as e:
            logger.error(f"Error starting AgentOps session trace: {e}")
            return None
    
    def end_session_trace(
        self, 
        trace_context: Optional[TraceContext] = None, 
        end_state: str = "Success"
    ) -> None:
        """End an AgentOps session trace."""
        if not self.enabled:
            return
            
        try:
            # Convert end_state to AgentOps format
            if end_state.lower() == "success":
                state = TraceState.SUCCESS
            elif end_state.lower() == "error":
                state = TraceState.ERROR
            else:
                state = TraceState.UNSET
            
            agentops.end_trace(trace_context=trace_context, end_state=state)
            
            # Clear current context if this was the current one
            if trace_context:
                current_context = self.get_context()
                if current_context == trace_context:
                    self.set_context(None)
                        
            logger.info("Ended AgentOps session trace")
        except Exception as e:
            logger.error(f"Error ending AgentOps session trace: {e}")
    
    def set_context(self, trace_context: Optional[TraceContext]) -> None:
        """Set the current trace context for this thread."""
        self._context_var.set(trace_context)
    
    def get_context(self) -> Optional[TraceContext]:
        """Get the current trace context for this thread."""
        return self._context_var.get()
    
    def serialize_context(self, trace_context: Optional[TraceContext] = None) -> Optional[Dict[str, Any]]:
        """Serialize trace context for transmission across process boundaries."""
        if not self.enabled:
            return None
            
        context_to_serialize = trace_context or self.get_context()
        if not context_to_serialize:
            return None
            
        try:
            # Convert TraceContext to serializable dictionary
            serialized = {
                "trace_id": getattr(context_to_serialize, 'trace_id', None),
                "session_id": getattr(context_to_serialize, 'session_id', None),
                "span_id": getattr(context_to_serialize, 'span_id', None),
                "parent_span_id": getattr(context_to_serialize, 'parent_span_id', None),
                "metadata": getattr(context_to_serialize, 'metadata', {}),
                "tags": getattr(context_to_serialize, 'tags', {}),
                "agentops_service_enabled": True
            }
            
            logger.debug(f"Serialized AgentOps context: {serialized}")
            return serialized
        except Exception as e:
            logger.error(f"Error serializing AgentOps context: {e}")
            return None
    
    def deserialize_context(self, serialized_context: Optional[Dict[str, Any]]) -> Optional[TraceContext]:
        """Deserialize trace context from serialized data."""
        if not self.enabled or not serialized_context:
            return None
            
        if not serialized_context.get("agentops_service_enabled"):
            return None
            
        try:
            logger.debug(f"Attempting to deserialize AgentOps context: {serialized_context}")
            
            # Try to start a new trace with the serialized metadata to maintain continuity
            trace_id = serialized_context.get("trace_id")
            session_id = serialized_context.get("session_id")
            tags = serialized_context.get("tags", {})
            metadata = serialized_context.get("metadata", {})
            
            if trace_id or session_id:
                # Start a new trace with the preserved metadata
                trace_name = f"worker_continuation_{session_id or trace_id}"
                
                # Merge tags and metadata for the new trace
                combined_tags = {**tags, **metadata}
                if trace_id:
                    combined_tags["original_trace_id"] = trace_id
                if session_id:
                    combined_tags["original_session_id"] = session_id
                
                trace_context = agentops.start_trace(
                    trace_name=trace_name,
                    tags=combined_tags
                )
                
                if trace_context:
                    logger.info(f"Successfully reconstructed AgentOps trace context in worker")
                    return trace_context
                else:
                    logger.warning("Failed to start new trace for context reconstruction")
                    return None
            else:
                logger.warning("No trace_id or session_id found in serialized context")
                return None
                
        except Exception as e:
            logger.error(f"Error deserializing AgentOps context: {e}")
            return None
    
    def record_event(self, name: str, level: str = "DEFAULT", status_message: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an event in AgentOps if a trace context is available.
        
        This function records events as metadata in the current trace context,
        providing parity with Langfuse event tracking.
        
        Args:
            name: Event name (e.g., "billing_limit_reached")
            level: Event level (DEFAULT, WARNING, ERROR, CRITICAL)
            status_message: Event message/status message
            metadata: Optional metadata dictionary
        """
        # Get trace context
        trace_context = self.get_context()
        
        if not self.enabled or not trace_context:
            return
        
        try:
            # Build event attributes with correct span kind
            event_attributes = {
                "event.name": name,
                "event.level": level,
                "event.message": status_message or name,
                "agentops.span.kind": SpanKind.OPERATION if SpanKind else "operation"
            }
            
            # Map levels to severity for better observability
            severity_map = {
                "DEFAULT": "INFO",
                "WARNING": "WARN",
                "ERROR": "ERROR",
                "CRITICAL": "FATAL"
            }
            event_attributes["severity"] = severity_map.get(level, "INFO")
            
            # Add metadata as attributes if provided
            if metadata:
                for key, value in metadata.items():
                    # Prefix metadata keys to avoid conflicts
                    if isinstance(value, (dict, list)):
                        import json
                        event_attributes[f"event.metadata.{key}"] = json.dumps(value)
                    else:
                        event_attributes[f"event.metadata.{key}"] = str(value)
            
            # For ERROR and CRITICAL levels, also set error attributes
            if level in ["ERROR", "CRITICAL"]:
                event_attributes["error"] = True
                event_attributes["error.message"] = status_message
            
            # Update trace metadata with event attributes
            success = agentops.update_trace_metadata(event_attributes)
            
            if success:
                logger.debug(f"Recorded AgentOps event '{name}' with level '{level}'")
            else:
                logger.warning(f"Failed to update trace metadata for event '{name}'")
            
        except Exception as e:
            logger.error(f"Failed to record AgentOps event '{name}': {str(e)}")
    
    def session_trace(self, name: Optional[str] = None):
        """Decorator for session-level tracing."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Check if enabled at runtime, not import time
                if not self.enabled or not AGENTOPS_AVAILABLE:
                    return await func(*args, **kwargs)
                
                try:
                    # Set span attributes with session span kind
                    self.set_span_attributes({
                        "agentops.span.kind": SpanKind.SESSION if SpanKind else "session",
                        "session.function": func.__name__
                    })
                    
                    # Record session start
                    self.record_event("session_start", {"function": func.__name__})
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record error in trace metadata
                    self.record_event("error", {"error_type": type(e).__name__, "error_message": str(e)})
                    raise
            return wrapper
        return decorator
    
    def agent_span(self, name: Optional[str] = None):
        """Decorator for agent-level spans."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Check if enabled at runtime, not import time
                if not self.enabled or not AGENTOPS_AVAILABLE:
                    return await func(*args, **kwargs)
                
                try:
                    # Set span attributes with agent span kind
                    self.set_span_attributes({
                        "agentops.span.kind": SpanKind.AGENT if SpanKind else "agent",
                        "agent.function": func.__name__
                    })
                    
                    # Apply AgentOps agent decorator at runtime if available
                    if agent:
                        decorated_func = agent(name=name or func.__name__)(func)
                        result = await decorated_func(*args, **kwargs)
                    else:
                        result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record error in trace metadata
                    self.record_event("agent_error", {"error_type": type(e).__name__, "error_message": str(e)})
                    raise
            return wrapper
        return decorator
    
    def thread_span(self, name: Optional[str] = None):
        """Decorator for thread-level spans."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Check if enabled at runtime, not import time
                if not self.enabled or not AGENTOPS_AVAILABLE:
                    return await func(*args, **kwargs)
                
                try:
                    # Set span attributes with task span kind (threads represent task execution)
                    self.set_span_attributes({
                        "agentops.span.kind": SpanKind.TASK if SpanKind else "task",
                        "task.function": func.__name__
                    })
                    
                    # Record thread start
                    self.record_event("thread_start", {"function": func.__name__})
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record error in trace metadata
                    self.record_event("thread_error", {"error_type": type(e).__name__, "error_message": str(e)})
                    raise
            return wrapper
        return decorator

    def llm_span(self, name: Optional[str] = None):
        """Decorator for LLM call spans."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Check if enabled at runtime, not import time
                if not self.enabled or not AGENTOPS_AVAILABLE:
                    return await func(*args, **kwargs)
                
                try:
                    # Extract model info from kwargs if available
                    model_name = kwargs.get('model_name') or kwargs.get('model')
                    
                    # Set span attributes with LLM span kind
                    self.set_span_attributes({
                        "agentops.span.kind": SpanKind.LLM if SpanKind else "llm",
                        "llm.function": func.__name__
                    })
                    
                    if model_name:
                        self.record_event("llm_call", {
                            "model_name": model_name,
                            "function": func.__name__
                        })
                    
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record error in trace metadata
                    self.record_event("llm_error", {
                        "error_type": type(e).__name__, 
                        "error_message": str(e),
                        "model_name": kwargs.get('model_name') or kwargs.get('model', 'unknown')
                    })
                    raise
            return wrapper
        return decorator

    def tool_span(self, name: Optional[str] = None):
        """Decorator for tool execution spans."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Check if enabled at runtime, not import time
                if not self.enabled or not AGENTOPS_AVAILABLE:
                    return await func(*args, **kwargs)
                
                try:
                    # Extract tool info from kwargs if available
                    tool_name = kwargs.get('tool_name') or func.__name__
                    
                    # Set span attributes with tool span kind
                    self.set_span_attributes({
                        "agentops.span.kind": SpanKind.TOOL if SpanKind else "tool",
                        "tool.function": func.__name__
                    })
                    
                    # Set tool attributes using semantic conventions
                    if ToolAttributes:
                        self.set_span_attributes({
                            ToolAttributes.TOOL_NAME: tool_name,
                            ToolAttributes.TOOL_ID: kwargs.get('tool_id', tool_name),
                        })
                    
                    if SpanAttributes:
                        self.set_span_attributes({
                            SpanAttributes.OPERATION_NAME: func.__name__
                        })
                    
                    # Record tool call event
                    self.record_event("tool_call", {
                        "tool_name": tool_name,
                        "function": func.__name__
                    })
                    
                    result = await func(*args, **kwargs)
                    
                    # Set result if available
                    if result and ToolAttributes:
                        self.set_span_attributes({
                            ToolAttributes.TOOL_RESULT: str(result)[:1000],  # Truncate large results
                            ToolAttributes.TOOL_STATUS: "success"
                        })
                    
                    return result
                except Exception as e:
                    # Set error attributes
                    if ToolAttributes:
                        self.set_span_attributes({
                            ToolAttributes.TOOL_STATUS: "error",
                            "error.type": type(e).__name__,
                            "error.message": str(e)
                        })
                    
                    # Record error event
                    self.record_event("tool_error", {
                        "error_type": type(e).__name__, 
                        "error_message": str(e),
                        "tool_name": kwargs.get('tool_name', func.__name__)
                    })
                    raise
            return wrapper
        return decorator
    
    def set_span_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set attributes on the current span using semantic conventions."""
        if not self.enabled or not attributes:
            return
            
        try:
            # Filter out None values and convert to proper types
            filtered_attributes = {}
            for key, value in attributes.items():
                if value is not None:
                    # Convert complex objects to strings
                    if isinstance(value, (dict, list)):
                        import json
                        filtered_attributes[key] = json.dumps(value)
                    elif not isinstance(value, (str, int, float, bool)):
                        filtered_attributes[key] = str(value)
                    else:
                        filtered_attributes[key] = value
            
            if filtered_attributes:
                success = agentops.update_trace_metadata(filtered_attributes)
                if success:
                    logger.debug(f"Set span attributes: {list(filtered_attributes.keys())}")
                else:
                    logger.warning(f"Failed to set span attributes: {list(filtered_attributes.keys())}")
                    
        except Exception as e:
            logger.error(f"Error setting span attributes: {e}")
    
    def set_agent_attributes(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None, 
                            agent_role: Optional[str] = None, tools: Optional[List[str]] = None) -> None:
        """Set agent-specific attributes using semantic conventions."""
        if not self.enabled:
            return
            
        attributes = {}
        if agent_id and AgentAttributes:
            attributes[AgentAttributes.AGENT_ID] = agent_id
        if agent_name and AgentAttributes:
            attributes[AgentAttributes.AGENT_NAME] = agent_name
        if agent_role and AgentAttributes:
            attributes[AgentAttributes.AGENT_ROLE] = agent_role
        if tools and AgentAttributes:
            attributes[AgentAttributes.AGENT_TOOLS] = tools
            
        if attributes:
            self.set_span_attributes(attributes)
    
    def set_llm_attributes(self, model: Optional[str] = None, temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None, system_prompt: Optional[str] = None) -> None:
        """Set LLM-specific attributes using semantic conventions."""
        if not self.enabled:
            return
            
        attributes = {}
        if model and SpanAttributes:
            attributes[SpanAttributes.LLM_REQUEST_MODEL] = model
            attributes[SpanAttributes.LLM_SYSTEM] = "suna"
        if temperature is not None and SpanAttributes:
            attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] = temperature
        if max_tokens and SpanAttributes:
            attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = max_tokens
        if system_prompt and SpanAttributes:
            # Truncate system prompt to avoid huge attributes
            attributes[SpanAttributes.LLM_REQUEST_SYSTEM_INSTRUCTION] = system_prompt[:500]
            
        if attributes:
            self.set_span_attributes(attributes)


# Global service instance
agentops_service = AgentOpsService()
