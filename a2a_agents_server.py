#!/usr/bin/env python3
"""
A2A Multi-Agent Customer Support Server
=======================================

A comprehensive multi-agent customer support system using the official A2A SDK.
This server orchestrates specialized AI agents for intelligent customer support operations.

Architecture:
- Triage Agent: Classifies and routes customer requests by priority and type
- Email Agent: Handles email operations and customer communications
- Documentation Agent: Manages ticket logging and knowledge base operations
- Resolution Agent: Generates solutions using vector RAG (FAQ + KB search)
- Escalation Agent: Handles complex issues requiring human intervention

Features:
- Vector-powered FAQ and Knowledge Base search with similarity ranking
- Customer intelligence via email history analysis for personalized responses
- Automatic escalation detection and notification system
- Comprehensive ticket logging to Google Sheets
- LangSmith observability integration for workflow tracing
- Environment-based configuration for multi-deployment support

Environment Variables Required:
- PORT: Server port (default: 8003)
- A2A_SERVER_URL: Public URL for agent card (for production deployment)
- MCP_GMAIL_URL: Gmail MCP server URL for email operations
- MCP_DRIVE_URL: Drive MCP server URL for document operations
- OPENAI_API_KEY: OpenAI API key for LLM and embedding operations
- DEFAULT_CUSTOMER_EMAIL: Fallback customer email for testing
- ESCALATION_EMAIL: Email address for urgent escalations
- SUPPORT_EMAIL: General support team email address
- LANGSMITH_API_KEY: (Optional) LangSmith API key for tracing
- LANGSMITH_PROJECT: (Optional) LangSmith project name
- LANGSMITH_ENDPOINT: (Optional) LangSmith endpoint URL

Usage:
    python a2a_agents_server.py

Deployment:
    Render Cloud Platform (Port 8003)
    Requires MCP Gmail and Drive servers to be deployed first

Author: Multi-Agent Customer Support System
Version: 2.0.0
"""

# Standard library imports
import os
import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Third-party imports
from pydantic import BaseModel, Field
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing_extensions import override

# Local imports
from vector_rag_system import get_rag_system  # Vector RAG system for FAQ/KB search

# A2A SDK imports
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.utils import new_agent_text_message
from starlette.responses import JSONResponse

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# LANGSMITH OBSERVABILITY INTEGRATION
# =============================================================================
# Optional LangSmith integration for end-to-end tracing and observability.
# Provides detailed workflow tracking for debugging and performance monitoring.
# Falls back gracefully if LangSmith is not installed or configured.

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    # Graceful fallback: Create no-op decorator if LangSmith not available
    def traceable(name=None):
        """
        No-op decorator when LangSmith is not available.
        
        Args:
            name (str): Name of the function to be traced (optional)
        
        Returns:
            function: Original function if LangSmith is not available
        """
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False

# Task and Agent Models
class TaskStatus(Enum):
    """Task processing status enumeration"""
    PENDING = "pending"        # Task created, awaiting processing
    PROCESSING = "processing"  # Task currently being processed by agent
    COMPLETED = "completed"    # Task successfully completed
    ESCALATED = "escalated"    # Task escalated to human specialist
    FAILED = "failed"          # Task failed during processing

class IssueType(Enum):
    """Customer issue type classification"""
    TECHNICAL = "technical"    # Technical support issues (bugs, errors, how-to)
    BILLING = "billing"        # Billing and payment related issues
    GENERAL = "general"        # General inquiries and information requests
    ESCALATION = "escalation"  # Issues requiring immediate escalation

class Priority(Enum):
    """Issue priority level classification"""
    LOW = "low"              # Non-urgent, can wait 24-48 hours
    MEDIUM = "medium"        # Standard priority, respond within 12 hours
    HIGH = "high"            # Important, respond within 4 hours
    URGENT = "urgent"        # Critical, immediate response required

@dataclass
class CustomerSupportTask:
    """
    Customer support task data model.
    
    Represents a complete customer support request with all necessary metadata
    for tracking, processing, and resolution by the multi-agent system.
    
    Attributes:
        task_id (str): Unique identifier for the task (e.g., "TRIAGE-abc123")
        customer_name (str): Customer's full name
        customer_email (str): Customer's email address (extracted from request context)
        request_type (str): Type of request (technical, billing, general, escalation)
        priority (str): Priority level (low, medium, high, urgent)
        description (str): Full description of the customer's issue or request
        status (str): Current processing status (pending, processing, completed, escalated)
        assigned_agent (str): Name of the agent currently handling the task
        created_at (str): ISO timestamp when the task was created
        updated_at (str): ISO timestamp when the task was last updated
        resolution (Optional[str]): Final resolution or solution provided to customer
        context_data (Optional[Dict[str, Any]]): Additional context and metadata
    """
    task_id: str
    customer_name: str
    customer_email: str  # Extracted from request context, fallback to DEFAULT_CUSTOMER_EMAIL
    request_type: str
    priority: str
    description: str
    status: str
    assigned_agent: str
    created_at: str
    updated_at: str
    resolution: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None

# =============================================================================
# MCP CLIENT FOR EXTERNAL SERVICE COMMUNICATION
# =============================================================================
# Client for communicating with Model Context Protocol (MCP) servers.
# Handles communication with Gmail and Drive MCP servers for email operations
# and document management.

class MCPClient:
    """
    Client for communicating with MCP (Model Context Protocol) servers.
    
    This client handles all communication with external MCP servers including:
    - Gmail MCP Server: Email operations (send, search, retrieve)
    - Drive MCP Server: Document operations (sheets, docs, drive files)
    
    Environment Variables Required:
    - MCP_GMAIL_URL: URL of the Gmail MCP server (e.g., https://gmail-server.onrender.com)
    - MCP_DRIVE_URL: URL of the Drive MCP server (e.g., https://drive-server.onrender.com)
    
    Attributes:
        gmail_url (str): Gmail MCP server URL from environment
        drive_url (str): Drive MCP server URL from environment
        client (httpx.AsyncClient): HTTP client for making requests
    """
    
    def __init__(self):
        """
        Initialize MCP client with server URLs from environment variables.
        
        Raises:
            ValueError: If required environment variables are not set
        """
        # Get MCP server URLs from environment variables
        self.gmail_url = os.getenv('MCP_GMAIL_URL')
        self.drive_url = os.getenv('MCP_DRIVE_URL')
        
        # Validate required environment variables
        if not self.gmail_url:
            raise ValueError(
                "MCP_GMAIL_URL environment variable is required. "
                "Set to your Gmail MCP server URL (e.g., https://gmail-server.onrender.com)"
            )
        if not self.drive_url:
            raise ValueError(
                "MCP_DRIVE_URL environment variable is required. "
                "Set to your Drive MCP server URL (e.g., https://drive-server.onrender.com)"
            )
        
        # Initialize HTTP client with appropriate timeout
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def call_mcp_tool(self, server_url: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.
        
        Makes an HTTP POST request to the specified MCP server to execute a tool
        with the provided parameters. Used for operations like sending emails,
        updating sheets, or searching documents.
        
        Args:
            server_url (str): Base URL of the MCP server
            tool_name (str): Name of the tool to execute
            **kwargs: Tool-specific parameters to pass in the request body
            
        Returns:
            Dict[str, Any]: Response from the MCP server tool execution
            
        Example:
            await client.call_mcp_tool(
                "https://gmail-server.onrender.com",
                "send_email",
                to="customer@example.com",
                subject="Support Response",
                body="Thank you for contacting us..."
            )
        """
        try:
            response = await self.client.post(
                f"{server_url}/tools/{tool_name}",
                json=kwargs
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error calling MCP tool {tool_name} on {server_url}: {e}")
            return {"error": str(e)}
    
    async def get_mcp_resource(self, server_url: str, resource_uri: str) -> str:
        """
        Get a resource from an MCP server.
        
        Makes an HTTP GET request to retrieve a resource (like documents, sheets,
        or email data) from the specified MCP server using a resource URI.
        
        Args:
            server_url (str): Base URL of the MCP server
            resource_uri (str): URI of the resource to retrieve
            
        Returns:
            str: Resource content as text (CSV for sheets, markdown for docs)
            
        Example:
            content = await client.get_mcp_resource(
                "https://drive-server.onrender.com",
                "sheet://tickets/active"
            )
        """
        try:
            response = await self.client.get(
                f"{server_url}/resources",
                params={"uri": resource_uri}
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"❌ Error getting MCP resource {resource_uri} from {server_url}: {e}")
            return f"Error: {str(e)}"

# Initialize MCP client
mcp_client = MCPClient()

# Setup LangSmith tracing
def setup_langsmith():
    """Setup LangSmith tracing if credentials are available"""
    if LANGSMITH_AVAILABLE:
        langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
        langsmith_project = os.getenv('LANGSMITH_PROJECT', 'MCP-A2A-Agents')
        
        if langsmith_api_key:
            os.environ['LANGSMITH_TRACING'] = 'true'
            os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
            os.environ['LANGSMITH_API_KEY'] = langsmith_api_key
            os.environ['LANGSMITH_PROJECT'] = langsmith_project
            print(f"✅ LangSmith tracing enabled for A2A server: {langsmith_project}")
        else:
            print("⚠️  LangSmith API key not found, A2A tracing disabled")
    else:
        print("⚠️  LangSmith not installed, A2A tracing disabled")

# Initialize LangSmith
setup_langsmith()

# Environment variable configuration
def get_required_env_var(var_name: str, description: str) -> str:
    """Get required environment variable with clear error message"""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} environment variable is required ({description})")
    return value

def get_env_var(var_name: str, default: str = None) -> str:
    """Get optional environment variable with default"""
    return os.getenv(var_name, default)

class TriageAgentExecutor(AgentExecutor):
    """Triage Agent - Request classification and priority routing"""
    
    def __init__(self):
        self.task_store = {}
    
    @override
    @traceable(name="triage_agent_execute")
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute triage and classification"""
        try:
            # Extract message from context
            message_text = ""
            if hasattr(context, 'message') and context.message:
                if hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'text'):
                            message_text += part.text
            
            # Create task
            task_id = f"TASK-{uuid.uuid4().hex[:8]}"
            
            # Classify request
            classification = self._classify_request(message_text)
            
            # Create structured task
            task = CustomerSupportTask(
                task_id=task_id,
                customer_name="Customer",  # Would be extracted from context
                customer_email=get_env_var('DEFAULT_CUSTOMER_EMAIL', 'customer@example.com'),  # Would be extracted from context
                request_type=classification["category"],
                priority=classification["priority"],
                description=message_text,
                status=TaskStatus.PROCESSING.value,
                assigned_agent="triage",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                context_data=classification
            )
            
            # Handle escalation workflow
            if "escalate" in message_text.lower() or "urgent" in message_text.lower():
                # Create escalation ticket
                escalation_data = {
                    "escalation_id": task_id,
                    "original_issue": message_text,
                    "escalated_at": datetime.now().isoformat(),
                    "priority": "urgent",
                    "status": "escalated"
                }
                
                self.task_store[task_id] = escalation_data
                
                # Send escalation email to support team
                await self._send_escalation_email(task_id, message_text)
                
                result = f"🚨 ESCALATED: Issue escalated to specialist team\nEscalation ID: {task_id}\nPriority: Urgent\nStatus: Under specialist review\nEscalation team notified via email"
            else:
                result = "🔍 ESCALATION: Issue reviewed for escalation criteria"
            
            # Store task
            self.task_store[task_id] = asdict(task)
            
            # Route to appropriate agent
            if classification["requires_escalation"]:
                result = f"🚨 ESCALATED: {classification['category'].upper()} issue (Priority: {classification['priority']})\n\nRouting to escalation specialist..."
            elif classification["category"] == "technical":
                result = f"🔧 TECHNICAL: Issue classified as technical support\n\nRouting to technical specialist..."
            elif classification["category"] == "billing":
                result = f"💳 BILLING: Issue classified as billing inquiry\n\nRouting to billing specialist..."
            else:
                result = f"💬 GENERAL: Issue classified as general support\n\nRouting to general support agent..."
            
            # Update task status
            self.task_store[task_id]["status"] = TaskStatus.COMPLETED.value
            self.task_store[task_id]["resolution"] = result
            
            # Send response
            response_message = new_agent_text_message(result)
            await event_queue.put(response_message)
            
        except Exception as e:
            error_message = f"❌ Triage Error: {str(e)}"
            response_message = new_agent_text_message(error_message)
            await event_queue.put(response_message)
    
    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation"""
        cancel_message = new_agent_text_message("Triage cancelled")
        await event_queue.put(cancel_message)
    
    def _classify_request(self, message: str) -> Dict[str, Any]:
        """Classify support request with enhanced logic"""
        message_lower = message.lower()
        
        # Escalation keywords
        if any(word in message_lower for word in ["urgent", "critical", "emergency", "down", "outage", "broken"]):
            return {
                "category": IssueType.ESCALATION.value,
                "priority": Priority.URGENT.value,
                "requires_escalation": True,
                "confidence": 0.9,
                "keywords": ["urgent", "critical", "emergency"]
            }
        
        # Technical keywords
        elif any(word in message_lower for word in ["error", "bug", "crash", "performance", "slow", "api", "integration"]):
            priority = Priority.HIGH.value if any(word in message_lower for word in ["crash", "error", "api"]) else Priority.MEDIUM.value
            return {
                "category": IssueType.TECHNICAL.value,
                "priority": priority,
                "requires_escalation": False,
                "confidence": 0.8,
                "keywords": ["technical", "error", "bug"]
            }
        
        # Billing keywords
        elif any(word in message_lower for word in ["billing", "payment", "invoice", "charge", "refund", "subscription"]):
            return {
                "category": IssueType.BILLING.value,
                "priority": Priority.MEDIUM.value,
                "requires_escalation": False,
                "confidence": 0.8,
                "keywords": ["billing", "payment", "invoice"]
            }
        
        # General support
        else:
            return {
                "category": IssueType.GENERAL.value,
                "priority": Priority.LOW.value,
                "requires_escalation": False,
                "confidence": 0.6,
                "keywords": ["general", "question", "help"]
            }

class EmailAgentExecutor(AgentExecutor):
    """Email Agent - Email operations and communication"""
    
    def __init__(self):
        self.task_store = {}
    
    @override
    @traceable(name="email_agent_execute")
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute email operations"""
        try:
            # Extract email task from context
            message_text = ""
            if hasattr(context, 'message') and context.message:
                if hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'text'):
                            message_text += part.text
            
            # Parse email request (would be more sophisticated in practice)
            task_data = json.loads(message_text) if message_text.startswith('{') else {"action": "send", "content": message_text}
            
            task_id = f"EMAIL-{uuid.uuid4().hex[:8]}"
            
            if task_data.get("action") == "send":
                # Send email via MCP Gmail server
                email_result = await mcp_client.call_mcp_tool(
                    mcp_client.gmail_url,
                    "send_email",
                    to=task_data.get("to", get_env_var('DEFAULT_CUSTOMER_EMAIL', 'customer@example.com')),
                    subject=task_data.get("subject", "Support Response"),
                    body=task_data.get("body", task_data.get("content", "Thank you for contacting support."))
                )
                
                result = f"📧 EMAIL SENT: {email_result.get('status', 'unknown')}\nMessage ID: {email_result.get('message_id', 'N/A')}"
                
            elif task_data.get("action") == "search":
                # Search emails via MCP Gmail server
                search_result = await mcp_client.call_mcp_tool(
                    mcp_client.gmail_url,
                    "search_emails",
                    query=task_data.get("query", ""),
                    max_results=task_data.get("max_results", 5)
                )
                
                emails = search_result if isinstance(search_result, list) else []
                result = f"🔍 EMAIL SEARCH: Found {len(emails)} emails\n\n" + "\n".join([f"- {email.get('subject', 'No Subject')} from {email.get('sender', 'Unknown')}" for email in emails[:3]])
                
            else:
                result = "📧 EMAIL: Unknown action requested"
            
            # Store task result
            self.task_store[task_id] = {
                "id": task_id,
                "action": task_data.get("action", "unknown"),
                "status": "completed",
                "result": result,
                "created_at": datetime.now().isoformat()
            }
            
            response_message = new_agent_text_message(result)
            await event_queue.put(response_message)
            
        except Exception as e:
            error_message = f"❌ Email Agent Error: {str(e)}"
            response_message = new_agent_text_message(error_message)
            await event_queue.put(response_message)
    
    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation"""
        cancel_message = new_agent_text_message("Email operation cancelled")
        await event_queue.put(cancel_message)

class DocumentationAgentExecutor(AgentExecutor):
    """Documentation Agent - Drive/Docs operations and knowledge management"""
    
    def __init__(self):
        self.task_store = {}
    
    @override
    @traceable(name="documentation_agent_execute")
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute documentation operations"""
        try:
            # Extract documentation task from context
            message_text = ""
            if hasattr(context, 'message') and context.message:
                if hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'text'):
                            message_text += part.text
            
            task_data = json.loads(message_text) if message_text.startswith('{') else {"action": "log_ticket", "content": message_text}
            task_id = f"DOC-{uuid.uuid4().hex[:8]}"
            
            if task_data.get("action") == "log_ticket":
                # Log ticket to sheet via MCP Drive server
                ticket_data = {
                    "ticket_id": task_data.get("ticket_id", f"TKT-{uuid.uuid4().hex[:6]}"),
                    "customer_name": task_data.get("customer_name", "Customer"),
                    "customer_email": task_data.get("customer_email", get_env_var('DEFAULT_CUSTOMER_EMAIL', 'customer@example.com')),
                    "issue_type": task_data.get("issue_type", "general"),
                    "priority": task_data.get("priority", "medium"),
                    "description": task_data.get("description", task_data.get("content", "")),
                    "status": "open",
                    "assigned_agent": "support",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                log_result = await mcp_client.call_mcp_tool(
                    mcp_client.drive_url,
                    "log_ticket_to_sheet",
                    ticket_data=ticket_data
                )
                
                result = f"📊 TICKET LOGGED: {log_result.get('status', 'unknown')}\nSheet: {log_result.get('sheet_id', 'N/A')}"
                
            elif task_data.get("action") == "update_faq":
                # Update FAQ document
                faq_result = await mcp_client.call_mcp_tool(
                    mcp_client.drive_url,
                    "update_faq_doc",
                    topic=task_data.get("topic", "General"),
                    content=task_data.get("content", "")
                )
                
                result = f"📝 FAQ UPDATED: {faq_result.get('status', 'unknown')}\nDocument: {faq_result.get('document_id', 'N/A')}"
                
            elif task_data.get("action") == "search_kb":
                # Search knowledge base
                search_result = await mcp_client.call_mcp_tool(
                    mcp_client.drive_url,
                    "search_knowledge_base",
                    query=task_data.get("query", "")
                )
                
                documents = search_result if isinstance(search_result, list) else []
                result = f"🔍 KNOWLEDGE BASE: Found {len(documents)} documents\n\n" + "\n".join([f"- {doc.get('title', 'Untitled')} ({doc.get('word_count', 0)} words)" for doc in documents[:3]])
                
            else:
                result = "📚 DOCUMENTATION: Unknown action requested"
            
            # Store task result
            self.task_store[task_id] = {
                "id": task_id,
                "action": task_data.get("action", "unknown"),
                "status": "completed",
                "result": result,
                "created_at": datetime.now().isoformat()
            }
            
            response_message = new_agent_text_message(result)
            await event_queue.put(response_message)
            
        except Exception as e:
            error_message = f"❌ Documentation Agent Error: {str(e)}"
            response_message = new_agent_text_message(error_message)
            await event_queue.put(response_message)
    
    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation"""
        cancel_message = new_agent_text_message("Documentation operation cancelled")
        await event_queue.put(cancel_message)

class ResolutionAgentExecutor(AgentExecutor):
    """Resolution Agent - RAG-powered solution generation with FAQ/KB search flow"""
    
    def __init__(self):
        self.task_store = {}
    
    @override
    @traceable(name="resolution_agent_execute")
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute RAG-powered solution generation with proper search flow"""
        try:
            # Extract resolution task from context
            message_text = ""
            # Extract customer email from context (from frontend form) with fallback
            customer_email = get_env_var('DEFAULT_CUSTOMER_EMAIL', 'customer@example.com')
            if hasattr(context, 'customer_email') and context.customer_email:
                customer_email = context.customer_email
            elif hasattr(context, 'message') and hasattr(context.message, 'customer_email'):
                customer_email = context.message.customer_email
            if hasattr(context, 'message') and context.message:
                if hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'text'):
                            message_text += part.text
            
            task_id = f"RES-{uuid.uuid4().hex[:8]}"
            
            # Step 0: Get customer context from email history (for personalization)
            customer_context = await self._get_customer_email_context(customer_email, message_text)
            
            # Step 1: Search FAQ first (vectorized similarity search)
            faq_result = await self._search_faq_vectorized(message_text)
            
            if faq_result["found"]:
                solution = faq_result["solution"]
                source = "FAQ Database"
                confidence = faq_result["confidence"]
            else:
                # Step 2: Search full knowledge base
                kb_result = await self._search_knowledge_base_full(message_text)
                
                if kb_result["found"]:
                    solution = kb_result["solution"]
                    source = "Knowledge Base"
                    confidence = kb_result["confidence"]
                else:
                    # Step 3: No solution found - escalate
                    escalation_result = await self._trigger_escalation(message_text, task_id)
                    solution = "Issue escalated to specialist team for manual review."
                    source = "Escalation"
                    confidence = 0.0
            
            # Step 3.5: Personalize solution based on customer context
            solution = self._personalize_solution(solution, customer_context, customer_email)
            
            # Step 4: Always send summary email to customer
            await self._send_customer_summary_email(customer_email, message_text, solution, task_id)
            
            # Step 5: Always send status email to support team
            await self._send_support_status_email(message_text, solution, source, confidence, task_id)
            
            # Create enhanced result with customer intelligence
            customer_intel = f"👤 Customer: {customer_context.get('customer_tier', 'standard').title()} ({customer_context.get('previous_interaction_count', 0)} prev. interactions)"
            if customer_context.get('escalation_history'):
                customer_intel += " ⚠️ Has escalation history"
            
            result = f"💡 SOLUTION GENERATED:\n\n{solution}\n\n📚 Source: {source} (Confidence: {confidence:.2f})\n{customer_intel}\n📧 Customer notified via email\n📊 Support team status updated"
            
            # Store task result
            self.task_store[task_id] = {
                "id": task_id,
                "issue": message_text,
                "solution": solution,
                "source": source,
                "confidence": confidence,
                "status": "completed",
                "created_at": datetime.now().isoformat()
            }
            
            response_message = new_agent_text_message(result)
            await event_queue.put(response_message)
            
        except Exception as e:
            error_message = f"❌ Resolution Agent Error: {str(e)}"
            response_message = new_agent_text_message(error_message)
            await event_queue.put(response_message)
    
    async def _search_faq_vectorized(self, query: str) -> Dict[str, Any]:
        """Search FAQ with vector similarity and reranking"""
        try:
            # Get vector RAG system
            rag_system = get_rag_system()
            
            # Search FAQ vector store
            result = await rag_system.search_faq(query)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in FAQ vector search: {e}")
            return {"found": False, "solution": "", "confidence": 0.0, "error": str(e)}
    
    async def _search_knowledge_base_full(self, query: str) -> Dict[str, Any]:
        """Search full knowledge base using vector similarity and reranking"""
        try:
            # Get vector RAG system
            rag_system = get_rag_system()
            
            # Search KB vector store
            result = await rag_system.search_knowledge_base(query)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in KB vector search: {e}")
            return {"found": False, "solution": "", "confidence": 0.0, "error": str(e)}
    
    async def _trigger_escalation(self, issue: str, task_id: str) -> Dict[str, Any]:
        """Trigger escalation when no solution is found"""
        try:
            # Send escalation email
            await mcp_client.call_mcp_tool(
                mcp_client.gmail_url,
                "send_email",
                to="escalation-team@company.com",
                subject=f"ESCALATION: No Solution Found - {task_id}",
                body=f"""ESCALATION REQUIRED

Task ID: {task_id}
Issue: {issue}
Reason: No solution found in FAQ or Knowledge Base
Escalated At: {datetime.now().isoformat()}

Please review and provide manual resolution.

Support System"""
            )
            
            return {"escalated": True, "task_id": task_id}
            
        except Exception as e:
            return {"escalated": False, "error": str(e)}
    
    async def _send_customer_summary_email(self, customer_email: str, issue: str, solution: str, task_id: str):
        """Always send summary email to customer"""
        try:
            await mcp_client.call_mcp_tool(
                mcp_client.gmail_url,
                "send_email",
                to=customer_email,
                subject=f"Support Request Summary - Ticket {task_id}",
                body=f"""Dear Customer,

Thank you for contacting our support team.

Ticket ID: {task_id}
Your Request: {issue}

Resolution:
{solution}

If you have any further questions, please don't hesitate to contact us.

Best regards,
Customer Support Team"""
            )
        except Exception as e:
            print(f"Failed to send customer summary email: {e}")
    
    async def _get_customer_email_context(self, customer_email: str, current_issue: str) -> Dict[str, Any]:
        """Get customer's email history for context and personalization"""
        try:
            # Search for previous emails from this customer
            previous_emails = await mcp_client.call_mcp_tool(
                mcp_client.gmail_url,
                "search_emails",
                query=f"from:{customer_email}",
                max_results=5
            )
            
            # Search for similar issues from any customer (for pattern recognition)
            issue_keywords = self._extract_keywords(current_issue)
            similar_issues = await mcp_client.call_mcp_tool(
                mcp_client.gmail_url,
                "search_emails",
                query=f"subject:({' OR '.join(issue_keywords[:3])})",
                max_results=3
            ) if issue_keywords else []
            
            # Analyze customer context
            context = {
                "is_returning_customer": len(previous_emails) > 0,
                "previous_interaction_count": len(previous_emails),
                "similar_issue_count": len(similar_issues),
                "customer_tier": self._determine_customer_tier(previous_emails),
                "communication_style": self._analyze_communication_style(previous_emails),
                "previous_issues": [email.get('subject', 'No Subject') for email in previous_emails[:3]],
                "escalation_history": self._check_escalation_history(previous_emails)
            }
            
            return context
            
        except Exception as e:
            print(f"⚠️ Error getting customer context: {e}")
            return {
                "is_returning_customer": False,
                "previous_interaction_count": 0,
                "similar_issue_count": 0,
                "customer_tier": "standard",
                "communication_style": "neutral",
                "previous_issues": [],
                "escalation_history": False
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from issue text for similarity search"""
        # Simple keyword extraction (could be enhanced with NLP)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        words = [word.lower().strip('.,!?;:') for word in text.split() if len(word) > 3 and word.lower() not in common_words]
        return list(set(words))[:10]  # Return unique keywords, max 10
    
    def _determine_customer_tier(self, previous_emails: List[Dict]) -> str:
        """Determine customer tier based on interaction history"""
        email_count = len(previous_emails)
        if email_count >= 10:
            return "vip"
        elif email_count >= 5:
            return "premium"
        elif email_count >= 2:
            return "regular"
        else:
            return "standard"
    
    def _analyze_communication_style(self, previous_emails: List[Dict]) -> str:
        """Analyze customer's communication style from previous emails"""
        if not previous_emails:
            return "neutral"
        
        # Simple analysis based on email content patterns
        total_length = sum(len(email.get('snippet', '')) for email in previous_emails)
        avg_length = total_length / len(previous_emails) if previous_emails else 0
        
        if avg_length > 200:
            return "detailed"
        elif avg_length > 100:
            return "moderate"
        else:
            return "brief"
    
    def _check_escalation_history(self, previous_emails: List[Dict]) -> bool:
        """Check if customer has escalation history"""
        escalation_keywords = ['urgent', 'escalate', 'manager', 'supervisor', 'complaint', 'unsatisfied', 'disappointed']
        for email in previous_emails:
            subject = email.get('subject', '').lower()
            snippet = email.get('snippet', '').lower()
            if any(keyword in subject or keyword in snippet for keyword in escalation_keywords):
                return True
        return False
    
    def _personalize_solution(self, base_solution: str, customer_context: Dict[str, Any], customer_email: str) -> str:
        """Personalize solution based on customer context and history"""
        try:
            # Start with base solution
            personalized = base_solution
            
            # Add personalized greeting based on customer tier and history
            if customer_context.get('is_returning_customer'):
                if customer_context.get('customer_tier') == 'vip':
                    greeting = f"Hello again! As one of our valued customers, I want to ensure we resolve this quickly for you."
                elif customer_context.get('customer_tier') == 'premium':
                    greeting = f"Thank you for reaching out again. I see you're a regular customer, so let me help you right away."
                else:
                    greeting = f"Hi there! I see we've helped you before, so let me get this sorted for you."
            else:
                greeting = f"Welcome! I'm here to help you with your request."
            
            # Add context about previous interactions if relevant
            context_note = ""
            if customer_context.get('previous_interaction_count', 0) > 0:
                prev_issues = customer_context.get('previous_issues', [])
                if prev_issues:
                    context_note = f"\n\n*Note: I can see you've contacted us before about: {', '.join(prev_issues[:2])}. If this is related, please let me know!*"
            
            # Add escalation sensitivity for customers with escalation history
            escalation_note = ""
            if customer_context.get('escalation_history'):
                escalation_note = f"\n\n*I want to make sure we get this resolved properly for you. If you need any additional assistance, please don't hesitate to ask.*"
            
            # Adjust communication style based on customer preference
            communication_style = customer_context.get('communication_style', 'neutral')
            if communication_style == 'detailed':
                style_note = f"\n\nI've provided a comprehensive solution above. If you need any clarification on any of the steps, please let me know."
            elif communication_style == 'brief':
                style_note = f"\n\nLet me know if you need any clarification on these steps."
            else:
                style_note = f"\n\nPlease feel free to reach out if you have any questions about this solution."
            
            # Add similar issue intelligence
            pattern_note = ""
            if customer_context.get('similar_issue_count', 0) > 1:
                pattern_note = f"\n\n*This appears to be a common issue we've been helping customers with recently, so the solution above should work well for you.*"
            
            # Combine all personalization elements
            personalized_solution = f"{greeting}\n\n{personalized}{context_note}{escalation_note}{style_note}{pattern_note}"
            
            return personalized_solution
            
        except Exception as e:
            print(f"⚠️ Error personalizing solution: {e}")
            return base_solution  # Return original solution if personalization fails
    
    async def _send_support_status_email(self, issue: str, solution: str, source: str, confidence: float, task_id: str):
        """Always send status email to support team"""
        try:
            await mcp_client.call_mcp_tool(
                mcp_client.gmail_url,
                "send_email",
                to="support-team@company.com",
                subject=f"Ticket Processed - {task_id}",
                body=f"""TICKET PROCESSING SUMMARY

Ticket ID: {task_id}
Processed At: {datetime.now().isoformat()}

Customer Issue:
{issue}

Resolution Provided:
{solution}

Resolution Source: {source}
Confidence Score: {confidence:.2f}

Status: {'Escalated' if source == 'Escalation' else 'Resolved'}

Support System"""
            )
        except Exception as e:
            print(f"Failed to send support status email: {e}")
    
    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation"""
        cancel_message = new_agent_text_message("Resolution generation cancelled")
        await event_queue.put(cancel_message)
    
    def _generate_solution(self, issue: str, kb_content: str, recent_tickets: str) -> str:
        """Generate solution based on issue and knowledge base"""
        issue_lower = issue.lower()
        
        # Simple keyword-based solution generation (would use LLM in practice)
        if "login" in issue_lower or "password" in issue_lower:
            return "Please try resetting your password using the 'Forgot Password' link on the login page. If the issue persists, please check your email for the reset instructions."
        elif "slow" in issue_lower or "performance" in issue_lower:
            return "Performance issues can often be resolved by clearing your browser cache and cookies. Please also check your internet connection and try using a different browser."
        elif "error" in issue_lower:
            return "Please provide the exact error message you're seeing. In the meantime, try refreshing the page and ensuring you have the latest version of your browser."
        elif "billing" in issue_lower or "payment" in issue_lower:
            return "For billing inquiries, please check your account settings and recent transactions. If you need a refund or have payment issues, our billing team will review your account."
        else:
            return "Thank you for contacting support. We've reviewed your request and will provide a detailed response shortly. Please check our FAQ section for immediate answers to common questions."

class EscalationAgentExecutor(AgentExecutor):
    """Escalation Agent - Complex issue handling and human handoff"""
    
    def __init__(self):
        self.task_store = {}
    
    @override
    @traceable(name="escalation_agent_execute")
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute escalation handling"""
        try:
            # Extract escalation task from context
            message_text = ""
            if hasattr(context, 'message') and context.message:
                if hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'text'):
                            message_text += part.text
            
            task_id = f"ESC-{uuid.uuid4().hex[:8]}"
            
            # Create escalation ticket
            escalation_data = {
                "ticket_id": task_id,
                "customer_name": "Customer",  # Would be extracted from context
                "customer_email": "customer@example.com",
                "issue_type": "escalation",
                "priority": "urgent",
                "description": f"ESCALATED ISSUE: {message_text}",
                "status": "escalated",
                "assigned_agent": "escalation_team",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Log escalation to sheet
            log_result = await mcp_client.call_mcp_tool(
                mcp_client.drive_url,
                "log_ticket_to_sheet",
                ticket_data=escalation_data
            )
            
            # Send escalation notification email
            email_result = await mcp_client.call_mcp_tool(
                mcp_client.gmail_url,
                "send_email",
                to="escalation-team@company.com",
                subject=f"URGENT: Escalated Support Ticket {task_id}",
                body=f"""URGENT ESCALATION REQUIRED

Ticket ID: {task_id}
Customer Issue: {message_text}
Escalated At: {datetime.now().isoformat()}

Please review and respond within 1 hour.

Support System"""
            )
            
            result = f"🚨 ESCALATED: Issue has been escalated to specialist team\n\nTicket ID: {task_id}\nExpected Response: Within 1 hour\nNotification Sent: {email_result.get('status', 'unknown')}"
            
            # Store escalation result
            self.task_store[task_id] = {
                "id": task_id,
                "issue": message_text,
                "status": "escalated",
                "escalated_at": datetime.now().isoformat(),
                "notification_sent": email_result.get('status') == 'success'
            }
            
            response_message = new_agent_text_message(result)
            await event_queue.put(response_message)
            
        except Exception as e:
            error_message = f"❌ Escalation Agent Error: {str(e)}"
            response_message = new_agent_text_message(error_message)
    
    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation"""
        cancel_message = new_agent_text_message("Escalation handling cancelled")
        await event_queue.put(cancel_message)
    
    async def _send_escalation_email(self, escalation_id: str, issue: str):
        """Send escalation notification to support team"""
        try:
            escalation_email = get_env_var('ESCALATION_EMAIL', 'escalation@company.com')
            
            await mcp_client.call_mcp_tool(
                mcp_client.gmail_url,
                "send_email",
                to=escalation_email,
                subject=f"URGENT: Customer Support Escalation - {escalation_id}",
                body=f"""URGENT ESCALATION ALERT

Escalation ID: {escalation_id}
Escalated At: {datetime.now().isoformat()}

Customer Issue:
{issue}

Reason: Complex issue requiring specialist attention
Priority: URGENT

Please review and provide immediate assistance.

Support System"""
            )
        except Exception as e:
            print(f"❌ Error sending escalation email: {e}")

def create_agent_card() -> AgentCard:
    """Create comprehensive Agent Card for multi-agent customer support system"""
    return AgentCard(
        name="Multi-Agent Customer Support System",
        description="Specialized AI agents for comprehensive customer support operations with MCP integration",
        version="2.0.0",
        url=os.getenv('A2A_SERVER_URL', f"http://localhost:{os.getenv('PORT', '8003')}"),

        defaultInputModes=["text"],
        defaultOutputModes=["text"],

        capabilities=AgentCapabilities(
            streaming=True,
            multi_agent=True,
            task_delegation=True
        ),

        skills=[
            AgentSkill(
                id="triage_requests",
                name="Triage Customer Requests",
                description="Classify and route customer requests by priority and type (technical, billing, general, escalation)",
                tags=["triage", "classification", "routing", "priority"],
            ),
            AgentSkill(
                id="email_operations",
                name="Email Operations",
                description="Send customer emails, search email history, and manage email communications",
                tags=["email", "communication", "gmail", "mcp"],
            ),
            AgentSkill(
                id="documentation_management",
                name="Documentation Management",
                description="Log tickets to sheets, update FAQ documents, and manage knowledge base",
                tags=["documentation", "tickets", "faq", "knowledge_base", "drive", "mcp"],
            ),
            AgentSkill(
                id="solution_generation",
                name="Solution Generation",
                description="Generate solutions based on knowledge base and historical ticket data",
                tags=["resolution", "solutions", "knowledge_base", "ai"],
            ),
            AgentSkill(
                id="escalation_handling",
                name="Escalation Handling",
                description="Handle complex issues requiring human intervention and specialist attention",
                tags=["escalation", "urgent", "specialist", "human_handoff"],
            )
        ]
    )

def main():
    """Main function to start A2A multi-agent server"""
    # Create agent card
    agent_card = create_agent_card()
    
    # Create task store
    task_store = InMemoryTaskStore()
    
    # Create request handler with triage agent as default
    request_handler = DefaultRequestHandler(
        agent_executor=TriageAgentExecutor(),
        task_store=task_store
    )
    
    # Create A2A Starlette application
    app_builder = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    # Build the app
    app = app_builder.build()
    
    # Health check endpoint
    @app.route("/health", methods=["GET"])
    async def health_check(request):
        return JSONResponse({"status": "healthy", "service": "A2A Multi-Agent Customer Support"})

    # Vector store management endpoints
    @app.route("/vector-store/faq/initialize", methods=["POST"])
    async def initialize_faq_vectorstore(request):
        """Initialize FAQ vector store from MCP Drive FAQ document"""
        try:
            # Get FAQ content from MCP Drive
            faq_content = await mcp_client.get_mcp_resource(
                mcp_client.drive_url,
                "doc://faq/all"
            )
            
            # Initialize FAQ vector store
            rag_system = get_rag_system()
            success = rag_system.create_faq_vectorstore(faq_content)
            
            payload = (
                {"status": "success", "message": "FAQ vector store initialized"}
                if success
                else {"status": "error", "message": "Failed to initialize FAQ vector store"}
            )
            return JSONResponse(payload)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing FAQ vector store: {str(e)}")

    @app.route("/vector-store/kb/initialize", methods=["POST"])
    async def initialize_kb_vectorstore(request):
        """Initialize KB vector store from PDF file"""
        try:
            body = await request.json()
            pdf_path = body.get("pdf_path", "knowledge_base.pdf")
            rag_system = get_rag_system()
            success = rag_system.create_kb_vectorstore(pdf_path)
            payload = (
                {"status": "success", "message": f"KB vector store initialized from {pdf_path}"}
                if success
                else {"status": "error", "message": "Failed to initialize KB vector store"}
            )
            return JSONResponse(payload)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing KB vector store: {str(e)}")

    @app.route("/vector-store/status", methods=["GET"])
    async def get_vectorstore_status(request):
        """Get status of vector stores"""
        try:
            status = get_rag_system().get_system_status()
            return JSONResponse(status)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting vector store status: {str(e)}")

    @app.route("/vector-store/faq/update", methods=["POST"])
    async def update_faq_vectorstore(request):
        """Update FAQ vector store when FAQ document changes"""
        try:
            # Get updated FAQ content from MCP Drive
            faq_content = await mcp_client.get_mcp_resource(
                mcp_client.drive_url,
                "doc://faq/all"
            )
            
            # Update FAQ vector store
            success = get_rag_system().update_faq_from_doc_change(faq_content)
            payload = (
                {"status": "success", "message": "FAQ vector store updated"}
                if success
                else {"status": "error", "message": "Failed to update FAQ vector store"}
            )
            return JSONResponse(payload)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating FAQ vector store: {str(e)}")

    port = os.getenv('PORT', '8003')
    print(f"🚀 Starting A2A Multi-Agent Customer Support Server on {os.getenv('A2A_SERVER_URL', 'http://localhost')}:{port}")
    print("🤖 Available Specialized Agents:")
    print("   • Triage Agent - Request classification and routing")
    print("   • Email Agent - Email operations and communication")
    print("   • Documentation Agent - Ticket logging and knowledge management")
    print("   • Resolution Agent - Solution generation and response")
    print("   • Escalation Agent - Complex issue handling")
    print(f"📋 Agent Card: {os.getenv('A2A_SERVER_URL', 'http://localhost')}:{port}/.well-known/agent.json")
    print("🔗 MCP Integration: Gmail (8001) + Drive (8002)")
    print("📊 Features: Multi-agent workflows, structured tasks, MCP tool integration")
    
    # Run the server
    port = int(os.getenv('PORT', 8003))
    print(f"🚀 Starting on {os.getenv('A2A_SERVER_URL', 'http://localhost')}:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()