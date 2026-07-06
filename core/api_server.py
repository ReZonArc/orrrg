"""
ORRRG REST API Server
=====================

FastAPI-based REST API providing external access to ORRRG's capabilities:
- System status and health monitoring
- Component management and control
- Knowledge queries via AtomSpace
- Inference requests via echonnxruntime
- Evolution Engine control
- Relevance Realization state
- WebSocket real-time event streaming

Usage:
    from core.api_server import create_app
    app = create_app(soc_instance)
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. API server unavailable. Install with: pip install fastapi uvicorn")


# --- Request/Response Models ---

class StatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    components: Dict[str, Any]
    evolution_active: bool
    relevance_state: Optional[Dict[str, Any]] = None


class ComponentRequest(BaseModel):
    component: str
    action: str
    data: Dict[str, Any] = {}


class InferenceRequest(BaseModel):
    model_name: str
    inputs: Dict[str, Any]
    output_names: Optional[List[str]] = None


class KnowledgeQueryRequest(BaseModel):
    pattern: Dict[str, Any] = {}
    limit: int = 10


class EvolutionRequest(BaseModel):
    action: str  # start, stop, status, evolve_once
    parameters: Dict[str, Any] = {}


# --- WebSocket Connection Manager ---

class ConnectionManager:
    """Manages WebSocket connections for real-time event streaming."""

    def __init__(self):
        self.active_connections: List[Any] = []

    async def connect(self, websocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket) -> None:
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast event to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


# --- App Factory ---

def create_app(soc=None) -> Any:
    """
    Create the FastAPI application.
    
    Args:
        soc: SelfOrganizingCore instance (can be set later via app.state.soc)
    
    Returns:
        FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="ORRRG API",
        description="Omnipotent Research and Reasoning Reactive Grid - REST API",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    app.state.soc = soc
    app.state.start_time = time.time()
    app.state.ws_manager = ConnectionManager()

    # --- Health & Status ---

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    @app.get("/status", response_model=StatusResponse)
    async def system_status():
        soc = app.state.soc
        if not soc:
            raise HTTPException(503, "System not initialized")

        uptime = time.time() - app.state.start_time
        components = {}
        if hasattr(soc, 'components'):
            for name, comp in soc.components.items():
                components[name] = {
                    'initialized': getattr(comp, 'initialized', False),
                }

        return StatusResponse(
            status="running",
            uptime_seconds=round(uptime, 2),
            components=components,
            evolution_active=hasattr(soc, 'evolution_engine') and soc.evolution_engine is not None,
        )

    # --- Component Operations ---

    @app.post("/components/{component_name}/process")
    async def process_component(component_name: str, request: ComponentRequest):
        soc = app.state.soc
        if not soc:
            raise HTTPException(503, "System not initialized")

        if not hasattr(soc, 'components') or component_name not in soc.components:
            raise HTTPException(404, f"Component '{component_name}' not found")

        component = soc.components[component_name]
        try:
            result = await component.process(request.data)
            await app.state.ws_manager.broadcast({
                'event': 'component_process',
                'component': component_name,
                'action': request.action,
                'timestamp': datetime.utcnow().isoformat(),
            })
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/components")
    async def list_components():
        soc = app.state.soc
        if not soc:
            raise HTTPException(503, "System not initialized")

        components = {}
        if hasattr(soc, 'components'):
            for name, comp in soc.components.items():
                info = comp.get_info() if hasattr(comp, 'get_info') else {'name': name}
                components[name] = info
        return {"components": components}

    # --- Inference ---

    @app.post("/inference/run")
    async def run_inference(request: InferenceRequest):
        soc = app.state.soc
        if not soc:
            raise HTTPException(503, "System not initialized")

        # Find echonnxruntime component
        onnx_comp = None
        if hasattr(soc, 'components'):
            onnx_comp = soc.components.get('echonnxruntime')
        if not onnx_comp:
            raise HTTPException(404, "echonnxruntime component not available")

        try:
            result = await onnx_comp.process({
                'action': 'infer',
                'model_name': request.model_name,
                'inputs': request.inputs,
                'output_names': request.output_names,
            })
            return result
        except Exception as e:
            raise HTTPException(500, str(e))

    # --- Knowledge ---

    @app.post("/knowledge/query")
    async def query_knowledge(request: KnowledgeQueryRequest):
        soc = app.state.soc
        if not soc:
            raise HTTPException(503, "System not initialized")

        oc_comp = None
        if hasattr(soc, 'components'):
            oc_comp = soc.components.get('oc-skintwin')
        if not oc_comp:
            raise HTTPException(404, "oc-skintwin component not available")

        try:
            result = await oc_comp.process({
                'operation': 'query',
                'pattern': request.pattern,
            })
            return result
        except Exception as e:
            raise HTTPException(500, str(e))

    # --- Evolution ---

    @app.post("/evolution")
    async def evolution_control(request: EvolutionRequest):
        soc = app.state.soc
        if not soc:
            raise HTTPException(503, "System not initialized")

        if not hasattr(soc, 'evolution_engine') or not soc.evolution_engine:
            raise HTTPException(404, "Evolution Engine not available")

        action = request.action
        engine = soc.evolution_engine

        if action == 'status':
            return {"status": "success", "engine": engine.get_statistics() if hasattr(engine, 'get_statistics') else {}}
        elif action == 'evolve_once':
            result = await engine.evolve() if hasattr(engine, 'evolve') else {}
            await app.state.ws_manager.broadcast({
                'event': 'evolution_cycle',
                'timestamp': datetime.utcnow().isoformat(),
            })
            return {"status": "success", "result": result}
        else:
            raise HTTPException(400, f"Unknown evolution action: {action}")

    # --- Relevance Realization ---

    @app.get("/relevance")
    async def relevance_state():
        soc = app.state.soc
        if not soc:
            raise HTTPException(503, "System not initialized")

        if hasattr(soc, 'relevance_integrator') and soc.relevance_integrator:
            state = soc.relevance_integrator.get_state() if hasattr(soc.relevance_integrator, 'get_state') else {}
            return {"status": "success", "relevance": state}
        return {"status": "success", "relevance": None}

    # --- WebSocket ---

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        manager = app.state.ws_manager
        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive, handle client messages
                data = await websocket.receive_text()
                # Echo or process client messages
                await websocket.send_json({
                    'event': 'ack',
                    'message': data,
                    'timestamp': datetime.utcnow().isoformat(),
                })
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    return app


async def start_server(soc=None, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the API server (for use within ORRRG)."""
    if not FASTAPI_AVAILABLE:
        logger.error("Cannot start API server: FastAPI not installed")
        return

    try:
        import uvicorn
    except ImportError:
        logger.error("Cannot start API server: uvicorn not installed")
        return

    app = create_app(soc)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
