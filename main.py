#!/usr/bin/env python3
"""
MCP Memory Server - Persistent knowledge graph and memory management

Based on the Open WebUI reference implementation with enhanced features.
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MEMORY_FILE = DATA_DIR / "memory.json"
KNOWLEDGE_FILE = DATA_DIR / "knowledge.json"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="MCP Memory Server",
    version="1.0.0",
    description="Persistent knowledge graph and memory management for AI applications",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Entity(BaseModel):
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    description: str = Field("", description="Entity description")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class Relation(BaseModel):
    from_entity: str = Field(..., description="Source entity name")
    to_entity: str = Field(..., description="Target entity name")
    relation_type: str = Field(..., description="Relation type")
    description: str = Field("", description="Relation description")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Relation attributes")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Memory ID")
    content: str = Field(..., description="Memory content")
    type: str = Field("general", description="Memory type")
    importance: float = Field(0.5, description="Memory importance (0.0 to 1.0)")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)

class MemoryStore(BaseModel):
    memories: List[Memory] = Field(default_factory=list)

class AddEntityRequest(BaseModel):
    entity: Entity

class AddRelationRequest(BaseModel):
    relation: Relation

class AddMemoryRequest(BaseModel):
    memory: Memory

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types")
    relation_types: Optional[List[str]] = Field(None, description="Filter by relation types")
    memory_types: Optional[List[str]] = Field(None, description="Filter by memory types")
    limit: int = Field(10, description="Maximum results to return")

class UpdateEntityRequest(BaseModel):
    name: str = Field(..., description="Entity name to update")
    updates: Dict[str, Any] = Field(..., description="Fields to update")

class UpdateMemoryRequest(BaseModel):
    memory_id: str = Field(..., description="Memory ID to update")
    updates: Dict[str, Any] = Field(..., description="Fields to update")

# Storage functions
def load_knowledge_graph() -> KnowledgeGraph:
    """Load knowledge graph from file"""
    if KNOWLEDGE_FILE.exists():
        try:
            with open(KNOWLEDGE_FILE, 'r') as f:
                data = json.load(f)
                return KnowledgeGraph(**data)
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
    return KnowledgeGraph()

def save_knowledge_graph(kg: KnowledgeGraph):
    """Save knowledge graph to file"""
    try:
        with open(KNOWLEDGE_FILE, 'w') as f:
            json.dump(kg.model_dump(), f, indent=2)
    except Exception as e:
        print(f"Error saving knowledge graph: {e}")

def load_memory_store() -> MemoryStore:
    """Load memory store from file"""
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                return MemoryStore(**data)
        except Exception as e:
            print(f"Error loading memory store: {e}")
    return MemoryStore()

def save_memory_store(ms: MemoryStore):
    """Save memory store to file"""
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(ms.model_dump(), f, indent=2)
    except Exception as e:
        print(f"Error saving memory store: {e}")

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mcp-memory"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "MCP Memory Server", "version": "1.0.0"}

@app.get("/stats")
async def get_stats():
    """Get memory and knowledge graph statistics"""
    kg = load_knowledge_graph()
    ms = load_memory_store()
    
    return {
        "entities": len(kg.entities),
        "relations": len(kg.relations),
        "memories": len(ms.memories),
        "data_dir": str(DATA_DIR),
        "files_exist": {
            "knowledge": KNOWLEDGE_FILE.exists(),
            "memory": MEMORY_FILE.exists()
        }
    }

# Knowledge Graph endpoints
@app.post("/entities")
async def add_entity(request: AddEntityRequest = Body(...)):
    """Add an entity to the knowledge graph"""
    kg = load_knowledge_graph()
    
    # Check if entity already exists
    existing = next((e for e in kg.entities if e.name == request.entity.name), None)
    if existing:
        raise HTTPException(status_code=400, detail="Entity already exists")
    
    kg.entities.append(request.entity)
    save_knowledge_graph(kg)
    
    return {"success": True, "message": f"Entity '{request.entity.name}' added"}

@app.post("/relations")
async def add_relation(request: AddRelationRequest = Body(...)):
    """Add a relation to the knowledge graph"""
    kg = load_knowledge_graph()
    
    # Check if entities exist
    from_exists = any(e.name == request.relation.from_entity for e in kg.entities)
    to_exists = any(e.name == request.relation.to_entity for e in kg.entities)
    
    if not from_exists:
        raise HTTPException(status_code=400, detail=f"From entity '{request.relation.from_entity}' not found")
    if not to_exists:
        raise HTTPException(status_code=400, detail=f"To entity '{request.relation.to_entity}' not found")
    
    kg.relations.append(request.relation)
    save_knowledge_graph(kg)
    
    return {"success": True, "message": f"Relation added: {request.relation.from_entity} -> {request.relation.to_entity}"}

@app.get("/entities")
async def get_entities():
    """Get all entities"""
    kg = load_knowledge_graph()
    return {"entities": kg.entities}

@app.get("/relations")
async def get_relations():
    """Get all relations"""
    kg = load_knowledge_graph()
    return {"relations": kg.relations}

@app.get("/knowledge-graph")
async def get_knowledge_graph():
    """Get complete knowledge graph"""
    kg = load_knowledge_graph()
    return kg

@app.post("/entities/update")
async def update_entity(request: UpdateEntityRequest = Body(...)):
    """Update an entity"""
    kg = load_knowledge_graph()
    
    entity = next((e for e in kg.entities if e.name == request.name), None)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Update fields
    for field, value in request.updates.items():
        if hasattr(entity, field):
            setattr(entity, field, value)
    
    entity.updated_at = datetime.now().isoformat()
    save_knowledge_graph(kg)
    
    return {"success": True, "message": f"Entity '{request.name}' updated"}

@app.delete("/entities/{entity_name}")
async def delete_entity(entity_name: str):
    """Delete an entity and its relations"""
    kg = load_knowledge_graph()
    
    # Remove entity
    kg.entities = [e for e in kg.entities if e.name != entity_name]
    
    # Remove relations involving this entity
    kg.relations = [r for r in kg.relations if r.from_entity != entity_name and r.to_entity != entity_name]
    
    save_knowledge_graph(kg)
    
    return {"success": True, "message": f"Entity '{entity_name}' deleted"}

# Memory endpoints
@app.post("/memories")
async def add_memory(request: AddMemoryRequest = Body(...)):
    """Add a memory"""
    ms = load_memory_store()
    ms.memories.append(request.memory)
    save_memory_store(ms)
    
    return {"success": True, "message": f"Memory '{request.memory.id}' added"}

@app.get("/memories")
async def get_memories():
    """Get all memories"""
    ms = load_memory_store()
    return {"memories": ms.memories}

@app.post("/memories/update")
async def update_memory(request: UpdateMemoryRequest = Body(...)):
    """Update a memory"""
    ms = load_memory_store()
    
    memory = next((m for m in ms.memories if m.id == request.memory_id), None)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Update fields
    for field, value in request.updates.items():
        if hasattr(memory, field):
            setattr(memory, field, value)
    
    memory.updated_at = datetime.now().isoformat()
    save_memory_store(ms)
    
    return {"success": True, "message": f"Memory '{request.memory_id}' updated"}

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory"""
    ms = load_memory_store()
    ms.memories = [m for m in ms.memories if m.id != memory_id]
    save_memory_store(ms)
    
    return {"success": True, "message": f"Memory '{memory_id}' deleted"}

# Search endpoints
@app.post("/search")
async def search(request: SearchRequest = Body(...)):
    """Search entities, relations, and memories"""
    kg = load_knowledge_graph()
    ms = load_memory_store()
    
    results = {
        "entities": [],
        "relations": [],
        "memories": []
    }
    
    query_lower = request.query.lower()
    
    # Search entities
    for entity in kg.entities:
        if (query_lower in entity.name.lower() or 
            query_lower in entity.description.lower() or
            (request.entity_types and entity.type in request.entity_types)):
            results["entities"].append(entity)
    
    # Search relations
    for relation in kg.relations:
        if (query_lower in relation.from_entity.lower() or 
            query_lower in relation.to_entity.lower() or
            query_lower in relation.description.lower() or
            (request.relation_types and relation.relation_type in request.relation_types)):
            results["relations"].append(relation)
    
    # Search memories
    for memory in ms.memories:
        if (query_lower in memory.content.lower() or 
            any(query_lower in tag.lower() for tag in memory.tags) or
            (request.memory_types and memory.type in request.memory_types)):
            results["memories"].append(memory)
    
    # Limit results
    for key in results:
        results[key] = results[key][:request.limit]
    
    return results

@app.post("/clear-all")
async def clear_all():
    """Clear all data (use with caution)"""
    try:
        if KNOWLEDGE_FILE.exists():
            KNOWLEDGE_FILE.unlink()
        if MEMORY_FILE.exists():
            MEMORY_FILE.unlink()
        
        return {"success": True, "message": "All data cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)