"""
Shared Memory System for LangGraph Agents

This module provides a shared memory system for agents to communicate
and share data across the conversion workflow.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import time
import json
import hashlib
from pathlib import Path
from loguru import logger


@dataclass
class MemoryEntry:
    """Individual memory entry."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None  # Time to live in seconds
    access_count: int = 0
    last_accessed: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEvent:
    """Memory event for notifications."""
    event_type: str  # "set", "get", "delete", "expire"
    key: str
    value: Any
    timestamp: float
    agent: str


class SharedMemory:
    """
    Thread-safe shared memory for agent communication.
    
    This class provides:
    - Thread-safe memory operations
    - TTL (Time To Live) support
    - Event notifications
    - Memory statistics
    - Automatic cleanup
    """
    
    def __init__(self, max_size: int = 10000, cleanup_interval: float = 60.0):
        """
        Initialize shared memory.
        
        Args:
            max_size: Maximum number of entries
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        # Memory storage
        self._memory: Dict[str, MemoryEntry] = {}
        self._tags: Dict[str, List[str]] = defaultdict(list)
        self._access_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event system
        self._event_handlers: List[Callable[[MemoryEvent], None]] = []
        
        # Statistics
        self._stats = {
            "sets": 0,
            "gets": 0,
            "deletes": 0,
            "expires": 0,
            "hits": 0,
            "misses": 0,
            "total_entries": 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
        
        self.logger = logger.bind(name="shared_memory")
        self.logger.info(f"Initialized shared memory (max_size={max_size})")
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, 
           tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
           agent: str = "unknown") -> bool:
        """
        Set a value in shared memory.
        
        Args:
            key: Memory key
            value: Value to store
            ttl: Time to live in seconds (None for no expiration)
            tags: Tags for the entry
            metadata: Additional metadata
            agent: Agent making the request
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                current_time = time.time()
                
                # Create memory entry
                entry = MemoryEntry(
                    key=key,
                    value=value,
                    timestamp=current_time,
                    ttl=ttl,
                    tags=tags or [],
                    metadata=metadata or {}
                )
                
                # Check if we need to make space
                if len(self._memory) >= self.max_size and key not in self._memory:
                    self._evict_oldest()
                
                # Store entry
                self._memory[key] = entry
                
                # Update tags
                for tag in entry.tags:
                    if key not in self._tags[tag]:
                        self._tags[tag].append(key)
                
                # Update statistics
                self._stats["sets"] += 1
                self._stats["total_entries"] = len(self._memory)
                
                # Emit event
                self._emit_event(MemoryEvent(
                    event_type="set",
                    key=key,
                    value=value,
                    timestamp=current_time,
                    agent=agent
                ))
                
                self.logger.debug(f"Set memory entry: {key} (agent={agent})")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to set memory entry {key}: {e}")
                return False
    
    def get(self, key: str, default: Any = None, agent: str = "unknown") -> Any:
        """
        Get a value from shared memory.
        
        Args:
            key: Memory key
            default: Default value if key not found
            agent: Agent making the request
            
        Returns:
            Value or default
        """
        with self._lock:
            try:
                current_time = time.time()
                
                if key not in self._memory:
                    self._stats["misses"] += 1
                    return default
                
                entry = self._memory[key]
                
                # Check if expired
                if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                    self._delete_entry(key, "expire")
                    self._stats["misses"] += 1
                    return default
                
                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = current_time
                self._access_history.append((key, current_time))
                
                # Update statistics
                self._stats["gets"] += 1
                self._stats["hits"] += 1
                
                # Emit event
                self._emit_event(MemoryEvent(
                    event_type="get",
                    key=key,
                    value=entry.value,
                    timestamp=current_time,
                    agent=agent
                ))
                
                self.logger.debug(f"Retrieved memory entry: {key} (agent={agent})")
                return entry.value
                
            except Exception as e:
                self.logger.error(f"Failed to get memory entry {key}: {e}")
                self._stats["misses"] += 1
                return default
    
    def delete(self, key: str, agent: str = "unknown") -> bool:
        """
        Delete a value from shared memory.
        
        Args:
            key: Memory key
            agent: Agent making the request
            
        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            try:
                if key not in self._memory:
                    return False
                
                self._delete_entry(key, "delete")
                
                # Emit event
                self._emit_event(MemoryEvent(
                    event_type="delete",
                    key=key,
                    value=None,
                    timestamp=time.time(),
                    agent=agent
                ))
                
                self.logger.debug(f"Deleted memory entry: {key} (agent={agent})")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete memory entry {key}: {e}")
                return False
    
    def _delete_entry(self, key: str, reason: str = "delete"):
        """Internal method to delete an entry."""
        entry = self._memory.pop(key, None)
        if entry:
            # Remove from tags
            for tag in entry.tags:
                if key in self._tags[tag]:
                    self._tags[tag].remove(key)
            
            # Update statistics
            self._stats["deletes"] += 1
            if reason == "expire":
                self._stats["expires"] += 1
            self._stats["total_entries"] = len(self._memory)
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in memory."""
        with self._lock:
            if key not in self._memory:
                return False
            
            entry = self._memory[key]
            current_time = time.time()
            
            # Check if expired
            if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                self._delete_entry(key, "expire")
                return False
            
            return True
    
    def get_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get all entries with a specific tag."""
        with self._lock:
            result = {}
            keys = self._tags.get(tag, [])
            
            for key in keys[:]:  # Copy to avoid modification during iteration
                if self.exists(key):
                    result[key] = self._memory[key].value
                else:
                    # Clean up invalid tag reference
                    keys.remove(key)
            
            return result
    
    def get_by_tags(self, tags: List[str], match_all: bool = True) -> Dict[str, Any]:
        """Get entries matching multiple tags."""
        with self._lock:
            if not tags:
                return {}
            
            if match_all:
                # Find intersection of all tag sets
                common_keys = set(self._tags.get(tags[0], []))
                for tag in tags[1:]:
                    common_keys &= set(self._tags.get(tag, []))
            else:
                # Find union of all tag sets
                common_keys = set()
                for tag in tags:
                    common_keys |= set(self._tags.get(tag, []))
            
            result = {}
            for key in common_keys:
                if self.exists(key):
                    result[key] = self._memory[key].value
            
            return result
    
    def clear(self, agent: str = "unknown"):
        """Clear all memory entries."""
        with self._lock:
            keys = list(self._memory.keys())
            for key in keys:
                self._delete_entry(key, "clear")
            
            self._tags.clear()
            self._access_history.clear()
            
            self.logger.info(f"Cleared all memory entries (agent={agent})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            current_time = time.time()
            active_entries = sum(1 for entry in self._memory.values() 
                               if not entry.ttl or (current_time - entry.timestamp) <= entry.ttl)
            
            hit_rate = (self._stats["hits"] / (self._stats["hits"] + self._stats["misses"]) 
                       if (self._stats["hits"] + self._stats["misses"]) > 0 else 0.0)
            
            return {
                "total_entries": len(self._memory),
                "active_entries": active_entries,
                "expired_entries": len(self._memory) - active_entries,
                "total_tags": len(self._tags),
                "operations": self._stats.copy(),
                "hit_rate": hit_rate,
                "memory_usage_estimate": sum(len(str(entry.value)) for entry in self._memory.values())
            }
    
    def _evict_oldest(self):
        """Evict the oldest entry to make space."""
        if not self._memory:
            return
        
        # Find oldest entry (least recently accessed)
        oldest_key = min(self._memory.keys(), 
                        key=lambda k: self._memory[k].last_accessed)
        self._delete_entry(oldest_key, "eviction")
        
        self.logger.debug(f"Evicted oldest entry: {oldest_key}")
    
    def _cleanup_expired(self):
        """Background thread to clean up expired entries."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                
                with self._lock:
                    current_time = time.time()
                    expired_keys = []
                    
                    for key, entry in self._memory.items():
                        if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._delete_entry(key, "expire")
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {e}")
    
    def _emit_event(self, event: MemoryEvent):
        """Emit memory event to registered handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in memory event handler: {e}")
    
    def add_event_handler(self, handler: Callable[[MemoryEvent], None]):
        """Add an event handler for memory events."""
        self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[MemoryEvent], None]):
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
    
    def export_to_file(self, file_path: Union[str, Path]) -> bool:
        """Export memory to JSON file."""
        try:
            with self._lock:
                export_data = {
                    "timestamp": time.time(),
                    "entries": {
                        key: {
                            "value": entry.value,
                            "timestamp": entry.timestamp,
                            "ttl": entry.ttl,
                            "access_count": entry.access_count,
                            "tags": entry.tags,
                            "metadata": entry.metadata
                        }
                        for key, entry in self._memory.items()
                    },
                    "tags": dict(self._tags),
                    "statistics": self.get_statistics()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Exported memory to: {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export memory: {e}")
            return False
    
    def import_from_file(self, file_path: Union[str, Path]) -> bool:
        """Import memory from JSON file."""
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            with self._lock:
                self.clear("import")
                
                # Import entries
                for key, entry_data in import_data.get("entries", {}).items():
                    entry = MemoryEntry(
                        key=key,
                        value=entry_data["value"],
                        timestamp=entry_data["timestamp"],
                        ttl=entry_data.get("ttl"),
                        access_count=entry_data.get("access_count", 0),
                        tags=entry_data.get("tags", []),
                        metadata=entry_data.get("metadata", {})
                    )
                    entry.last_accessed = entry.timestamp
                    self._memory[key] = entry
                    
                    # Update tags
                    for tag in entry.tags:
                        if key not in self._tags[tag]:
                            self._tags[tag].append(key)
                
                self.logger.info(f"Imported memory from: {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to import memory: {e}")
            return False


class MemoryManager:
    """
    High-level memory manager for the conversion workflow.
    
    This class provides convenient methods for common memory operations
    in the MATLAB2C++ conversion context.
    """
    
    def __init__(self, shared_memory: SharedMemory):
        """Initialize memory manager with shared memory instance."""
        self.shared_memory = shared_memory
        self.logger = logger.bind(name="memory_manager")
    
    def store_analysis_result(self, project_name: str, analysis_data: Dict[str, Any], 
                            agent: str = "analyzer") -> bool:
        """Store MATLAB analysis result."""
        key = f"analysis:{project_name}"
        return self.shared_memory.set(
            key=key,
            value=analysis_data,
            ttl=3600,  # 1 hour TTL
            tags=["analysis", "matlab", project_name],
            metadata={"agent": agent, "type": "analysis_result"},
            agent=agent
        )
    
    def get_analysis_result(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get MATLAB analysis result."""
        key = f"analysis:{project_name}"
        return self.shared_memory.get(key)
    
    def store_conversion_plan(self, project_name: str, plan_data: Dict[str, Any], 
                            agent: str = "planner") -> bool:
        """Store conversion plan."""
        key = f"plan:{project_name}"
        return self.shared_memory.set(
            key=key,
            value=plan_data,
            ttl=1800,  # 30 minutes TTL
            tags=["plan", "conversion", project_name],
            metadata={"agent": agent, "type": "conversion_plan"},
            agent=agent
        )
    
    def get_conversion_plan(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get conversion plan."""
        key = f"plan:{project_name}"
        return self.shared_memory.get(key)
    
    def store_generated_code(self, project_name: str, code_data: Dict[str, Any], 
                           version: str = "latest", agent: str = "generator") -> bool:
        """Store generated C++ code."""
        key = f"code:{project_name}:{version}"
        return self.shared_memory.set(
            key=key,
            value=code_data,
            ttl=7200,  # 2 hours TTL
            tags=["code", "cpp", project_name, version],
            metadata={"agent": agent, "type": "generated_code", "version": version},
            agent=agent
        )
    
    def get_generated_code(self, project_name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get generated C++ code."""
        key = f"code:{project_name}:{version}"
        return self.shared_memory.get(key)
    
    def store_quality_assessment(self, project_name: str, assessment_data: Dict[str, Any], 
                               version: str = "latest", agent: str = "assessor") -> bool:
        """Store quality assessment."""
        key = f"assessment:{project_name}:{version}"
        return self.shared_memory.set(
            key=key,
            value=assessment_data,
            ttl=3600,  # 1 hour TTL
            tags=["assessment", "quality", project_name, version],
            metadata={"agent": agent, "type": "quality_assessment", "version": version},
            agent=agent
        )
    
    def get_quality_assessment(self, project_name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get quality assessment."""
        key = f"assessment:{project_name}:{version}"
        return self.shared_memory.get(key)
    
    def store_intermediate_result(self, operation: str, result: Dict[str, Any], 
                                agent: str = "unknown") -> str:
        """Store intermediate operation result."""
        result_id = hashlib.md5(f"{operation}_{time.time()}".encode()).hexdigest()[:8]
        key = f"intermediate:{operation}:{result_id}"
        
        self.shared_memory.set(
            key=key,
            value=result,
            ttl=900,  # 15 minutes TTL
            tags=["intermediate", operation],
            metadata={"agent": agent, "type": "intermediate_result", "operation": operation},
            agent=agent
        )
        
        return result_id
    
    def get_project_data(self, project_name: str) -> Dict[str, Any]:
        """Get all data for a project."""
        return {
            "analysis": self.get_analysis_result(project_name),
            "plan": self.get_conversion_plan(project_name),
            "code": self.get_generated_code(project_name),
            "assessment": self.get_quality_assessment(project_name)
        }
    
    def cleanup_project_data(self, project_name: str):
        """Clean up all data for a project."""
        tags_to_clean = [project_name]
        keys_to_delete = []
        
        for tag in tags_to_clean:
            project_data = self.shared_memory.get_by_tag(tag)
            keys_to_delete.extend(project_data.keys())
        
        for key in keys_to_delete:
            self.shared_memory.delete(key, "cleanup")
        
        self.logger.info(f"Cleaned up data for project: {project_name}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory usage."""
        stats = self.shared_memory.get_statistics()
        
        # Get data by type
        analysis_data = self.shared_memory.get_by_tag("analysis")
        plan_data = self.shared_memory.get_by_tag("plan")
        code_data = self.shared_memory.get_by_tag("code")
        assessment_data = self.shared_memory.get_by_tag("assessment")
        
        return {
            "statistics": stats,
            "data_types": {
                "analysis": len(analysis_data),
                "plans": len(plan_data),
                "code": len(code_data),
                "assessments": len(assessment_data)
            },
            "active_projects": len(set(key.split(":")[1] for key in analysis_data.keys()))
        }
