"""
Streaming Executor with async/await support.

Enables real-time token streaming from LLM for better UX.
"""

import asyncio
import json
from typing import List, Dict, Any, AsyncIterator, Optional
from dataclasses import dataclass
from enum import Enum
import sys

from llm.local_llm_client import LocalLLMClient


class StreamStatus(Enum):
    """Status of streaming operation."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StreamChunk:
    """A chunk of streamed content."""
    token: str
    status: StreamStatus
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StreamingExecutor:
    """
    Asynchronous executor for streaming LLM responses.
    
    Replaces blocking executor for analysis/planning tasks.
    """
    
    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.buffer = []
        
    async def stream_llm_analysis(
        self, 
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream LLM response token by token.
        
        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Yields:
            StreamChunk objects with individual tokens
        """
        
        try:
            yield StreamChunk(
                token="",
                status=StreamStatus.STARTED,
                metadata={"max_tokens": max_tokens}
            )
            
            # Call LLM with streaming enabled
            # NOTE: Requires local_llm_client to support streaming
            # For now, we buffer and yield chunks
            
            response = await asyncio.to_thread(
                self.llm_client.generate_text,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Simulate streaming by yielding character chunks
            chunk_size = 20  # Characters per chunk
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i+chunk_size]
                yield StreamChunk(
                    token=chunk,
                    status=StreamStatus.IN_PROGRESS,
                    metadata={"offset": i}
                )
                await asyncio.sleep(0.01)  # Simulate network latency
            
            yield StreamChunk(
                token="",
                status=StreamStatus.COMPLETE,
                metadata={"total_length": len(response)}
            )
            
        except Exception as e:
            yield StreamChunk(
                token="",
                status=StreamStatus.ERROR,
                error=str(e)
            )
    
    async def stream_code_patch(
        self,
        file_content: str,
        task_description: str,
        context_chunks: List[Dict[str, Any]]
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream code patch generation.
        
        Args:
            file_content: Current file content
            task_description: What to change
            context_chunks: Relevant code context
            
        Yields:
            StreamChunk objects with patch tokens
        """
        
        context_str = "\n".join([
            f"// {c.get('file_path', 'unknown')}:\n{c.get('content', '')[:300]}"
            for c in context_chunks
        ])
        
        prompt = f"""You are an expert code editor. Generate a precise code patch.

Task: {task_description}

Current File:
```
{file_content[:1000]}
```

Related Code Context:
{context_str}

Generate ONLY the modified code section(s) with clear /* CHANGE */ markers.
Keep changes minimal and focused."""
        
        async for chunk in self.stream_llm_analysis(prompt, max_tokens=1024):
            yield chunk
    
    async def collect_stream(
        self,
        stream: AsyncIterator[StreamChunk]
    ) -> tuple[str, bool, Optional[str]]:
        """
        Collect a full stream into a string.
        
        Returns:
            (full_text, success, error_msg)
        """
        full_text = []
        error = None
        
        try:
            async for chunk in stream:
                if chunk.status == StreamStatus.ERROR:
                    error = chunk.error
                    return "".join(full_text), False, error
                    
                if chunk.status in [
                    StreamStatus.IN_PROGRESS,
                    StreamStatus.COMPLETE
                ]:
                    full_text.append(chunk.token)
            
            return "".join(full_text), True, None
            
        except Exception as e:
            return "".join(full_text), False, str(e)


# Synchronous wrapper for CLI compatibility
class SyncStreamingExecutor:
    """Wrapper that makes streaming executor work in sync code."""
    
    def __init__(self, llm_client: LocalLLMClient = None):
        self.executor = StreamingExecutor(llm_client)
    
    def stream_llm_analysis_to_stdout(
        self,
        prompt: str,
        max_tokens: int = 2048,
    ) -> tuple[str, bool]:
        """
        Stream to stdout with real-time output.
        
        Returns:
            (full_response, success)
        """
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            stream = self.executor.stream_llm_analysis(prompt, max_tokens)
            
            full_text = []
            async def consume_stream():
                async for chunk in stream:
                    if chunk.status == StreamStatus.IN_PROGRESS:
                        sys.stdout.write(chunk.token)
                        sys.stdout.flush()
                        full_text.append(chunk.token)
                    elif chunk.status == StreamStatus.ERROR:
                        return False
                return True
            
            success = loop.run_until_complete(consume_stream())
            return "".join(full_text), success
            
        finally:
            loop.close()
    
    def stream_code_patch_to_stdout(
        self,
        file_content: str,
        task_description: str,
        context_chunks: List[Dict[str, Any]]
    ) -> tuple[str, bool]:
        """Stream code patch to stdout."""
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            stream = self.executor.stream_code_patch(
                file_content,
                task_description,
                context_chunks
            )
            
            full_text = []
            async def consume():
                async for chunk in stream:
                    if chunk.status == StreamStatus.IN_PROGRESS:
                        sys.stdout.write(chunk.token)
                        sys.stdout.flush()
                        full_text.append(chunk.token)
                return True
            
            success = loop.run_until_complete(consume())
            return "".join(full_text), success
            
        finally:
            loop.close()
