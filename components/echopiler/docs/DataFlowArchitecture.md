# Compiler Explorer Data Flow Architecture

This document complements the [Technical Architecture](TechnicalArchitecture.md) by focusing on data flows, request processing, and interaction patterns within Compiler Explorer.

## Table of Contents

- [Request Processing Flow](#request-processing-flow)
- [Compilation Data Flow](#compilation-data-flow)
- [WebSocket Communication](#websocket-communication)
- [Caching Strategies](#caching-strategies)
- [Configuration Loading](#configuration-loading)
- [Error Handling Flow](#error-handling-flow)

## Request Processing Flow

### Standard Web Request Flow

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant CDN
    participant LoadBalancer
    participant WebServer
    participant CompilerService
    participant Cache
    participant Storage

    User->>Browser: Navigate to Compiler Explorer
    Browser->>CDN: Request static assets
    CDN-->>Browser: Static assets (cached)
    Browser->>LoadBalancer: API request (compile code)
    LoadBalancer->>WebServer: Route request
    WebServer->>Cache: Check compilation cache
    alt Cache Hit
        Cache-->>WebServer: Return cached result
        WebServer-->>Browser: Compilation result
    else Cache Miss
        WebServer->>CompilerService: Initiate compilation
        CompilerService->>CompilerService: Execute compiler
        CompilerService-->>WebServer: Compilation result
        WebServer->>Cache: Store result
        WebServer->>Storage: Store large results
        WebServer-->>Browser: Compilation result
    end
```

### Asynchronous Compilation Flow (SQS Workers)

```mermaid
sequenceDiagram
    participant Browser
    participant WebServer
    participant SQS
    participant Worker
    participant S3
    participant WebSocket

    Browser->>WebServer: Compilation request
    WebServer->>SQS: Queue compilation job
    WebServer-->>Browser: Job queued response
    
    Worker->>SQS: Poll for jobs
    SQS-->>Worker: Compilation job
    Worker->>Worker: Execute compilation
    
    alt Large Result
        Worker->>S3: Store result
        Worker->>WebSocket: Send result reference
    else Normal Result
        Worker->>WebSocket: Send result directly
    end
    
    WebSocket-->>Browser: Push result to client
    Browser->>Browser: Update UI with result
```

## Compilation Data Flow

### Single Compilation Process

```mermaid
flowchart TD
    subgraph "Input Processing"
        A[Source Code] --> B[Language Detection]
        B --> C[Compiler Selection]
        C --> D[Argument Parsing]
    end
    
    subgraph "Pre-compilation"
        D --> E[Source Validation]
        E --> F[Cache Lookup]
        F --> G{Cache Hit?}
    end
    
    subgraph "Compilation"
        G -->|No| H[Temp File Creation]
        H --> I[Compiler Execution]
        I --> J[Output Parsing]
        J --> K[Assembly Processing]
    end
    
    subgraph "Post-processing"
        K --> L[Line Mapping]
        L --> M[Tool Execution]
        M --> N[Result Formatting]
    end
    
    subgraph "Output"
        G -->|Yes| O[Cached Result]
        N --> P[Store in Cache]
        P --> Q[Return Result]
        O --> Q
    end
    
    subgraph "Error Handling"
        I --> R{Compilation Failed?}
        R -->|Yes| S[Error Processing]
        S --> T[Error Formatting]
        T --> Q
        R -->|No| K
    end
```

### Multi-compiler Compilation

```mermaid
flowchart LR
    subgraph "Source Management"
        SOURCE[Source Code]
        SHARED[Shared Libraries]
        HEADERS[Header Files]
    end
    
    subgraph "Parallel Compilation"
        COMP1[Compiler 1]
        COMP2[Compiler 2]
        COMP3[Compiler 3]
        COMPN[Compiler N]
    end
    
    subgraph "Result Aggregation"
        MERGE[Result Merger]
        DIFF[Diff Generator]
        ANALYSIS[Analysis Tools]
    end
    
    SOURCE --> COMP1
    SOURCE --> COMP2
    SOURCE --> COMP3
    SOURCE --> COMPN
    
    SHARED --> COMP1
    SHARED --> COMP2
    SHARED --> COMP3
    SHARED --> COMPN
    
    HEADERS --> COMP1
    HEADERS --> COMP2
    HEADERS --> COMP3
    HEADERS --> COMPN
    
    COMP1 --> MERGE
    COMP2 --> MERGE
    COMP3 --> MERGE
    COMPN --> MERGE
    
    MERGE --> DIFF
    MERGE --> ANALYSIS
```

## WebSocket Communication

### Real-time Updates Flow

```mermaid
sequenceDiagram
    participant Frontend
    participant WebServer
    participant Worker
    participant WebSocketManager
    
    Frontend->>WebServer: Connect WebSocket
    WebServer->>WebSocketManager: Register client
    
    Frontend->>WebServer: Submit compilation job
    WebServer->>Worker: Queue job with client ID
    
    Worker->>Worker: Process compilation
    Worker->>WebSocketManager: Send result with client ID
    WebSocketManager->>Frontend: Push compilation result
    
    Frontend->>Frontend: Update UI in real-time
    
    Note over Frontend,WebSocketManager: Connection maintained for live updates
```

### WebSocket Event Types

```mermaid
graph TB
    subgraph "WebSocket Events"
        subgraph "Compilation Events"
            COMPILE_START[compile:start]
            COMPILE_PROGRESS[compile:progress]
            COMPILE_RESULT[compile:result]
            COMPILE_ERROR[compile:error]
        end
        
        subgraph "Execution Events"
            EXEC_START[execution:start]
            EXEC_OUTPUT[execution:output]
            EXEC_COMPLETE[execution:complete]
            EXEC_ERROR[execution:error]
        end
        
        subgraph "System Events"
            SYSTEM_STATUS[system:status]
            QUEUE_SIZE[queue:size]
            WORKER_STATUS[worker:status]
        end
        
        subgraph "Client Events"
            CLIENT_CONNECT[client:connect]
            CLIENT_DISCONNECT[client:disconnect]
            CLIENT_ERROR[client:error]
        end
    end
```

## Caching Strategies

### Multi-tier Caching Architecture

```mermaid
flowchart TB
    subgraph "Caching Tiers"
        subgraph "L1 - Memory Cache"
            MEM1[Configuration Cache]
            MEM2[Compiler Info Cache]
            MEM3[Recent Results Cache]
        end
        
        subgraph "L2 - Redis Cache"
            REDIS1[Compilation Results]
            REDIS2[Assembly Output]
            REDIS3[Tool Results]
        end
        
        subgraph "L3 - File System Cache"
            FS1[Binary Artifacts]
            FS2[Temporary Files]
            FS3[Preprocessed Source]
        end
        
        subgraph "L4 - S3 Storage"
            S3_1[Large Results]
            S3_2[Historical Data]
            S3_3[Backup Storage]
        end
    end
    
    REQUEST[Compilation Request]
    
    REQUEST --> MEM1
    MEM1 -->|Miss| REDIS1
    REDIS1 -->|Miss| FS1
    FS1 -->|Miss| S3_1
    
    S3_1 -->|Store| FS1
    FS1 -->|Store| REDIS1
    REDIS1 -->|Store| MEM1
```

### Cache Key Strategy

```mermaid
graph TB
    subgraph "Cache Key Components"
        COMPILER[Compiler ID + Version]
        SOURCE[Source Code Hash]
        OPTIONS[Compiler Options]
        LIBS[Library Dependencies]
        LANG[Language Version]
    end
    
    subgraph "Key Generation"
        HASH[SHA-256 Hasher]
        KEY[Cache Key]
    end
    
    COMPILER --> HASH
    SOURCE --> HASH
    OPTIONS --> HASH
    LIBS --> HASH
    LANG --> HASH
    
    HASH --> KEY
    
    KEY --> LOOKUP[Cache Lookup]
```

## Configuration Loading

### Configuration Cascade

```mermaid
flowchart TD
    subgraph "Configuration Sources"
        DEFAULT[defaults.properties]
        ENV[environment.properties]
        PLATFORM[platform.properties]
        LOCAL[local.properties]
    end
    
    subgraph "Loading Process"
        LOADER[Configuration Loader]
        PARSER[Property Parser]
        VALIDATOR[Configuration Validator]
    end
    
    subgraph "Runtime Configuration"
        MEMORY[In-Memory Config]
        CACHE[Config Cache]
        RELOAD[Hot Reload Trigger]
    end
    
    DEFAULT --> LOADER
    ENV --> LOADER
    PLATFORM --> LOADER
    LOCAL --> LOADER
    
    LOADER --> PARSER
    PARSER --> VALIDATOR
    VALIDATOR --> MEMORY
    
    MEMORY --> CACHE
    RELOAD --> LOADER
```

### Configuration Inheritance

```mermaid
graph TB
    subgraph "Group Inheritance"
        BASE_GROUP[&base_compiler]
        LANG_GROUP[&cpp_compiler]
        SPECIFIC[compiler.gcc121]
    end
    
    subgraph "Property Resolution"
        INHERIT[Inheritance Resolver]
        OVERRIDE[Override Handler]
        FINAL[Final Configuration]
    end
    
    BASE_GROUP --> INHERIT
    LANG_GROUP --> INHERIT
    SPECIFIC --> OVERRIDE
    
    INHERIT --> OVERRIDE
    OVERRIDE --> FINAL
    
    FINAL --> COMPILER_INSTANCE[Compiler Instance]
```

## Error Handling Flow

### Compilation Error Processing

```mermaid
flowchart TD
    subgraph "Error Sources"
        COMPILER_ERROR[Compiler Error]
        SYSTEM_ERROR[System Error]
        TIMEOUT_ERROR[Timeout Error]
        RESOURCE_ERROR[Resource Error]
    end
    
    subgraph "Error Processing"
        DETECTOR[Error Detector]
        CLASSIFIER[Error Classifier]
        FORMATTER[Error Formatter]
    end
    
    subgraph "Error Response"
        USER_MESSAGE[User-friendly Message]
        TECHNICAL_DETAILS[Technical Details]
        SUGGESTIONS[Suggestions/Fixes]
    end
    
    subgraph "Monitoring"
        LOGGER[Error Logger]
        METRICS[Error Metrics]
        ALERTS[Alert System]
    end
    
    COMPILER_ERROR --> DETECTOR
    SYSTEM_ERROR --> DETECTOR
    TIMEOUT_ERROR --> DETECTOR
    RESOURCE_ERROR --> DETECTOR
    
    DETECTOR --> CLASSIFIER
    CLASSIFIER --> FORMATTER
    
    FORMATTER --> USER_MESSAGE
    FORMATTER --> TECHNICAL_DETAILS
    FORMATTER --> SUGGESTIONS
    
    DETECTOR --> LOGGER
    LOGGER --> METRICS
    METRICS --> ALERTS
```

### Error Recovery Patterns

```mermaid
stateDiagram-v2
    [*] --> Normal
    Normal --> CompilationError: Compilation fails
    Normal --> SystemError: System failure
    Normal --> TimeoutError: Request timeout
    
    CompilationError --> ErrorReporting: Log and report
    SystemError --> Retry: Attempt retry
    TimeoutError --> Cleanup: Clean resources
    
    Retry --> Normal: Success
    Retry --> FallbackMode: Max retries exceeded
    
    ErrorReporting --> Normal: Continue operation
    Cleanup --> Normal: Resources cleaned
    
    FallbackMode --> Normal: Manual intervention
    FallbackMode --> [*]: Service restart required
```

This data flow documentation provides detailed insights into how information moves through Compiler Explorer, complementing the structural architecture documentation with behavioral and processing patterns.