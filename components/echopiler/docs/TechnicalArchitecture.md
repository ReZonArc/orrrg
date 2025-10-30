# Compiler Explorer Technical Architecture

This document provides a comprehensive overview of Compiler Explorer's technical architecture, including system components, data flows, and key design patterns.

## Table of Contents

- [Overview](#overview)
- [High-Level System Architecture](#high-level-system-architecture)
- [Frontend Architecture](#frontend-architecture)
- [Backend Architecture](#backend-architecture)
- [Compilation System](#compilation-system)
- [Storage and Caching](#storage-and-caching)
- [Execution System](#execution-system)
- [Deployment Architecture](#deployment-architecture)
- [Key Design Patterns](#key-design-patterns)
- [Related Documentation](#related-documentation)

## Overview

Compiler Explorer is a web-based interactive compiler exploration tool built with:

- **Frontend**: TypeScript with Monaco Editor and GoldenLayout
- **Backend**: Node.js with Express
- **Configuration**: Hierarchical `.properties` files
- **Compilation**: Multi-compiler support with worker queues
- **Storage**: S3, local filesystem, and caching layers
- **Deployment**: Docker containers with AWS infrastructure

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Client Browser"
        UI[Monaco Editor UI]
        GL[GoldenLayout Panels]
        WS[WebSocket Client]
    end
    
    subgraph "Load Balancer"
        LB[AWS ALB/CloudFront]
    end
    
    subgraph "Web Servers"
        WS1[Web Server 1]
        WS2[Web Server 2]
        WSN[Web Server N]
    end
    
    subgraph "Compilation Workers"
        CW1[Compilation Worker 1]
        CW2[Compilation Worker 2]
        CWN[Compilation Worker N]
    end
    
    subgraph "Execution Workers"
        EW1[Execution Worker 1]
        EW2[Execution Worker 2]
        EWN[Execution Worker N]
    end
    
    subgraph "Message Queues"
        CSQ[Compilation SQS Queue]
        ESQ[Execution SQS Queue]
    end
    
    subgraph "Storage"
        S3[S3 Bucket]
        CACHE[Redis Cache]
        FS[Local File System]
    end
    
    subgraph "External Services"
        METRICS[Prometheus/Grafana]
        SENTRY[Sentry]
        SPONSORS[Sponsors API]
    end
    
    UI --> LB
    LB --> WS1
    LB --> WS2
    LB --> WSN
    
    WS1 --> CSQ
    WS2 --> CSQ
    WSN --> CSQ
    
    WS1 --> ESQ
    WS2 --> ESQ
    WSN --> ESQ
    
    CSQ --> CW1
    CSQ --> CW2
    CSQ --> CWN
    
    ESQ --> EW1
    ESQ --> EW2
    ESQ --> EWN
    
    CW1 --> S3
    CW2 --> S3
    CWN --> S3
    
    WS1 --> S3
    WS2 --> S3
    WSN --> S3
    
    WS1 --> CACHE
    WS2 --> CACHE
    WSN --> CACHE
    
    CW1 --> FS
    CW2 --> FS
    CWN --> FS
    
    WS1 --> METRICS
    WS1 --> SENTRY
    WS1 --> SPONSORS
```

## Frontend Architecture

The frontend is built with TypeScript and uses Monaco Editor for code editing with GoldenLayout for flexible UI arrangement.

```mermaid
graph TB
    subgraph "Frontend Application"
        subgraph "Core Components"
            MAIN[main.ts - App Entry Point]
            HUB[Hub - Event Management]
            COMPONENTS[Components - UI Factory]
        end
        
        subgraph "Editor Layer"
            MONACO[Monaco Editor]
            THEMES[Theme System]
            LANG[Language Support]
        end
        
        subgraph "UI Panels (GoldenLayout)"
            EDITOR[Editor Pane]
            COMPILER[Compiler Pane]
            OUTPUT[Output Pane]
            TOOLS[Tool Panes]
            DIFF[Diff View]
            EXEC[Execution Pane]
        end
        
        subgraph "Services"
            API[API Client]
            SHARING[Sharing Service]
            LOCAL[Local Storage]
            SETTINGS[Settings]
        end
        
        subgraph "Event System"
            EVHUB[Event Hub]
            EVMAP[Event Map]
            WS[WebSocket Handler]
        end
    end
    
    MAIN --> HUB
    MAIN --> COMPONENTS
    HUB --> EVHUB
    COMPONENTS --> EDITOR
    COMPONENTS --> COMPILER
    COMPONENTS --> OUTPUT
    COMPONENTS --> TOOLS
    COMPONENTS --> DIFF
    COMPONENTS --> EXEC
    
    EDITOR --> MONACO
    EDITOR --> LANG
    COMPILER --> API
    OUTPUT --> API
    TOOLS --> API
    EXEC --> API
    
    API --> WS
    SHARING --> API
    SETTINGS --> LOCAL
    
    EVHUB --> EVMAP
    EVMAP --> WS
```

### Key Frontend Components

- **main.ts**: Application entry point, initializes all systems
- **Hub**: Central event management and component coordination
- **Monaco Editor**: Code editor with syntax highlighting and IntelliSense
- **GoldenLayout**: Flexible panel layout system
- **Event System**: Manages communication between components
- **API Client**: Handles REST API and WebSocket communication

## Backend Architecture

The backend is a Node.js application with Express serving REST APIs and managing compilation/execution workflows.

```mermaid
graph TB
    subgraph "Backend Application"
        subgraph "Application Layer"
            APP[app.ts - Main Entry]
            SERVER[Express Server]
            ROUTES[Route Handlers]
        end
        
        subgraph "API Layer"
            API[API Controllers]
            MIDDLEWARE[Middleware]
            AUTH[Authentication]
        end
        
        subgraph "Service Layer"
            COMP_FINDER[Compiler Finder]
            LANG_SERVICE[Language Service]
            FORMAT_SERVICE[Formatting Service]
            STORAGE_SERVICE[Storage Service]
        end
        
        subgraph "Compilation System"
            BASE_COMPILER[Base Compiler]
            COMP_IMPLS[Compiler Implementations]
            COMP_ARGS[Argument Parsers]
        end
        
        subgraph "Execution System"
            EXEC_ENV[Execution Environment]
            REMOTE_EXEC[Remote Execution]
            EXEC_QUEUE[Execution Queue]
        end
        
        subgraph "Infrastructure"
            CONFIG[Configuration System]
            LOGGING[Logger]
            METRICS[Metrics]
            SENTRY_BACKEND[Sentry Integration]
        end
    end
    
    APP --> SERVER
    SERVER --> ROUTES
    ROUTES --> API
    API --> MIDDLEWARE
    API --> AUTH
    
    API --> COMP_FINDER
    API --> LANG_SERVICE
    API --> FORMAT_SERVICE
    API --> STORAGE_SERVICE
    
    COMP_FINDER --> BASE_COMPILER
    BASE_COMPILER --> COMP_IMPLS
    BASE_COMPILER --> COMP_ARGS
    
    API --> EXEC_ENV
    EXEC_ENV --> REMOTE_EXEC
    EXEC_ENV --> EXEC_QUEUE
    
    SERVER --> CONFIG
    SERVER --> LOGGING
    SERVER --> METRICS
    SERVER --> SENTRY_BACKEND
```

### Key Backend Components

- **Express Server**: HTTP server with REST API endpoints
- **Compiler Finder**: Discovers and manages available compilers
- **Base Compiler**: Abstract base class for all compiler implementations
- **Configuration System**: Hierarchical `.properties` file system
- **Storage Service**: Manages S3, local, and cached storage

## Compilation System

Compiler Explorer supports multiple compilation modes with both synchronous and asynchronous processing.

```mermaid
graph TB
    subgraph "Compilation Flow"
        subgraph "Request Processing"
            REQ[Compilation Request]
            VALIDATE[Request Validation]
            PARSE[Request Parsing]
        end
        
        subgraph "Compiler Selection"
            FINDER[Compiler Finder]
            CONFIG[Compiler Config]
            INSTANCE[Compiler Instance]
        end
        
        subgraph "Compilation Modes"
            SYNC[Synchronous Mode]
            ASYNC[Asynchronous Mode]
            WORKER[Worker Mode]
        end
        
        subgraph "Processing"
            PREP[Source Preparation]
            COMPILE[Compilation]
            POST[Post-processing]
        end
        
        subgraph "Output Generation"
            ASM[Assembly Output]
            BINARY[Binary Output]
            TOOLS[Tool Results]
            ERRORS[Error Messages]
        end
        
        subgraph "Storage & Caching"
            CACHE_CHECK[Cache Check]
            CACHE_STORE[Cache Store]
            S3_STORE[S3 Storage]
        end
    end
    
    REQ --> VALIDATE
    VALIDATE --> PARSE
    PARSE --> FINDER
    FINDER --> CONFIG
    CONFIG --> INSTANCE
    
    INSTANCE --> SYNC
    INSTANCE --> ASYNC
    INSTANCE --> WORKER
    
    SYNC --> PREP
    ASYNC --> PREP
    WORKER --> PREP
    
    PREP --> CACHE_CHECK
    CACHE_CHECK --> COMPILE
    COMPILE --> POST
    
    POST --> ASM
    POST --> BINARY
    POST --> TOOLS
    POST --> ERRORS
    
    ASM --> CACHE_STORE
    BINARY --> CACHE_STORE
    TOOLS --> CACHE_STORE
    
    CACHE_STORE --> S3_STORE
```

### Compilation Worker Architecture

For high-load scenarios, Compiler Explorer uses SQS-based worker queues:

```mermaid
graph LR
    subgraph "Web Server"
        API[API Request]
        QUEUE_PUB[Queue Publisher]
        WS_HANDLER[WebSocket Handler]
    end
    
    subgraph "AWS SQS"
        COMP_QUEUE[Compilation Queue]
        EXEC_QUEUE[Execution Queue]
    end
    
    subgraph "Compilation Workers"
        WORKER1[Worker 1]
        WORKER2[Worker 2]
        WORKERN[Worker N]
    end
    
    subgraph "Results"
        WS_SEND[WebSocket Sender]
        S3_RESULT[S3 Result Storage]
    end
    
    API --> QUEUE_PUB
    QUEUE_PUB --> COMP_QUEUE
    QUEUE_PUB --> EXEC_QUEUE
    
    COMP_QUEUE --> WORKER1
    COMP_QUEUE --> WORKER2
    COMP_QUEUE --> WORKERN
    
    WORKER1 --> WS_SEND
    WORKER2 --> WS_SEND
    WORKERN --> WS_SEND
    
    WORKER1 --> S3_RESULT
    WORKER2 --> S3_RESULT
    WORKERN --> S3_RESULT
    
    WS_SEND --> WS_HANDLER
```

## Storage and Caching

Compiler Explorer uses a multi-tier storage system for optimal performance:

```mermaid
graph TB
    subgraph "Storage Architecture"
        subgraph "Client Layer"
            BROWSER[Browser Cache]
            LOCAL_STORAGE[Local Storage]
        end
        
        subgraph "CDN Layer"
            CLOUDFRONT[CloudFront CDN]
            STATIC_CACHE[Static Asset Cache]
        end
        
        subgraph "Application Layer"
            MEMORY_CACHE[In-Memory Cache]
            REDIS[Redis Cache]
            FILE_CACHE[File System Cache]
        end
        
        subgraph "Persistent Storage"
            S3_PRIMARY[S3 Primary Storage]
            S3_BACKUP[S3 Backup Storage]
            LOCAL_FS[Local File System]
        end
        
        subgraph "External Storage"
            SHORTENER[URL Shortener Storage]
            CONFIG_STORAGE[Configuration Storage]
        end
    end
    
    BROWSER --> CLOUDFRONT
    LOCAL_STORAGE --> MEMORY_CACHE
    
    CLOUDFRONT --> STATIC_CACHE
    STATIC_CACHE --> S3_PRIMARY
    
    MEMORY_CACHE --> REDIS
    REDIS --> FILE_CACHE
    FILE_CACHE --> S3_PRIMARY
    
    S3_PRIMARY --> S3_BACKUP
    S3_PRIMARY --> LOCAL_FS
    
    MEMORY_CACHE --> SHORTENER
    MEMORY_CACHE --> CONFIG_STORAGE
```

### Storage Types

1. **S3 Storage**: Primary storage for compilation results and large artifacts
2. **Redis Cache**: Fast access cache for frequently used data
3. **File System Cache**: Local caching for compilers and temporary files
4. **Memory Cache**: In-process caching for configuration and metadata

## Execution System

The execution system supports both local and remote code execution with security isolation:

```mermaid
graph TB
    subgraph "Execution Architecture"
        subgraph "Request Processing"
            EXEC_REQ[Execution Request]
            VALIDATION[Security Validation]
            ENV_SETUP[Environment Setup]
        end
        
        subgraph "Execution Environments"
            LOCAL_ENV[Local Environment]
            REMOTE_ENV[Remote Environment]
            DOCKER_ENV[Docker Environment]
        end
        
        subgraph "Security & Isolation"
            SANDBOX[Sandboxing]
            TIMEOUT[Timeout Control]
            RESOURCE_LIMITS[Resource Limits]
        end
        
        subgraph "Execution Modes"
            DIRECT[Direct Execution]
            WINE[Wine Emulation]
            QEMU[QEMU Emulation]
        end
        
        subgraph "Output Handling"
            STDOUT[Standard Output]
            STDERR[Standard Error]
            EXIT_CODE[Exit Code]
            METRICS_OUT[Execution Metrics]
        end
    end
    
    EXEC_REQ --> VALIDATION
    VALIDATION --> ENV_SETUP
    
    ENV_SETUP --> LOCAL_ENV
    ENV_SETUP --> REMOTE_ENV
    ENV_SETUP --> DOCKER_ENV
    
    LOCAL_ENV --> SANDBOX
    REMOTE_ENV --> SANDBOX
    DOCKER_ENV --> SANDBOX
    
    SANDBOX --> TIMEOUT
    TIMEOUT --> RESOURCE_LIMITS
    
    RESOURCE_LIMITS --> DIRECT
    RESOURCE_LIMITS --> WINE
    RESOURCE_LIMITS --> QEMU
    
    DIRECT --> STDOUT
    WINE --> STDOUT
    QEMU --> STDOUT
    
    STDOUT --> STDERR
    STDERR --> EXIT_CODE
    EXIT_CODE --> METRICS_OUT
```

## Deployment Architecture

Compiler Explorer is deployed using containerized microservices on AWS:

```mermaid
graph TB
    subgraph "AWS Cloud Infrastructure"
        subgraph "Edge & Load Balancing"
            CF[CloudFront CDN]
            ALB[Application Load Balancer]
            R53[Route 53 DNS]
        end
        
        subgraph "Compute (ECS/EC2)"
            subgraph "Web Tier"
                WEB1[Web Server 1]
                WEB2[Web Server 2]
                WEBN[Web Server N]
            end
            
            subgraph "Worker Tier"
                COMP_WORKERS[Compilation Workers]
                EXEC_WORKERS[Execution Workers]
            end
        end
        
        subgraph "Message Queues"
            SQS_COMP[SQS Compilation Queue]
            SQS_EXEC[SQS Execution Queue]
        end
        
        subgraph "Storage Services"
            S3_BUCKET[S3 Storage Bucket]
            S3_OVERFLOW[S3 Overflow Bucket]
            EFS[EFS File System]
        end
        
        subgraph "Caching & Databases"
            ELASTICACHE[ElastiCache Redis]
            DYNAMODB[DynamoDB]
        end
        
        subgraph "Monitoring & Logging"
            CLOUDWATCH[CloudWatch]
            PROMETHEUS[Prometheus]
            GRAFANA[Grafana]
        end
        
        subgraph "Security & Networking"
            VPC[VPC]
            SECURITY_GROUPS[Security Groups]
            IAM[IAM Roles]
        end
    end
    
    R53 --> CF
    CF --> ALB
    ALB --> WEB1
    ALB --> WEB2
    ALB --> WEBN
    
    WEB1 --> SQS_COMP
    WEB2 --> SQS_COMP
    WEBN --> SQS_COMP
    
    WEB1 --> SQS_EXEC
    WEB2 --> SQS_EXEC
    WEBN --> SQS_EXEC
    
    SQS_COMP --> COMP_WORKERS
    SQS_EXEC --> EXEC_WORKERS
    
    COMP_WORKERS --> S3_BUCKET
    EXEC_WORKERS --> S3_BUCKET
    
    WEB1 --> S3_BUCKET
    WEB2 --> S3_BUCKET
    WEBN --> S3_BUCKET
    
    COMP_WORKERS --> S3_OVERFLOW
    EXEC_WORKERS --> S3_OVERFLOW
    
    WEB1 --> ELASTICACHE
    WEB2 --> ELASTICACHE
    WEBN --> ELASTICACHE
    
    WEB1 --> EFS
    WEB2 --> EFS
    WEBN --> EFS
    
    COMP_WORKERS --> EFS
    EXEC_WORKERS --> EFS
    
    WEB1 --> CLOUDWATCH
    COMP_WORKERS --> CLOUDWATCH
    EXEC_WORKERS --> CLOUDWATCH
    
    CLOUDWATCH --> PROMETHEUS
    PROMETHEUS --> GRAFANA
    
    ALL --> VPC
    ALL --> SECURITY_GROUPS
    ALL --> IAM
```

## Key Design Patterns

### 1. Configuration Hierarchy

Compiler Explorer uses a sophisticated configuration system with inheritance:

```mermaid
graph TB
    subgraph "Configuration Hierarchy"
        DEFAULTS[defaults.properties]
        ENV[environment.properties]
        PLATFORM[platform.properties]
        HOST[hostname.properties]
        LOCAL[local.properties]
    end
    
    DEFAULTS --> ENV
    ENV --> PLATFORM
    PLATFORM --> HOST
    HOST --> LOCAL
    
    LOCAL --> FINAL[Final Configuration]
```

### 2. Plugin Architecture

Compilers and tools follow a plugin pattern:

```mermaid
graph TB
    subgraph "Plugin System"
        BASE[Base Compiler Interface]
        
        subgraph "Compiler Implementations"
            GCC[GCC Compiler]
            CLANG[Clang Compiler]
            MSVC[MSVC Compiler]
            RUST[Rust Compiler]
            GO[Go Compiler]
            OTHER[Other Compilers...]
        end
        
        subgraph "Tool Implementations"
            OBJDUMP[Objdump Tool]
            READELF[Readelf Tool]
            STRINGS[Strings Tool]
            CUSTOM[Custom Tools...]
        end
    end
    
    BASE --> GCC
    BASE --> CLANG
    BASE --> MSVC
    BASE --> RUST
    BASE --> GO
    BASE --> OTHER
    
    BASE --> OBJDUMP
    BASE --> READELF
    BASE --> STRINGS
    BASE --> CUSTOM
```

### 3. Event-Driven Architecture

The frontend uses event-driven communication:

```mermaid
graph TB
    subgraph "Event System"
        HUB[Event Hub]
        
        subgraph "Event Types"
            COMPILE[Compilation Events]
            EDIT[Editor Events]
            LAYOUT[Layout Events]
            SHARING[Sharing Events]
        end
        
        subgraph "Listeners"
            COMPILER_PANE[Compiler Pane]
            EDITOR_PANE[Editor Pane]
            OUTPUT_PANE[Output Pane]
            TOOLS_PANE[Tools Pane]
        end
    end
    
    HUB --> COMPILE
    HUB --> EDIT
    HUB --> LAYOUT
    HUB --> SHARING
    
    COMPILE --> COMPILER_PANE
    EDIT --> EDITOR_PANE
    LAYOUT --> OUTPUT_PANE
    SHARING --> TOOLS_PANE
```

## Related Documentation

- [Configuration System](Configuration.md) - Detailed configuration documentation
- [API Documentation](API.md) - REST API specification
- [What is Compiler Explorer](WhatIsCompilerExplorer.md) - User-facing overview
- [Adding a Compiler](AddingACompiler.md) - Guide for adding new compilers
- [Adding a Language](AddingALanguage.md) - Guide for adding new languages
- [Privacy Policy](Privacy.md) - GDPR compliance and data handling

For implementation details, see the source code in:
- `/lib/` - Backend implementation
- `/static/` - Frontend implementation
- `/etc/config/` - Configuration files
- `/docs/` - Additional documentation