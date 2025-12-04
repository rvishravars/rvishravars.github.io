---
layout: page
title: Design
permalink: /software/design/
---

# Design

[← Back to Software](/software/)

## Immutable datastructures
The paper “Making Data Structures Persistent” introduces methods to preserve all versions of data structures instead of overwriting them [3]. It defines partial and full persistence and proposes two key techniques — fat nodes and node copying — to achieve this with minimal time and space overhead. Using these methods, the authors create persistent balanced trees like red-black trees, enabling access and updates across versions efficiently. Foundational work laid the groundwork for immutable data structures used in modern functional programming and versioned systems.

## Model-Driven Software Development

- **Domain-Specific Modeling**: Creating abstract representations of business processes and data flows using Platform-Independent Models.
- **Automated Code Generation**: Converting high-level models into executable code through transformation engines
- **Rapid Customization**: Modifying business logic through model alterations rather than direct code changes
- **Consistent Architecture**: Maintaining architectural coherence across multiple tenant deployments in SaaS environments

The paper examines how ERP firms in emerging economies, especially India’s Ramco Systems Limited, leveraged model-driven development and cloud computing to overcome structural disadvantages [1].

## Spring

> "Rod Johnson drafted much of his book while trekking to Everest Base Camp." [2]

Spring's core design principle is Inversion of Control (IoC), realized through dependency injection (DI). It emphasizes simple POJO-based components for loose coupling and easier testing. Aspect-Oriented Programming (AOP) is used alongside IoC to separate cross-cutting concerns.
