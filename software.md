---
layout: page
title: Software
permalink: /software/
---

# Software

[← Back to Home](/)

This page collects my notes, tutorials, tips, and things I learn in software engineering.

## Table of contents

- [Machine Learning](#machine-learning)
- [Architecture](#architecture)
- [Design](#design)
- [References](#references)

## Architecture

## Design
### Immutable datastructures
The paper “Making Data Structures Persistent” introduces methods to preserve all versions of data structures instead of overwriting them [3]. It defines partial and full persistence and proposes two key techniques — fat nodes and node copying — to achieve this with minimal time and space overhead. Using these methods, the authors create persistent balanced trees like red-black trees, enabling access and updates across versions efficiently. Foundational work laid the groundwork for immutable data structures used in modern functional programming and versioned systems.

### Model-Driven Software Development

- **Domain-Specific Modeling**: Creating abstract representations of business processes and data flows using Platform-Independent Models.
- **Automated Code Generation**: Converting high-level models into executable code through transformation engines
- **Rapid Customization**: Modifying business logic through model alterations rather than direct code changes
- **Consistent Architecture**: Maintaining architectural coherence across multiple tenant deployments in SaaS environments

The paper examines how ERP firms in emerging economies, especially India’s Ramco Systems Limited, leveraged model-driven development and cloud computing to overcome structural disadvantages [1].

### Spring

> "Rod Johnson drafted much of his book while trekking to Everest Base Camp." [2]

Spring's core design principle is Inversion of Control (IoC), realized through dependency injection (DI). It emphasizes simple POJO-based components for loose coupling and easier testing. Aspect-Oriented Programming (AOP) is used alongside IoC to separate cross-cutting concerns.

## Databases

# References

[1] S. Pinjala, P. Seetharaman, and R. Roy, “Impact of Cloud on Firm Evolution: A Causal Model of a Latecomer ERP Firm in an Emerging Economy,” Proceedings of the 38th International Conference on Information Systems (ICIS), Seoul, 2017.

[2] R. Johnson, Expert One-on-One J2EE Design and Development. Wiley, 2002.

[3] Driscoll, J. R.; Sarnak, N.; Sleator, D. D.; and Tarjan, R. E. 1989. *Making Data Structures Persistent*. Journal of Computer and System Sciences 38, 1 (Feb. 1989), 86–124. DOI:[10.1016/0022-0000(89)90034-2](https://doi.org/10.1016/0022-0000(89)90034-2)