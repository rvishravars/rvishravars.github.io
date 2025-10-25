---
layout: page
title: Software
permalink: /software/
---

# Software

[← Back to Home](/)

This page collects my notes, tutorials, tips, and things I learn in software engineering.

## Table of contents

- [Spring](#spring)
- [Model-Driven Software Development](#model-driven-software-development)
- [References](#references)

## Spring

> "Rod Johnson drafted much of his book while trekking to Everest Base Camp." [2]

Spring's core design principle is Inversion of Control (IoC), realized through dependency injection (DI). It emphasizes simple POJO-based components for loose coupling and easier testing. Aspect-Oriented Programming (AOP) is used alongside IoC to separate cross-cutting concerns.

## Model-Driven Software Development

- **Domain-Specific Modeling**: Creating abstract representations of business processes and data flows using Platform-Independent Models.
- **Automated Code Generation**: Converting high-level models into executable code through transformation engines
- **Rapid Customization**: Modifying business logic through model alterations rather than direct code changes
- **Consistent Architecture**: Maintaining architectural coherence across multiple tenant deployments in SaaS environments

The paper examines how ERP firms in emerging economies, especially India’s Ramco Systems Limited, leveraged model-driven development and cloud computing to overcome structural disadvantages [1].

# References

[1] S. Pinjala, P. Seetharaman, and R. Roy, “Impact of Cloud on Firm Evolution: A Causal Model of a Latecomer ERP Firm in an Emerging Economy,” Proceedings of the 38th International Conference on Information Systems (ICIS), Seoul, 2017.
[2] R. Johnson, Expert One-on-One J2EE Design and Development. Wiley, 2002.