# Data Science Challenge 2026

This repository contains materials for the **Data Science Challenge 2026** at **Lawrence Livermore National Laboratory (LLNL)**. The challenge is part of the 2026 Data Science Summer Institute (DSSI) and focuses on applying **agentic AI** to a materials science workflow.

## Goal

The goal of this challenge is to build an AI-assisted workflow for analyzing X-ray CT data from additively manufactured lattice structures. Participants use coding agents, Model Context Protocol (MCP) tools, skills, and subagents to automate steps such as:

- Segmenting CT volumes into material and background regions
- Visualizing 2D slices from 3D datasets
- Skeletonizing segmented lattice structures
- Inspecting lattice defects such as missing, thin, broken, or bent struts
- Generating structured non-destructive evaluation (NDE) reports

By the end of the challenge, participants should understand how to move from standalone scientific scripts toward reusable, agent-driven workflows that can reason over data, call tools, and produce traceable analysis outputs.

## Repository Contents

- `DATA_SCIENCE_CHALLENGE_2026.md/pdf` - main challenge instructions
- `src/` - starter Python code for MCP tools and skeletonization
- `data/` - sample CT, mesh, and lattice structure data
- `images/` - example visualizations used in the challenge materials
- `presentation/` - introductory challenge slides
- `.agents/skills/` - project-specific Codex skills
- `.codex/agents/` - project-specific Codex subagent definitions

## Contact

Haichao Miao
Research Scientist at LLNL
miao1 (at) llnl.gov
