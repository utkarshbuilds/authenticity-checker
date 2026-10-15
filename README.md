
# Multimodal Content Authenticity Checker

A student proof-of-concept for analyzing the authenticity of digital images and text.

## Goal

Estimate whether submitted content is likely to be:
- AUTHENTIC
- MANIPULATED
- SYNTHETIC
- UNVERIFIABLE

The system will explain its signals, communicate uncertainty, compare content with a small trusted reference set, and log analysis results.

## Planned Features

- Image metadata inspection
- Error Level Analysis (ELA)
- Perceptual image hashing
- Text and claim analysis
- Explainable score fusion
- Analysis history and evaluation metrics
- Streamlit user interface

## Tech Stack

Python, Pillow, OpenCV, ImageHash, NumPy, scikit-learn,
pandas, Streamlit, pytest, and SQLite or JSON.

## Project Status

Phase 0: Initial project setup.

## Setup

Create and activate a Python virtual environment, then install:

    python -m pip install -r requirements.txt

## Important Disclaimer

This is an experimental tool. Its output is an estimate, not
proof that an image or claim is genuine or false.
