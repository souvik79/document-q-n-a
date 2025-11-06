#!/bin/bash

# Test script for CLI application
# This script demonstrates all CLI functionality

PYTHON="/home/souvik/projects/document_q_n_a/env/bin/python"
CLI="cli.py"

echo "==========================================="
echo "Testing Document Q&A CLI"
echo "==========================================="
echo ""

# Test 1: Show help
echo "Test 1: Show help"
echo "-------------------------------------------"
$PYTHON $CLI --help | head -20
echo ""

# Test 2: List projects
echo "Test 2: List projects"
echo "-------------------------------------------"
$PYTHON $CLI project list
echo ""

# Test 3: Create a new project
echo "Test 3: Create new project"
echo "-------------------------------------------"
$PYTHON $CLI project create "CLI Test Project" --description "Testing CLI functionality"
echo ""

# Test 4: List projects again
echo "Test 4: List projects (should show new project)"
echo "-------------------------------------------"
$PYTHON $CLI project list
echo ""

# Test 5: Get project ID (we'll use the latest one)
PROJECT_ID=$(sqlite3 data/app.db "SELECT id FROM projects ORDER BY id DESC LIMIT 1")
echo "Test 5: Using Project ID: $PROJECT_ID"
echo ""

# Test 6: Select project and show stats
echo "Test 6: Select project and show stats"
echo "-------------------------------------------"
$PYTHON $CLI project select $PROJECT_ID
$PYTHON $CLI stats
echo ""

echo "==========================================="
echo "CLI Tests Complete!"
echo "==========================================="
echo ""
echo "Available commands:"
echo "  $PYTHON $CLI project list"
echo "  $PYTHON $CLI project create \"My Project\""
echo "  $PYTHON $CLI project select <id>"
echo "  $PYTHON $CLI document add <file>"
echo "  $PYTHON $CLI document add-url <url>"
echo "  $PYTHON $CLI ask \"Your question\""
echo "  $PYTHON $CLI chat"
echo "  $PYTHON $CLI stats"
echo ""
