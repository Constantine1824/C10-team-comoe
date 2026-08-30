# Medical SLM Document Chunking

# This directory handles data serialization and semantic chunking for our Domain-Specific Small Language Model (SLM).

## Overview
# Because medical guidelines contain critical safety rules, dosage instructions, and contraindications, traditional token-#splitting can sever important context. Each row in our dataset represents an atomic clinical guideline, so we use a 1 row = 1 #chunk strategy enriched with structural metadata.
