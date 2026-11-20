import os
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "variantforge.db"
)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Projects table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        campaign_name TEXT,
        category TEXT DEFAULT 'general',
        brand_name TEXT,
        description TEXT,
        target_platforms TEXT, -- JSON list
        target_regions TEXT,   -- JSON list
        preferred_language TEXT DEFAULT 'en',
        original_image_id TEXT,
        original_image_url TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    # 2. Original Assets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS original_assets (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        file_path TEXT,
        filename TEXT,
        width INTEGER,
        height INTEGER,
        format TEXT,
        file_size INTEGER,
        created_at TEXT
    )
    """)

    # 3. Creative Analysis (Qwen2.5-VL)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS creative_analyses (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        critique_score INTEGER,
        strengths TEXT, -- JSON
        weaknesses TEXT, -- JSON
        recommendations TEXT, -- JSON
        color_palette TEXT, -- JSON
        visual_hierarchy TEXT,
        typography_notes TEXT,
        raw_output TEXT,
        created_at TEXT
    )
    """)

    # 4. OCR Results (PaddleOCR / EasyOCR)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ocr_results (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        elements TEXT, -- JSON array of {text, confidence, box}
        corrected_text TEXT,
        created_at TEXT
    )
    """)

    # 5. Protected Regions (Logos, pricing, headlines)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS protected_regions (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        region_type TEXT, -- logo, headline, cta, price, disclaimer
        box TEXT, -- JSON [x, y, w, h] or polygon
        content TEXT,
        is_locked INTEGER DEFAULT 1,
        created_at TEXT
    )
    """)

    # 6. Editing Layers (Fabric.js layers: text, image, shape, logo, background)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS editing_layers (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        layer_index INTEGER,
        layer_type TEXT, -- text, image, shape, logo, background
        fabric_json TEXT,
        is_visible INTEGER DEFAULT 1,
        is_locked INTEGER DEFAULT 0,
        created_at TEXT
    )
    """)

    # 7. Generated Variants
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS generated_variants (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        platform TEXT,
        width INTEGER,
        height INTEGER,
        language TEXT DEFAULT 'en',
        design_style TEXT,
        generation_model TEXT, -- SDXL Inpaint, ControlNet, FLUX.2 Klein
        image_id TEXT,
        image_url TEXT,
        created_at TEXT
    )
    """)

    # 8. Brand Assets (reusable)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS brand_assets (
        id TEXT PRIMARY KEY,
        brand_name TEXT,
        asset_type TEXT, -- logo, color_palette, font, product_image
        asset_value TEXT,
        created_at TEXT
    )
    """)

    # 9. Project History / Audit Log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS project_history (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        operation TEXT,
        model TEXT,
        duration_ms REAL,
        output_reference TEXT,
        status TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Database initialized successfully at {DB_PATH}")

# Initialize tables immediately upon module load
init_db()
