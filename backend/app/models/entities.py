import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.database import get_connection

class ProjectRepository:
    @staticmethod
    def create(data: Dict[str, Any]) -> Dict[str, Any]:
        project_id = data.get("id") or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO projects (
                id, name, campaign_name, category, brand_name, description,
                target_platforms, target_regions, preferred_language,
                original_image_id, original_image_url, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id,
            data.get("name", "Untitled Campaign"),
            data.get("campaign_name", ""),
            data.get("category", "general"),
            data.get("brand_name", ""),
            data.get("description", ""),
            json.dumps(data.get("target_platforms", ["instagram_square"])),
            json.dumps(data.get("target_regions", ["Global"])),
            data.get("preferred_language", "en"),
            data.get("original_image_id"),
            data.get("original_image_url"),
            now,
            now
        ))
        conn.commit()
        conn.close()
        return ProjectRepository.get(project_id)

    @staticmethod
    def get(project_id: str) -> Optional[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        d["target_platforms"] = json.loads(d["target_platforms"]) if d.get("target_platforms") else []
        d["target_regions"] = json.loads(d["target_regions"]) if d.get("target_regions") else []
        return d

    @staticmethod
    def list_all() -> List[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["target_platforms"] = json.loads(d["target_platforms"]) if d.get("target_platforms") else []
            d["target_regions"] = json.loads(d["target_regions"]) if d.get("target_regions") else []
            results.append(d)
        return results

    @staticmethod
    def update(project_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        fields = []
        values = []
        for k, v in updates.items():
            if k in ["target_platforms", "target_regions"] and isinstance(v, (list, dict)):
                v = json.dumps(v)
            fields.append(f"{k} = ?")
            values.append(v)
        fields.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(project_id)
        cursor.execute(f"UPDATE projects SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return ProjectRepository.get(project_id)

    @staticmethod
    def delete(project_id: str) -> bool:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        cursor.execute("DELETE FROM editing_layers WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM generated_variants WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM creative_analyses WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM ocr_results WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM protected_regions WHERE project_id = ?", (project_id,))
        conn.commit()
        conn.close()
        return True


class AnalysisRepository:
    @staticmethod
    def save(project_id: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        aid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO creative_analyses (
                id, project_id, critique_score, strengths, weaknesses,
                recommendations, color_palette, visual_hierarchy,
                typography_notes, raw_output, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            aid,
            project_id,
            analysis.get("critique_score", 85),
            json.dumps(analysis.get("strengths", [])),
            json.dumps(analysis.get("weaknesses", [])),
            json.dumps(analysis.get("recommendations", [])),
            json.dumps(analysis.get("color_palette", [])),
            analysis.get("visual_hierarchy", ""),
            analysis.get("typography_notes", ""),
            analysis.get("raw_output", ""),
            now
        ))
        conn.commit()
        conn.close()
        return analysis

    @staticmethod
    def get_by_project(project_id: str) -> Optional[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM creative_analyses WHERE project_id = ? ORDER BY created_at DESC LIMIT 1", (project_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        for field in ["strengths", "weaknesses", "recommendations", "color_palette"]:
            if d.get(field):
                d[field] = json.loads(d[field])
        return d


class VariantRepository:
    @staticmethod
    def create(variant: Dict[str, Any]) -> Dict[str, Any]:
        vid = variant.get("id") or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO generated_variants (
                id, project_id, platform, width, height, language,
                design_style, generation_model, image_id, image_url, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vid,
            variant.get("project_id"),
            variant.get("platform", "instagram_square"),
            variant.get("width", 1080),
            variant.get("height", 1080),
            variant.get("language", "en"),
            variant.get("design_style", "modern"),
            variant.get("generation_model", "SDXL Inpaint"),
            variant.get("image_id"),
            variant.get("image_url"),
            now
        ))
        conn.commit()
        conn.close()
        return {**variant, "id": vid, "created_at": now}

    @staticmethod
    def list_by_project(project_id: str) -> List[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM generated_variants WHERE project_id = ? ORDER BY created_at DESC", (project_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def list_all() -> List[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM generated_variants ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
