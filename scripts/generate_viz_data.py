#!/usr/bin/env python3
"""
Generate JSON data for the DeepKT + WD-IRT Visualization Dashboard.
Reads parquet artifacts from reports/ and writes JSONs to docs/data/.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add project root to path for imports
sys.path.append(str(Path.cwd()))

def main():
    reports_dir = Path("reports")
    output_dir = Path("docs/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading artifacts from {reports_dir}...")
    
    # 1. SAKT Mastery (Sample Student)
    # We'll pick a student with interesting activity
    mastery_path = reports_dir / "sakt_student_state.parquet"
    if mastery_path.exists():
        df = pd.read_parquet(mastery_path)
        # Find a student with many interactions
        top_student = df["user_id"].value_counts().index[0]
        student_df = df[df["user_id"] == top_student].sort_values("position")
        
        # Select top 3 skills
        top_skills = student_df["item_id"].unique()[:3] # Simplified: mapping item to skill would be better but item_id proxy works for viz
        
        # Reshape for line chart
        # We need: position, skill, mastery
        # Since mastery is per-item, we might want to aggregate or just show item mastery
        # For the viz, let's just dump the raw sequence for this student
        
        mastery_data = {
            "student_id": str(top_student),
            "interactions": student_df[["position", "item_id", "response", "mastery"]].to_dict(orient="records")
        }
        
        with open(output_dir / "sakt_mastery.json", "w") as f:
            json.dump({"data": mastery_data}, f, indent=2)
        print(f"✓ Generated sakt_mastery.json (Student {top_student})")

    # 2. WD-IRT Parameters
    params_path = reports_dir / "item_params.parquet"
    if params_path.exists():
        df = pd.read_parquet(params_path)
        # Sample if too large
        if len(df) > 2000:
            df = df.sample(2000, random_state=42)
            
        params_data = df[["item_id", "topic", "difficulty", "guessing"]].to_dict(orient="records")
        
        with open(output_dir / "wd_irt_params.json", "w") as f:
            json.dump({"data": params_data}, f, indent=2)
        print(f"✓ Generated wd_irt_params.json ({len(df)} items)")

    # 3. Attention Heatmap
    attn_path = reports_dir / "sakt_attention.parquet"
    if attn_path.exists():
        df = pd.read_parquet(attn_path)
        # Pick one interesting example (e.g., last position of a student)
        if not df.empty:
            sample = df.iloc[0]
            # The parquet has 'top_influences' list, not the full matrix
            # For the heatmap viz, we ideally want the full matrix.
            # However, if we only have top influences, we can visualize those.
            # But the plan asked for an "Animated Heatmap".
            # If the parquet only has sparse data, we might need to mock the full matrix or use what we have.
            # Let's check the structure.
            # The `export.py` saves `top_influences`.
            # To get a full heatmap, we'd need to change export or just visualize the sparse influences.
            # Let's visualize the sparse influences as a "Focus Map".
            
            attn_data = {
                "user_id": str(sample["user_id"]),
                "position": int(sample["position"]),
                "influences": sample["top_influences"].tolist() if isinstance(sample["top_influences"], np.ndarray) else sample["top_influences"]
            }
            
            with open(output_dir / "attention_data.json", "w") as f:
                json.dump({"data": attn_data}, f, indent=2)
            print("✓ Generated attention_data.json")

    # 4. Gaming Alerts
    # We can run the gaming check script or just mock some if the file doesn't exist
    # Let's try to run the detection logic on the events
    events_path = Path("data/interim/edm_cup_2023_42_events.parquet")
    if events_path.exists():
        from src.common.gaming_detection import generate_gaming_report
        events_df = pd.read_parquet(events_path)
        # Run on a sample to be fast
        sample_events = events_df[events_df["user_id"].isin(events_df["user_id"].unique()[:50])]
        alerts_df = generate_gaming_report(sample_events)
        
        alerts_data = alerts_df.to_dict(orient="records")
        with open(output_dir / "gaming_alerts.json", "w") as f:
            json.dump({"data": alerts_data, "metadata": {"total_flagged": len(alerts_df)}}, f, indent=2)
        print(f"✓ Generated gaming_alerts.json ({len(alerts_df)} alerts)")

    # 5. RL Recommendations
    # We'll use the bandit state if available
    bandit_path = reports_dir / "bandit_state.npz"
    if bandit_path.exists() and params_path.exists():
        # We need to simulate a recommendation
        from src.common.bandit import LinUCBBandit, build_student_context, items_to_arms
        from src.common.recommendation import recommend_items_rl
        
        bandit = LinUCBBandit.load(bandit_path)
        items_df = pd.read_parquet(params_path)
        
        # Pick a student and topic
        if events_path.exists():
            events_df = pd.read_parquet(events_path)
            student_id = events_df["user_id"].iloc[0]
            # Find a topic they've interacted with
            student_events = events_df[events_df["user_id"] == student_id]
            # Extract skill from skill_ids list
            skills = []
            for s in student_events["skill_ids"]:
                if isinstance(s, list): skills.extend(s)
                elif isinstance(s, str): skills.append(s)
            
            topic = skills[0] if skills else items_df["topic"].iloc[0]
            
            recs = recommend_items_rl(
                user_id=student_id,
                target_skill=topic,
                item_params=items_df,
                events_df=events_df,
                bandit=bandit,
                max_items=10
            )
            
            recs_data = []
            for r in recs:
                recs_data.append({
                    "item_id": r.item.item_id,
                    "expected_reward": r.expected_reward,
                    "uncertainty": r.uncertainty,
                    "mode": "Explore" if r.is_exploration else "Exploit",
                    "reason": r.reason
                })
                
            with open(output_dir / "rl_recommendations.json", "w") as f:
                json.dump({"data": recs_data}, f, indent=2)
            print(f"✓ Generated rl_recommendations.json")

    # Check for embed flag
    import sys
    if "--embed" in sys.argv:
        print("Embedding data into data.js...")
        # Collect all data
        all_data = {}
        
        # Load what we just generated
        for filename in ["sakt_mastery.json", "wd_irt_params.json", "attention_data.json", "gaming_alerts.json", "rl_recommendations.json"]:
            path = output_dir / filename
            if path.exists():
                with open(path) as f:
                    all_data[filename] = json.load(f)["data"]
        
        # Write to JS file
        js_content = f"const VIZ_DATA = {json.dumps(all_data)};"
        with open(output_dir.parent / "js/data.js", "w") as f:
            f.write(js_content)
        print(f"✓ Generated data.js with {len(all_data)} datasets")

if __name__ == "__main__":
    main()
