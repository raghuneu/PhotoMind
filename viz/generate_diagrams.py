"""
Generate Mermaid diagram source files for PhotoMind architecture.

These diagrams are embedded inline in TECHNICAL_REPORT.md as ```mermaid blocks
and render natively in GitHub, VS Code, and most markdown viewers.

This script exports standalone .mmd files for use with mermaid-cli (mmdc)
or the Mermaid Live Editor (https://mermaid.live) if PNG/SVG export is needed.

Usage:
    python viz/generate_diagrams.py          # writes .mmd files to viz/diagrams/
    mmdc -i viz/diagrams/arch.mmd -o viz/figures/arch.png   # optional PNG export
"""

import os

DIAGRAMS = {
    "high_level_architecture": """\
graph TB
    subgraph PhotoMind
        subgraph Ingestion["INGESTION PIPELINE<br/>(Process.sequential)"]
            PA["Photo Analyst Agent"]
            PA -->|DirectoryReadTool| Scan["Scan photos/"]
            Scan -->|PhotoVisionTool| Analyze["Analyze via GPT-4o Vision"]
            Analyze --> Index["Index → JSON KB"]
            KR1["Knowledge Retriever Agent"]
        end
        subgraph Query["QUERY PIPELINE<br/>(Process.hierarchical)"]
            Controller["Controller Agent<br/>(Manager/Orchestrator)"]
            Controller --> Retriever["Knowledge Retriever Agent"]
            Controller --> Synthesizer["Insight Synthesizer Agent"]
        end
        KB[("Knowledge Base<br/>photo_index.json")]
        Index --> KB
        Retriever --> KB
    end
""",
    "data_flow_ingestion": """\
graph TD
    A["photos/ directory"] -->|DirectoryReadTool| B["List of image files<br/>(JPG, PNG, HEIC)"]
    B -->|"PhotoVisionTool → GPT-4o Vision API"| C["Per-photo analysis:<br/>{image_type, ocr_text,<br/>description, entities, confidence}"]
    C -->|"FileReadTool + direct write"| D[("knowledge_base/photo_index.json<br/>[25 photos indexed]")]
""",
    "data_flow_query": """\
graph TD
    Q["User natural-language query"] -->|Controller classifies intent| T["Query type:<br/>factual | semantic | behavioral"]
    T -->|PhotoKnowledgeBaseTool| S["Search results with<br/>relevance scores"]
    S -->|Insight Synthesizer| R["Structured answer:<br/>{answer, confidence_grade,<br/>source_photos, reasoning}"]
""",
    "rl_architecture": """\
graph TD
    UQ["User Query"] --> QFE["QueryFeatureExtractor<br/>→ 12-dim feature vector"]
    QFE --> CB["Contextual Bandit<br/>(Thompson Sampling / UCB1 / ε-Greedy)"]
    CB -->|"arm: factual / semantic / behavioral"| PKB["PhotoKnowledgeBaseTool<br/>3 search strategies"]
    PKB --> CS["ConfidenceState → 8-dim state vector"]
    CS --> DQN["ConfidenceDQN<br/>FC(8→64→64→5)"]
    DQN -->|"accept_high / accept_moderate / hedge / decline"| Terminal["Terminal (done=True)"]
    DQN -->|"requery"| Requery["Non-terminal: pick alternate arm,<br/>re-observe results (max 2 steps)"]
    Requery -->|"alternate strategy"| PKB
    Terminal --> IS["Insight Synthesizer<br/>→ graded answer with source attribution"]
""",
    "training_pipeline": """\
graph LR
    subgraph Offline["Offline Simulation"]
        KB2[("photo_index.json")] --> Sim["PhotoMindSimulator<br/>pre-compute 3 strategies × 56 queries"]
        Aug["Query Augmentation<br/>10× synonym + entity swap"] --> Sim
    end
    subgraph Training
        Sim --> Bandit["Train Contextual Bandit<br/>2000 episodes × 5 seeds"]
        Sim --> DQN2["Train DQN Confidence<br/>2000 episodes × 5 seeds<br/>(multi-step with requery)"]
    end
    subgraph Evaluation
        Bandit --> Ablation["7-config Ablation<br/>42 train / 14 held-out"]
        DQN2 --> Ablation
        Ablation --> Stats["Statistical Analysis<br/>95% CI, paired t-test, Cohen's d"]
    end
""",
}


def generate(output_dir: str = "./viz/diagrams"):
    os.makedirs(output_dir, exist_ok=True)
    for name, source in DIAGRAMS.items():
        path = os.path.join(output_dir, f"{name}.mmd")
        with open(path, "w") as f:
            f.write(source)
        print(f"Wrote {path}")
    print(f"\n{len(DIAGRAMS)} diagrams exported to {output_dir}/")
    print("To render PNGs: mmdc -i <file>.mmd -o <file>.png")


if __name__ == "__main__":
    generate()
