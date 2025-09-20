import json
from papers.question_answering_fullwiki_papers import papers as qa_papers
from papers.depth_perception_papers import papers as depth_papers
from papers.image_segmentation_anomaly_track_papers import papers as seg_papers
from papers.GLUE_papers import papers as glue_papers
from itertools import combinations

output = []

def add_pairs(papers, task_name, metric):
    for paper1, paper2 in combinations(papers, 2):
        topic = f"What is the better method for {task_name}: {paper1['method_name']} or {paper2['method_name']}?"
        entry = {
            "topic": topic,
            "paper1": {
                "arxiv_link": paper1["arxiv_link"],
                "title": paper1["title"],
                "abstract": paper1["abstract"],
                "introduction": paper1["introduction"]
            },
            "paper2": {
                "arxiv_link": paper2["arxiv_link"],
                "title": paper2["title"],
                "abstract": paper2["abstract"],
                "introduction": paper2["introduction"]
            },
            "ground_truth": "paper1" if paper1[metric] >= paper2[metric] else "paper2"
        }
        output.append(entry)

add_pairs(qa_papers, "open-domain question answering tasks", "joint_f1")
add_pairs(depth_papers, "depth perception tasks", "SILog")
add_pairs(seg_papers, "image segmentation anomaly detection tasks", "mean_F1")
add_pairs(glue_papers, "general language understanding tasks", "score")

print(len(output))

with open("dataset.json", "w") as f:
    json.dump(output, f, indent=2)

