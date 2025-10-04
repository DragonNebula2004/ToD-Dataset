import json
from papers.question_answering_fullwiki_papers import papers as qa_papers
from papers.depth_perception_papers import papers as depth_papers
from papers.image_segmentation_anomaly_track_papers import papers as seg_papers
from papers.GLUE_papers import papers as glue_papers
from itertools import combinations

output = []

task_descriptions = {
    "open-domain question answering tasks": "Open-domain question answering involves developing models that can answer questions using information from large, unstructured text corpora, requiring advanced retrieval and reasoning capabilities.",
    "depth perception tasks": "Depth perception tasks focus on estimating the distance of objects from a single image or sequence, which is crucial for applications in robotics, autonomous driving, and 3D scene understanding.",
    "image segmentation anomaly detection tasks": "Image segmentation anomaly detection aims to identify and segment unusual or unexpected regions in images, supporting safety-critical applications in medical imaging, manufacturing, and autonomous systems.",
    "general language understanding tasks": "General language understanding tasks evaluate models on a broad set of natural language processing challenges, such as sentiment analysis, inference, and textual similarity, to measure overall language comprehension."
}

def add_pairs(papers, task_name, metric, methodology_chunks):
    description = task_descriptions.get(task_name, "")
    for paper1, paper2 in combinations(papers, 2):
        topic = f"{description} What is the better method for {task_name}: {paper1['method_name']} or {paper2['method_name']}?"
        entry = {
            "topic": topic,
            "paper1": {
                "arxiv_link": paper1["arxiv_link"],
                "title": paper1["title"],
                "abstract": paper1["abstract"],
                "methodology": methodology_chunks.get(paper1["method_name"], {}).get("chunks", [])
            },
            "paper2": {
                "arxiv_link": paper2["arxiv_link"],
                "title": paper2["title"],
                "abstract": paper2["abstract"],
                "methodology": methodology_chunks.get(paper2["method_name"], {}).get("chunks", [])
            },
            "ground_truth": "paper1" if paper1[metric] >= paper2[metric] else "paper2"
        }
        output.append(entry)

# Example usage:
with open('methodology_chunks.json') as f:
    methodology_chunks = json.load(f)
# add_pairs(qa_papers, "open-domain question answering tasks", "joint_f1", methodology_chunks)
add_pairs(qa_papers, "open-domain question answering tasks", "joint_f1", methodology_chunks)
add_pairs(depth_papers, "depth perception tasks", "SILog", methodology_chunks)
add_pairs(seg_papers, "image segmentation anomaly detection tasks", "mean_F1", methodology_chunks)
add_pairs(glue_papers, "general language understanding tasks", "score", methodology_chunks)

print(len(output))

with open("dataset.json", "w") as f:
    json.dump(output, f, indent=2)

