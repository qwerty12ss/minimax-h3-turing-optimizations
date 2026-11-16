"""Build visual-editor workflows from the API graphs and installed node schemas."""
import argparse
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def convert(graph, schemas):
    nodes, by_id, links = [], {}, []
    for index, (key, entry) in enumerate(graph.items()):
        schema = schemas[entry["class_type"]]
        node = dict(id=int(key), type=entry["class_type"], pos=[(index % 5) * 420, (index // 5) * 400],
                    size=[380, 320], flags={}, order=index, mode=0, inputs=[], outputs=[],
                    properties={"Node name for S&R": entry["class_type"]}, widgets_values=[])
        linked = {name for name, value in entry["inputs"].items() if isinstance(value, list)}
        for section in ("required", "optional"):
            definitions = schema.get("input", {}).get(section, {})
            for name in schema.get("input_order", {}).get(section, list(definitions)):
                definition = definitions[name]
                kind = definition[0]
                options = definition[1] if len(definition) > 1 else {}
                widget = isinstance(kind, list) or kind in {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}
                if name in linked or not widget or options.get("forceInput"):
                    node["inputs"].append(dict(name=name, type=kind, link=None))
                elif not options.get("hidden"):
                    default = options.get("default", kind[0] if isinstance(kind, list) else {"STRING": "", "BOOLEAN": False}.get(kind, 0))
                    node["widgets_values"].append(entry["inputs"].get(name, default))
                    if name in ("noise_seed", "seed"):
                        node["widgets_values"].append("fixed")
                    if name == "format":
                        node["widgets_values"].extend([entry["inputs"].get("format.codec", "h264"), "auto"])
        for slot, kind in enumerate(schema.get("output", [])):
            names = schema.get("output_name", schema["output"])
            node["outputs"].append(dict(name=names[slot], type=kind, links=[], slot_index=slot))
        nodes.append(node)
        by_id[key] = node
    for key, entry in graph.items():
        target = by_id[key]
        for name, value in entry["inputs"].items():
            if not isinstance(value, list):
                continue
            source_key, source_slot = value
            target_slot = next(i for i, slot in enumerate(target["inputs"]) if slot["name"] == name)
            link_id = len(links) + 1
            kind = by_id[source_key]["outputs"][source_slot]["type"]
            links.append([link_id, int(source_key), source_slot, int(key), target_slot, kind])
            by_id[source_key]["outputs"][source_slot]["links"].append(link_id)
            target["inputs"][target_slot]["link"] = link_id
    return dict(last_node_id=max(n["id"] for n in nodes), last_link_id=len(links), nodes=nodes, links=links,
                groups=[], config={}, extra={"ds": {"scale": 0.65, "offset": [30, 30]}}, version=0.4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    with urllib.request.urlopen(args.url.rstrip("/") + "/object_info") as response:
        schemas = json.load(response)
    for path in (ROOT / "workflows").glob("*_api.json"):
        graph = json.loads(path.read_text())
        target = path.with_name(path.name.replace("_api", "_gui"))
        target.write_text(json.dumps(convert(graph, schemas), indent=2) + "\n", newline="\n")


if __name__ == "__main__":
    main()
