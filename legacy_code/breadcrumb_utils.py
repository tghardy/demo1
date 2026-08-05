def build_breadcrumb_string(nodes, rels):
    """Build the exact breadcrumb string that should be passed to the LLM."""
    if not nodes:
        return "Root (No lineage found)"

    root_node = nodes[0]
    root_content = root_node.get("content", "Root")
    root_type = root_node.get("type", "unknown")
    breadcrumb_string = f"{root_content} [{root_type}]"

    for i, rel in enumerate(rels):
        if i + 1 >= len(nodes):
            break

        next_node = nodes[i + 1]
        next_node_name = next_node.get("content", "Unknown")
        next_node_type = next_node.get("type", "unknown")

        if next_node_type == "category":
            next_node_name = f"{next_node_name} (category)"

        rel_type = rel.get("rel_type", "RELATED_TO")
        breadcrumb_string += f" -[{rel_type}]-> {next_node_name}"

    return breadcrumb_string


def escape_rich_markup(text):
    """Escape bracketed text so Rich can display it literally."""
    escaped = text.replace("[", "\\[").replace("]", "\\]")
    return escaped
