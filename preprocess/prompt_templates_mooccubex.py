"""Prompt templates for MOOCCubeX multi-level memory knowledge generation."""


def _format_list(values, empty_text="none observed"):
    values = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(values) if values else empty_text


def build_user_prompt(user_descriptor, sensory_courses, working_courses, long_term_courses):
    """Build the learner profile prompt used for offline LLM generation."""
    return (
        f"Given a user {user_descriptor}, this user's course selections are organized by the "
        "Atkinson-Shiffrin Memory Model into three levels: "
        f"SENSORY MEMORY (immediate exploration needs): {_format_list(sensory_courses)}; "
        f"WORKING MEMORY (current learning session and short-term skill goals): {_format_list(working_courses)}; "
        f"LONG-TERM MEMORY (strategic career planning): {_format_list(long_term_courses)}. "
        "Analyze this user's learning preferences considering factors such as subject domain, "
        "instructional approach, complexity level, pacing and duration, depth versus breadth, "
        "assessment methods, and real-world applications. Provide clear explanations based on "
        "the multilevel memory patterns. Your response must be in English without subtitles, "
        "bullet points, or Chinese text. Translate any Chinese course names to English in your analysis."
    )


def build_item_prompt(title, main_field, prerequisites):
    """Build the static course-profile prompt."""
    return (
        f"Introduce course {title} in the {main_field} domain and describe its cognitive attributes "
        "from the Atkinson-Shiffrin Memory Model perspective considering SENSORY MEMORY impact "
        "(immediate appeal and first impressions), WORKING MEMORY demands (cognitive load and "
        "practical skill building), and LONG-TERM MEMORY value (career development and domain "
        "expertise). Particularly emphasize the prerequisite knowledge requirements and prerequisite "
        f"course dependencies ({prerequisites}), as these are unique characteristics of courses that "
        "determine learning progression and memory consolidation pathways. Explain how prerequisites "
        "relate to different memory levels and learning readiness. Your response must be in English "
        "without subtitles, bullet points, or numbered lists."
    )


def build_mtr_prompt(
    user_descriptor,
    sensory_courses,
    working_courses,
    long_term_courses,
    long_term_fields,
):
    """Build the Memory Transition Reflection prompt."""
    fields = _format_list(long_term_fields)
    courses = _format_list(long_term_courses)
    return (
        f"Given a user {user_descriptor}, this user's learning behaviors are categorized by the "
        "Atkinson-Shiffrin Memory Model: "
        f"SENSORY MEMORY (immediate browsing): {_format_list(sensory_courses)}; "
        f"WORKING MEMORY (current learning session): {_format_list(working_courses)}; "
        f"LONG-TERM MEMORY (established expertise in domains: {fields}, courses: {courses}). "
        "Analyze the memory transition dynamics across the three levels considering attention selection "
        "from sensory to working memory (which browsed courses show signs of active encoding through "
        "topic alignment or prerequisite relevance rather than passive exploration), consolidation from "
        "working to long-term memory (which current learning activities show rehearsal patterns and deep "
        "processing that strengthen existing domain expertise), and retrieval influence from long-term "
        "memory on new learning (how established expertise facilitates or interferes with the processing "
        "of new courses in sensory and working memory). Based on these transition patterns, describe the "
        "overall trajectory from exploration through consolidation to expertise and indicate which current "
        "interests are likely to persist. Your response must be in English without subtitles, bullet points, "
        "or Chinese text. Translate any Chinese course names to English in your analysis."
    )
