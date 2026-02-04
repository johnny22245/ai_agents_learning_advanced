"""
Prompts to Solve GSM8K task by braking them into sub-problems.
"""
from jinja2 import Template

def generate_prompt(flow, data):
    prompt_planner = """### Instruction
    You are a planning assistant. Break down the Main Task into a logical, numbered list of subtasks. Do not provide explanations, just the list.

    ### Main Task
    {{main_task}}

    ### Last feedback
    {{feedback}}
    
    ### Output Format
    ["Subtask 1", "Subtask 2", "...", "..."]
    ** Output should be a list.
    """

    solve_subtask_prompt = """### Instruction
    Solve the following Subtask efficiently.
    
    ### Subtask
    {{subtask}}

    ### Response
    Let's think step by step to solve this."""

    aggregator_prompt = """### Instruction
    You are an aggregator. Combine the Subtask Answers below to answer the Main Goal.
    Output ONLY valid JSON. Do not write an intro or outro.

    ### Main Goal
    {{main_task}}

    ### Subtask Answers
    {{list_of_subtask_answers}}
    
    Let's think step by step to solve this.

    ### JSON Output
    {
    "final_answer": "YOUR SYNTHESIS HERE, ONLY ANSWER."
    }"""

    evaluator_prompt = """### Instruction
    Review the Answer below for the Task. Check for factual inconsistencies or logic errors.

    ### Task
    {{main_task}}

    ### Answer
    {{final_answer}}

    ### Evaluation
    First, output "Status: True" if correct, or "Status: False" if incorrect.
    Then, provide "Reasoning:". If False, explain how to fix it.

    Status:"""
    
    if flow == "plan":
        tm = Template(prompt_planner)
    elif flow == "solve_subtask":
        tm = Template(solve_subtask_prompt)
    elif flow == "aggregator":
        tm = Template(aggregator_prompt)
    elif flow == "evaluate":
        tm = Template(evaluator_prompt)
    
    output_prompt = tm.render(data)
    
    return output_prompt
    