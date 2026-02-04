# import libraries
from prompt_template import generate_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import re
import ast
import json


import logging
import sys

# 1. Setup the logger
logger = logging.getLogger("LLM_Parser")
logger.setLevel(logging.INFO)

# 2. Create formatters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# goal for an Agent
goal_defined = "Solved"

def parse_raw(output, flow):
    if flow == "plan":
        text = output.strip()
        try:
            result = ast.literal_eval(text)
            if isinstance(result, list):
                return [str(x).strip() for x in result]
        except Exception:
            pass  # fall through
        
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                bracket_text = text[start:end + 1]
                result = ast.literal_eval(bracket_text)
                if isinstance(result, list):
                    return [str(x).strip() for x in result]
        except Exception:
            pass  # fall through
        
        lines = text.splitlines()
        numbered_items = []

        for line in lines:
            match = re.match(r"^\s*\d+\.\s+(.*)", line)
            if match:
                numbered_items.append(match.group(1).strip())

        if len(numbered_items) > 0:
            return numbered_items
        
        return f"ERROR: Could not parse plan output:\n{text}"
    
    if flow == "aggregator":
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            # Get start and end indices
            start_index = match.start()
            end_index = match.end()
            
            json_string = match.group(0)
            
            try:
                data = json.loads(json_string)
                return data
            except json.JSONDecodeError as e:
                return f"ERROR: Found braces but JSON is invalid: {str(e)}"
        else:
            return f"ERROR: No proper JSON generated."
        
    if flow == "evaluate":
        matches = re.findall(r'\b(true|false)\b', output, re.IGNORECASE)
    
        # Normalize to lowercase to compare sets
        unique_matches = set(m.lower() for m in matches)
        
        # Logic: If both are present, or neither is present, return None
        if len(unique_matches) != 1:
            return None
        
        # Return the single value found as a boolean
        found = unique_matches.pop()
        return True if found == 'true' else False

"""
LLM initialized here and will be used below for multiple LLM-calls.
"""
# calls LLM
class LLMCallModel():
    def __init__(self, model_path = None):
        if not model_path:
            self.path = "./../models/Llama_3.2_1B_instruct"
        else:
            self.path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path)
        
    def generate(self, prompt, max_seq_op=128):
        logger.info(prompt)
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_seq_op)
        result = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        logger.info(result)
        return result
        

# Agent loop
class AgentLoop():
    def __init__(self, task = None, sub_task = None):
        self.main_task = task
        self.sub_task = sub_task #list of sub-tasks
        self.sub_task_answer = None
        self.state = "initialise"
        self.counter = 0
        self.model = LLMCallModel()
        self.states = defaultdict(list)
        self.final_answer = None
        self.feedback = None
        self.max_steps = 15
        self.step_count = 0
        
    def to_dict(self):
        """JSON-safe representation"""
        return {
            "main_task": str(self.main_task),
            "state": str(self.state),
            "step_count": str(self.step_count),
            "max_steps": str(self.max_steps),
            "subtasks": str(self.sub_task),
            "subtask_answers": str(self.sub_task_answer),
            "final_answer": str(self.final_answer),
            "feedback": str(self.feedback)
        }
        
        
    def agent_call(self):
        counter = 0
        while (self.state != goal_defined) and (counter <= 3) and (self.step_count <= self.max_steps):
            logger.info(f"Present step: {self.step_count} & counter: {counter}")
            ## observe
            ## decide
            ## action
            ## result
            ## update state
            ## termination mode
            if self.state == "initialise":
                data = {"main_task": self.main_task, "feedback": self.feedback}
                prompt = generate_prompt(flow="plan", data=data)
                output = self.model.generate(prompt=prompt, max_seq_op=512)
                logger.info(f"Planning LL o/p: {output}")
                plan_list = parse_raw(output, "plan") # -- observe
                logger.info(f"Plan List: {plan_list}")
                if isinstance(plan_list, list): # -- decide
                    self.sub_task = plan_list # action & result
                    self.sub_task_answer = {x:"" for x in range(len(self.sub_task))}
                    self.state = "subtask" # update state
                else:
                    counter += 1
                    continue
                
            elif self.state == "subtask":
                counter = 0
                for i in range(len(self.sub_task)):
                    try:
                        subtask_counter = 1
                        data = {"subtask": self.sub_task[i]}
                        prompt = generate_prompt(flow="solve_subtask", data=data)
                        output = self.model.generate(prompt=prompt)
                        self.sub_task_answer[i] = output
                        logger.info(f"Subtask {i}: {self.sub_task[i]}")
                        logger.info(f"Model O/p for {i}: {output}")
                    except:
                        subtask_counter += 1
                        while subtask_counter < 3:
                            data = {"subtask": self.sub_task[i]}
                            prompt = generate_prompt(flow="solve_subtask", data=data)
                            output = self.model.generate(prompt=prompt)
                            self.sub_task_answer[i] = output
                            subtask_counter += 1
                        return f"ERROR: in state subtask."
                self.state = "aggregate"
                            
                
            elif self.state == "aggregate":
                counter = 0
                
                try:
                    data = {"main_task": self.main_task, 
                        "list_of_subtask_answers": [self.sub_task[i] + " Answer: " + self.sub_task_answer[i] for i in range(len(self.sub_task))]}
                    prompt = generate_prompt(flow="aggregator", data=data)
                    output = self.model.generate(prompt=prompt)
                    logger.info(f"Output raw ~ Aggregator: {output}")
                    out_fin_ans = parse_raw(output=output, flow="aggregator")
                    
                    logger.info(f"Output parsed: {out_fin_ans}")
                    if isinstance(out_fin_ans, dict):
                        if "final_answer" in out_fin_ans:
                            self.final_answer = out_fin_ans["final_answer"]
                            self.state = "evaluate"
                        else:
                            counter += 1
                    else:
                        counter += 1
                except:
                    counter += 1
                    continue
                
            elif self.state == "evaluate":
                counter = 0
                data = {"main_task": self.main_task, 
                        "final_answer": self.final_answer}
                prompt = generate_prompt(flow="evaluate", data=data)
                output = self.model.generate(prompt=prompt)
                logger.info(f"Output ~ Evaluator: {output}")
                
                op_eval = parse_raw(output=output, flow="evaluate")
                logger.info(f"O/P: {op_eval}")
                if op_eval == True:
                    self.state = "Solved"
                    return self.final_answer
                elif op_eval == False:
                    self.feedback = str(output)
                    self.state = "initialise"
                else:
                    counter += 1
                    continue
            self.step_count += 1
            
def main():
    # Main function or method
    task = """
    Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    """

    agent = AgentLoop(task=task)
    agent.agent_call()
    logger.info(f"Final Answer: {str(agent.final_answer)}")

    with open('phase_0_object.json', 'w') as f:
        json.dump(agent.to_dict(), f, indent=2)
                

if __name__ == "__main__":
    main()