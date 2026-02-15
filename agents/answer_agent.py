#!/usr/bin/python3

import re
import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent
# from .answer_model_llama import AAgent

class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""

    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        self.select_prompt1 = select_prompt1

    def build_prompt(self, question_data: Dict[str, str | Any]) -> Tuple[str, str]:
        """Generate an answer to the given MCQ question with confidence and reasoning"""

        sys_prompt1 = ("You are an expert in quantitative aptitude for competitive exams, solving MCQs with step-by-step reasoning before selecting the correct answer,You are an expert logical reasoning solver,Solve carefully and respond strictly in valid JSON.")
        sys_prompt2 = (
            "You are an expert answer agent specializing in solving multiple-choice questions (MCQs) that test "
            "quantitative aptitude skills, as seen in top-tier competitive exams. "
            "You have a deep understanding of logical reasoning, puzzles, and analytical problem-solving under exam conditions. "
            "For each question, think step by step using a clear chain-of-thought approach. "
            "Break down the problem, analyze all options, eliminate distractors, and then confidently select the correct answer. "
            "Always explain your reasoning before finalizing your choice."
        )

        tmpl = (
            "INSTRUCTIONS FOR ANSWERING:\n"
            "1. Carefully read and understand what is being asked.\n"
            "2. Consider why each choice might be correct or incorrect.\n"
            "3. There is only **ONE OPTION** correct.\n"
            "4. Provide reasoning within 100 words\n\n"
        
            "IMPORTANT RULES (follow strictly):\n"
            "A) OUTPUT ONLY JSON. Do NOT add any extra text, headings, markdown, or option text.\n"
            "B) The 'answer' value must be ONLY ONE LETTER: A or B or C or D.\n"
            "C) Keep reasoning short and factual (<= 100 words).\n\n"
        
            "Carefully resolve family relations step by step before deciding the final relation.\n"
            "FAMILY-RELATION RULES (to avoid common mistakes):\n"
            "1) Resolve the chain LEFT-to-RIGHT.\n"
            "2) 'my mother's only son' = identify that person first (often 'me').\n"
            "3) 'son's wife' means spouse of the son (NOT mother).\n"
            "4) After resolving the person, apply the next relation to that person only.\n\n"
        
            "SEATING/ARRANGEMENT RULES:\n"
            "1) If people face outward, LEFT/RIGHT reverses for them.\n"
            "2) Use opposite/left/right constraints consistently before choosing.\n"
            "3) Do a quick final check that ALL constraints are satisfied.\n\n"
        
            "Now answer the following question:\n"
            "Question: {}\n"
            "Choices: {}\n\n"
            "RESPONSE FORMAT: Strictly generate a valid JSON object as shown below:\n"
            "{{\n"
            '    "answer": "A",\n'
            '    "reasoning": "Brief explanation within 100 words"\n'
            "}}\n"
        )


        prompt = tmpl.format(
            question_data["question"], self._format_choices(question_data["choices"])
        )

        return prompt, sys_prompt1 if self.select_prompt1 else sys_prompt2

    def answer_question(self, question_data, **gen_kwargs):
    
        prompt, system_prompt = self.build_prompt(question_data)
    
        response, token_len, gen_time = self.agent.generate_response(
            prompt,
            system_prompt=system_prompt,
            **gen_kwargs
        )
    
        if isinstance(response, list):
            response = response[0]
    
        response = response.strip()
    
        try:
            result = json.loads(response)
        except:
            # Try extracting answer letter manually
            letter_match = re.search(r"[ABCD]", response)
            answer_letter = letter_match.group(0) if letter_match else "A"
        
            return json.dumps({
                "answer": answer_letter,
                "reasoning": "Auto-extracted due to JSON parse error."
            }), token_len, gen_time

    
        answer = str(result.get("answer", "")).strip().upper()
        reasoning = str(result.get("reasoning", "")).strip()
    
        if answer not in ["A", "B", "C", "D"]:
            answer = "A"
    
        words = reasoning.split()
        if len(words) > 100:
            reasoning = " ".join(words[:100])
    
        final_json = {
            "answer": answer,
            "reasoning": reasoning
        }
    
        return json.dumps(final_json), token_len, gen_time

    def answer_batches(
        self,
        questions: List[Dict[str, Any]],
        batch_size: int = 5,
        **gen_kwargs
    ):
    
        outputs = []
        tls = []
        gts = []
    
        total_batches = (len(questions) + batch_size - 1) // batch_size
    
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
    
            for q in batch:
                answer, tl, gt = self.answer_question(q, **gen_kwargs)
    
                # answer is JSON string → convert to dict
                try:
                    outputs.append(answer)
                except:
                    outputs.append({
                        "answer": "A",
                        "reasoning": "Parsing failed. Default answer selected."
                    })
    
                tls.append(tl if tl else 0)
                gts.append(gt if gt else 0)
    
        return outputs, tls, gts

    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_answers(self, ans: List[str | Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format"""

        def basic_checks(a1: Dict[str, str]) -> bool:
            # check required keys
            required_keys = ["answer"]
            if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
                if len(a1["answer"]) == 1 and (a1["answer"] not in "ABCDabcd"):
                    return False
                check_len = self.count_tokens_a(a1["answer"])
                if check_len < 50:
                    check_len += self.count_tokens_a(a1.get("reasoning", "None"))
                    if check_len < 128:
                        # check answer format - EXTRA checks
                        # if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
                        return True
            return False

        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                if basic_checks(a):
                    filtered_answers.append(a)
                else:
                    filtered_answers.append(None)
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                        print(f"Skipping invalid answer at index {i}: {a}")
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {a}")
                    filtered_answers.append(None)
                    continue
            else:
                # If the answer is neither a dict nor a str, skip it
                print(f"Skipping unsupported type at index {i}: {type(a)}")
                filtered_answers.append(None)
        return filtered_answers

    def save_answers(self, answers: List[str], file_path: str | Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump([a for a in answers], f, indent=4)

    def _format_choices(self, choices: List[str]) -> str:
        r"""Format the choices for better readability"""
        formatted = []
        for choice in choices:
            # Ensure each choice starts with a letter if not already formatted
            if not re.match(r"^[A-D]\)", choice.strip()):
                # Extract letter from existing format or assign based on position
                letter = chr(65 + len(formatted))  # A, B, C, D
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return " ".join(formatted)


# Example usage
if __name__ == "__main__":
    import json
    import yaml
    import argparse
    from utils.build_prompt import auto_json, option_extractor_prompt

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
    argparser.add_argument(
        "--input_file",
        type=str,
        default="outputs/filtered_questions.json",
        help="Path to the input JSON file with questions",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/answers.json",
        help="Path to save the answers",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for processing questions"
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    args = argparser.parse_args()

    SELECT_PROMPT1 = False  # Use the first system prompt for answering

    # Load sample questions (assuming they're saved from QuestioningAgent)
    with open(args.input_file, "r") as f:
        sample_questions = json.load(f)

    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1)

    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions, batch_size=args.batch_size, **gen_kwargs
    )
    ans = []
    for idx, (q, a) in enumerate(zip(sample_questions, answer)):
        if args.verbose:
            print(f"\n=== Question {idx+1} ===")
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected: {q.get('answer', 'N/A')}")
            print(f"Model Answer:\n{a}")
        try:
            a = json.loads(a)
            if all(k in a for k in ["answer", "reasoning"]):
                # ++++++++++++++++++++++++++
                # TODO: IMPROVE THE FOLLOWING
                if len(a["answer"]) != 1:
                    a["answer"] = agent.agent.generate_response(
                        option_extractor_prompt(a["answer"], q["choices"])
                    )
                # ++++++++++++++++++++++++++
            else:
                # the dictionary is not as expected. So extract it using the same model: Self-Reflection
                prompt = (
                    "Extract **ONLY** the answer and reasoning while discarding the rest.\n\n"
                    "String:\n"
                    "{}\n\n"
                    "Given Format:\n"
                    "{{\n"
                    '    "answer": "Only the option letter (A, B, C, or D)",\n'
                    '    "reasoning": "..."\n'
                    "}}"
                )
                a = agent.agent.generate_response(
                    prompt.format(json.dumps(a, indent=4))
                )
        except json.JSONDecodeError:
            a = agent.agent.generate_response(auto_json(a))
        ans.append(a)

    if args.verbose:
        if gen_kwargs.get("tgps_show", False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                print(f"TGPS: {tl/gt:.3f} seconds")
            print("\n" + "=" * 50)
            print(
                f"Total Time: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds"
            )

    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace(
        "answers.json", "filtered_answers.json"
    )
    agent.save_answers(agent.filter_answers(ans), filtered_file_name)
