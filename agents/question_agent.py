#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any
from .question_model import QAgent
import random
import json
import re


class QuestioningAgent(object):
    r"""Agent responsible for generating questions (Enhanced with SFT + RL filtering)"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)
        self.max_retries = 3
        self.reward_threshold = 0.75

    # ============================================================
    # 🔥 SFT SYNTHETIC DATASET INJECTION
    # ============================================================

    def _synthetic_sft_examples(self) -> str:
        """High quality curated examples (acts like SFT guidance)"""
        return """
High Quality Example 1:
{
  "topic": "Syllogisms",
  "question": "Statements: All metals are elements. Some elements are radioactive. Conclusion: Some metals are radioactive?",
  "choices": ["A) Follows", "B) Does not follow", "C) Either follows", "D) Cannot be determined"],
  "expected_answer": "B",
  "explanation": "No guaranteed overlap between metals and radioactive elements."
}

High Quality Example 2:
{
  "topic": "Mixed Series (Alphanumeric)",
  "question": "Find the missing term: AB, CDE, FGHI, ?, MNOPQR",
  "choices": ["A) IJKL", "B) JKLM", "C) LMNO", "D) KLMN"],
  "expected_answer": "D",
  "explanation": "Letter blocks increase by one: 2,3,4,5,6."
}
"""

    # ============================================================
    # PROMPT BUILDER (UNCHANGED FORMAT + SFT ADDED)
    # ============================================================

    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:

        correct_option = random.choice(["A", "B", "C", "D"])
        distractors = ", ".join(
            [opt for opt in ["A", "B", "C", "D"] if opt != correct_option]
        )

        sys_prompt = """
You are an expert competitive exam question setter.

STRICT RULES:
- Output ONLY valid JSON.
- Do NOT include markdown.
- Do NOT include code fences.
- Do NOT add extra keys.
- Use key "expected_answer".
- Exactly four options labeled A), B), C), D).
- Only one correct answer.
"""

        prompt = f"""
Generate an EXTREMELY DIFFICULT MCQ on topic: {topic}

Requirements:
1. Only option {correct_option} is correct.
2. Distractors ({distractors}) must reflect realistic misconceptions.
3. Explanation must be concise (<100 words).

{self._synthetic_sft_examples()}

RESPONSE FORMAT:
{{
  "topic": "{topic}",
  "question": "...?",
  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "expected_answer": "{correct_option}",
  "explanation": "..."
}}
"""

        if wicl and inc_samples:
            prompt += "\nREFERENCE EXAMPLES:\n"
            for sample in inc_samples:
                prompt += json.dumps(sample, indent=2) + "\n"

        return prompt, sys_prompt

    # ============================================================
    # 🔥 RL-STYLE REWARD FUNCTION
    # ============================================================

    def _reward_function(self, q: Dict) -> float:
        score = 0

        if isinstance(q.get("choices"), list) and len(q["choices"]) == 4:
            score += 0.25

        if q.get("expected_answer") in ["A", "B", "C", "D"]:
            score += 0.25

        if len(q.get("explanation", "")) > 20:
            score += 0.25

        if "?" in q.get("question", ""):
            score += 0.15

        # Bonus if distractors look different
        if len(set(q.get("choices", []))) == 4:
            score += 0.10

        return score

    # ============================================================
    # JSON VALIDATION
    # ============================================================

    def _validate_json(self, text: str):
        try:
            obj = json.loads(text.strip())
            required = [
                "topic",
                "question",
                "choices",
                "expected_answer",
                "explanation",
            ]
            if not all(k in obj for k in required):
                return None
            return obj
        except:
            return None

    # ============================================================
    # GENERATE SINGLE QUESTION (RL LOOP)
    # ============================================================

    def generate_question(
        self,
        topic,
        wadvsys,
        wicl,
        inc_samples,
        **gen_kwargs,
    ):

        prompt, sys_prompt = self.build_prompt(
            topic, wadvsys, wicl, inc_samples
        )

        best_output = None
        best_score = 0

        for _ in range(self.max_retries):

            resp, tl, gt = self.agent.generate_response(
                prompt,
                sys_prompt,
                max_new_tokens=300,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                tgps_show=gen_kwargs.get("tgps_show", False),
            )

            if isinstance(resp, list):
                resp = resp[0]

            obj = self._validate_json(resp)

            if obj:
                reward = self._reward_function(obj)

                if reward > best_score:
                    best_score = reward
                    best_output = obj

                if reward >= self.reward_threshold:
                    return json.dumps(obj), tl, gt

        # return best candidate
        if best_output:
            return json.dumps(best_output), tl, gt

        return resp, tl, gt

    # ============================================================
    # BATCH GENERATION (SAFE)
    # ============================================================

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ):

        extended_topics = self.populate_topics(topics, num_questions)

        questions = []
        tls, gts = [], []

        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):

            batch_topics = extended_topics[i:i + batch_size]

            for topic_pair in batch_topics:

                topic_str = f"{topic_pair[0]}/{topic_pair[1]}"

                resp, tl, gt = self.generate_question(
                    topic_str,
                    wadvsys,
                    wicl,
                    inc_samples.get(topic_pair[1], []) if inc_samples else None,
                    **kwargs
                )

                questions.append(resp)
                tls.append(tl)
                gts.append(gt)

            pbar.update(1)

        pbar.close()
        return questions, tls, gts

    # ============================================================
    # POPULATE TOPICS (UNCHANGED)
    # ============================================================

    def populate_topics(
        self, topics: Dict[str, List[str]], num_questions: int
    ):
        all_subtopics = [
            (t, st) for t, sublist in topics.items() for st in sublist
        ]
        return random.choices(all_subtopics, k=num_questions)

    @staticmethod
    def load_icl_samples(file_path: str | Path):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, "r") as f:
            return json.load(f)

# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(
        description="Generate questions using the QuestioningAgent."
    )
    argparser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Total number of questions to generate.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/questions.json",
        help="Output file name to save the generated questions.",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for generating questions."
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "=" * 50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(
                f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds\n\n"
            )
        print("\n" + "+" * 50 + "\n")

    # check if question is JSON format
    ques = []
    for q in question:
        try:
            json.loads(q)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in question: {q}\nError: {e}")
            # use agent itself to extract JSON: Self-Reflection
            # the dictionary is not as expected.
            # TODO: IMPROVE THE FOLLOWING
            prompt = (
                "Extract **ONLY** the topic, question, choices, answer, and explanation while discarding the rest.\n"
                "Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n"
                "String:\n"
                "{}\n\n"
                "Given Format:\n"
                "{{\n"
                '  "topic": "...",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "Only the option letter (A, B, C, or D)",\n'
                '  "explanation": "..."\n'
                "}}"
            )
            q = agent.agent.generate_response(
                prompt.format(q),
                "You are an expert JSON extractor.",
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
            )
        ques.append(q)
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace(
        "questions.json", "filtered_questions.json"
    )
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================
