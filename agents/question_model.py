# Starting with Qwen3-4B
import time
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer

# question_model.py

import time
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(42)


class QAgent(object):
    """
    High-capacity LLM wrapper with safe sampling control.
    """

    def __init__(self, **kwargs):
        model_name = "Qwen/Qwen3-4B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate_response(
        self,
        message: str | List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ):

        if system_prompt is None:
            system_prompt = "You are a world-class competitive reasoning examiner."

        if isinstance(message, str):
            message = [message]

        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        texts = [
            self.tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=True,
            )
            for m in all_messages
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        tgps_show = kwargs.get("tgps_show", False)

        if tgps_show:
            start_time = time.time()

        # ===============================
        # SAFE SAMPLING LOGIC
        # ===============================

        temperature = kwargs.get("temperature", 0.8)
        top_p = kwargs.get("top_p", 0.95)

        if temperature is None or temperature <= 0:
            do_sample = False
            temperature = None
            top_p = None
        else:
            do_sample = True

        generate_kwargs = dict(
            **model_inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 150),
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        generated_ids = self.model.generate(**generate_kwargs)

        if tgps_show:
            generation_time = time.time() - start_time

        outputs = []
        total_tokens = 0

        for input_ids, generated in zip(
            model_inputs.input_ids, generated_ids
        ):
            output_ids = generated[len(input_ids):]
            total_tokens += len(output_ids)

            content = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            ).strip()

            outputs.append(content)

        if tgps_show:
            return (
                outputs[0] if len(outputs) == 1 else outputs,
                total_tokens,
                generation_time if generation_time > 0 else 1e-6,
            )

        return outputs[0] if len(outputs) == 1 else outputs, None, None


if __name__ == "__main__":
    # Single example generation
    model = QAgent()
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": "A",
            "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
        }}
    """

    response, tl, tm = model.generate_response(
        prompt,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("Single example response:")
    print("Response: ", response)
    print(
        f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec"
    )
    print("+-------------------------------------------------\n\n")

    # Multi example generation
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, tm = model.generate_response(
        prompts,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("\nMulti example responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i+1}: {resp}")
    print(
        f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec"
    )
