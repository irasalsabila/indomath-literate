import torch
import pandas as pd
import json
import re, csv
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from unsloth import FastLanguageModel
import math
from collections import Counter
import argparse
import openai

language_prompts = {
        "bn": "আসুন ধাপে ধাপে চিন্তা করি।",
        "de": "Denken wir Schritt für Schritt.",
        "en": "Let's think step by step.",
        "es": "Pensemos paso a paso.",
        "fr": "Réfléchissons étape par étape.",
        "ja": "段階的に考えてみましょう。",
        "ru": "Давайте думать поэтапно.",
        "sw": "Hebu fikiria hatua kwa hatua.",
        "te": "అంచెలంచెలుగా ఆలోచిద్దాం.",
        "th": "ลองคิดทีละขั้นตอน",
        "zh": "让我们一步步思考。",
    }

negative_prompts = {
    "en": "Solve the following questions: {question}\nThe following reasoning has an inconsistency:\n{original_response}\n{update_wait1}",
    "bn": "নিম্নলিখিত প্রশ্নগুলি সমাধান করুন: {question}\nনিম্নলিখিত যুক্তিটিতে একটি অসঙ্গতি রয়েছে:\n{original_response}\n{update_wait1}",
    "de": "Los die folgenden Fragen: {question}\nDie folgende Argumentation enthält eine Inkonsistenz:\n{original_response}\n{update_wait1}",
    "es": "Resuelve las siguientes preguntas: {question}\nEl siguiente razonamiento tiene una inconsistencia:\n{original_response}\n{update_wait1}",
    "fr": "Résolvez les questions suivantes: {question}\nLe raisonnement suivant présente une incohérence :\n{original_response}\n{update_wait1}",
    "ja": "次の質問を解いてください: {question}\n次の推論には矛盾があります:\n{original_response}\n{update_wait1}",
    "ru": "Решите следующие вопросы: {question}\nСледующее рассуждение имеет несоответствие:\n{original_response}\n{update_wait1}",
    "sw": "Tatua maswali yafuatayo: {question}\nHoja ifuatayo ina hali ya kutofautiana:\n{original_response}\n{update_wait1}",
    "te": "కింది ప్రశ్నలను పరిష్కరించండి: {question}\nకింది తార్కికంలో అస్థిరత ఉంది:\n{original_response}\n{update_wait1}",
    "th": "แก้คำถามต่อไปนี้: {question}\nการใช้เหตุผลต่อไปนี้ไม่สอดคล้องกัน:\n{original_response}\n{update_wait1}",
    "zh": "解答以下问题: {question}\n以下推理存在不一致之处：\n{original_response}\n{update_wait1}",
}

update_wait_messages = {
    "en": "Wait …",
    "bn": "অপেক্ষা করুন …",
    "de": "Warte …",
    "es": "Espera …",
    "fr": "Attendez …",
    "ja": "待ってください …",
    "ru": "Подождите …",
    "sw": "Subiri …",
    "te": "ఆగండి …",
    "th": "รอสักครู่ …",
    "zh": "等待 …",
}

update_wait_again = {
    "en": "I made a mistake; try another way…",
    "bn": "আমি ভুল করেছি; অন্যভাবে চেষ্টা করুন…",
    "de": "Ik heb een fout gemaakt. Probeer het op een andere manier…",
    "es": "Cometí un error; prueba de otra manera…",
    "fr": "J'ai fait une erreur, essayez une autre méthode...",
    "ja": "間違いを犯しました。別の方法を試してください...",
    "ru": "Я допустил ошибку. Попробуйте по-другому...",
    "sw": "Nilifanya makosa; jaribu njia nyingine...",
    "te": "నేను పొరపాటు చేసాను; వేరే విధంగా ప్రయత్నించండి...",
    "th": "ฉันทำผิดแล้ว ลองวิธีอื่นดู...",
    "zh": "我犯了一个错误；请尝试另一种方法......",
}

LANGUAGES = ["en", "es", "ja", "th", "bn", "te"]

def load_gpt_model():
    """Loads the GPT model and sets the API key."""
    try:
        with open("api_key.txt", "r") as file:
            openai.api_key = file.read().strip()
        return "gpt-4o"  # or "gpt-3.5-turbo"
    except Exception as e:
        print(f"Error loading API key: {e}")
        exit(1)

def load_sbert_model():
    """Loads the Sentence-BERT model."""
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def load_datasets():
    """Loads datasets for different languages."""
    languages = LANGUAGES
    return {lang: load_dataset("juletxara/mgsm", lang) for lang in languages}

def inspect_samples(datasets):
    """Displays a sample of training and test data for each language."""
    for lang, dataset in datasets.items():
        print(f"\nLanguage: {lang}")
        print("Sample Training Example:")
        print(dataset["train"][0])
        print("Sample Test Example:")
        print(dataset["test"][0])

def prepare_questions(datasets):
    """Prepares training and test questions by language."""
    train_questions_by_lang = {lang: [{"question": ex["question"], "answer": ex["answer"]} for ex in dataset["train"]] for lang, dataset in datasets.items()}
    test_questions_by_lang = {lang: [{"question": ex["question"]} for ex in dataset["test"]] for lang, dataset in datasets.items()}
    return train_questions_by_lang, test_questions_by_lang

def compute_train_embeddings(train_questions_by_lang, model_sbert):
    """Computes SBERT embeddings for training data."""
    return {lang: model_sbert.encode([ex["question"] for ex in train_questions], convert_to_tensor=True)
            for lang, train_questions in train_questions_by_lang.items()}

def get_top_few_shot_examples(test_question, test_lang, train_questions_by_lang, train_embeddings_by_lang, model_sbert, k=2):
    """Retrieves the most relevant few-shot examples using SBERT similarity."""
    if test_lang not in train_embeddings_by_lang:
        return []

    test_embedding = model_sbert.encode(test_question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(test_embedding, train_embeddings_by_lang[test_lang])[0]
    top_indices = torch.argsort(-similarities)[:k]

    return [{"question": train_questions_by_lang[test_lang][i]["question"], "answer": train_questions_by_lang[test_lang][i]["answer"]} for i in top_indices]

def generate_inference_prompts(test_questions_by_lang, train_questions_by_lang, train_embeddings_by_lang, 
                               model_sbert, language_prompts, test_index=0):
    """
    Creates inference prompts for a specified test question index across all languages.
    """
    inference_results = []

    for lang in test_questions_by_lang.keys():
        if lang not in train_questions_by_lang:
            continue
        
        test_samples = test_questions_by_lang[lang]
        if len(test_samples) <= test_index:  # Skip if index is out of range
            continue

        test_question = test_samples[test_index]["question"]
        few_shot_examples = get_top_few_shot_examples(test_question, lang, train_questions_by_lang, 
                                                      train_embeddings_by_lang, model_sbert, k=2)

        step_by_step_prompt = language_prompts.get(lang, "Let's think step by step.")
        few_shot_examples_text = "\n\n".join([f"{ex['question']}\nStep-by-Step Answer: {ex['answer']}\n" 
                                              for ex in few_shot_examples])

        prompt = f"""
        {few_shot_examples_text}

        Question: {test_question}
        {step_by_step_prompt}
        """.strip()

        inference_results.append({
            "language": lang,
            "test_question": test_question,
            "few_shot_prompt1": few_shot_examples[0] if few_shot_examples else None,
            "few_shot_prompt2": few_shot_examples[1] if len(few_shot_examples) > 1 else None,
            "final_prompt": prompt
        })

    return pd.DataFrame(inference_results)

def run_inference(df_inference_results, model_name):
    """Generates answers using OpenAI GPT."""
    final_prompts = df_inference_results["final_prompt"].tolist()
    generated_responses = []

    for prompt in final_prompts:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0.7
        )
        generated_text = response['choices'][0]['message']['content']
        generated_responses.append(generated_text)

    df_inference_results["generated_answer"] = generated_responses
    return df_inference_results

def extract_numbers(df_inference_results, language_prompts):
    """Filters generated responses and extracts numbers."""
    filtered_responses = []

    for lang, response in zip(df_inference_results["language"], df_inference_results["generated_answer"]):
        step_by_step_prompt = language_prompts.get(lang, "Let's think step by step.")
        match = re.search(re.escape(step_by_step_prompt) + r"(.*)", response, re.DOTALL)
        filtered_response = match.group(1).strip() if match else response
        filtered_responses.append(filtered_response)

    df_inference_results["original_response"] = filtered_responses
    return df_inference_results

def check_consistency(df_inference_results, correct_answers):
    """Checks if generated answers contain the correct numeric answer."""
    consistency_results = []

    for index, row in df_inference_results.iterrows():
        response = row["original_response"]
        lang = row["language"]
        correct_answer = correct_answers.get(lang, None)
        numbers = list(map(int, re.findall(r'\d+', response))) if response else []
        is_correct = correct_answer in numbers if correct_answer is not None else False
        status = "Correct" if is_correct else f"Incorrect"

        consistency_results.append({
            "language": lang,
            "original_response": response,
            "extracted_numbers": numbers,
            "expected_answer": correct_answer,
            "initial_status": status,
        })

    return pd.DataFrame(consistency_results)

def perform_self_revision(df_consistency, df_inference_results, model_name, 
                          negative_prompts, update_wait_messages, update_wait_again, 
                          max_waits=5, max_total_waits=15):
    """
    Performs self-revision for incorrect responses using iterative refinement.
    """

    revised_responses = []
    wait_counts = []
    new_statuses = []
    extracted_numbers_list = []

    incorrect_rows = df_consistency[df_consistency["initial_status"].str.startswith("Incorrect")]

    for index, row in incorrect_rows.iterrows():
        lang = row["language"]
        correct_answer = row["expected_answer"]
        question = df_inference_results.loc[index, "test_question"]
        original_response = row["original_response"]

        update_wait = update_wait_messages.get(lang, "Wait …")
        prompt_template = negative_prompts.get(lang, negative_prompts["en"])

        wait_count = 0
        is_correct = False

        while not is_correct and wait_count < max_waits:
            # Generate the full prompt
            full_prompt = prompt_template.format(
                question=question,
                original_response=original_response,
                update_wait1=update_wait
            )

            try:
                # Use GPT-4 to generate the revised response
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=128,
                    temperature=0.7
                )
                revised_response = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error during revision: {e}")
                revised_response = "Error"

            # Extract numbers from the revised response
            numbers = re.findall(r'\d+', revised_response)
            numbers = list(map(int, numbers)) if numbers else []

            # Check if the correct answer is present in the revised response
            is_correct = correct_answer in numbers

            if not is_correct:
                wait_count += 1
                # Update the prompt for the next iteration
                if wait_count == max_waits:
                    update_wait += " " + update_wait_again.get(lang, "I made a mistake; try another way…")
                elif wait_count < max_total_waits:
                    update_wait += " " + update_wait_messages.get(lang, "Wait …")
                else:
                    break
            else:
                break

        # Determine the final status
        new_status = "Correct" if is_correct else "Incorrect"

        revised_responses.append(revised_response)
        wait_counts.append(wait_count)
        new_statuses.append(new_status)
        extracted_numbers_list.append(numbers)

    # Construct the revised consistency DataFrame
    df_revised_consistency = pd.DataFrame({
        "language": incorrect_rows["language"].values,
        "original_response": incorrect_rows["original_response"].values,
        "revised_response": revised_responses,
        "wait_count": wait_counts,
        "extracted_numbers": extracted_numbers_list,
        "expected_answer": incorrect_rows["expected_answer"].values,
        "new_status": new_statuses,
    })

    return df_revised_consistency


def merge_consistency_results(df_consistency, df_revised_consistency):
    """
    Merges the initial consistency results with self-revision results.
    """

    if "expected_answer" not in df_revised_consistency.columns:
        df_revised_consistency["expected_answer"] = df_consistency["expected_answer"]

    df_final = df_consistency.merge(df_revised_consistency, on=["language", "original_response"], how="left")

    if "expected_answer" not in df_final.columns:
        df_final["expected_answer"] = df_consistency["expected_answer"]

    df_final["revised_response"] = df_final["revised_response"].fillna(df_final["original_response"])
    df_final["wait_count"] = df_final["wait_count"].fillna(0).astype(int)
    df_final["new_status"] = df_final["new_status"].fillna(df_final["initial_status"])

    expected_columns = ["language", "original_response", "revised_response", "expected_answer", 
                        "initial_status", "new_status", "wait_count"]

    existing_columns = [col for col in expected_columns if col in df_final.columns]
    df_final = df_final[existing_columns]

    return df_final

def compute_perplexity(text):
    """
    Computes an approximation of perplexity without using a pre-trained model.
    Uses Shannon entropy of token distributions to estimate fluency.
    """
    tokens = text.split()  
    token_counts = Counter(tokens)  
    total_tokens = len(tokens)

    if total_tokens == 0:
        return float("inf")  

    token_probs = {word: count / total_tokens for word, count in token_counts.items()}
    entropy = -sum(prob * math.log2(prob) for prob in token_probs.values())
    perplexity = 2 ** entropy
    return perplexity

def classify_errors(df_final):
    """Classifies errors into logical and numerical errors."""
    logical_errors = []
    numerical_errors = []
    revision_attempts = {}

    for _, row in df_final.iterrows():
        lang = row["language"]
        response = row["revised_response"]
        expected_answer = row["expected_answer"]
        
        extracted_numbers = list(map(int, re.findall(r'\d+', response))) if response else []

        # Logical Errors: If the response has reasoning but reaches the wrong conclusion
        if expected_answer not in extracted_numbers:
            logical_errors.append(lang)

        # Numerical Errors: If there are numbers but incorrect calculations
        if extracted_numbers and expected_answer not in extracted_numbers:
            numerical_errors.append(lang)

        # Count how many attempts were needed to revise the answer
        if row["new_status"] == "Correct":
            revision_attempts[lang] = revision_attempts.get(lang, 0) + row["wait_count"]

    return logical_errors, numerical_errors, revision_attempts

def compute_metrics(df_final):
    """
    Computes evaluation metrics such as accuracy and self-revision success rate.
    """
    total_responses = len(df_final)

    # Filter responses
    correct_responses = df_final[df_final["initial_status"] == "Correct"]
    correct_after_revision = df_final[df_final["new_status"] == "Correct"]
    incorrect_after_revision = df_final[
        (df_final["new_status"].str.startswith("Incorrect")) &  # Still incorrect
        ~(df_final["new_status"] == "Correct")  # Exclude corrected ones
    ]

    # Compute counts
    correct_responses_count = len(correct_responses)
    correct_after_revision_count = len(correct_after_revision)
    incorrect_after_revision_count = len(incorrect_after_revision)

    # Calculate final answer accuracy
    final_answer_accuracy = (correct_responses_count + correct_after_revision_count) / total_responses

    # Calculate self-revision rate (only considering previously incorrect responses)
    self_revision_rate = correct_after_revision_count / (total_responses - correct_responses_count) if (total_responses - correct_responses_count) > 0 else 0

    # perplexities = [compute_perplexity(resp) for resp in df_final["revised_response"] if resp]

    perplexity_per_language = {}
    for lang in df_final['language'].unique():
        lang_response = df_final[df_final["language"] == lang]["revised_response"].dropna().tolist()
        if lang_response:
            perplexities = [compute_perplexity(resp) for resp in lang_response]
            perplexity_per_language[lang] = sum(perplexities) / len(perplexities)

    all_perplexities = list(perplexity_per_language.values())
    # avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float("inf")
    perplexity_per_language["avg"] = sum(all_perplexities) / len(all_perplexities) if all_perplexities else float("inf")

    logical_errors, numerical_errors, revision_attempts = classify_errors(df_final)

    # Compute metrics per language
    correct_per_language = correct_responses.groupby("language").size().to_dict()
    correct_after_revision_per_language = correct_after_revision.groupby("language").size().to_dict()
    incorrect_per_language = incorrect_after_revision.groupby("language").size().to_dict()

    return {
        "total_responses": total_responses,
        "correct_responses": correct_responses_count,
        "correct_after_revision": correct_after_revision_count,
        "final_answer_accuracy": final_answer_accuracy,
        "self_revision_rate": self_revision_rate,
        "perplexity_per_language": perplexity_per_language,  # Perplexity for each language
        "logical_errors": logical_errors,
        "numerical_errors": numerical_errors,
        "revision_attempts": revision_attempts,
        "correct_per_language": correct_per_language,
        "correct_after_revision_per_language": correct_after_revision_per_language,
        "incorrect_per_language": incorrect_per_language,
    }

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run multilingual NLG evaluation.")
    
    parser.add_argument("--num_tests", type=int, default=2, help="Number of test cases to run.")
    parser.add_argument("--model", type=str, default="laihuiyuan/mCoT", help="Model name or path.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Custom dataset path (optional).")

    return parser.parse_args()

def main():
    args = parse_args()

    print("Loading models...")
    model_name = load_gpt_model()
    model_sbert = load_sbert_model()
    datasets = load_datasets()

    print("Preparing questions and embeddings...")
    train_questions_by_lang, test_questions_by_lang = prepare_questions(datasets)
    train_embeddings_by_lang = compute_train_embeddings(train_questions_by_lang, model_sbert)

    all_metrics = []

    for test_index in range(args.num_tests):
        print(f"\n🚀 Running Test {test_index + 1}/{args.num_tests}...\n")

        print("Generating inference prompts...")
        df_inference_results = generate_inference_prompts(test_questions_by_lang, train_questions_by_lang, 
                                                          train_embeddings_by_lang, model_sbert, language_prompts, 
                                                          test_index=test_index)

        print("Running inference...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name.to(device)
        df_inference_results = run_inference(df_inference_results, model_name)

        print("Filtering generated responses...")
        df_inference_results = extract_numbers(df_inference_results, language_prompts)

        print("Checking consistency...")
        correct_answers = {lang: datasets[lang]["test"][test_index]["answer_number"] for lang in datasets.keys()}
        df_consistency = check_consistency(df_inference_results, correct_answers)

        print("Performing self-revision for incorrect responses...")
        df_revised_consistency = perform_self_revision(
            df_consistency, df_inference_results, model_name, device,
            negative_prompts, update_wait_messages, update_wait_again
        )

        print("Merging consistency results...")
        df_final = merge_consistency_results(df_consistency, df_revised_consistency)
        
        # Save results for this test iteration
        df_final.to_csv(f"results_gpt/df_final_results_test_{test_index+1}.csv", index=False)
        print(f"✅ df_final saved to results_gpt/df_final_results_test_{test_index+1}.csv")

        print("Computing metrics...")
        results = compute_metrics(df_final)

        # Store metrics for this test iteration
        results["test_index"] = test_index + 1  # Add test index for tracking
        all_metrics.append(results)

        # Print evaluation metrics
        print("\n📊 **Evaluation Metrics:**")
        print(f"Total Responses: {results['total_responses']}")
        print(f"Correct Before Revision: {results['correct_responses']}")
        print(f"Correct After Revision: {results['correct_after_revision']}")
        print(f"Final Answer Accuracy: {results['final_answer_accuracy']:.2%}")
        print(f"Self-Revision Rate: {results['self_revision_rate']:.2%}\n")

        print("✅ **Per-Language Metrics:**")
        for lang, count in results["correct_per_language"].items():
            print(f"✔ {lang}: {count} correct before revision")

        for lang, count in results["correct_after_revision_per_language"].items():
            print(f"🔄 {lang}: {count} correct after revision")

        for lang, count in results["incorrect_per_language"].items():
            print(f"❌ {lang}: {count} still incorrect after revision")

        print(f"\n✅ Finished Test {test_index + 1}/5\n")

    metrics_file = "results_gpt/metrics_results.csv"
    with open(metrics_file, "w", newline="") as csvfile:
        fieldnames = ["test_index", "total_responses", "correct_responses", "correct_after_revision",
                      "final_answer_accuracy", "self_revision_rate", "perplexity_per_language", "logical_errors", "numerical_errors", "revision_attempts"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in all_metrics:
            writer.writerow({
                "test_index": result["test_index"],
                "total_responses": result["total_responses"],
                "correct_responses": result["correct_responses"],
                "correct_after_revision": result["correct_after_revision"],
                "final_answer_accuracy": result["final_answer_accuracy"],
                "self_revision_rate": result["self_revision_rate"],
                "perplexity_per_language": result['perplexity_per_language'],
                "logical_errors": result['logical_errors'],
                "numerical_errors": result['numerical_errors'],
                "revision_attempts": result['revision_attempts']
            })

    print(f"\n📁 ✅ All metrics saved to `{metrics_file}`!\n")

if __name__ == "__main__":
    main()